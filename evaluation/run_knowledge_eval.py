"""
evaluation/run_knowledge_eval.py

Knowledge Agent 노드(app/agents/knowledge.py)를 Gold Set 으로 E2E 평가하는 러너.

무엇을 하는가:
    - Gold Set(RAG/SQL/multi)을 feature/gold-set 브랜치에서 "읽기 전용"으로 로드한다.
      (현재 워킹트리에는 gold_set/*.json 을 커밋하지 않는다 — git show 로 직접 읽음.)
    - 각 gold 항목의 query 로 실제 knowledge_node(state) 를 호출한다.
      → 내부적으로 ReAct 에이전트가 vector_rag / graph_rag / text_to_sql tool 을
        스스로 선택·호출하는 진짜 경로를 그대로 탄다(E2E).
    - 결과를 아래 지표로 채점한다:
        1) route_correct : gold 의 route_type 에 맞는 tool 을 실제로 호출했는가
        2) char_f1       : 최종 답변과 expected_answer 의 문자 bigram F1(한국어 견고)
        3) number_recall : expected_answer 의 숫자 사실이 답변에 나타난 비율(SQL 사실 검증)
        4) errored       : 노드가 예외로 실패했는가

전제(E2E 실행 스택이 모두 떠 있어야 함):
    - Ollama(11434, OPENAI_BASE_URL)      : knowledge_node 의 qwen2.5:1.5b ReAct LLM
    - Backend MCP 서버(9000, MCP_SERVER_URL): vector/graph/text2sql tool 실제 검색
    - Qdrant / Neo4j / vehicle_data.db     : 위 tool 들의 백엔드 스토어
    스택이 꺼져 있으면 각 항목은 errored=True 로 집계되고, 러너는 죽지 않는다.

실행:
    .venv/bin/python -m evaluation.run_knowledge_eval                 # RAG+SQL+multi 전체
    .venv/bin/python -m evaluation.run_knowledge_eval --sets rag sql  # 일부만
    .venv/bin/python -m evaluation.run_knowledge_eval --limit 5       # 세트별 앞 5건만(빠른 점검)
    .venv/bin/python -m evaluation.run_knowledge_eval --gold-ref origin/feature/gold-set
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()  # app.core.config 가 import 시점에 os.getenv 를 읽으므로 가장 먼저 실행

from langchain_core.messages import HumanMessage

from app.agents.knowledge import knowledge_node

# gold-set 파일명 ↔ route 대응. tool 세트는 execution 노드용이라 여기선 제외.
GOLD_FILES = {
    "rag": "evaluation/gold_set/RAG_gold_set.json",
    "sql": "evaluation/gold_set/SQL_gold_set.json",
    "multi": "evaluation/gold_set/multi_gold_set.json",
}

# knowledge_node 가 노출하는 tool 이름 분류
RETRIEVAL_TOOLS = {"vector_rag_search", "graph_rag_search"}
SQL_TOOLS = {"text_to_sql_query"}


# ---------------------------------------------------------------------------
# Gold Set 로드 (읽기 전용: git show 로 브랜치 blob 을 직접 읽음)
# ---------------------------------------------------------------------------

def load_gold_set(set_name: str, gold_ref: str) -> List[Dict[str, Any]]:
    path = GOLD_FILES[set_name]
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{gold_ref}:{path}"],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace").strip()
        raise SystemExit(
            f"[gold-set 로드 실패] git show {gold_ref}:{path}\n{stderr}\n"
            f"→ 브랜치가 있는지 확인: git fetch origin {gold_ref.split('/')[-1]}"
        )
    return json.loads(raw)


# ---------------------------------------------------------------------------
# 채점 유틸
# ---------------------------------------------------------------------------

def _char_bigrams(text: str) -> List[str]:
    # 공백/문장부호 제거 후 문자 bigram. 한국어 형태소 경계에 견고한 근사치.
    s = re.sub(r"[\s\W]+", "", text or "")
    return [s[i : i + 2] for i in range(len(s) - 1)] if len(s) >= 2 else list(s)


def char_bigram_f1(pred: str, gold: str) -> float:
    p, g = _char_bigrams(pred), _char_bigrams(gold)
    if not p or not g:
        return 0.0
    from collections import Counter

    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(cp.values())
    recall = overlap / sum(cg.values())
    return 2 * precision * recall / (precision + recall)


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def number_recall(pred: str, gold: str) -> Optional[float]:
    # expected_answer 의 숫자 사실(연료 45.3, 공기압 33 등)이 답변에 나타난 비율.
    gold_nums = set(_NUM_RE.findall(gold or ""))
    if not gold_nums:
        return None  # 숫자 사실이 없는 항목은 이 지표에서 제외
    pred_nums = set(_NUM_RE.findall(pred or ""))
    return len(gold_nums & pred_nums) / len(gold_nums)


def extract_called_tools(new_messages: List[Any]) -> List[str]:
    """knowledge_node 가 반환한 new_messages 에서 ReAct 가 호출한 tool 이름을 모은다."""
    names: List[str] = []
    for m in new_messages:
        for tc in getattr(m, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                names.append(name)
    return names


def route_correct(route_type: str, tools_called: set) -> bool:
    used_retrieval = bool(tools_called & RETRIEVAL_TOOLS)
    used_sql = bool(tools_called & SQL_TOOLS)
    if route_type == "rag":
        return used_retrieval
    if route_type == "sql":
        return used_sql
    if route_type == "multi":
        return used_retrieval and used_sql
    return False


# ---------------------------------------------------------------------------
# 한 항목 평가
# ---------------------------------------------------------------------------

async def eval_item(item: Dict[str, Any]) -> Dict[str, Any]:
    state = {
        "messages": [HumanMessage(content=item["query"])],
        "plan": [],
        "context_data": {},
        "error_count": {},
    }
    started = time.perf_counter()
    errored, error_msg = False, ""
    answer = ""
    tools_called: List[str] = []

    try:
        result = await knowledge_node(state)
        answer = result.get("context_data", {}).get("last_knowledge_result", "") or ""
        tools_called = extract_called_tools(result.get("messages", []))
        # 노드가 잡아서 feedback 으로 돌려준 실패도 실패로 집계
        if result.get("feedback"):
            errored, error_msg = True, str(result["feedback"])[:200]
    except Exception as exc:  # 방어적: 스택 미기동 등
        errored, error_msg = True, f"{type(exc).__name__}: {exc}"[:200]

    latency = time.perf_counter() - started
    gold = item.get("expected_answer", "") or ""
    return {
        "id": item.get("id"),
        "route_type": item.get("route_type"),
        "query": item.get("query"),
        "answer": answer,
        "expected_answer": gold,
        "tools_called": tools_called,
        "route_correct": route_correct(item.get("route_type", ""), set(tools_called)),
        "char_f1": round(char_bigram_f1(answer, gold), 3),
        "number_recall": number_recall(answer, gold),
        "errored": errored,
        "error_msg": error_msg,
        "latency_s": round(latency, 2),
    }


# ---------------------------------------------------------------------------
# 집계 & 리포트
# ---------------------------------------------------------------------------

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def agg(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(subset)
        if n == 0:
            return {"n": 0}
        ok = [r for r in subset if not r["errored"]]
        nr = [r["number_recall"] for r in subset if r["number_recall"] is not None]
        return {
            "n": n,
            "errored": sum(r["errored"] for r in subset),
            "route_acc": round(sum(r["route_correct"] for r in subset) / n, 3),
            "char_f1": round(sum(r["char_f1"] for r in subset) / n, 3),
            "number_recall": round(sum(nr) / len(nr), 3) if nr else None,
            "avg_latency_s": round(sum(r["latency_s"] for r in ok) / len(ok), 2) if ok else None,
        }

    by_route = {}
    for rt in ("rag", "sql", "multi"):
        by_route[rt] = agg([r for r in rows if r["route_type"] == rt])
    return {"overall": agg(rows), "by_route": by_route}


def print_report(summary: Dict[str, Any]) -> None:
    def line(label: str, s: Dict[str, Any]) -> str:
        if s.get("n", 0) == 0:
            return f"  {label:<8} (없음)"
        return (
            f"  {label:<8} n={s['n']:<4} err={s['errored']:<3} "
            f"route_acc={s['route_acc']:<6} char_f1={s['char_f1']:<6} "
            f"num_recall={s['number_recall']} avg_latency={s['avg_latency_s']}s"
        )

    print("\n" + "=" * 78)
    print("Knowledge Node · Gold Set 평가 결과")
    print("=" * 78)
    for rt in ("rag", "sql", "multi"):
        print(line(rt, summary["by_route"][rt]))
    print("-" * 78)
    print(line("OVERALL", summary["overall"]))
    print("=" * 78)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def run(sets: List[str], gold_ref: str, limit: Optional[int], out_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for set_name in sets:
        items = load_gold_set(set_name, gold_ref)
        if limit:
            items = items[:limit]
        print(f"[{set_name}] {len(items)}건 평가 중...")
        for i, item in enumerate(items, 1):
            row = await eval_item(item)
            flag = "ERR " if row["errored"] else "OK  " if row["route_correct"] else "MISS"
            print(f"  ({i}/{len(items)}) {row['id']} {flag} "
                  f"f1={row['char_f1']} tools={row['tools_called']}")
            rows.append(row)

    summary = summarize(rows)
    print_report(summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n상세 리포트 저장: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Knowledge Node Gold Set 평가 러너")
    ap.add_argument("--sets", nargs="+", default=["rag", "sql", "multi"],
                    choices=list(GOLD_FILES.keys()), help="평가할 gold set 종류")
    ap.add_argument("--gold-ref", default="origin/feature/gold-set",
                    help="gold_set/*.json 을 읽어올 git ref (읽기 전용)")
    ap.add_argument("--limit", type=int, default=None, help="세트별 앞 N건만 평가")
    ap.add_argument("--out", default="evaluation/reports/knowledge_eval.json",
                    help="상세 리포트 저장 경로")
    args = ap.parse_args()

    asyncio.run(run(args.sets, args.gold_ref, args.limit, Path(args.out)))


if __name__ == "__main__":
    main()
