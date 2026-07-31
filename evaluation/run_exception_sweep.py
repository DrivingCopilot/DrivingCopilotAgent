"""
evaluation/run_exception_sweep.py

RAG_gold_set.json 의 모든 query 를 실제 knowledge_node(E2E: MCP→Qdrant/Neo4j→7B 융합)
경로에 태워, "나올 수 있는 모든 예외/실패 케이스"를 발굴·분류하는 스윕 하니스.

품질 채점(char_f1 등)이 목적인 run_knowledge_eval.py 와 달리, 이 러너는 오직
"이 쿼리가 어떤 방식으로 실패/이상 종료할 수 있는가"에 집중한다. 이번 세션의 주제인
'침묵(무응답) 버그' 및 knowledge 2단계(검색 1.5B → Fusion/CoT 7B) 리팩터가 실제
매뉴얼 쿼리 80건에서 어떤 예외를 남기는지 한 번에 훑는다.

분류 카테고리(우선순위 높은 것부터 하나로 배타 분류):
  - node_exception       : knowledge_node 호출 자체가 예외를 던짐(하드 크래시)
  - silent_empty         : 최종 답변이 비었는데도 실패 처리조차 안 됨(진짜 '침묵')
  - knowledge_failed     : 노드가 실패 계약(knowledge_failed=True / feedback)으로 종료
  - item_timeout         : 단일 쿼리가 하드 타임아웃(기본 120s) 초과
  - ok                   : 비어있지 않은 grounded 답변으로 정상 종료

추가로 각 항목에 '관찰 플래그'(배타 아님)를 붙인다:
  - hallucination_no_toolcall : 1.5B ReAct 가 tool 을 안 불러 결정적 검색 폴백이 발동
  - used_7b_fusion            : 복잡(관계형) 질의로 7B Graph Fusion/CoT 경로를 탐
  - fusion_weak_fallback      : 7B 융합 결과가 부실해 1.5B 답변으로 폴백
  - tool_timeout              : 검색 tool(주로 vector_rag_search)이 MCP 타임아웃

전제(실행 스택이 떠 있어야 함): MODEL_SERVER_URL(원격 7B/1.5B), MCP 서버(9000),
Qdrant/Neo4j/SQLite. 스택이 꺼져 있으면 대부분 knowledge_failed/tool_timeout 로 잡힌다.

실행:
    .venv/bin/python -m evaluation.run_exception_sweep                # 전체 80건
    .venv/bin/python -m evaluation.run_exception_sweep --limit 5      # 앞 5건(빠른 점검)
    .venv/bin/python -m evaluation.run_exception_sweep --item-timeout 90
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from app.agents.knowledge import knowledge_node
from app.graph import ws as _ws

GOLD_PATH = "evaluation/gold_set/RAG_gold_set.json"
REPORT_PATH = "evaluation/reports/rag_exception_sweep.json"

# knowledge_node 가 남기는 경로 로그 + WS status 메시지를 후킹해 관찰 플래그를 잡는다.
# ("Fusing graph+vector..."는 logger 가 아니라 WS status 로 나가므로 둘 다 캡처한다.)
_LOG_SIGNALS = {
    "hallucination_no_toolcall": "결정적 검색 폴백",
    "used_7b_fusion": "Fusing graph+vector",
    "fusion_weak_fallback": "7B 융합 결과가 부실",
    "tool_timeout": "MCP timeout",
}
_WATCHED_LOGGERS = ("app.agents.knowledge", "app.core.mcp_client", "app.agents.crag")


class _SignalCapture(logging.Handler):
    """한 쿼리 실행 동안 watched 로거의 메시지에서 신호 문자열을 탐지한다."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):
            self.messages.append(record.getMessage())

    def flags(self) -> dict[str, bool]:
        blob = "\n".join(self.messages)
        return {flag: (needle in blob) for flag, needle in _LOG_SIGNALS.items()}


def _looks_empty_or_malformed(text: str) -> bool:
    t = (text or "").strip()
    return (not t) or ("<tool_call>" in t)


def _is_prompt_echo(text: str) -> bool:
    """7B(VL) 융합이 답변 대신 입력 프롬프트 템플릿을 그대로 echo한 경우."""
    t = text or ""
    return ("[Retrieved Context]" in t) or t.lstrip().startswith("[User Query]")


async def sweep_item(item: dict[str, Any], item_timeout: float) -> dict[str, Any]:
    state = {
        "messages": [HumanMessage(content=item["query"])],
        "plan": [],
        "context_data": {},
        "error_count": {},
    }

    cap = _SignalCapture()
    loggers = [logging.getLogger(name) for name in _WATCHED_LOGGERS]
    prev_levels = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.addHandler(cap)
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)

    # WS status 메시지(예: "Fusing graph+vector context (7B CoT)...")도 신호로 캡처.
    orig_send = _ws.websocket_manager.send_status

    async def _capture_send(message: str):
        cap.messages.append(message)
        return await orig_send(message)

    _ws.websocket_manager.send_status = _capture_send

    started = time.perf_counter()
    category = "ok"
    detail = ""
    answer = ""
    knowledge_failed = False

    try:
        result = await asyncio.wait_for(knowledge_node(state), timeout=item_timeout)
        ctx = result.get("context_data", {}) or {}
        answer = ctx.get("last_knowledge_result", "") or ""
        knowledge_failed = bool(ctx.get("knowledge_failed"))
        feedback = result.get("feedback")

        if knowledge_failed or feedback:
            category = "knowledge_failed"
            detail = str(feedback or answer)[:200]
        elif _looks_empty_or_malformed(answer):
            # 실패 처리조차 안 되고 빈/깨진 답이 통과했다면 그것이 진짜 '침묵'.
            category = "silent_empty"
            detail = repr(answer)[:120]
        elif _is_prompt_echo(answer):
            # 7B 융합이 프롬프트를 그대로 echo한 malformed 답변이 사용자에게 노출됨.
            category = "malformed_echo"
            detail = repr(answer)[:120]
        else:
            category = "ok"
    except TimeoutError:
        category = "item_timeout"
        detail = f">{item_timeout}s"
    except Exception as exc:  # 하드 크래시
        category = "node_exception"
        detail = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        _ws.websocket_manager.send_status = orig_send
        for lg in loggers:
            lg.removeHandler(cap)
        for lg, lvl in prev_levels:
            lg.setLevel(lvl)

    latency = time.perf_counter() - started
    return {
        "id": item.get("id"),
        "query": item.get("query"),
        "category": category,
        "detail": detail,
        "knowledge_failed": knowledge_failed,
        "answer_len": len(answer),
        "answer_preview": (answer or "")[:160],
        "flags": cap.flags(),
        "latency_s": round(latency, 2),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG gold-set 예외 스윕")
    parser.add_argument("--limit", type=int, default=0, help="앞 N건만(0=전체)")
    parser.add_argument("--item-timeout", type=float, default=120.0)
    parser.add_argument("--gold", default=GOLD_PATH)
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args()

    items = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if args.limit > 0:
        items = items[: args.limit]

    print(f"[sweep] {len(items)}건 시작 (item-timeout={args.item_timeout}s)\n")

    results: list[dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        r = await sweep_item(item, args.item_timeout)
        results.append(r)
        flag_str = ",".join(f for f, on in r["flags"].items() if on) or "-"
        mark = "OK " if r["category"] == "ok" else "!! "
        print(
            f"{mark}[{i:>2}/{len(items)}] {r['id']:<8} {r['category']:<16} "
            f"{r['latency_s']:>6.1f}s flags={flag_str}  {r['query'][:30]}"
        )
        if r["category"] != "ok":
            print(f"        └ {r['detail']}")

    # 집계
    from collections import Counter

    cat_counts = Counter(r["category"] for r in results)
    flag_counts = Counter()
    for r in results:
        for f, on in r["flags"].items():
            if on:
                flag_counts[f] += 1

    exception_ids = {
        cat: [r["id"] for r in results if r["category"] == cat]
        for cat in cat_counts
        if cat != "ok"
    }

    summary = {
        "total": len(results),
        "category_counts": dict(cat_counts),
        "flag_counts": dict(flag_counts),
        "exception_ids": exception_ids,
        "silence_invariant_ok": cat_counts.get("silent_empty", 0) == 0,
        "results": results,
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("예외 스윕 요약")
    print("=" * 60)
    for cat, n in cat_counts.most_common():
        print(f"  {cat:<18} {n:>3}")
    print("  ---- 관찰 플래그 ----")
    for f, n in flag_counts.most_common():
        print(f"  {f:<26} {n:>3}")
    print(f"\n  침묵(silent_empty) 없음: {summary['silence_invariant_ok']}")
    if exception_ids:
        print("  예외 항목 ID:")
        for cat, ids in exception_ids.items():
            print(f"    {cat}: {', '.join(ids)}")
    print(f"\n리포트 저장: {args.report}")


if __name__ == "__main__":
    asyncio.run(main())
