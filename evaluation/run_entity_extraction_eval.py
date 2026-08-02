"""
evaluation/run_entity_extraction_eval.py

app/memory/entity.py 의 extract_preferences() 프롬프트를 미니 평가셋(gold set)으로
채점하는 러너. entity 메모리 프롬프트 개선(Round 0 baseline → Round N 개선) 실험용.

무엇을 하는가:
    - evaluation/entity_extraction_mini_set.json (utterance/expected 쌍, 19건)을 로드.
    - 각 utterance 로 실제 extract_preferences() 를 호출한다
      (finalize.py 의 _get_extractor_llm() 과 동일하게 ChatOpenAI(EXTRACTOR_MODEL_NAME),
       OPENAI_BASE_URL 로 로컬 Ollama 를 가리키는 실제 LLM 호출 — mock 없음).
    - 완전 일치 대신 세 가지를 각각 O/X 로 채점한다:
        (a) key_recall   : expected 의 모든 키가 실제 출력에도 등장했는가
        (b) no_extra_key : expected 에 없는 키를 임의로 만들어내지 않았는가
        (c) negation_ok  : expected 에서 null 인 키가 실제로도 명시적으로 null 로 나왔는가
                            (해당 케이스에 negation 대상이 없으면 "N/A")
      case_pass = (a) and (b) and ((c) or N/A) — 세 기준을 모두 만족해야 케이스 전체 통과.

전제:
    - Ollama(11434, OPENAI_BASE_URL) 가 떠 있어야 함(qwen2.5:1.5b, EXTRACTOR_MODEL_NAME).
    - 이 스크립트는 순수 함수(extract_preferences) 단위 평가라 그래프/서버 불필요.

실행:
    .venv/Scripts/python.exe -m evaluation.run_entity_extraction_eval --round round0
    .venv/Scripts/python.exe -m evaluation.run_entity_extraction_eval --round round1 --out evaluation/reports/entity_extraction_round1.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949) 유니코드 출력 깨짐/충돌 방지

from dotenv import load_dotenv

load_dotenv()  # app.core.config 가 import 시점에 os.getenv 를 읽으므로 가장 먼저 실행

from app.core.config import EXTRACTOR_MODEL_NAME
from app.memory.entity import extract_preferences

DEFAULT_GOLD_PATH = "evaluation/entity_extraction_mini_set.json"
DEFAULT_OUT_PATH = "evaluation/reports/entity_extraction_eval.json"


def load_gold_set(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=EXTRACTOR_MODEL_NAME, temperature=0.0)


async def eval_item(item: Dict[str, Any], llm) -> Dict[str, Any]:
    expected: Dict[str, Any] = item.get("expected", {})
    expected_keys = set(expected.keys())
    negation_keys = {k for k, v in expected.items() if v is None}

    started = time.perf_counter()
    errored, error_msg = False, ""
    actual: Dict[str, Any] = {}
    try:
        actual = await extract_preferences(item["utterance"], llm)
    except Exception as exc:  # 방어적: LLM 미기동 등
        errored, error_msg = True, f"{type(exc).__name__}: {exc}"[:200]
    latency = time.perf_counter() - started

    actual_keys = set(actual.keys())

    key_recall_ok = expected_keys.issubset(actual_keys)
    no_extra_key_ok = actual_keys.issubset(expected_keys)
    if negation_keys:
        negation_ok: Any = all(k in actual and actual[k] is None for k in negation_keys)
    else:
        negation_ok = "N/A"

    case_pass = key_recall_ok and no_extra_key_ok and (negation_ok in (True, "N/A"))

    return {
        "id": item.get("id"),
        "category": item.get("category"),
        "utterance": item["utterance"],
        "expected": expected,
        "actual": actual,
        "key_recall_ok": key_recall_ok,
        "no_extra_key_ok": no_extra_key_ok,
        "negation_ok": negation_ok,
        "case_pass": case_pass,
        "errored": errored,
        "error_msg": error_msg,
        "latency_s": round(latency, 2),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def rate(key: str, applicable_only: bool = False) -> float:
        vals = [r[key] for r in rows if not applicable_only or r[key] != "N/A"]
        vals = [v for v in vals if v != "N/A"]
        return round(sum(bool(v) for v in vals) / len(vals), 3) if vals else None

    n = len(rows)
    by_category: Dict[str, Any] = {}
    categories = sorted({r["category"] for r in rows})
    for cat in categories:
        sub = [r for r in rows if r["category"] == cat]
        by_category[cat] = {
            "n": len(sub),
            "case_pass_rate": round(sum(r["case_pass"] for r in sub) / len(sub), 3),
        }

    return {
        "overall": {
            "n": n,
            "errored": sum(r["errored"] for r in rows),
            "key_recall_rate": rate("key_recall_ok"),
            "no_extra_key_rate": rate("no_extra_key_ok"),
            "negation_rate": rate("negation_ok", applicable_only=True),
            "case_pass_rate": round(sum(r["case_pass"] for r in rows) / n, 3) if n else None,
        },
        "by_category": by_category,
    }


def print_report(round_label: str, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 90)
    print(f"entity extraction 미니 평가셋 결과 — {round_label}")
    print("=" * 90)
    print(f"{'id':<38}{'a(recall)':<11}{'b(no_extra)':<13}{'c(negation)':<13}{'PASS':<6}")
    print("-" * 90)
    for r in rows:
        def mark(v: Any) -> str:
            if v == "N/A":
                return "N/A"
            return "O" if v else "X"

        print(
            f"{r['id']:<38}{mark(r['key_recall_ok']):<11}{mark(r['no_extra_key_ok']):<13}"
            f"{mark(r['negation_ok']):<13}{'O' if r['case_pass'] else 'X':<6}"
        )
    print("-" * 90)
    o = summary["overall"]
    print(
        f"OVERALL n={o['n']} errored={o['errored']} "
        f"key_recall={o['key_recall_rate']} no_extra_key={o['no_extra_key_rate']} "
        f"negation={o['negation_rate']} case_pass_rate={o['case_pass_rate']}"
    )
    print("=" * 90)


async def run(gold_path: Path, out_path: Path, round_label: str, limit: int | None) -> None:
    items = load_gold_set(gold_path)
    if limit:
        items = items[:limit]

    llm = _get_llm()
    print(f"[{round_label}] {len(items)}건 평가 중 (model={EXTRACTOR_MODEL_NAME})...")

    rows = []
    for i, item in enumerate(items, 1):
        row = await eval_item(item, llm)
        flag = "ERR " if row["errored"] else "OK  " if row["case_pass"] else "MISS"
        print(f"  ({i}/{len(items)}) {row['id']} {flag} actual={row['actual']}")
        rows.append(row)

    summary = summarize(rows)
    print_report(round_label, summary, rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"round": round_label, "model": EXTRACTOR_MODEL_NAME, "summary": summary, "rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n상세 리포트 저장: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="entity extraction 미니 평가셋 러너")
    ap.add_argument("--gold", default=DEFAULT_GOLD_PATH, help="평가셋 JSON 경로")
    ap.add_argument("--out", default=DEFAULT_OUT_PATH, help="상세 리포트 저장 경로")
    ap.add_argument("--round", default="round0", help="라운드 라벨 (리포트에 기록)")
    ap.add_argument("--limit", type=int, default=None, help="앞 N건만 평가 (빠른 점검용)")
    args = ap.parse_args()

    asyncio.run(run(Path(args.gold), Path(args.out), args.round, args.limit))


if __name__ == "__main__":
    main()
