"""
evaluation/run_tool_eval.py

Execution Agent(app/agents/execution.py 의 run_execution)를 tool_gold_set 으로
E2E 평가하는 러너.

무엇을 하는가:
    - tool Gold Set 을 evaluation/_gold.py 의 load_gold_set() 으로 로드한다
      (워킹트리 우선, 없으면 git ref 폴백 — provenance 로 어느 쪽인지 리포트에 남긴다).
    - 각 gold 항목의 query 를 단일 plan step 으로 삼아 실제 run_execution(state) 를
      호출한다 → 내부적으로 LLM 이 tool/파라미터를 추출하고 MCP 서버(vehicle 12종)를
      실제로 호출하는 진짜 경로를 그대로 탄다(E2E).
    - 계획서 G3(Tool명 + 파라미터 Exact Match) 기준으로 채점한다.

run_knowledge_eval.py 와의 차이:
    knowledge_node 는 tool 호출을 LangChain 메시지의 `.tool_calls` 속성에 남기지만,
    run_execution 은 반환 dict 의 `tool_calls` 리스트에 이미 구조화된 형태
    ({"tool", "params", "result", "status", ["error_type"], ["error_msg"]})로 직접 담아
    반환한다(execution.py:239-249). 그래서 run_knowledge_eval.py 의 extract_called_tools()
    를 확장하지 않고 별도 함수 extract_tool_invocations() 를 둔다.

계획서 §5 error_type 계약과 이 경로의 실제 한계:
    call_mcp_tool_once(app/core/mcp_client.py)가 만드는 error_type 은 "timeout" 또는
    "parameter" 뿐이다. "sql" 은 text2sql 전용이라 이 경로엔 없다. 또한 LLM 이 고른
    tool 이름이 MCP_TOOLS 에 없을 때(execution.py:192-203) run_execution 은 1회
    재추출을 시도하고, 그래도 실패하면 tool_calls 에 아무 항목도 남기지 않고 그 plan
    step 을 조용히 건너뛴다 — 즉 "LLM 이 tool 없음으로 판단(null)"과 "재추출까지 실패한
    invalid_tool"이 반환값만으로는 구분되지 않는다. 이 러너는 black-box 로 run_execution
    을 호출하므로 이 둘을 모두 failure="no_tool_call" 로 통합한다. error_type_dist 는
    timeout/parameter/invalid_tool/sql 4개 키를 항상 갖지만, invalid_tool/sql 은 이 경로
    에서 구조적으로 항상 0이다.

알려진 stale gold 항목(STALE_GOLD_IDS, 6건) — result_exact 채점은 하되 overall/by_tool
집계에서는 제외하고 known_stale 로 별도 표시한다. 근거는 두 갈래로 서로 다르다:
    - tool_022~025(get_vehicle_status): 실제 응답 템플릿(mcp_server.py:82-92)이 항상
      "타이어 압력(...)...  경고등 ..." 절을 문장 끝에 붙이는데, gold expected_answer 는
      "...주행 모드 normal."에서 끝나고 이 절 자체가 없다. vehicle_state 값이 무엇이든
      영구적으로 문자열이 일치할 수 없는 구조적 문제.
    - tool_056/057(query_dashboard, speed/rpm): 구조적 문제가 아니다 — 이 두 metric 은
      애초에 tire_pressure/warning_lights 를 포함하지 않는 짧은 문장이다. 실제 원인은
      gold 가 speed=0/rpm=0 을 기대하지만 실제 초기값(app/services/vehicle.py:26-27 의
      _DEFAULTS)이 speed=60.0/rpm=2000 인 상태 값 드리프트다. vehicle_state 가 우연히
      0 이 되면 일치할 수도 있어 tool_022~025 와 달리 "영구 불일치"는 아니다.

전제(E2E 실행 스택이 모두 떠 있어야 함):
    - Ollama(11434, OPENAI_BASE_URL)       : execution_node 의 tool 추출용 LLM
    - Backend MCP 서버(9000, MCP_SERVER_URL): vehicle tool 12종 실제 실행
    스택이 꺼져 있으면 각 항목은 errored=True 로 집계되고, 러너는 죽지 않는다.

실행:
    .venv/bin/python -m evaluation.run_tool_eval                 # 전체
    .venv/bin/python -m evaluation.run_tool_eval --limit 5       # 앞 5건만(빠른 점검)
    .venv/bin/python -m evaluation.run_tool_eval --item-timeout 90
    .venv/bin/python -m evaluation.run_tool_eval --gold-ref origin/feature/gold-set
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()  # app.core.config 가 import 시점에 os.getenv 를 읽으므로 가장 먼저 실행

from app.agents.execution import MCP_TOOLS, run_execution

from evaluation._gold import load_gold_set

# mcp_server.py 실제 시그니처의 기본값. control_climate 만 on=True 기본값을 가짐.
TOOL_DEFAULTS: Dict[str, Dict[str, Any]] = {"control_climate": {"on": True}}

# run_execution 이 실제로 만들 수 있는 §5 error_type. invalid_tool/sql 은 이 경로에서
# 구조적으로 발생하지 않지만(모듈 docstring 참고), 리포트에는 항상 4개 키를 노출한다.
_ERROR_TYPES = ("timeout", "parameter", "invalid_tool", "sql")

# 응답 문자열(result_exact) 채점에서 제외할 알려진 stale gold 항목.
# 근거는 모듈 docstring "알려진 stale gold 항목" 절 참고 — tool_022~025 와 tool_056/057
# 은 서로 다른 이유로 결과 문자열이 gold 와 일치하지 않는다.
STALE_GOLD_IDS = {"tool_022", "tool_023", "tool_024", "tool_025", "tool_056", "tool_057"}


# ---------------------------------------------------------------------------
# tool_calls 추출 (run_execution 반환 dict 전용 — extract_called_tools() 와 무관)
# ---------------------------------------------------------------------------

def extract_tool_invocations(
    result: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any], str, str, str]]:
    """run_execution 반환값의 tool_calls 리스트에서 (name, params, status, error_type, result) 추출."""
    calls = result.get("tool_calls") or []
    return [
        (
            c["tool"],
            c.get("params", {}),
            c.get("status", ""),
            c.get("error_type", ""),
            c.get("result", ""),
        )
        for c in calls
    ]


# ---------------------------------------------------------------------------
# 채점 (계획서 G3: Tool명 + 파라미터 Exact Match)
# ---------------------------------------------------------------------------

def score_tool_item(
    expected_name: str,
    expected_params: Dict[str, Any],
    calls: List[Tuple[str, Dict[str, Any], str, str, str]],
    expected_answer: str = "",
) -> Dict[str, Any]:
    """단일 gold 항목을 채점한다.

    calls 는 extract_tool_invocations() 의 반환값(호출 순서대로)이며, 첫 번째 호출만 본다
    (plan 이 단일 step 이므로 run_execution 이 성공적으로 처리하면 tool_calls 는 최대 1건).

    result_exact 는 tool_correct 이고 params_effective 도 True 일 때만 계산한다 — 즉
    "tool도 맞고 파라미터도(기본값 흡수 포함) 맞은 경우에만 응답 문자열까지 정확히
    일치하는지"를 보는 지표다. 그 외(no_tool_call, status=="error", hallucinated_tool,
    wrong_tool, param_mismatch)는 응답 문자열 비교 자체가 의미 없으므로 전부 None.

    Returns:
        {"tool_correct", "params_exact", "params_effective", "result_exact", "failure", "error_type"}
    """
    if not calls:
        return {
            "tool_correct": False,
            "params_exact": False,
            "params_effective": False,
            "result_exact": None,
            "failure": "no_tool_call",
            "error_type": None,
        }

    name, params, status, error_type, result_text = calls[0]

    if status == "error":
        return {
            "tool_correct": False,
            "params_exact": False,
            "params_effective": False,
            "result_exact": None,
            "failure": error_type or "unknown_error",
            "error_type": error_type or None,
        }

    if name not in MCP_TOOLS:
        return {
            "tool_correct": False,
            "params_exact": False,
            "params_effective": False,
            "result_exact": None,
            "failure": "hallucinated_tool",
            "error_type": None,
        }

    if name != expected_name:
        return {
            "tool_correct": False,
            "params_exact": False,
            "params_effective": False,
            "result_exact": None,
            "failure": "wrong_tool",
            "error_type": None,
        }

    params_exact = params == expected_params
    effective = {**TOOL_DEFAULTS.get(name, {}), **params}
    params_effective = all(effective.get(k) == v for k, v in expected_params.items())
    result_exact = result_text.strip() == expected_answer.strip() if params_effective else None

    return {
        "tool_correct": True,
        "params_exact": params_exact,
        "params_effective": params_effective,
        "result_exact": result_exact,
        "failure": None if params_effective else "param_mismatch",
        "error_type": None,
    }


# ---------------------------------------------------------------------------
# 한 항목 평가
# ---------------------------------------------------------------------------

async def eval_item(item: Dict[str, Any], item_timeout: float) -> Dict[str, Any]:
    """gold 항목 하나를 run_execution(state) 로 E2E 실행하고 채점한다.

    run_execution 은 state["plan"](plan step 문자열 리스트)과 state["context_data"] 만
    읽으므로(execution.py:171-173), eval state 는 그 두 필드만 채운다.

    item_timeout 은 러너 자체의 방어적 타임아웃이며, 계획서 §5 의 tool 호출 타임아웃
    (call_mcp_tool_once 내부, MCP_TOOL_TIMEOUT)과는 별개 개념이다 — 그래서 여기서 발생하는
    failure 는 "eval_timeout"/"eval_exception"으로 §5 4종과 분리해서 표시한다.
    """
    state: Dict[str, Any] = {"plan": [item["query"]], "context_data": {}}

    started = time.perf_counter()
    errored, error_msg = False, ""
    calls: List[Tuple[str, Dict[str, Any], str, str, str]] = []
    score: Dict[str, Any] = {}

    try:
        result = await asyncio.wait_for(run_execution(state), timeout=item_timeout)
        calls = extract_tool_invocations(result)
        score = score_tool_item(
            item["expected_tool_name"],
            item["expected_tool_params"],
            calls,
            item.get("expected_answer", ""),
        )
    except asyncio.TimeoutError:
        errored, error_msg = True, f">{item_timeout}s"
        score = {
            "tool_correct": False,
            "params_exact": False,
            "params_effective": False,
            "result_exact": None,
            "failure": "eval_timeout",
            "error_type": None,
        }
    except Exception as exc:  # 방어적: 스택 미기동 등
        errored, error_msg = True, f"{type(exc).__name__}: {exc}"[:200]
        score = {
            "tool_correct": False,
            "params_exact": False,
            "params_effective": False,
            "result_exact": None,
            "failure": "eval_exception",
            "error_type": None,
        }

    latency = time.perf_counter() - started
    if calls:
        called_name, called_params, called_status, _called_error_type, called_result = calls[0]
    else:
        called_name, called_params, called_status, called_result = None, None, None, None

    return {
        "id": item.get("id"),
        "query": item.get("query"),
        "expected_tool_name": item.get("expected_tool_name"),
        "expected_tool_params": item.get("expected_tool_params"),
        "expected_answer": item.get("expected_answer"),
        "known_stale": item.get("id") in STALE_GOLD_IDS,
        "called_tool": called_name,
        "called_params": called_params,
        "called_status": called_status,
        "called_result": called_result,
        **score,
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
        # result_exact 는 tool_correct and params_effective 인 항목에서만 계산되므로(score_tool_item
        # 참고), result_exact_rate 는 "완전히 맞은 항목 중 응답 문자열까지 일치한 비율"이다 —
        # n 전체에 대한 비율이 아니다.
        re_vals = [r["result_exact"] for r in subset if r["result_exact"] is not None]
        return {
            "n": n,
            "tool_acc": round(sum(r["tool_correct"] for r in subset) / n, 3),
            "params_exact_rate": round(sum(r["params_exact"] for r in subset) / n, 3),
            "params_effective_rate": round(sum(r["params_effective"] for r in subset) / n, 3),
            "result_exact_rate": round(sum(re_vals) / len(re_vals), 3) if re_vals else None,
            "failure_dist": dict(Counter(r["failure"] for r in subset if r["failure"])),
            "avg_latency_s": round(sum(r["latency_s"] for r in subset) / n, 2),
        }

    # known_stale(STALE_GOLD_IDS) 은 result_exact 계산은 하되 overall/by_tool 에서는 뺀다
    # (모듈 docstring "알려진 stale gold 항목" 절 참고). error_type_dist 는 tool 호출
    # 성공/실패 여부와는 무관한 지표라 stale 여부와 상관없이 전체 rows 기준으로 집계한다.
    normal_rows = [r for r in rows if not r["known_stale"]]
    stale_rows = [r for r in rows if r["known_stale"]]

    by_tool = {tool: agg([r for r in normal_rows if r["expected_tool_name"] == tool]) for tool in MCP_TOOLS}

    error_type_dist: Dict[str, int] = {k: 0 for k in _ERROR_TYPES}
    for r in rows:
        et = r.get("error_type")
        if et in error_type_dist:
            error_type_dist[et] += 1

    known_stale = agg(stale_rows)
    known_stale["ids"] = sorted(r["id"] for r in stale_rows)

    return {
        "overall": agg(normal_rows),
        "by_tool": by_tool,
        "error_type_dist": error_type_dist,
        "known_stale": known_stale,
    }


def print_report(summary: Dict[str, Any]) -> None:
    def line(label: str, s: Dict[str, Any]) -> str:
        if s.get("n", 0) == 0:
            return f"  {label:<20} (없음)"
        return (
            f"  {label:<20} n={s['n']:<4} tool_acc={s['tool_acc']:<6} "
            f"exact={s['params_exact_rate']:<6} effective={s['params_effective_rate']:<6} "
            # result_exact 는 "완전히 맞은 항목(tool_correct and params_effective) 중
            # 응답 문자열까지 일치한 비율" — n 전체 대비 비율이 아님.
            f"result_exact={s['result_exact_rate']:<6} avg_latency={s['avg_latency_s']}s"
        )

    print("\n" + "=" * 88)
    print("Execution Agent(run_execution) · Tool Gold Set 평가 결과")
    print("=" * 88)
    for tool in MCP_TOOLS:
        print(line(tool, summary["by_tool"][tool]))
    print("-" * 88)
    print(line("OVERALL", summary["overall"]))
    print("-" * 88)
    print("  §5 error_type 분포:", summary["error_type_dist"])
    print("-" * 88)
    stale = summary["known_stale"]
    print(f"  known_stale(overall/by_tool 제외) ids={stale['ids']}")
    if stale.get("n", 0) > 0:
        print(f"    n={stale['n']} result_exact={stale['result_exact_rate']}")
    print("=" * 88)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def run(limit: Optional[int], item_timeout: float, gold_ref: str, out_path: Path) -> None:
    items, provenance = load_gold_set("tool", gold_ref)
    if limit:
        items = items[:limit]

    print(f"[tool] {len(items)}건 평가 중 (item-timeout={item_timeout}s, source={provenance['source']})...")

    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        row = await eval_item(item, item_timeout)
        flag = "ERR " if row["errored"] else "OK  " if row["tool_correct"] and row["params_effective"] else "MISS"
        print(
            f"  ({i}/{len(items)}) {row['id']} {flag} "
            f"expected={row['expected_tool_name']} called={row['called_tool']} "
            f"failure={row['failure']}"
        )
        rows.append(row)

    summary = summarize(rows)
    print_report(summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"provenance": provenance, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n상세 리포트 저장: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Execution Agent(run_execution) Tool Gold Set 평가 러너")
    ap.add_argument("--limit", type=int, default=None, help="앞 N건만 평가")
    ap.add_argument("--item-timeout", type=float, default=60.0, help="항목별 하드 타임아웃(초)")
    ap.add_argument("--gold-ref", default="origin/feature/gold-set",
                    help="워킹트리에 gold_set/tool_gold_set.json 이 없을 때 읽어올 git ref")
    ap.add_argument("--out", default="evaluation/reports/tool_eval.json",
                    help="상세 리포트 저장 경로")
    args = ap.parse_args()

    asyncio.run(run(args.limit, args.item_timeout, args.gold_ref, Path(args.out)))


if __name__ == "__main__":
    main()
