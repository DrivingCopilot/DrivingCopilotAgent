"""
evaluation/run_ragas_eval.py

Knowledge Agent 노드(app/agents/knowledge.py)의 RAG 답변을 RAGAS 메트릭으로
평가하는 러너. 계획서 G4의 "Faithfulness > 0.85 (RAGAS 자동 계산)" 목표를
실제로 측정하기 위한 파이프라인이다.

무엇을 측정하는가 (RAGAS 0.4.x):
    - faithfulness                        : 답변의 각 주장이 검색된 context에 근거하는가 (환각 억제)
    - answer_relevancy (ResponseRelevancy): 답변이 질문에 얼마나 직접적으로 답하는가 (임베딩 기반)
    - context_precision (w/ reference)    : 검색된 context 중 정답에 유용한 것이 상위에 왔는가
    - context_recall                      : 정답을 뒷받침할 정보가 검색 context에 들어왔는가

판정(Judge) 백엔드 — 두 가지 선택:
    1) local (기본): 로컬 모델 서버(MODEL_SERVER_URL, OpenAI 호환)를 judge LLM으로,
       bge-m3(HuggingFace)를 judge 임베딩으로 사용. 완전 오프라인. 단, 판정 품질은
       로컬 7B 수준에 좌우된다(참고치로 활용).
    2) openai: OPENAI_API_KEY 가 있으면 GPT-4o(계획서 명시 Judge)로 채점. 임베딩은
       그래도 로컬 bge-m3를 쓴다(문서 정합성 유지 + 비용 절감).

두 단계로 분리 — E2E 생성(느림)과 RAGAS 채점(반복)을 떼어놨다:
    - 생성:  knowledge_node 를 gold query 로 E2E 실행 → (질문/답변/context/정답) 샘플 생성.
             --build-only 로 샘플만 만들어 --samples-out 에 저장할 수 있다.
    - 채점:  RAGAS 메트릭 실행. --score-from <samples.json> 으로 이미 만든 샘플을 재채점만
             할 수 있다(judge/메트릭만 바꿔 빠르게 반복).

전제(생성 단계에 한함 — 채점만 할 땐 스택 불필요):
    knowledge_node E2E 스택(모델 서버 + MCP 서버 + Qdrant/Neo4j/DB)이 떠 있어야 한다.
    run_knowledge_eval.py 와 동일. 스택이 꺼져 있으면 해당 항목은 스킵된다.

실행 예:
    # 로컬 judge, RAG 앞 10건 빠른 점검
    .venv/bin/python -m evaluation.run_ragas_eval --sets rag --limit 10

    # 샘플만 먼저 생성해 캐싱(E2E 1회) → 이후 judge만 바꿔 재채점
    # (RAGAS 는 RAG 세트 전용 — 복합/SQL 은 크로스 역량이라 E2E/Multi-Turn 으로 별도 검증)
    .venv/bin/python -m evaluation.run_ragas_eval --sets rag --build-only \
        --samples-out evaluation/reports/ragas_samples.json
    .venv/bin/python -m evaluation.run_ragas_eval \
        --score-from evaluation/reports/ragas_samples.json --judge openai
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()  # app.core.config 가 import 시점에 os.getenv 를 읽으므로 가장 먼저 실행

from langchain_core.messages import HumanMessage, ToolMessage

from app.agents.knowledge import _deterministic_retrieve, _is_unusable_result, knowledge_node
from app.core.config import MODEL_NAME, MODEL_SERVER_URL, QWEN_VL_MODEL_NAME
from evaluation._gold import load_gold_set

# RAGAS Faithfulness/Recall/Precision 은 계획서 G4에서 "RAG 품질 메트릭"으로 정의된다 —
# 답변이 '검색된 매뉴얼 context'에 근거하는지를 잰다. 따라서 순수 매뉴얼 QA인 RAG 세트에만
# 의미가 있다. 복합(multi)은 gold 스키마가 expected_sql + relevant_chunk_ids 를 둘 다 갖는
# 크로스 역량 질의로(예: "지금 공기압"=텔레메트리/execution + "표준 공기압"=매뉴얼/knowledge),
# 계획서상 G3 Multi-Turn 완수율(최종 VehicleState 일치)·G5 E2E 시나리오로 검증하도록 설계됐다.
# 이 러너는 knowledge_node 단독을 돌리므로 복합을 여기서 채점하면 (a) 다른 노드가 답할
# 라이브 데이터를 knowledge에 강제하고 (b) 복합 쿼리가 검색을 오염시켜 Faithfulness 를
# 구조적으로 과소 측정한다. 그래서 기본은 RAG 전용이다(복합은 --sets multi 로 강제 시 경고).
DEFAULT_SETS = ["rag"]
# 크로스 역량이라 knowledge 단독 RAGAS 로는 과소 측정되는 세트(강제 시 경고).
_CROSS_CAPABILITY_SETS = {"multi", "sql"}


# ---------------------------------------------------------------------------
# 1) 생성 단계 — knowledge_node E2E → RAGAS 샘플
# ---------------------------------------------------------------------------

def _contexts_from_messages(messages: List[Any]) -> List[str]:
    """knowledge_node 가 반환한 new_messages 에서 실제 사용된 검색 context를 뽑는다.

    각 ToolMessage(vector/graph/sql tool 출력) 하나를 context 한 조각으로 본다.
    에러·'결과 없음' 출력은 grounding 에 못 쓰므로 제외한다(knowledge 노드와 동일 기준).
    RAGAS 의 retrieved_contexts 로 그대로 넘긴다.
    """
    contexts: List[str] = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        if _is_unusable_result(content):
            continue
        contexts.append(content.strip())
    return contexts


async def build_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """gold 항목 하나를 knowledge_node 로 E2E 실행해 RAGAS 샘플 dict 를 만든다.

    retrieved_contexts는 우선 new_messages의 ToolMessage에서 뽑는다(1.5B ReAct가
    스스로 tool을 호출한 "react_grounded" 경로 — 실제 합성 입력과 정확히 일치).
    1.5B가 tool을 안 부르면 knowledge_node는 내부적으로 _deterministic_retrieve()
    안전망으로 넘어가는데, 이 경로는 tool_ctx 딕셔너리를 바로 합성에 쓰고
    new_messages에 ToolMessage를 남기지 않는다 — 그래서 메시지 기반 추출이
    비어버린다. 이 경우 knowledge_node가 내부에서 쓴 것과 동일한
    _deterministic_retrieve()를 여기서도 호출해 같은 context를 복원한다
    (knowledge_node가 실제로 합성에 사용한 것과 동일한 함수·같은 query).
    """
    state = {
        "messages": [HumanMessage(content=item["query"])],
        "plan": [],
        "context_data": {},
        "error_count": {},
    }
    started = time.perf_counter()
    answer, contexts, errored, error_msg = "", [], False, ""
    try:
        result = await knowledge_node(state)
        answer = result.get("context_data", {}).get("last_knowledge_result", "") or ""
        contexts = _contexts_from_messages(result.get("messages", []))
        if not contexts and answer and not result.get("feedback"):
            # react_grounded=False 로 deterministic 폴백을 탄 경우 — 같은 함수로 재구성.
            tool_ctx = await _deterministic_retrieve(item["query"])
            contexts = [c for c in (tool_ctx["graph"], tool_ctx["vector"]) if c]
        if result.get("feedback"):
            errored, error_msg = True, str(result["feedback"])[:200]
    except Exception as exc:  # 스택 미기동 등 — 러너는 죽지 않는다
        errored, error_msg = True, f"{type(exc).__name__}: {exc}"[:200]

    return {
        "id": item.get("id"),
        "route_type": item.get("route_type"),
        "user_input": item.get("query", ""),
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": item.get("expected_answer", "") or "",
        "errored": errored,
        "error_msg": error_msg,
        "latency_s": round(time.perf_counter() - started, 2),
    }


async def build_samples(sets: List[str], gold_ref: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for set_name in sets:
        items, prov = load_gold_set(set_name, gold_ref)
        if limit:
            items = items[:limit]
        print(f"[{set_name}] {len(items)}건 생성 중... (source={prov['source']})")
        for i, item in enumerate(items, 1):
            s = await build_sample(item)
            flag = "ERR " if s["errored"] else ("NOCTX" if not s["retrieved_contexts"] else "OK  ")
            print(f"  ({i}/{len(items)}) {s['id']} {flag} "
                  f"ctx={len(s['retrieved_contexts'])} ans_len={len(s['response'])}")
            samples.append(s)
    return samples


# ---------------------------------------------------------------------------
# 2) 채점 단계 — RAGAS judge/임베딩 구성 + evaluate
# ---------------------------------------------------------------------------

def build_judge_llm(judge: str, model: Optional[str]):
    """RAGAS judge LLM 을 구성해 LangchainLLMWrapper 로 감싼다.

    judge="local"  : 로컬 모델 서버(OpenAI 호환). 완전 오프라인.
    judge="openai" : OPENAI_API_KEY 필요. 계획서 명시 GPT-4o 계열.
    judge="gemini" : GOOGLE_API_KEY 필요. Google AI Studio 무료 티어(gemini-2.5-flash).
                     GPT-4o 대안 — 강한 judge를 무료로. 무료 RPM 제한 → concurrency 낮게.
    """
    from ragas.llms import LangchainLLMWrapper

    if judge == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise SystemExit("--judge gemini 인데 GOOGLE_API_KEY 가 없습니다.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model or "gemini-2.5-flash",
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=5,  # 무료 티어 429(rate limit) 완화
            timeout=120,
        )
        return LangchainLLMWrapper(llm)

    from langchain_openai import ChatOpenAI
    if judge == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("--judge openai 인데 OPENAI_API_KEY 가 없습니다.")
        llm = ChatOpenAI(model=model or "gpt-4o-mini", temperature=0.0)
    else:  # local
        llm = ChatOpenAI(
            model=model or os.getenv("RAGAS_JUDGE_MODEL", QWEN_VL_MODEL_NAME),
            base_url=MODEL_SERVER_URL,
            api_key=os.getenv("OPENAI_API_KEY", "sk-local-noauth"),
            temperature=0.0,
            timeout=120,
            max_retries=1,
        )
    return LangchainLLMWrapper(llm)


def build_judge_embeddings():
    """answer_relevancy 계산용 임베딩 — 로컬 bge-m3(문서 인덱싱과 동일 모델)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    emb = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(emb)


def score_with_ragas(
    samples: List[Dict[str, Any]],
    judge: str,
    model: Optional[str],
    concurrency: int,
) -> Dict[str, Any]:
    """RAGAS 로 샘플을 채점하고 요약/행별 점수를 반환한다."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    # context가 하나도 없는 샘플은 context 기반 메트릭이 무의미(NaN 유발)하므로 제외.
    usable = [s for s in samples if s["retrieved_contexts"] and s["response"] and not s["errored"]]
    skipped = [s for s in samples if s not in usable]
    if not usable:
        raise SystemExit(
            f"채점 가능한 샘플이 없습니다(전체 {len(samples)}건 모두 context/answer 없음 또는 error).\n"
            "→ 생성 단계에서 E2E 스택이 떠 있었는지, retrieved_contexts 가 채워졌는지 확인하세요."
        )

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["user_input"],
            response=s["response"],
            retrieved_contexts=s["retrieved_contexts"],
            reference=s["reference"],
        )
        for s in usable
    ])

    llm = build_judge_llm(judge, model)
    embeddings = build_judge_embeddings()
    # ResponseRelevancy 는 기본 strictness=3(질문 3개 생성 후 self-consistency)이라
    # OpenAI 의 n>1(다중 completion) 파라미터를 요구한다. 이 레포의 로컬 모델 서버는
    # transformers 백엔드를 직접 감싼 경량 OpenAI 호환 서버라 n>1 을 지원하지 않아
    # local judge 에서 InternalServerError/불완전 생성이 났다 — strictness=1 로 낮춰
    # 단일 호출로 우회한다. openai judge 는 다중 completion 을 지원하니 그대로 3 유지.
    relevancy_strictness = 1 if judge == "local" else 3
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=relevancy_strictness),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    # 로컬 모델 서버는 프로세스당 1개 모델을 공유하므로 동시요청을 낮게 잡아 큐 폭주를 막는다.
    # gemini 무료 티어는 RPM 제한이 낮아 429 백오프가 길어진다 — timeout/max_wait/max_retries를
    # 크게 잡아 백오프가 잡 타임아웃에 걸려 TimeoutError로 죽지 않게 한다.
    if judge == "gemini":
        run_config = RunConfig(max_workers=1, timeout=900, max_retries=15, max_wait=120)
    else:
        run_config = RunConfig(max_workers=concurrency, timeout=180)

    print(f"\nRAGAS 채점 시작 — judge={judge}, 샘플 {len(usable)}건"
          f"{f' (스킵 {len(skipped)}건)' if skipped else ''}...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in
                   ("user_input", "response", "retrieved_contexts", "reference")]

    def _mean(col: str) -> Optional[float]:
        vals = [v for v in df[col].tolist() if v == v]  # NaN 제외
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {m: _mean(m) for m in metric_cols}

    # 행별 점수(리포트 저장용) — id/route 를 usable 순서로 다시 붙인다.
    rows = []
    for s, (_, r) in zip(usable, df.iterrows()):
        rows.append({
            "id": s["id"],
            "route_type": s["route_type"],
            **{m: (None if r[m] != r[m] else round(float(r[m]), 4)) for m in metric_cols},
        })

    return {
        "judge": judge,
        "judge_model": model or (os.getenv("RAGAS_JUDGE_MODEL", QWEN_VL_MODEL_NAME) if judge == "local" else "gpt-4o-mini"),
        "n_scored": len(usable),
        "n_skipped": len(skipped),
        "metrics": summary,
        "rows": rows,
    }


def print_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"RAGAS 평가 결과 · judge={report['judge']} ({report['judge_model']})")
    print("=" * 72)
    print(f"  채점 {report['n_scored']}건 / 스킵 {report['n_skipped']}건")
    print("-" * 72)
    targets = {"faithfulness": 0.85}  # 계획서 G4 목표
    for name, val in report["metrics"].items():
        tgt = targets.get(name)
        mark = ""
        if tgt is not None and val is not None:
            mark = f"   (목표 >{tgt} {'✅' if val >= tgt else '❌'})"
        print(f"  {name:<40} {val if val is not None else 'N/A'}{mark}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Knowledge Node RAGAS 평가 러너")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS, choices=["rag", "multi", "sql"],
                    help="평가할 gold set (기본: rag). 복합(multi)·SQL은 크로스 역량이라 "
                         "knowledge 단독 RAGAS로는 과소 측정 — E2E/Multi-Turn 평가 권장.")
    ap.add_argument("--gold-ref", default="origin/feature/gold-set",
                    help="워킹트리에 gold set 이 없을 때 읽어올 git ref")
    ap.add_argument("--limit", type=int, default=None, help="세트별 앞 N건만")
    ap.add_argument("--judge", choices=["local", "openai", "gemini"], default="local",
                    help="RAGAS judge LLM 백엔드 (기본 local=로컬 모델 서버, gemini=무료 gemini-2.5-flash)")
    ap.add_argument("--judge-model", default=None,
                    help="judge 모델명 (local 기본: 7B / openai 기본: gpt-4o-mini / gemini 기본: gemini-2.5-flash)")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="RAGAS 동시 요청 수 (로컬 서버 큐 폭주 방지, 기본 2)")
    ap.add_argument("--build-only", action="store_true",
                    help="RAGAS 채점 없이 샘플만 생성해 --samples-out 에 저장")
    ap.add_argument("--score-from", default=None,
                    help="이미 생성된 샘플 JSON을 읽어 채점만 수행(생성 단계 스킵)")
    ap.add_argument("--samples-out", default="evaluation/reports/ragas_samples.json",
                    help="생성한 샘플 저장 경로")
    ap.add_argument("--out", default="evaluation/reports/ragas_eval.json",
                    help="채점 리포트 저장 경로")
    args = ap.parse_args()

    # --- 샘플 확보: 파일에서 읽거나 E2E 생성 ---
    if args.score_from:
        samples = json.loads(Path(args.score_from).read_text(encoding="utf-8"))
        print(f"샘플 로드: {args.score_from} ({len(samples)}건)")
    else:
        forced_cross = _CROSS_CAPABILITY_SETS & set(args.sets)
        if forced_cross:
            print(
                f"⚠️  경고: {sorted(forced_cross)} 는 크로스 역량 세트입니다. 이 러너는 "
                "knowledge_node 단독을 돌리므로(supervisor 분해 없음) Faithfulness 가 "
                "구조적으로 과소 측정됩니다 — 계획서상 복합은 G3 Multi-Turn 완수율·G5 E2E 로 "
                "검증하세요. RAGAS 는 RAG 세트에만 사용하는 것을 권장합니다."
            )
        samples = asyncio.run(build_samples(args.sets, args.gold_ref, args.limit))
        out = Path(args.samples_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n샘플 저장: {out} ({len(samples)}건)")

    if args.build_only:
        print("--build-only: 채점 생략.")
        return

    # --- RAGAS 채점 ---
    report = score_with_ragas(samples, args.judge, args.judge_model, args.concurrency)
    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n상세 리포트 저장: {out}")


if __name__ == "__main__":
    main()
