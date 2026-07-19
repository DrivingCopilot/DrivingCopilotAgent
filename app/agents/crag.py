"""
app/agents/crag.py

Corrective RAG(CRAG) 메커니즘의 노드 및 조건부 엣지 함수.

knowledge_node(검색기)가 채운 검색 결과(context_data["last_knowledge_result"])의
품질을 LLM으로 명시적으로 평가하고, 부실하면 쿼리를 재작성해 재검색시키는
교정 루프를 구성한다. builder.py 오케스트레이션(add_node/add_edge)은 이 파일에서
다루지 않는다 — 아래 "그래프 배선 계약"을 참고해 별도로 배선한다.

────────────────────────────────────────────────────────────────────────────
그래프 배선 계약 (builder.py 담당자용)
────────────────────────────────────────────────────────────────────────────
노드 등록:
    graph.add_node("grade_retrieval", grade_retrieval_node)
    graph.add_node("transform_query", transform_query_node)
    graph.add_node("refine_knowledge", refine_knowledge_node)

엣지:
    # 기존 knowledge → observe 정적 엣지를 아래로 교체
    graph.add_edge("knowledge", "grade_retrieval")

    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_grade,
        {"refine": "refine_knowledge", "transform": "transform_query"},
    )
    graph.add_edge("transform_query", "knowledge")   # 재검색 루프
    graph.add_edge("refine_knowledge", "observe")    # 기존 검증 흐름 합류

목표 토폴로지:
    knowledge → grade_retrieval →
        correct                → refine_knowledge → observe
        incorrect | ambiguous  → transform_query  → knowledge (재검색, MAX_CRAG_ATTEMPTS 캡)
        (재검색 소진 시)        → refine_knowledge → observe   # best-effort

knowledge_node 연동(별도 담당):
    transform_query_node 는 context_data["refined_query"] 에 재작성된 쿼리를 기록한다.
    knowledge_node 는 재검색 시 context_data["refined_query"] 가 있으면 그것을
    instruction 으로 우선 사용하도록 소폭 수정이 필요하다(plan 스텝 재-pop 방지 포함).

State 영향:
    AgentState 스키마 변경 불필요. 모든 CRAG 상태는 기존 context_data(얕은 merge
    리듀서 merge_context) 하위 키에 저장한다: crag_grade / refined_query / crag_attempts.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.graph.state import AgentState
from app.graph import ws as _ws
from app.core.json_utils import extract_first_json_object

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 설정 상수 (모듈 로컬 — config.py 미수정 방침)
# ---------------------------------------------------------------------------

# 재검색(transform_query → knowledge) 최대 반복 횟수. 무한 루프 방지용.
MAX_CRAG_ATTEMPTS = 1

# 평가/재작성/정제에 사용하는 LLM. knowledge Executor와 동일 계열(qwen2.5:1.5b)로 맞춘다.
# CRAG_MODEL 로 오버라이드 가능 — knowledge_node의 KNOWLEDGE_MODEL 규약과 동일.
CRAG_GRADER_MODEL = os.getenv("CRAG_MODEL", "qwen2.5:1.5b")

# grade 파싱 실패 등 예외 시 기본 판정 — transform(재검색)으로 흘려 교정 기회를 준다.
_DEFAULT_GRADE = "ambiguous"


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

GRADE_SYSTEM_PROMPT = """You are the Retrieval Evaluator in a Corrective RAG (CRAG) pipeline for an on-device driving copilot.
Given the user's query and the retrieved knowledge, judge how well the retrieved knowledge answers the query.

Grade definitions:
- "correct":   The retrieved knowledge is relevant and sufficient to answer the query.
- "incorrect": The retrieved knowledge is irrelevant, empty, or an error — it does not help answer the query.
- "ambiguous": The retrieved knowledge is partially relevant but incomplete or uncertain.

Respond with ONLY a single JSON object, no prose, no markdown fences:
{"grade": "correct|incorrect|ambiguous", "score": 0.0, "reasoning": "one short sentence"}
- "score" is your confidence that the knowledge answers the query, from 0.0 (useless) to 1.0 (fully sufficient).
"""

TRANSFORM_QUERY_PROMPT = """You are the Query Rewriter in a Corrective RAG (CRAG) pipeline for an on-device driving copilot.
The previous retrieval was insufficient. Rewrite the user's query into a single, clearer, retrieval-optimized query
that is more likely to match the vehicle manual / knowledge base.
Keep the user's original intent. Output ONLY the rewritten query text — no explanation, no quotes.
"""

REFINE_PROMPT = """You are the Knowledge Refiner in a Corrective RAG (CRAG) pipeline for an on-device driving copilot.
Given the user's query and the retrieved knowledge, extract and recompose ONLY the parts relevant to the query.
Remove unrelated strips, boilerplate, and noise. Preserve concrete facts, numbers, and steps.
Output ONLY the refined knowledge as plain text — no preamble, no commentary.
"""


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _latest_user_query(messages: List[Any]) -> str:
    """messages에서 가장 최근 HumanMessage 내용을 반환 (supervisor 관례와 동일)."""
    return next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )


def _effective_query(state: AgentState) -> str:
    """평가/재작성 대상 쿼리.

    knowledge_node 가 이번에 실제로 실행한 쿼리를 context_data["crag_query"] 로
    기록하므로 그것을 최우선으로 쓴다 — knowledge 가 완료 스텝을 pop 한 뒤에는
    plan[0] 이 '다음' 스텝으로 바뀌어 grader 가 엉뚱한 스텝을 평가하기 때문이다.
    crag_query 가 없으면 plan 첫 스텝, 그것도 없으면 최근 사용자 쿼리로 폴백한다.
    """
    crag_query = state.get("context_data", {}).get("crag_query", "")
    if crag_query:
        return str(crag_query)
    plan = state.get("plan", [])
    if plan:
        return str(plan[0])
    return _latest_user_query(state.get("messages", []))


def _parse_grade(content: str) -> Dict[str, Any]:
    """LLM 출력에서 첫 JSON 객체를 안전 파싱해 grade dict로 정규화한다."""
    text = content.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].strip()

    parsed = json.loads(extract_first_json_object(text))

    grade = str(parsed.get("grade", _DEFAULT_GRADE)).lower().strip()
    if grade not in ("correct", "incorrect", "ambiguous"):
        grade = _DEFAULT_GRADE

    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    return {
        "grade": grade,
        "score": max(0.0, min(1.0, score)),
        "reasoning": str(parsed.get("reasoning", "")),
    }


# ---------------------------------------------------------------------------
# 노드
# ---------------------------------------------------------------------------

async def grade_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    검색 결과 품질 평가자. context_data["last_knowledge_result"]를 LLM으로 평가해
    context_data["crag_grade"] = {grade, score, reasoning}를 기록한다.
    라우팅은 route_after_grade가 grade 값으로 판정하므로 next_agent는 쓰지 않는다.
    """
    await _ws.websocket_manager.send_status(
        json.dumps({"type": "status", "data": "Evaluating retrieval quality (CRAG)..."})
    )

    context_data = state.get("context_data", {})
    retrieved = context_data.get("last_knowledge_result", "")
    query = _effective_query(state)

    llm = ChatOpenAI(
        model=CRAG_GRADER_MODEL,
        temperature=0.0,
        max_tokens=200,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    eval_msg = HumanMessage(
        content=f"[User Query]\n{query}\n\n[Retrieved Knowledge]\n{retrieved}"
    )

    try:
        response = await llm.ainvoke([SystemMessage(content=GRADE_SYSTEM_PROMPT), eval_msg])
        grade = _parse_grade(response.content)
    except Exception as e:  # noqa: BLE001 — 평가 실패 시 재검색으로 흘린다
        logger.warning("grade_retrieval: 평가 실패 — %s. 기본 grade=%s", e, _DEFAULT_GRADE)
        grade = {"grade": _DEFAULT_GRADE, "score": 0.0, "reasoning": f"grader error: {e}"}

    logger.info("grade_retrieval: grade=%s score=%.2f", grade["grade"], grade["score"])
    await _ws.websocket_manager.send_status(
        json.dumps({"type": "status", "data": f"CRAG grade: {grade['grade']}"})
    )

    # 신규 키만 반환 → merge_context 리듀서가 context_data에 병합
    return {"context_data": {"crag_grade": grade}}


async def transform_query_node(state: AgentState) -> Dict[str, Any]:
    """
    쿼리 재작성 + 재검색 카운터 증가. context_data["refined_query"]에 재작성된 쿼리를
    기록하고 crag_attempts를 +1 한다. 실제 재검색은 그래프 배선(transform_query→knowledge)이
    수행하며 knowledge_node가 refined_query를 우선 사용한다.
    """
    await _ws.websocket_manager.send_status(
        json.dumps({"type": "status", "data": "Reformulating query (CRAG)..."})
    )

    context_data = state.get("context_data", {})
    attempts = int(context_data.get("crag_attempts", 0)) + 1
    original_query = _effective_query(state)

    llm = ChatOpenAI(model=CRAG_GRADER_MODEL, temperature=0.2, max_tokens=200)

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=TRANSFORM_QUERY_PROMPT),
             HumanMessage(content=original_query)]
        )
        refined_query = (response.content or "").strip() or original_query
    except Exception as e:  # noqa: BLE001 — 실패 시 원 쿼리로 재검색
        logger.warning("transform_query: 재작성 실패 — %s. 원 쿼리 사용", e)
        refined_query = original_query

    logger.info(
        "transform_query: attempts=%d/%d, refined=%r",
        attempts, MAX_CRAG_ATTEMPTS, refined_query,
    )

    return {"context_data": {"refined_query": refined_query, "crag_attempts": attempts}}


async def refine_knowledge_node(state: AgentState) -> Dict[str, Any]:
    """
    지식 정제(decompose-recompose). 검색 텍스트에서 무관한 부분을 제거하고 핵심만
    재구성해 context_data["last_knowledge_result"]를 정제본으로 덮어쓴다.
    """
    await _ws.websocket_manager.send_status(
        json.dumps({"type": "status", "data": "Refining knowledge (CRAG)..."})
    )

    context_data = state.get("context_data", {})
    retrieved = context_data.get("last_knowledge_result", "")
    query = _effective_query(state)

    # 정제할 내용이 없으면 원본 유지
    if not retrieved:
        logger.info("refine_knowledge: 검색 결과 없음 — 정제 생략")
        return {"context_data": {}}

    llm = ChatOpenAI(model=CRAG_GRADER_MODEL, temperature=0.0, max_tokens=512)

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=REFINE_PROMPT),
             HumanMessage(content=f"[User Query]\n{query}\n\n[Retrieved Knowledge]\n{retrieved}")]
        )
        refined = (response.content or "").strip() or retrieved
    except Exception as e:  # noqa: BLE001 — 실패 시 원본 유지
        logger.warning("refine_knowledge: 정제 실패 — %s. 원본 유지", e)
        refined = retrieved

    logger.info("refine_knowledge: %d→%d chars", len(retrieved), len(refined))
    return {"context_data": {"last_knowledge_result": refined}}


# ---------------------------------------------------------------------------
# 조건부 엣지 함수
# ---------------------------------------------------------------------------

def route_after_grade(state: AgentState) -> str:
    """
    grade_retrieval_node의 평가(context_data["crag_grade"])와 재검색 횟수를 보고
    다음 노드를 결정한다. 기존 route_next / route_after_reflect와 동일한 관례.

    Returns:
        "refine"    → refine_knowledge (정제 후 observe로)
        "transform" → transform_query  (쿼리 재작성 후 knowledge 재검색)
    """
    cd = state.get("context_data", {})
    grade = (cd.get("crag_grade") or {}).get("grade", _DEFAULT_GRADE)
    attempts = int(cd.get("crag_attempts", 0))

    # 재검색 캡 소진 → grade 무관하게 정제 후 종료 (무한 루프 방지, best-effort)
    if attempts >= MAX_CRAG_ATTEMPTS:
        logger.info("route_after_grade: 재검색 캡 소진(%d) → refine", attempts)
        return "refine"

    if grade == "correct":
        logger.info("route_after_grade: correct → refine")
        return "refine"

    logger.info("route_after_grade: %s → transform", grade)
    return "transform"
