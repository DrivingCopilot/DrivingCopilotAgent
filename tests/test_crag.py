"""
tests/test_crag.py

CRAG 노드 및 조건부 엣지(app/agents/crag.py) 단위 테스트.

핵심 원칙(test_knowledge.py와 동일):
    노드는 실제 LLM(ChatOpenAI/vLLM)에 의존한다. 단위 테스트에서는 LLM 경계를
    mock 처리하고 "노드의 오케스트레이션/파싱 로직"과 라우팅 결정만 검증한다.
    (builder.py 배선은 다른 담당자 소관이므로 여기서는 그래프를 조립하지 않는다.)

실행:
    .venv/bin/python -m pytest tests/test_crag.py -v
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.crag import (
    grade_retrieval_node,
    transform_query_node,
    refine_knowledge_node,
    route_after_grade,
    _effective_query,
    MAX_CRAG_ATTEMPTS,
    _DEFAULT_GRADE,
)


# ---------------------------------------------------------------------------
# 헬퍼: ChatOpenAI(...) 를 대체하는 가짜 LLM
#   ainvoke(messages) 는 .content 를 가진 AIMessage 를 돌려준다.
# ---------------------------------------------------------------------------

def _fake_llm(content="", raise_exc=None):
    llm = MagicMock()
    if raise_exc is not None:
        llm.ainvoke = AsyncMock(side_effect=raise_exc)
    else:
        llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return llm


def _patch_llm(fake):
    """crag 모듈의 ChatOpenAI 생성을 mock (모든 노드가 동일 팩토리를 사용)."""
    return patch("app.agents.crag.ChatOpenAI", MagicMock(return_value=fake))


# ---------------------------------------------------------------------------
# grade_retrieval_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grade_parses_correct_json():
    fake = _fake_llm('{"grade": "correct", "score": 0.92, "reasoning": "충분함"}')
    state = {
        "messages": [HumanMessage(content="엔진 경고등이 뭐야?")],
        "plan": ["매뉴얼에서 엔진 경고등 검색"],
        "context_data": {"last_knowledge_result": "엔진 경고등은 엔진 점검이 필요함을 의미합니다."},
    }
    with _patch_llm(fake):
        result = await grade_retrieval_node(state)

    grade = result["context_data"]["crag_grade"]
    assert grade["grade"] == "correct"
    assert grade["score"] == pytest.approx(0.92)
    # 신규 키만 반환한다 (merge_context 리듀서가 병합)
    assert set(result["context_data"].keys()) == {"crag_grade"}


@pytest.mark.asyncio
async def test_grade_handles_markdown_fenced_json():
    fake = _fake_llm('```json\n{"grade": "incorrect", "score": 0.1, "reasoning": "무관"}\n```')
    state = {
        "messages": [HumanMessage(content="질문")],
        "context_data": {"last_knowledge_result": "관련 없는 텍스트"},
    }
    with _patch_llm(fake):
        result = await grade_retrieval_node(state)

    assert result["context_data"]["crag_grade"]["grade"] == "incorrect"


@pytest.mark.asyncio
async def test_grade_clamps_score_and_normalizes_unknown_grade():
    # 알 수 없는 grade + 범위 밖 score → 기본 grade, score 0~1 클램프
    fake = _fake_llm('{"grade": "maybe", "score": 5.0}')
    state = {"messages": [HumanMessage(content="질문")], "context_data": {"last_knowledge_result": "x"}}
    with _patch_llm(fake):
        result = await grade_retrieval_node(state)

    grade = result["context_data"]["crag_grade"]
    assert grade["grade"] == _DEFAULT_GRADE
    assert grade["score"] == 1.0


@pytest.mark.asyncio
async def test_grade_falls_back_on_llm_exception():
    fake = _fake_llm(raise_exc=RuntimeError("vLLM down"))
    state = {"messages": [HumanMessage(content="질문")], "context_data": {"last_knowledge_result": "x"}}
    with _patch_llm(fake):
        result = await grade_retrieval_node(state)

    # 예외 시 기본 grade 로 흘려 재검색 기회를 준다
    assert result["context_data"]["crag_grade"]["grade"] == _DEFAULT_GRADE


# ---------------------------------------------------------------------------
# transform_query_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transform_writes_refined_query_and_increments_attempts():
    fake = _fake_llm("엔진 경고등 의미와 대처 방법")
    state = {
        "messages": [HumanMessage(content="이거 뭐야")],
        "plan": ["엔진 경고등"],
        "context_data": {"crag_attempts": 0},
    }
    with _patch_llm(fake):
        result = await transform_query_node(state)

    cd = result["context_data"]
    assert cd["refined_query"] == "엔진 경고등 의미와 대처 방법"
    assert cd["crag_attempts"] == 1


@pytest.mark.asyncio
async def test_transform_falls_back_to_original_on_empty_output():
    fake = _fake_llm("   ")  # 공백만 반환 → 원 쿼리 유지
    state = {
        "messages": [HumanMessage(content="원래 질문")],
        "plan": [],
        "context_data": {},
    }
    with _patch_llm(fake):
        result = await transform_query_node(state)

    assert result["context_data"]["refined_query"] == "원래 질문"
    assert result["context_data"]["crag_attempts"] == 1


# ---------------------------------------------------------------------------
# refine_knowledge_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refine_overwrites_last_knowledge_result():
    fake = _fake_llm("정제된 핵심 지식")
    state = {
        "messages": [HumanMessage(content="질문")],
        "context_data": {"last_knowledge_result": "장황하고 노이즈 많은 원본 텍스트"},
    }
    with _patch_llm(fake):
        result = await refine_knowledge_node(state)

    assert result["context_data"]["last_knowledge_result"] == "정제된 핵심 지식"


@pytest.mark.asyncio
async def test_refine_noop_when_no_retrieval():
    fake = _fake_llm("사용되면 안 됨")
    state = {"messages": [HumanMessage(content="질문")], "context_data": {}}
    with _patch_llm(fake) as p:
        result = await refine_knowledge_node(state)

    # 검색 결과가 없으면 LLM 호출 없이 빈 병합 반환
    assert result["context_data"] == {}
    p.assert_not_called()


# ---------------------------------------------------------------------------
# route_after_grade (조건부 엣지 함수) — 순수 함수, LLM 불필요
# ---------------------------------------------------------------------------

def _state_with(grade, attempts=0):
    return {"context_data": {"crag_grade": {"grade": grade}, "crag_attempts": attempts}}


def test_route_correct_goes_to_refine():
    assert route_after_grade(_state_with("correct")) == "refine"


def test_route_incorrect_goes_to_transform():
    assert route_after_grade(_state_with("incorrect")) == "transform"


def test_route_ambiguous_goes_to_transform():
    assert route_after_grade(_state_with("ambiguous")) == "transform"


def test_route_caps_reretrieval_to_refine():
    # 재검색 캡 소진 → grade 무관하게 refine (무한 루프 방지)
    assert route_after_grade(_state_with("incorrect", attempts=MAX_CRAG_ATTEMPTS)) == "refine"


def test_route_default_grade_when_missing():
    # crag_grade 없음 → 기본 grade(ambiguous) → transform
    assert route_after_grade({"context_data": {}}) == "transform"


# ---------------------------------------------------------------------------
# _effective_query — 평가 기준 쿼리 선택 (crag_query 우선)
# ---------------------------------------------------------------------------

def test_effective_query_prefers_crag_query():
    # knowledge 가 완료 스텝을 pop 해 plan[0] 이 '다음' 스텝으로 바뀌어도,
    # crag_query(실제 실행된 쿼리)가 있으면 그것을 평가 대상으로 쓴다.
    state = {
        "context_data": {"crag_query": "실행된 스텝"},
        "plan": ["다음 스텝"],
        "messages": [HumanMessage(content="사용자 원본")],
    }
    assert _effective_query(state) == "실행된 스텝"


def test_effective_query_falls_back_to_plan_then_user():
    assert _effective_query(
        {"context_data": {}, "plan": ["스텝"], "messages": []}
    ) == "스텝"
    assert _effective_query(
        {"context_data": {}, "plan": [], "messages": [HumanMessage(content="질문")]}
    ) == "질문"
