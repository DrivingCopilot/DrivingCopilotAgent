"""
tests/test_knowledge.py

Knowledge Agent 노드(app/agents/knowledge.py) 단위 테스트.

핵심 원칙:
    knowledge_node는 실제 LLM(ChatOpenAI/vLLM), Qdrant, Neo4j, SQLite에 의존한다.
    단위 테스트에서는 이 외부 경계를 mock 처리하고, "노드의 오케스트레이션 로직"만 검증한다.
    즉, LLM 성능이 아니라 다음을 본다:
      - plan을 읽어 지시문을 만드는가
      - ReAct 에이전트를 호출하는가
      - 결과를 context_data에 올바르게 담는가
      - 완료된 plan 스텝을 pop 하는가
      - 항상 supervisor로 제어를 반환하는가
      - 실패 시 error_count / feedback 을 채우는가

실행:
    .venv/bin/python -m pytest tests/test_knowledge.py -v
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.knowledge import (
    knowledge_node,
    vector_rag_search,
    graph_rag_search,
    text_to_sql_query,
)


# ---------------------------------------------------------------------------
# 테스트용 가짜 ReAct 에이전트
# create_react_agent(...) 가 반환하는 객체를 대체한다.
# ainvoke 는 입력 messages 뒤에 AI 답변을 덧붙여 돌려준다(실제 ReAct 동작 모사).
# ---------------------------------------------------------------------------

class FakeReactAgent:
    def __init__(self, answer="엔진 경고등은 엔진 점검이 필요함을 의미합니다.",
                 raise_exc=None, return_empty=False):
        self.answer = answer
        self.raise_exc = raise_exc
        self.return_empty = return_empty
        self.received = None  # ainvoke 에 전달된 payload 캡처

    async def ainvoke(self, payload):
        self.received = payload
        if self.raise_exc is not None:
            raise self.raise_exc
        msgs = list(payload["messages"])
        if self.return_empty:
            # 새 메시지를 생성하지 못한 경우(빈 응답) 모사
            return {"messages": msgs}
        return {"messages": msgs + [AIMessage(content=self.answer)]}


def _patch_agent(fake_agent):
    """knowledge 모듈의 LLM 생성과 에이전트 생성을 동시에 mock."""
    return patch.multiple(
        "app.agents.knowledge",
        create_react_agent=MagicMock(return_value=fake_agent),
        ChatOpenAI=MagicMock(return_value=MagicMock()),
    )


# ---------------------------------------------------------------------------
# 시나리오 1: Plan 기반 실행 (정상 경로)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_executes_plan_step():
    fake = FakeReactAgent(answer="엔진 경고등 설명입니다.")
    state = {
        "messages": [HumanMessage(content="엔진 경고등이 뭐야?")],
        "plan": ["매뉴얼에서 엔진 경고등 검색", "관련 부품 그래프 탐색"],
        "context_data": {},
        "error_count": {},
    }

    with _patch_agent(fake):
        result = await knowledge_node(state)

    # 지시문에 plan[0] 이 포함되어 에이전트로 전달되었는가
    last_instruction = fake.received["messages"][-1].content
    assert "매뉴얼에서 엔진 경고등 검색" in last_instruction

    # 완료된 첫 스텝이 pop 되어 나머지만 남는가
    assert result["plan"] == ["관련 부품 그래프 탐색"]

    # 결과가 context_data 에 저장되는가
    assert result["context_data"]["last_knowledge_result"] == "엔진 경고등 설명입니다."

    # 항상 supervisor 로 복귀
    assert result["next_agent"] == "supervisor"

    # 마지막 메시지가 에이전트의 최종 답변인가
    assert result["messages"][-1].content == "엔진 경고등 설명입니다."


# ---------------------------------------------------------------------------
# 시나리오 2: Plan 없음 (기본 경로)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_without_plan_uses_default_instruction():
    fake = FakeReactAgent(answer="기본 질의 응답입니다.")
    state = {
        "messages": [HumanMessage(content="가까운 정비소 알려줘")],
        "plan": [],
        "context_data": {},
        "error_count": {},
    }

    with _patch_agent(fake):
        result = await knowledge_node(state)

    last_instruction = fake.received["messages"][-1].content
    assert "latest query" in last_instruction  # 기본 지시문 사용
    assert result["plan"] == []                # plan 은 비어 있는 그대로
    assert result["next_agent"] == "supervisor"
    assert result["context_data"]["last_knowledge_result"] == "기본 질의 응답입니다."


# ---------------------------------------------------------------------------
# 시나리오 3: context_data 병합 안전성 (기존 결과 보존)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_preserves_existing_context():
    fake = FakeReactAgent(answer="새 검색 결과")
    state = {
        "messages": [HumanMessage(content="질문")],
        "plan": [],
        "context_data": {
            "vector_results": ["기존 벡터 결과"],
            "vehicle_state": {"speed": 60},
        },
        "error_count": {},
    }

    with _patch_agent(fake):
        result = await knowledge_node(state)

    ctx = result["context_data"]
    # 기존 키가 덮어써지지 않고 보존되어야 한다
    assert ctx["vector_results"] == ["기존 벡터 결과"]
    assert ctx["vehicle_state"] == {"speed": 60}
    # 새 결과도 함께 존재
    assert ctx["last_knowledge_result"] == "새 검색 결과"


# ---------------------------------------------------------------------------
# 시나리오 4: 예외 처리 (Retry 로직)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_handles_agent_exception():
    fake = FakeReactAgent(raise_exc=RuntimeError("vLLM endpoint down"))
    state = {
        "messages": [HumanMessage(content="질문")],
        "plan": ["검색 스텝"],
        "context_data": {},
        "error_count": {"parameter": 1},  # 이미 1회 실패한 상태
    }

    with _patch_agent(fake):
        result = await knowledge_node(state)

    # parameter 에러 카운트가 1 증가하여 2가 되어야 한다
    assert result["error_count"]["parameter"] == 2
    # Reflexion 용 feedback 이 채워졌는가
    assert "vLLM endpoint down" in result["feedback"]
    # 실패해도 제어는 supervisor 로
    assert result["next_agent"] == "supervisor"
    # 에러 안내 메시지를 반환
    assert "error" in result["messages"][0].content.lower()


# ---------------------------------------------------------------------------
# 시나리오 5: 빈 응답 fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_empty_response_fallback():
    fake = FakeReactAgent(return_empty=True)
    state = {
        "messages": [HumanMessage(content="질문")],
        "plan": [],
        "context_data": {},
        "error_count": {},
    }

    with _patch_agent(fake):
        result = await knowledge_node(state)

    # 새 메시지가 없으면 기본 문구로 대체
    assert result["context_data"]["last_knowledge_result"] == "Knowledge retrieval completed."
    assert result["next_agent"] == "supervisor"


# ---------------------------------------------------------------------------
# 시나리오 6: Tool - MCP 위임 (vector / graph)
# 검색 로직은 공용 MCP 서버에 있으므로, 툴은 call_mcp_tool_once 로 위임만 한다.
# 여기서는 "올바른 tool 이름/파라미터로 호출하고 결과 텍스트를 그대로 반환하는가"를 본다.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vector_rag_delegates_to_mcp():
    mock_call = AsyncMock(return_value=("매뉴얼 검색 결과입니다.", "success", "", ""))
    with patch("app.agents.knowledge.call_mcp_tool_once", mock_call):
        result = await vector_rag_search.ainvoke({"query": "엔진 경고등"})

    # 올바른 MCP tool 이름과 파라미터로 호출했는가
    mock_call.assert_awaited_once_with("vector_rag_search", {"query": "엔진 경고등"})
    # 서버 결과를 그대로 반환
    assert result == "매뉴얼 검색 결과입니다."


# ---------------------------------------------------------------------------
# 시나리오 7: CRAG 연동 (crag_query 기록 / 재검색 / 실패 시 상태 초기화)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_records_crag_query_for_grader():
    # 멀티스텝 plan 에서 완료 스텝을 pop 하더라도, grade/transform 이 평가할
    # 쿼리(crag_query)에는 '실제 실행된 스텝'이 기록되어야 한다.
    fake = FakeReactAgent(answer="결과")
    state = {
        "messages": [HumanMessage(content="원본 질문")],
        "plan": ["스텝A", "스텝B"],
        "context_data": {},
        "error_count": {},
    }
    with _patch_agent(fake):
        result = await knowledge_node(state)

    assert result["context_data"]["crag_query"] == "스텝A"  # 실행된 스텝 기록
    assert result["plan"] == ["스텝B"]                       # 완료 스텝 pop


@pytest.mark.asyncio
async def test_knowledge_node_reretrieval_uses_refined_query_keeps_plan():
    fake = FakeReactAgent(answer="재검색 결과")
    state = {
        "messages": [HumanMessage(content="원본")],
        "plan": ["스텝A"],
        "context_data": {"refined_query": "재작성된 쿼리", "crag_query": "스텝A", "crag_attempts": 1},
        "error_count": {},
    }
    with _patch_agent(fake):
        result = await knowledge_node(state)

    # 재검색 지시문에 refined_query 사용
    assert "재작성된 쿼리" in fake.received["messages"][-1].content
    # plan 재-pop 안 함, refined_query 소비, crag_query 원본 유지
    assert result["plan"] == ["스텝A"]
    assert result["context_data"]["refined_query"] == ""
    assert result["context_data"]["crag_query"] == "스텝A"


@pytest.mark.asyncio
async def test_knowledge_node_exception_clears_refined_query():
    # 재검색 중 실패해도 refined_query 를 초기화해 다음 위임으로 새지 않게 한다.
    fake = FakeReactAgent(raise_exc=RuntimeError("boom"))
    state = {
        "messages": [HumanMessage(content="질문")],
        "plan": ["스텝"],
        "context_data": {"refined_query": "재작성"},
        "error_count": {},
    }
    with _patch_agent(fake):
        result = await knowledge_node(state)

    assert result["context_data"]["refined_query"] == ""


@pytest.mark.asyncio
async def test_graph_rag_forwards_entities():
    mock_call = AsyncMock(return_value=("관계 정보", "success", "", ""))
    with patch("app.agents.knowledge.call_mcp_tool_once", mock_call):
        result = await graph_rag_search.ainvoke(
            {"query": "엔진 경고등 부품", "entities": ["엔진", "점화플러그"]}
        )

    mock_call.assert_awaited_once_with(
        "graph_rag_search", {"query": "엔진 경고등 부품", "entities": ["엔진", "점화플러그"]}
    )
    assert result == "관계 정보"


@pytest.mark.asyncio
async def test_graph_rag_defaults_entities_to_empty_list():
    """entities 미지정 시 빈 리스트로 정규화되어 전달된다."""
    mock_call = AsyncMock(return_value=("관계 정보", "success", "", ""))
    with patch("app.agents.knowledge.call_mcp_tool_once", mock_call):
        await graph_rag_search.ainvoke({"query": "타이어 관련 정비"})

    _, params = mock_call.await_args.args
    assert params["entities"] == []


# ---------------------------------------------------------------------------
# 시나리오 7: Tool - MCP 실패 시 에이전트가 읽을 오류 문자열 반환
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_returns_error_string_on_mcp_failure():
    """MCP 호출이 실패하면 예외를 던지지 않고 오류 문자열을 반환한다(ReAct 재시도용)."""
    mock_call = AsyncMock(return_value=("타임아웃", "fail", "timeout", "10초 초과"))
    with patch("app.agents.knowledge.call_mcp_tool_once", mock_call):
        result = await text_to_sql_query.ainvoke({"query": "이번 달 총 주행거리"})

    assert "text_to_sql_query 실패" in result
    assert "timeout" in result
    assert "10초 초과" in result
