import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import app.graph.builder as builder_module
from app.a2a.dispatch import dispatch_task
import app.a2a.dispatch as dispatch_module
from app.graph.builder import run_graph


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_graph_cache():
    builder_module.build_graph.cache_clear()
    yield
    builder_module.build_graph.cache_clear()


@pytest.fixture(autouse=True)
def route_a2a_through_dispatch(monkeypatch):
    """
    A2AClient.send_task를 실제 HTTP 대신 in-process dispatch_task로 우회한다.

    테스트 환경에는 knowledge/execution/perception 독립 서버(8002~8004)가
    떠 있지 않으므로, 실제 HTTP 라운드트립 대신 dispatch_task를 직접 호출해
    그래프 라우팅 + 각 노드의 실제 로직(아래 MCP/LLM mock으로 경계만 대체)을
    계속 같은 방식으로 검증한다.
    """
    class _InProcessA2AClient:
        def __init__(self, *args, **kwargs):
            pass

        async def send_task(self, request):
            return await dispatch_task(request)

    monkeypatch.setattr("app.graph.a2a_nodes.A2AClient", _InProcessA2AClient)


@pytest.fixture
def patch_supervisor(monkeypatch):
    def _setup(responses: list[dict]):
        tracker = {"count": 0, "received_states": []}

        async def fake_supervisor(state):
            tracker["received_states"].append(dict(state))
            idx = tracker["count"]
            tracker["count"] += 1
            if idx >= len(responses):
                return {"next_agent": "__end__", "plan": [], "feedback": ""}
            return responses[idx]

        monkeypatch.setattr(builder_module, "supervisor_node", fake_supervisor)
        builder_module.build_graph.cache_clear()
        return tracker

    return _setup


# ---------------------------------------------------------------------------
# routing tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_react_one_loop_execution(patch_supervisor):
    # execution 노드는 실제 실행하되 MCP/LLM 경계만 mock (test_execution.py 패턴).
    tracker = patch_supervisor([
        {"next_agent": "execution", "plan": ["에어컨을 22도로 켠다"], "feedback": ""},
        {"next_agent": "__end__", "plan": [], "feedback": ""},
    ])

    with (
        patch(
            "app.agents.execution._extract_tool_call",
            new_callable=AsyncMock,
            return_value={"tool_name": "control_climate", "params": {"temperature": 22, "on": True}},
        ),
        patch(
            "app.core.mcp_client.call_mcp_tool_raw",
            new_callable=AsyncMock,
            return_value=("에어컨을 22℃로 켰어요.", "success"),
        ),
        patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
    ):
        result = await run_graph("에어컨 켜줘")

    assert tracker["count"] == 2, "supervisor should be called exactly twice"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "control_climate"
    assert result["tool_calls"][0]["status"] == "success"
    # 성공 시 vehicle_state 에 last_<tool> 기록 (run_execution)
    assert result["context_data"]["vehicle_state"]["last_control_climate"] == "에어컨을 22℃로 켰어요."


@pytest.mark.asyncio
async def test_react_one_loop_knowledge(patch_supervisor, monkeypatch):
    # knowledge 는 LLM/MCP(ReAct) 경계를 갖고, 이어 CRAG 서브그래프
    # (grade_retrieval → refine_knowledge → observe)를 거친다. 라우팅만 검증하므로
    # knowledge/CRAG 노드는 결정적 stub 으로 대체한다.
    async def fake_knowledge(state):
        return {
            "context_data": {"last_knowledge_result": "매뉴얼 검색 결과"},
            "plan": [],
            "next_agent": "supervisor",
        }

    async def fake_grade(state):
        # correct → route_after_grade 가 refine 으로 보낸다 (재검색 없음)
        return {"context_data": {"crag_grade": {"grade": "correct", "score": 1.0}}}

    async def fake_refine(state):
        return {"context_data": {}}

    monkeypatch.setattr(builder_module, "knowledge_node", fake_knowledge)
    monkeypatch.setattr(builder_module, "grade_retrieval_node", fake_grade)
    monkeypatch.setattr(builder_module, "refine_knowledge_node", fake_refine)
    builder_module.build_graph.cache_clear()

    tracker = patch_supervisor([
        {"next_agent": "knowledge", "plan": ["s1"], "feedback": ""},
        {"next_agent": "__end__", "plan": [], "feedback": ""},
    ])

    result = await run_graph("매뉴얼 검색해줘")

    assert tracker["count"] == 2
    # knowledge → grade_retrieval(correct) → refine_knowledge → observe → supervisor → end
    assert result["context_data"]["last_knowledge_result"] == "매뉴얼 검색 결과"


@pytest.mark.asyncio
async def test_crag_reretrieval_loop(patch_supervisor, monkeypatch):
    # grade 가 incorrect 를 반환하면 transform_query → knowledge 로 재검색하고,
    # crag_attempts 가 MAX_CRAG_ATTEMPTS(1) 에 도달하면 refine 으로 빠져 종료한다.
    from app.agents.crag import MAX_CRAG_ATTEMPTS

    knowledge_calls = {"count": 0}

    async def fake_knowledge(state):
        knowledge_calls["count"] += 1
        return {
            "context_data": {"last_knowledge_result": f"결과 {knowledge_calls['count']}"},
            "next_agent": "supervisor",
        }

    async def fake_grade(state):
        return {"context_data": {"crag_grade": {"grade": "incorrect", "score": 0.0}}}

    async def fake_transform(state):
        cd = state.get("context_data", {})
        attempts = int(cd.get("crag_attempts", 0)) + 1
        return {"context_data": {"refined_query": "재작성된 쿼리", "crag_attempts": attempts}}

    async def fake_refine(state):
        return {"context_data": {}}

    monkeypatch.setattr(builder_module, "knowledge_node", fake_knowledge)
    monkeypatch.setattr(builder_module, "grade_retrieval_node", fake_grade)
    monkeypatch.setattr(builder_module, "transform_query_node", fake_transform)
    monkeypatch.setattr(builder_module, "refine_knowledge_node", fake_refine)
    builder_module.build_graph.cache_clear()

    tracker = patch_supervisor([
        {"next_agent": "knowledge", "plan": ["s1"], "feedback": ""},
        {"next_agent": "__end__", "plan": [], "feedback": ""},
    ])

    result = await run_graph("애매한 질문")

    # 최초 검색 1회 + 재검색 1회 = knowledge 2회 (MAX_CRAG_ATTEMPTS=1 캡)
    assert knowledge_calls["count"] == MAX_CRAG_ATTEMPTS + 1
    assert tracker["count"] == 2


@pytest.mark.asyncio
async def test_react_one_loop_perception(patch_supervisor):
    tracker = patch_supervisor([
        {"next_agent": "perception", "plan": ["s1"], "feedback": ""},
        {"next_agent": "__end__", "plan": [], "feedback": ""},
    ])

    # hazards 빈 배열 — rain 등 hazard가 있으면 perception이 HAZARD_PLAN_STEPS로
    # plan을 채워 route_after_perception이 execution으로 직행시켜 tracker count가 깨진다.
    mock_vlm_response = MagicMock()
    mock_vlm_response.content = (
        '{"answer": "전방 도로가 맑고 특이사항 없습니다.", "hazards": []}'
    )
    mock_vlm = MagicMock()
    mock_vlm.ainvoke = AsyncMock(return_value=mock_vlm_response)

    with (
        patch(
            "app.agents.perception.call_mcp_tool_once",
            new_callable=AsyncMock,
            return_value=("/9j/fake_base64_frame==", "success", "", ""),
        ),
        patch(
            "app.agents.perception._get_vision_llm",
            return_value=mock_vlm,
        ),
        patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
    ):
        result = await run_graph("주변 상황 보여줘")

    assert tracker["count"] == 2

    assert "vision_results" in result["context_data"]
    vision = result["context_data"]["vision_results"]
    assert vision["status"] == "success"
    assert vision["answer"] == "전방 도로가 맑고 특이사항 없습니다."
    assert "hazards" in vision
    assert vision["hazards"] == []


@pytest.mark.asyncio
async def test_observe_routes_to_end_on_retry_limit(patch_supervisor, monkeypatch):
    # invalid_tool limit=1이므로 첫 실패에서 observe가 바로 __end__로 종료
    async def failing_execution(state):
        return {
            "tool_calls": [
                *state.get("tool_calls", []),
                {
                    "tool": "wiper",
                    "params": {},
                    "status": "error",
                    "error_type": "invalid_tool",
                    "error_msg": "wrong tool",
                },
            ],
            "context_data": {**state.get("context_data", {})},
        }

    monkeypatch.setitem(dispatch_module._DISPATCH_TABLE, "execution", failing_execution)
    builder_module.build_graph.cache_clear()

    tracker = patch_supervisor([
        {"next_agent": "execution", "plan": ["s1"], "feedback": ""},
    ])

    result = await run_graph("와이퍼 켜줘")

    assert tracker["count"] == 1, "supervisor called once; observe should terminate"
    assert result["next_agent"] == "__end__"
    assert result["error_count"]["invalid_tool"] == 1
    assert len(result["messages"]) >= 1
    assert result["plan"] == []


@pytest.mark.asyncio
async def test_supervisor_self_loop(patch_supervisor):
    tracker = patch_supervisor([
        {"next_agent": "supervisor", "plan": [], "feedback": "retry"},
        {"next_agent": "__end__", "plan": [], "feedback": ""},
    ])

    result = await run_graph("ping")

    assert tracker["count"] == 2, "supervisor self-loop once then end"


@pytest.mark.asyncio
async def test_immediate_end(patch_supervisor):
    tracker = patch_supervisor([
        {"next_agent": "__end__", "plan": [], "feedback": ""},
    ])

    result = await run_graph("안녕")

    assert tracker["count"] == 1
    assert result["tool_calls"] == []
    assert result["next_agent"] == "__end__"


@pytest.mark.asyncio
async def test_unmappable_plan_terminates_via_retry_limit_not_recursion(patch_supervisor):
    """
    supervisor가 (mock으로) 계속 'execution'만 반복 지시해도, plan을 MCP tool로
    매핑 못 하는 상황은 execution.py의 invalid_tool 폴백 → observe.py의 재시도
    한도에서 정상 종료돼야 한다. 예전엔 매핑 실패가 빈 tool_calls로 "성공"처럼
    통과돼 이 안전장치를 못 타고 LangGraph의 recursion_limit(하드 크래시)까지
    가야 멈췄는데, 지금은 그 전에 정상적으로 __end__로 끝나야 한다.
    """
    patch_supervisor(
        [{"next_agent": "execution", "plan": ["s1"], "feedback": ""}] * 30
    )

    result = await run_graph("loop test")

    assert result["next_agent"] == "__end__"
    assert any(tc.get("error_type") == "invalid_tool" for tc in result["tool_calls"])
