import pytest
from unittest.mock import AsyncMock, patch

import app.graph.builder as builder_module
from app.graph.builder import run_graph


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_graph_cache():
    builder_module.build_graph.cache_clear()
    yield
    builder_module.build_graph.cache_clear()


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
            "app.agents.execution._call_mcp_tool_raw",
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
async def test_react_one_loop_knowledge(patch_supervisor):
    tracker = patch_supervisor([
        {"next_agent": "knowledge", "plan": ["s1"], "feedback": ""},
        {"next_agent": "__end__", "plan": [], "feedback": ""},
    ])

    result = await run_graph("매뉴얼 검색해줘")

    assert tracker["count"] == 2
    assert any(tc["tool"] == "mock_knowledge_search" for tc in result["tool_calls"])
    assert "vector_results" in result["context_data"]
    assert "graph_results" in result["context_data"]


@pytest.mark.asyncio
async def test_react_one_loop_perception(patch_supervisor):
    tracker = patch_supervisor([
        {"next_agent": "perception", "plan": ["s1"], "feedback": ""},
        {"next_agent": "__end__", "plan": [], "feedback": ""},
    ])

    result = await run_graph("주변 상황 보여줘")

    assert tracker["count"] == 2

    # Tool 2종 모두 호출되어야 함
    tool_names = [tc["tool"] for tc in result["tool_calls"]]
    assert "analyze_camera_feed" in tool_names
    assert "detect_objects" in tool_names

    # vision_results는 구조화된 dict
    assert "vision_results" in result["context_data"]
    vision = result["context_data"]["vision_results"]
    assert "scene" in vision
    assert "objects" in vision
    assert vision["scene"]["weather"] == "rain"  # mock scenario 확인


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

    monkeypatch.setattr(builder_module, "run_execution", failing_execution)
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
async def test_recursion_limit_safety(patch_supervisor):
    patch_supervisor(
        [{"next_agent": "execution", "plan": ["s1"], "feedback": ""}] * 30
    )

    with pytest.raises(Exception) as exc_info:
        await run_graph("loop test")

    err_str = str(exc_info.value).lower()
    assert (
        "recursion" in err_str or "limit" in err_str or "maximum" in err_str
    ), f"expected recursion-related error, got: {exc_info.value}"
