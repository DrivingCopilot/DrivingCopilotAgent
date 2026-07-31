"""
tests/test_dispatch.py

app/a2a/dispatch.py 단위 테스트.
knowledge_node/run_execution/perception_node의 시그니처를 건드리지 않고
A2ATaskRequest/Response 경계 변환만 담당하므로, 이 파일에서는 그 변환 로직만
검증한다 (각 노드 내부 로직은 tests/test_knowledge.py 등에서 이미 검증됨).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.a2a.dispatch import _DISPATCH_TABLE, dispatch_task
from app.a2a.models import A2ATaskRequest


def _make_request(agent_name: str, **context_kwargs) -> A2ATaskRequest:
    return A2ATaskRequest(
        task_id="t-1",
        agent_name=agent_name,
        instruction="test",
        context=context_kwargs,
    )


class TestDispatchTaskUnknownAgent:
    async def test_unknown_agent_returns_invalid_tool_error(self):
        req = _make_request("bogus")
        resp = await dispatch_task(req)

        assert resp.status == "error"
        assert resp.error_type == "invalid_tool"
        assert resp.task_id == "t-1"


class TestDispatchTaskSuccess:
    async def test_success_serializes_messages_in_result(self):
        """노드가 messages(BaseMessage 리스트)를 반환하면 dict로 직렬화해서 감싼다."""
        fake_node = AsyncMock(return_value={
            "messages": [AIMessage(content="지식 답변")],
            "plan": [],
            "next_agent": "supervisor",
        })
        with patch.dict(_DISPATCH_TABLE, {"knowledge": fake_node}):
            req = _make_request("knowledge", messages=[], plan=["질문에 답하라"])
            resp = await dispatch_task(req)

        assert resp.status == "success"
        assert isinstance(resp.result["messages"], list)
        assert isinstance(resp.result["messages"][0], dict)  # BaseMessage가 아니라 dict
        assert resp.result["messages"][0]["data"]["content"] == "지식 답변"

    async def test_success_without_messages_passes_through(self):
        """messages 키가 없는 결과(execution/perception 흔한 케이스)는 그대로 통과."""
        fake_node = AsyncMock(return_value={
            "tool_calls": [{"tool": "control_wiper", "status": "success"}],
            "context_data": {"vehicle_state": {}},
        })
        with patch.dict(_DISPATCH_TABLE, {"execution": fake_node}):
            req = _make_request("execution", plan=["와이퍼를 켠다"])
            resp = await dispatch_task(req)

        assert resp.status == "success"
        assert resp.result["tool_calls"][0]["tool"] == "control_wiper"

    async def test_build_state_reconstructs_context_fields(self):
        """_build_state가 request.context의 필드를 AgentState로 정확히 옮기는지 확인."""
        received_state = {}

        async def capture_state(state):
            received_state.update(state)
            return {}

        with patch.dict(_DISPATCH_TABLE, {"perception": AsyncMock(side_effect=capture_state)}):
            req = _make_request(
                "perception",
                route_type="vision",
                plan=["기존 plan"],
                context_data={"vision_results": {"status": "ok"}},
                error_count={"timeout": 1},
            )
            await dispatch_task(req)

        assert received_state["route_type"] == "vision"
        assert received_state["plan"] == ["기존 plan"]
        assert received_state["context_data"] == {"vision_results": {"status": "ok"}}
        assert received_state["error_count"] == {"timeout": 1}

    async def test_messages_round_trip_through_dispatch(self):
        """request.context의 messages(dict)가 노드에는 BaseMessage로 복원되어 전달된다."""
        received_state = {}

        async def capture_state(state):
            received_state.update(state)
            return {}

        from app.a2a.serde import serialize_messages

        with patch.dict(_DISPATCH_TABLE, {"knowledge": AsyncMock(side_effect=capture_state)}):
            req = _make_request(
                "knowledge",
                messages=serialize_messages([HumanMessage(content="안녕")]),
            )
            await dispatch_task(req)

        assert received_state["messages"][0].content == "안녕"


class TestDispatchTaskNodeException:
    async def test_node_exception_becomes_error_response(self):
        with patch.dict(
            _DISPATCH_TABLE,
            {"execution": AsyncMock(side_effect=RuntimeError("boom"))},
        ):
            req = _make_request("execution", plan=["뭔가 실행"])
            resp = await dispatch_task(req)

        assert resp.status == "error"
        assert resp.error_type == "parameter"
        assert "boom" in resp.error


class TestDispatchTaskRealExecutionIntegration:
    """실제 run_execution을 기존 MCP/LLM mock 패턴으로 한 번 관통시켜보는 sanity 테스트."""

    async def test_dispatch_execution_end_to_end(self):
        req = _make_request("execution", plan=["와이퍼를 켠다"])

        with (
            patch(
                "app.agents.execution._extract_tool_call",
                new_callable=AsyncMock,
                return_value={"tool_name": "control_wiper", "params": {"on": True}},
            ),
            patch(
                "app.core.mcp_client.call_mcp_tool_raw",
                new_callable=AsyncMock,
                return_value=("와이퍼를 켰습니다.", "success"),
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            resp = await dispatch_task(req)

        assert resp.status == "success"
        assert resp.result["tool_calls"][0]["tool"] == "control_wiper"
        assert resp.result["tool_calls"][0]["status"] == "success"
