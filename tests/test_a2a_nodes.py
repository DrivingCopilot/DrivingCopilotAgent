"""
tests/test_a2a_nodes.py

app/graph/a2a_nodes.py 단위 테스트.
A2AClient.send_task를 mock해서 실제 HTTP 없이:
  1. WS tool_start/tool_result가 A2A 호출 경계에서 합성되는지
  2. 성공 응답의 messages가 BaseMessage로 복원되어 state에 병합되는지
  3. 통신 실패(status="error") 시 tool_calls가 합성되어 observe_node가
     감지할 수 있는지
를 검증한다.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from app.a2a.models import A2ATaskResponse
from app.a2a.serde import serialize_messages
from app.graph.a2a_nodes import execution_a2a_node, knowledge_a2a_node, perception_a2a_node


def _make_state(**overrides):
    state = {
        "messages": [],
        "route_type": "",
        "plan": [],
        "next_agent": "",
        "tool_calls": [],
        "context_data": {},
        "error_count": {},
        "feedback": "",
    }
    state.update(overrides)
    return state


class TestRemoteAgentSuccess:
    async def test_ws_tool_start_and_result_emitted(self):
        ws_calls: list[str] = []

        async def capture_ws(msg: str) -> None:
            ws_calls.append(msg)

        fake_response = A2ATaskResponse(task_id="t-1", status="success", result={"tool_calls": []})
        with (
            patch("app.graph.ws.websocket_manager.send_status", side_effect=capture_ws),
            patch("app.graph.a2a_nodes.A2AClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.send_task = AsyncMock(return_value=fake_response)
            await execution_a2a_node(_make_state(plan=["와이퍼를 켠다"]))

        types = [json.loads(m)["type"] for m in ws_calls]
        assert types == ["tool_start", "tool_result"]

    async def test_success_result_messages_deserialized(self):
        serialized = serialize_messages([AIMessage(content="지식 답변")])
        fake_response = A2ATaskResponse(
            task_id="t-1", status="success", result={"messages": serialized, "plan": []}
        )
        with (
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
            patch("app.graph.a2a_nodes.A2AClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.send_task = AsyncMock(return_value=fake_response)
            result = await knowledge_a2a_node(_make_state())

        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "지식 답변"

    async def test_request_context_carries_state_fields(self):
        """state의 plan/context_data/error_count가 A2ATaskRequest.context로 실린다."""
        captured_request = {}

        async def capture_send_task(request):
            captured_request["req"] = request
            return A2ATaskResponse(task_id=request.task_id, status="success", result={})

        with (
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
            patch("app.graph.a2a_nodes.A2AClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.send_task = AsyncMock(side_effect=capture_send_task)
            await execution_a2a_node(_make_state(
                plan=["와이퍼를 켠다"],
                context_data={"vehicle_state": {"a": 1}},
                error_count={"timeout": 1},
            ))

        req = captured_request["req"]
        assert req.agent_name == "execution"
        assert req.context["plan"] == ["와이퍼를 켠다"]
        assert req.context["context_data"] == {"vehicle_state": {"a": 1}}
        assert req.context["error_count"] == {"timeout": 1}


class TestRemoteAgentTransportFailure:
    async def test_error_status_synthesizes_tool_calls_entry(self):
        """A2A 통신 실패 시 observe_node가 감지할 수 있도록 tool_calls를 합성한다."""
        fake_response = A2ATaskResponse(
            task_id="t-1", status="error", error="connection refused", error_type="timeout"
        )
        with (
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
            patch("app.graph.a2a_nodes.A2AClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.send_task = AsyncMock(return_value=fake_response)
            result = await execution_a2a_node(_make_state(plan=["와이퍼를 켠다"]))

        tc = result["tool_calls"][0]
        assert tc["tool"] == "execution"
        assert tc["status"] == "error"
        assert tc["error_type"] == "timeout"
        assert tc["error_msg"] == "connection refused"

    async def test_error_status_clears_stale_plan(self):
        """
        전송 실패 시 plan을 비우지 않으면, perception 실패 후 supervisor가 넣어둔
        낡은 plan이 남아 route_after_perception이 이를 hazard 감지로 오인해
        execution으로 잘못 튄다 (실제로 겪은 회귀).
        """
        fake_response = A2ATaskResponse(
            task_id="t-1", status="error", error="timed out", error_type="timeout"
        )
        with (
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
            patch("app.graph.a2a_nodes.A2AClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.send_task = AsyncMock(return_value=fake_response)
            result = await perception_a2a_node(_make_state(plan=["delegate_to_perception"]))

        assert result["plan"] == []

    async def test_client_uses_a2a_task_timeout_not_default(self):
        """Agent Card 조회용 기본 5초가 아니라 A2A_TASK_TIMEOUT(LLM/VLM 추론 고려)을 써야 한다."""
        from app.core.config import A2A_TASK_TIMEOUT

        fake_response = A2ATaskResponse(task_id="t-1", status="success", result={})
        with (
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
            patch("app.graph.a2a_nodes.A2AClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.send_task = AsyncMock(return_value=fake_response)
            await perception_a2a_node(_make_state())

        _, kwargs = mock_client_cls.call_args
        assert kwargs["timeout"] == A2A_TASK_TIMEOUT
        assert A2A_TASK_TIMEOUT > 5.0

    async def test_knowledge_transport_failure_also_synthesizes_tool_calls(self):
        """knowledge는 평소 tool_calls를 안 쓰지만, 통신 실패는 예외적으로 관측 가능해야 한다."""
        fake_response = A2ATaskResponse(
            task_id="t-1", status="error", error="down", error_type="parameter"
        )
        with (
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
            patch("app.graph.a2a_nodes.A2AClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.send_task = AsyncMock(return_value=fake_response)
            result = await knowledge_a2a_node(_make_state())

        assert result["tool_calls"][0]["error_type"] == "parameter"
