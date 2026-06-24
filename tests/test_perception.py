"""
tests/test_perception.py

Perception Agent 단위 테스트.

테스트 구성:
    [Unit]  Mock MCP 서버 — app.agents.perception.call_mcp_tool_once 를 patch
            Mock Vision LLM — app.agents.perception._VISION_LLM 모듈 변수를 교체

실행 방법:
    pytest tests/test_perception.py -v
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_state(context_data: dict | None = None) -> Dict[str, Any]:
    """테스트용 최소 AgentState 생성."""
    return {
        "messages": [],
        "route_type": "vision",
        "plan": [],
        "next_agent": "perception",
        "tool_calls": [],
        "context_data": context_data or {},
        "error_count": {},
        "feedback": "",
    }


def _mock_vision_llm(response_content: str):
    """
    _VISION_LLM 모듈 변수 자체를 MagicMock 으로 교체하는 컨텍스트 매니저 반환.
    ChatOpenAI 는 Pydantic 모델이라 인스턴스 속성을 직접 patch 할 수 없으므로,
    execution.py 의 _mock_llm 헬퍼와 동일하게 모듈 변수를 통째로 교체한다.
    """
    mock_response = MagicMock()
    mock_response.content = response_content

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    return patch("app.agents.perception._VISION_LLM", new=mock_llm)


class TestPerceptionNode:
    async def test_success_returns_vision_description(self):
        """camera_feed 성공 + VLM 정상 응답 → vision_results.status='success'."""
        with (
            patch(
                "app.agents.perception.call_mcp_tool_once",
                new_callable=AsyncMock,
                return_value=("base64FRAME==", "success", "", ""),
            ),
            _mock_vision_llm("비가 내리고 도로가 젖어있습니다."),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.perception import perception_node

            result = await perception_node(_make_state())

        vision_results = result["context_data"]["vision_results"]
        assert vision_results["status"] == "success"
        assert "비" in vision_results["description"]

    async def test_camera_frame_fetch_fail_skips_vlm_call(self):
        """camera_feed 조회 실패 시 VLM 을 호출하지 않고 fail 상태를 반환한다."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()

        with (
            patch(
                "app.agents.perception.call_mcp_tool_once",
                new_callable=AsyncMock,
                return_value=("", "fail", "timeout", "'get_camera_frame' 호출 타임아웃"),
            ),
            patch("app.agents.perception._VISION_LLM", new=mock_llm),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.perception import perception_node

            result = await perception_node(_make_state())

        vision_results = result["context_data"]["vision_results"]
        assert vision_results["status"] == "fail"
        assert vision_results["error_type"] == "timeout"
        mock_llm.ainvoke.assert_not_called()

    async def test_vlm_exception_returns_fail(self):
        """camera_feed 는 성공했지만 VLM 호출이 예외를 던지면 fail/vlm 으로 분류한다."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("connection refused"))

        with (
            patch(
                "app.agents.perception.call_mcp_tool_once",
                new_callable=AsyncMock,
                return_value=("base64FRAME==", "success", "", ""),
            ),
            patch("app.agents.perception._VISION_LLM", new=mock_llm),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.perception import perception_node

            result = await perception_node(_make_state())

        vision_results = result["context_data"]["vision_results"]
        assert vision_results["status"] == "fail"
        assert vision_results["error_type"] == "vlm"
        assert "connection refused" in vision_results["error_msg"]

    async def test_context_data_preserved(self):
        """기존 context_data(vector_results 등)가 덮어씌워지지 않는지 확인."""
        with (
            patch(
                "app.agents.perception.call_mcp_tool_once",
                new_callable=AsyncMock,
                return_value=("base64FRAME==", "success", "", ""),
            ),
            _mock_vision_llm("맑은 날씨입니다."),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.perception import perception_node

            result = await perception_node(
                _make_state(context_data={"vector_results": ["매뉴얼 청크"]})
            )

        assert result["context_data"]["vector_results"] == ["매뉴얼 청크"]

    async def test_ws_tokens_emitted(self):
        """tool_start / tool_result / status 토큰이 WS 로 송출되는지 확인."""
        ws_calls: list[str] = []

        async def capture_ws(msg: str) -> None:
            ws_calls.append(msg)

        with (
            patch(
                "app.agents.perception.call_mcp_tool_once",
                new_callable=AsyncMock,
                return_value=("base64FRAME==", "success", "", ""),
            ),
            _mock_vision_llm("맑은 날씨입니다."),
            patch("app.graph.ws.websocket_manager.send_status", side_effect=capture_ws),
        ):
            from app.agents.perception import perception_node

            await perception_node(_make_state())

        types = [json.loads(m).get("type") for m in ws_calls]
        assert "tool_start" in types
        assert "tool_result" in types
        assert "status" in types
