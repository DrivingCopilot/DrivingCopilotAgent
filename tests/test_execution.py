"""
tests/test_execution.py

Execution Agent 단위 테스트.

테스트 구성:
    [Unit]  Mock MCP 서버 — 실제 프로세스 없이 _call_mcp_tool_raw 를 patch
    [Integ] 실 MCP 서버  — pytest.mark.integration, BACKEND 환경 필요 시 실행

실행 방법:
    # Unit 테스트만 (기본)
    pytest tests/test_execution.py -v

    # Integration 테스트 포함 (Backend 실행 필요)
    pytest tests/test_execution.py -v -m integration
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

def _make_state(
    plan: list[str] | None = None,
    tool_calls: list | None = None,
    context_data: dict | None = None,
    error_count: dict | None = None,
) -> Dict[str, Any]:
    """테스트용 최소 AgentState 생성."""
    return {
        "messages": [],
        "route_type": "tool",
        "plan": plan or [],
        "next_agent": "execution",
        "tool_calls": tool_calls or [],
        "context_data": context_data or {},
        "error_count": error_count or {},
        "feedback": "",
    }


# ---------------------------------------------------------------------------
# Unit 테스트 — Mock MCP 서버
# ---------------------------------------------------------------------------

def _mock_llm(response_content: str):
    """
    _EXTRACTION_LLM 모듈 변수 자체를 MagicMock 으로 교체하는 컨텍스트 매니저 반환.
    ChatOpenAI 는 Pydantic 모델이라 인스턴스 속성을 직접 patch 할 수 없으므로,
    모듈 변수를 통째로 교체해 _get_extraction_llm() 이 mock 을 반환하게 한다.
    """
    mock_response = MagicMock()
    mock_response.content = response_content

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    return patch("app.agents.execution._EXTRACTION_LLM", new=mock_llm)


class TestExtractToolCall:
    """_extract_tool_call: LLM 응답을 파싱해 tool 이름/파라미터를 추출하는지 확인."""

    async def test_valid_tool_extraction(self):
        """정상 LLM 응답 → 올바른 tool_name / params 반환."""
        payload = json.dumps({
            "tool_name": "control_climate",
            "params": {"temperature": 22, "on": True},
        })
        with _mock_llm(payload):
            from app.agents.execution import _extract_tool_call
            result = await _extract_tool_call("에어컨을 22도로 설정한다")

        assert result is not None
        assert result["tool_name"] == "control_climate"
        assert result["params"]["temperature"] == 22
        assert result["params"]["on"] is True

    async def test_no_tool_returns_none(self):
        """tool_name: null 응답 → None 반환."""
        payload = json.dumps({"tool_name": None, "params": {}})
        with _mock_llm(payload):
            from app.agents.execution import _extract_tool_call
            result = await _extract_tool_call("그냥 잡담")

        assert result is None

    async def test_malformed_json_returns_none(self):
        """LLM 이 JSON 이 아닌 텍스트 반환 → None 반환 (파싱 실패 시 안전 처리)."""
        with _mock_llm("죄송합니다, 잘 모르겠습니다."):
            from app.agents.execution import _extract_tool_call
            result = await _extract_tool_call("알 수 없는 요청")

        assert result is None

    async def test_code_block_stripped(self):
        """LLM 이 ```json ... ``` 코드 블록으로 감싸서 반환해도 정상 파싱."""
        raw = '```json\n{"tool_name": "control_wiper", "params": {"on": true}}\n```'
        with _mock_llm(raw):
            from app.agents.execution import _extract_tool_call
            result = await _extract_tool_call("와이퍼를 켜줘")

        assert result is not None
        assert result["tool_name"] == "control_wiper"


class TestCallMcpToolOnce:
    """
    _call_mcp_tool_once: 단일 호출 + 예외 → error_type 매핑 검증.
    retry 로직 없음 — 재시도는 observe_node 책임.
    """

    async def test_success(self):
        """정상 호출 → status='success', error_type=''."""
        with patch(
            "app.agents.execution._call_mcp_tool_raw",
            new_callable=AsyncMock,
            return_value=("에어컨을 켜고 온도를 22℃로 설정했어요.", "success"),
        ):
            from app.agents.execution import _call_mcp_tool_once

            result_text, status, error_type, error_msg = await _call_mcp_tool_once(
                "control_climate", {"temperature": 22, "on": True}
            )

        assert status == "success"
        assert error_type == ""
        assert error_msg == ""
        assert "22℃" in result_text

    async def test_mcp_server_error_classified_as_parameter(self):
        """MCP 서버가 isError 응답 → status='fail', error_type='parameter'."""
        with patch(
            "app.agents.execution._call_mcp_tool_raw",
            new_callable=AsyncMock,
            return_value=("invalid temperature value", "error"),
        ):
            from app.agents.execution import _call_mcp_tool_once

            result_text, status, error_type, error_msg = await _call_mcp_tool_once(
                "control_climate", {"temperature": 99}
            )

        assert status == "fail"
        assert error_type == "parameter"
        assert error_msg != ""

    async def test_timeout_classified(self):
        """TimeoutError → status='fail', error_type='timeout'."""
        with patch("asyncio.wait_for", new_callable=AsyncMock, side_effect=asyncio.TimeoutError):
            from app.agents.execution import _call_mcp_tool_once

            result_text, status, error_type, error_msg = await _call_mcp_tool_once(
                "control_climate", {}
            )

        assert status == "fail"
        assert error_type == "timeout"
        assert "타임아웃" in result_text

    async def test_validation_exception_classified_as_parameter(self):
        """'validation' 키워드 예외 → error_type='parameter'."""
        with patch(
            "asyncio.wait_for",
            new_callable=AsyncMock,
            side_effect=ValueError("validation error: field 'temperature'"),
        ):
            from app.agents.execution import _call_mcp_tool_once

            _, status, error_type, _ = await _call_mcp_tool_once(
                "control_climate", {"temperature": 99}
            )

        assert status == "fail"
        assert error_type == "parameter"

    async def test_unknown_exception_classified_as_parameter(self):
        """예상치 못한 예외 → status='fail', error_type='parameter'."""
        with patch(
            "asyncio.wait_for",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected crash"),
        ):
            from app.agents.execution import _call_mcp_tool_once

            _, status, error_type, _ = await _call_mcp_tool_once(
                "control_climate", {}
            )

        assert status == "fail"
        assert error_type == "parameter"


class TestRunExecution:
    """run_execution: 상태 병합, WS 토큰 송출, 다양한 시나리오 통합 검증."""

    async def test_tool_calls_accumulated(self):
        """기존 tool_calls 에 새 결과가 누적되는지 확인."""
        state = _make_state(
            plan=["에어컨을 22도로 켠다"],
            tool_calls=[{"tool": "old_tool", "status": "success", "result": "이전 결과"}],
        )

        with (
            patch(
                "app.agents.execution._extract_tool_call",
                new_callable=AsyncMock,
                return_value={"tool_name": "control_climate", "params": {"temperature": 22, "on": True}},
            ),
            patch(
                "app.agents.execution._call_mcp_tool_raw",
                new_callable=AsyncMock,
                return_value=("에어컨을 켜고 온도를 22℃로 설정했어요.", "success"),
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.execution import run_execution
            result = await run_execution(state)

        tool_calls = result["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["tool"] == "old_tool"
        assert tool_calls[1]["tool"] == "control_climate"
        assert tool_calls[1]["status"] == "success"

    async def test_fail_sets_error_type_and_error_msg(self):
        """tool 실패 시 error_type · error_msg 필드가 tool_calls 에 포함되는지 확인."""
        state = _make_state(plan=["에어컨을 켠다"])

        with (
            patch(
                "app.agents.execution._extract_tool_call",
                new_callable=AsyncMock,
                return_value={"tool_name": "control_climate", "params": {}},
            ),
            patch(
                "asyncio.wait_for",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError,
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.execution import run_execution
            result = await run_execution(state)

        tc = result["tool_calls"][-1]
        assert tc["status"] == "fail"
        assert tc["error_type"] == "timeout"
        assert tc["error_msg"] != ""

    async def test_no_error_count_in_return(self):
        """run_execution 반환값에 error_count 가 없어야 한다 (observe 책임)."""
        state = _make_state(plan=["와이퍼를 켠다"])

        with (
            patch(
                "app.agents.execution._extract_tool_call",
                new_callable=AsyncMock,
                return_value={"tool_name": "control_wiper", "params": {"on": True}},
            ),
            patch(
                "app.agents.execution._call_mcp_tool_raw",
                new_callable=AsyncMock,
                return_value=("와이퍼를 켰습니다.", "success"),
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.execution import run_execution
            result = await run_execution(state)

        assert "error_count" not in result

    async def test_vehicle_state_updated_on_success(self):
        """tool 성공 시 vehicle_state 에 last_{tool_name} 키가 생성되는지 확인."""
        state = _make_state(plan=["와이퍼를 켠다"])

        with (
            patch(
                "app.agents.execution._extract_tool_call",
                new_callable=AsyncMock,
                return_value={"tool_name": "control_wiper", "params": {"on": True}},
            ),
            patch(
                "app.agents.execution._call_mcp_tool_raw",
                new_callable=AsyncMock,
                return_value=("와이퍼를 켰습니다.", "success"),
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.execution import run_execution
            result = await run_execution(state)

        vehicle_state = result["context_data"]["vehicle_state"]
        assert vehicle_state.get("last_control_wiper") == "와이퍼를 켰습니다."

    async def test_vehicle_state_not_updated_on_fail(self):
        """tool 실패 시 vehicle_state 가 갱신되지 않아야 한다."""
        state = _make_state(plan=["에어컨을 켠다"])

        with (
            patch(
                "app.agents.execution._extract_tool_call",
                new_callable=AsyncMock,
                return_value={"tool_name": "control_climate", "params": {}},
            ),
            patch(
                "asyncio.wait_for",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError,
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.execution import run_execution
            result = await run_execution(state)

        vehicle_state = result["context_data"]["vehicle_state"]
        assert "last_control_climate" not in vehicle_state

    async def test_ws_tokens_emitted(self):
        """tool_start / tool_result 토큰이 WS 로 송출되는지 확인."""
        state = _make_state(plan=["창문을 연다"])
        ws_calls: list[str] = []

        async def capture_ws(msg: str) -> None:
            ws_calls.append(msg)

        with (
            patch(
                "app.agents.execution._extract_tool_call",
                new_callable=AsyncMock,
                return_value={"tool_name": "control_window", "params": {"is_open": True}},
            ),
            patch(
                "app.agents.execution._call_mcp_tool_raw",
                new_callable=AsyncMock,
                return_value=("창문을 열었습니다.", "success"),
            ),
            patch(
                "app.graph.ws.websocket_manager.send_status",
                side_effect=capture_ws,
            ),
        ):
            from app.agents.execution import run_execution
            await run_execution(state)

        types = [json.loads(m).get("type") for m in ws_calls]
        assert "tool_start" in types
        assert "tool_result" in types

    async def test_unknown_tool_skipped_after_retry(self):
        """알 수 없는 tool 이름 추출 시 재추출 실패 → skip (tool_calls 에 추가 안 됨)."""
        state = _make_state(plan=["알 수 없는 작업"])

        with (
            patch(
                "app.agents.execution._extract_tool_call",
                new_callable=AsyncMock,
                return_value={"tool_name": "unknown_tool_xyz", "params": {}},
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.execution import run_execution
            result = await run_execution(state)

        assert result["tool_calls"] == []

    async def test_empty_plan_returns_no_tool_calls(self):
        """plan 이 비어 있으면 tool_calls 변화 없음."""
        state = _make_state(plan=[])

        with patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock):
            from app.agents.execution import run_execution
            result = await run_execution(state)

        assert result["tool_calls"] == []

    async def test_get_vehicle_status_updates_last_status_report(self):
        """get_vehicle_status 성공 시 last_status_report 키로 vehicle_state 갱신."""
        state = _make_state(plan=["차량 상태를 조회한다"])

        with (
            patch(
                "app.agents.execution._extract_tool_call",
                new_callable=AsyncMock,
                return_value={"tool_name": "get_vehicle_status", "params": {}},
            ),
            patch(
                "app.agents.execution._call_mcp_tool_raw",
                new_callable=AsyncMock,
                return_value=("속도 60km/h, RPM 2000, 연료 80%", "success"),
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.execution import run_execution
            result = await run_execution(state)

        vehicle_state = result["context_data"]["vehicle_state"]
        assert "last_status_report" in vehicle_state
        assert "60km/h" in vehicle_state["last_status_report"]

    async def test_context_data_preserved(self):
        """기존 context_data(vector_results 등)가 덮어씌워지지 않는지 확인."""
        state = _make_state(
            plan=["조명을 켠다"],
            context_data={"vector_results": ["매뉴얼 청크"], "vehicle_state": {}},
        )

        with (
            patch(
                "app.agents.execution._extract_tool_call",
                new_callable=AsyncMock,
                return_value={"tool_name": "control_lighting", "params": {"on": True}},
            ),
            patch(
                "app.agents.execution._call_mcp_tool_raw",
                new_callable=AsyncMock,
                return_value=("실내등을 켰습니다.", "success"),
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.execution import run_execution
            result = await run_execution(state)

        assert result["context_data"].get("vector_results") == ["매뉴얼 청크"]


# ---------------------------------------------------------------------------
# Integration 테스트 — 실 MCP 서버
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRunExecutionIntegration:
    """
    실 MCP 서버(DrivingCopilotBackend/mcp_server.py)를 사용하는 통합 테스트.

    실행 전 환경 변수 확인:
        MCP_SERVER_PYTHON  — Backend venv python 경로
        MCP_SERVER_SCRIPT  — mcp_server.py 절대 경로

    실행 방법:
        pytest tests/test_execution.py -m integration -v
    """

    async def test_get_vehicle_status_real(self):
        """실 MCP 서버로 get_vehicle_status 호출 — 차량 상태 문자열 반환 확인."""
        from app.agents.execution import _call_mcp_tool_raw

        result_text, status = await _call_mcp_tool_raw("get_vehicle_status", {})

        assert status == "success"
        assert any(kw in result_text for kw in ("속도", "연료", "배터리", "km/h"))

    async def test_control_climate_real(self):
        """실 MCP 서버로 control_climate 호출 — 성공 메시지 반환 확인."""
        from app.agents.execution import _call_mcp_tool_raw

        result_text, status = await _call_mcp_tool_raw(
            "control_climate", {"temperature": 24, "on": True}
        )

        assert status == "success"
        assert "24" in result_text or "에어컨" in result_text

    async def test_control_wiper_real(self):
        """실 MCP 서버로 control_wiper 호출."""
        from app.agents.execution import _call_mcp_tool_raw

        result_text, status = await _call_mcp_tool_raw("control_wiper", {"on": True})

        assert status == "success"
        assert "와이퍼" in result_text

    async def test_all_12_tools_callable(self):
        """12종 tool 모두 호출 가능 (파라미터 최솟값으로)."""
        from app.agents.execution import _call_mcp_tool_raw, MCP_TOOLS

        sample_params = {
            "control_climate":   {"temperature": 20, "on": True},
            "set_navigation":    {"destination": "서울역"},
            "control_media":     {"action": "play"},
            "get_vehicle_status": {},
            "control_window":    {"is_open": False},
            "control_lighting":  {"on": False},
            "control_seat":      {"direction": "forward"},
            "control_parking":   {"enable": False},
            "trigger_emergency": {"kind": "alert"},
            "set_driving_mode":  {"mode": "normal"},
            "control_wiper":     {"on": False},
            "query_dashboard":   {"metric": "speed"},
        }

        errors: list[str] = []
        for tool in MCP_TOOLS:
            params = sample_params.get(tool, {})
            try:
                _, status = await _call_mcp_tool_raw(tool, params)
                if status == "error":
                    errors.append(f"{tool}: error 응답")
            except Exception as exc:
                errors.append(f"{tool}: {exc}")

        assert errors == [], "호출 실패한 tool:\n" + "\n".join(errors)
