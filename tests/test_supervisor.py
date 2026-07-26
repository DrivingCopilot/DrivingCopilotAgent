"""
tests/test_supervisor.py

Supervisor Agent 단위 테스트.

테스트 구성:
    Mock LLM — app.agents.supervisor.ChatOpenAI 클래스를 patch 해
               with_structured_output(...).ainvoke() 가 미리 정해둔
               {"parsed": SupervisorDecision(...), "parsing_error": None}
               을 반환하도록 한다 (구조화 출력 흉내).
    Mock A2A 발견 — _a2a_client.fetch_all_cards 를 patch.

실행 방법:
    pytest tests/test_supervisor.py -v
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


def _make_state(
    context_data: dict | None = None,
    next_agent: str = "",
    user_query: str = "",
) -> Dict[str, Any]:
    return {
        "messages": [HumanMessage(content=user_query)] if user_query else [],
        "route_type": "vision",
        "plan": [],
        "next_agent": next_agent,
        "tool_calls": [],
        "context_data": context_data or {},
        "error_count": {},
        "feedback": "",
    }


def _mock_llm_returning(json_payload: dict):
    """
    with_structured_output(...).ainvoke() 가 json_payload로 만든 SupervisorDecision을
    {"parsed": ..., "parsing_error": None} 형태로 반환하는 ChatOpenAI Mock을 patch.
    """
    from app.agents.supervisor import SupervisorDecision

    parsed = SupervisorDecision(**json_payload)
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        return_value={
            "raw": AIMessage(content=json.dumps(json_payload)),
            "parsed": parsed,
            "parsing_error": None,
        }
    )
    mock_instance = MagicMock()
    mock_instance.with_structured_output = MagicMock(return_value=mock_structured)

    mock_cls = MagicMock(return_value=mock_instance)
    return patch("app.agents.supervisor.ChatOpenAI", new=mock_cls)


class TestSupervisorNode:
    async def test_repeated_perception_delegation_is_blocked(self):
        """
        vision_results 가 이미 채워진 상태에서 LLM 이 Rule 2 를 어기고 다시
        'perception' 을 선택해도, 코드 레벨 안전장치가 __end__ 로 강제 전환해야
        한다 (7B 모델의 instruction-following 불안정으로 인한 무한 루프 방지).
        """
        llm_payload = {
            "reasoning": "still no results from perception, delegating again",
            "plan": ["delegate_to_perception"],
            "next_agent": "perception",
        }
        state = _make_state(
            context_data={
                "vision_results": {
                    "status": "success",
                    "description": "맑은 날씨입니다.",
                    "hazards": [],
                }
            },
            next_agent="perception",
            user_query="지금 비와?",
        )

        with (
            _mock_llm_returning(llm_payload),
            patch(
                "app.agents.supervisor._a2a_client.fetch_all_cards",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.supervisor import supervisor_node

            result = await supervisor_node(state)

        assert result["next_agent"] == "__end__"
        assert result["plan"] == []
        # 안전장치가 발동해도 모델의 (틀린) reasoning이 아니라, 사용자가 물어본
        # "비" 에 대해 직접 "아니요"로 답하는 결정적 답변이 나가야 한다.
        final_text = result["messages"][0].content
        assert "비" in final_text
        assert "아니요" in final_text
        assert "delegating again" not in final_text

    async def test_first_perception_delegation_is_allowed(self):
        """vision_results 가 비어있을 때는 'perception' 위임이 그대로 통과해야 한다."""
        llm_payload = {
            "reasoning": "need to check the camera for weather conditions",
            "plan": ["delegate_to_perception"],
            "next_agent": "perception",
        }
        state = _make_state(context_data={})

        with (
            _mock_llm_returning(llm_payload),
            patch(
                "app.agents.supervisor._a2a_client.fetch_all_cards",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.supervisor import supervisor_node

            result = await supervisor_node(state)

        assert result["next_agent"] == "perception"

    async def test_repeated_execution_same_tool_is_blocked(self):
        """
        직전에 execution이 get_vehicle_status를 성공 실행했는데 LLM이 또 같은
        tool로 execution 위임을 시도하면 — get_vehicle_status는 매번 success라
        error_count/MAX_RETRY 백스톱이 안 걸려 recursion_limit까지 무한 반복한다 —
        코드 레벨 안전장치가 __end__로 강제 전환해야 한다(execution 재호출 차단).
        """
        llm_payload = {
            "reasoning": "타이어 공기압을 확인하기 위해 상태를 다시 조회합니다.",
            "plan": ["get_vehicle_status"],
            "next_agent": "execution",
        }
        state = {
            "messages": [HumanMessage(content="타이어 공기압 체크는 어떻게 해?")],
            "route_type": "",
            "plan": [],
            "next_agent": "supervisor",
            "tool_calls": [{
                "tool": "get_vehicle_status",
                "params": {},
                "result": "타이어 압력(psi) 33.0/33.0/32.0/33.0",
                "status": "success",
            }],
            "context_data": {},
            "error_count": {},
            "feedback": "",
        }

        with (
            _mock_llm_returning(llm_payload),
            patch(
                "app.agents.supervisor._a2a_client.fetch_all_cards",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.supervisor import supervisor_node

            result = await supervisor_node(state)

        assert result["next_agent"] == "__end__"
        assert result["plan"] == []
        # 무한루프 대신 직전 성공 tool 결과로 결정적 답변이 나가야 한다(침묵 방지).
        assert result["messages"][0].content.strip()

    async def test_execution_different_tool_is_allowed(self):
        """
        멀티스텝 plan의 정당한 다음 단계 — 직전 성공 tool과 다른 tool로 execution을
        위임하는 경우 — 는 재호출 차단에 걸리지 않고 그대로 통과해야 한다.
        """
        llm_payload = {
            "reasoning": "이제 창문을 닫습니다.",
            "plan": ["control_window is_open=false"],
            "next_agent": "execution",
        }
        state = {
            "messages": [HumanMessage(content="에어컨 켜고 창문 닫아줘")],
            "route_type": "",
            "plan": [],
            "next_agent": "supervisor",
            "tool_calls": [{
                "tool": "control_climate",
                "params": {"temperature": 22, "on": True},
                "result": "에어컨을 켜고 온도를 22℃로 설정했어요.",
                "status": "success",
            }],
            "context_data": {},
            "error_count": {},
            "feedback": "",
        }

        with (
            _mock_llm_returning(llm_payload),
            patch(
                "app.agents.supervisor._a2a_client.fetch_all_cards",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.supervisor import supervisor_node

            result = await supervisor_node(state)

        assert result["next_agent"] == "execution"

    async def test_vision_answer_confirms_asked_hazard_when_present(self):
        """비가 감지된 상태에서 '비와?'를 물으면 '네, 비가 감지되었습니다' 류로 답해야 한다."""
        llm_payload = {
            "reasoning": "still delegating, ignoring rule 2",
            "plan": ["delegate_to_perception"],
            "next_agent": "perception",
        }
        state = _make_state(
            context_data={
                "vision_results": {
                    "status": "success",
                    "description": "비가 내리고 있습니다.",
                    "hazards": ["rain"],
                }
            },
            next_agent="perception",
            user_query="지금 비와?",
        )

        with (
            _mock_llm_returning(llm_payload),
            patch(
                "app.agents.supervisor._a2a_client.fetch_all_cards",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.supervisor import supervisor_node

            result = await supervisor_node(state)

        final_text = result["messages"][0].content
        assert "네" in final_text
        assert "비" in final_text

    async def test_vision_answer_is_deterministic_even_without_rule2_violation(self):
        """
        모델이 Rule 2를 어기지 않고 스스로 '__end__'를 선택해도, 사용자에게
        보일 답변은 모델의 reasoning이 아니라 vision_results 기반 결정적 답변이어야
        한다 — 답변 생성 로직이 Rule 2 위반 감지와 독립적임을 검증한다.
        """
        llm_payload = {
            "reasoning": "The scene looks calm and there is no rain visible.",
            "plan": [],
            "next_agent": "__end__",
        }
        state = _make_state(
            context_data={
                "vision_results": {
                    "status": "success",
                    "description": "맑은 날씨입니다.",
                    "hazards": [],
                }
            },
            next_agent="perception",
            user_query="지금 비와?",
        )

        with (
            _mock_llm_returning(llm_payload),
            patch(
                "app.agents.supervisor._a2a_client.fetch_all_cards",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.supervisor import supervisor_node

            result = await supervisor_node(state)

        assert result["next_agent"] == "__end__"
        final_text = result["messages"][0].content
        assert "아니요" in final_text
        assert "비" in final_text
        assert final_text != llm_payload["reasoning"]


def _mock_llm_parsing_error(error: Exception):
    """with_structured_output(...).ainvoke() 가 parsing_error를 채워 반환하는 Mock을 patch."""
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        return_value={"raw": AIMessage(content="broken"), "parsed": None, "parsing_error": error}
    )
    mock_instance = MagicMock()
    mock_instance.with_structured_output = MagicMock(return_value=mock_structured)
    mock_cls = MagicMock(return_value=mock_instance)
    return patch("app.agents.supervisor.ChatOpenAI", new=mock_cls)


class TestSupervisorParsingError:
    """구조화 출력 파싱 실패(parsing_error) 시 재시도/포기 분기 검증."""

    async def test_retries_when_under_limit(self):
        state = _make_state()
        state["error_count"] = {"parameter": 0}

        with (
            _mock_llm_parsing_error(ValueError("boom")),
            patch(
                "app.agents.supervisor._a2a_client.fetch_all_cards",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.supervisor import supervisor_node

            result = await supervisor_node(state)

        assert result["next_agent"] == "supervisor"
        assert result["error_count"]["parameter"] == 1

    async def test_gives_up_at_limit(self):
        state = _make_state()
        state["error_count"] = {"parameter": 1}  # MAX_RETRY["parameter"] == 2, 이번이 마지막

        with (
            _mock_llm_parsing_error(ValueError("boom")),
            patch(
                "app.agents.supervisor._a2a_client.fetch_all_cards",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.supervisor import supervisor_node

            result = await supervisor_node(state)

        assert result["next_agent"] == "__end__"
        assert result["error_count"]["parameter"] == 2
        assert "실패했습니다" in result["messages"][0].content


class TestComposeToolResultSummary:
    """last_tool_call로부터 결정적 최종 답변을 구성하는 _compose_tool_result_summary 검증."""

    def test_success_uses_result_text_as_is(self):
        from app.agents.supervisor import _compose_tool_result_summary

        last_tool_call = {"tool": "control_wiper", "params": {"on": True}, "result": "와이퍼를 켰습니다.", "status": "success"}
        assert _compose_tool_result_summary(last_tool_call) == "와이퍼를 켰습니다."

    def test_error_includes_tool_and_error_msg(self):
        from app.agents.supervisor import _compose_tool_result_summary

        last_tool_call = {
            "tool": "control_climate",
            "params": {},
            "result": "",
            "status": "error",
            "error_type": "parameter",
            "error_msg": "temperature 값이 범위를 벗어났습니다.",
        }
        summary = _compose_tool_result_summary(last_tool_call)
        assert "control_climate" in summary
        assert "temperature 값이 범위를 벗어났습니다." in summary


class TestFilterInternalPlanSteps:
    """plan 배열에서 LangGraph 내부 라우팅 예약어를 걸러내는 _filter_internal_plan_steps 검증."""

    def test_removes_end_token(self):
        from app.agents.supervisor import _filter_internal_plan_steps

        assert _filter_internal_plan_steps(["control_wiper on=true", "__end__"]) == ["control_wiper on=true"]

    def test_keeps_normal_steps_untouched(self):
        from app.agents.supervisor import _filter_internal_plan_steps

        steps = ["control_wiper on=true", "control_lighting on=true"]
        assert _filter_internal_plan_steps(steps) == steps


class TestSupervisorFinalTextPriority:
    """final_text 우선순위(vision_results > last_tool_call > last_knowledge_result > reasoning) 검증."""

    async def test_uses_last_tool_call_over_reasoning(self):
        state = _make_state()
        state["tool_calls"] = [
            {"tool": "control_wiper", "params": {"on": True}, "result": "와이퍼를 켰습니다.", "status": "success"}
        ]

        llm_payload = {
            "reasoning": "The user's request to turn on the wiper was already fulfilled.",
            "plan": [],
            "next_agent": "__end__",
        }

        with (
            _mock_llm_returning(llm_payload),
            patch(
                "app.agents.supervisor._a2a_client.fetch_all_cards",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.graph.ws.websocket_manager.send_status", new_callable=AsyncMock),
        ):
            from app.agents.supervisor import supervisor_node

            result = await supervisor_node(state)

        final_text = result["messages"][0].content
        assert final_text == "와이퍼를 켰습니다."
        assert final_text != llm_payload["reasoning"]
