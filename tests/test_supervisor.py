"""
tests/test_supervisor.py

Supervisor Agent 단위 테스트.

테스트 구성:
    Mock LLM — app.agents.supervisor.ChatOllama 클래스를 patch 해
               with_structured_output(...).ainvoke() 가 미리 정해둔
               {"parsed": SupervisorDecision(...), "parsing_error": None}
               을 반환하도록 한다 (grammar-constrained 구조화 출력 흉내).
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
    messages: list | None = None,
) -> Dict[str, Any]:
    if messages is None:
        messages = [HumanMessage(content=user_query)] if user_query else []
    return {
        "messages": messages,
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
    {"parsed": ..., "parsing_error": None} 형태로 반환하는 ChatOllama Mock을 patch.
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
    return patch("app.agents.supervisor.ChatOllama", new=mock_cls)


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
                    "related_hazard": "rain",
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
                    "related_hazard": "rain",
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
                    "related_hazard": "rain",
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
    return patch("app.agents.supervisor.ChatOllama", new=mock_cls)


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


class TestComposeVisionSummary:
    """
    vision_results로부터 최종 답변을 구성하는 _compose_vision_summary 검증.
    related_hazard는 perception.py의 VLM이 이미지+질문을 보고 직접 판단해
    vision_results에 채워 넣는 값이라, 여기서는 텍스트 키워드 매칭 없이 그
    값을 그대로 신뢰해야 한다.
    """

    def test_related_hazard_present_and_detected_confirms_yes(self):
        from app.agents.supervisor import _compose_vision_summary

        vision_results = {
            "status": "success",
            "description": "비가 내리고 있습니다.",
            "hazards": ["rain"],
            "related_hazard": "rain",
        }
        result = _compose_vision_summary(vision_results)
        assert "네" in result
        assert "비" in result

    def test_related_hazard_present_but_not_detected_confirms_no(self):
        from app.agents.supervisor import _compose_vision_summary

        vision_results = {
            "status": "success",
            "description": "맑은 날씨입니다.",
            "hazards": [],
            "related_hazard": "rain",
        }
        result = _compose_vision_summary(vision_results)
        assert "아니요" in result
        assert "비" in result

    def test_no_related_hazard_uses_vlm_answer(self):
        """3종 hazard 어휘 밖의 질문(예: 도로 표지판)은 VLM이 직접 작성한 answer를 그대로 쓴다."""
        from app.agents.supervisor import _compose_vision_summary

        vision_results = {
            "status": "success",
            "description": "도로에 속도제한 50 표지판이 보입니다.",
            "hazards": [],
            "related_hazard": None,
            "answer": "전방 표지판은 속도제한 50 표지판입니다.",
        }
        result = _compose_vision_summary(vision_results)
        assert result == "전방 표지판은 속도제한 50 표지판입니다."

    def test_no_related_hazard_and_no_answer_falls_back_to_hazard_summary(self):
        """answer가 없는(구버전 vision_results 등) 경우 기존 hazard 통보 fallback으로 떨어진다."""
        from app.agents.supervisor import _compose_vision_summary

        vision_results = {
            "status": "success",
            "description": "터널 안입니다.",
            "hazards": ["tunnel"],
        }
        result = _compose_vision_summary(vision_results)
        assert "터널" in result
        assert "감지되었습니다" in result

    def test_no_hazard_no_answer_reports_no_risk(self):
        from app.agents.supervisor import _compose_vision_summary

        vision_results = {"status": "success", "description": "맑음", "hazards": []}
        result = _compose_vision_summary(vision_results)
        assert "감지되지 않았습니다" in result

    def test_failure_status_reports_error(self):
        from app.agents.supervisor import _compose_vision_summary

        vision_results = {"status": "fail", "error_msg": "카메라 타임아웃"}
        result = _compose_vision_summary(vision_results)
        assert "카메라 타임아웃" in result


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

    async def test_vision_summary_ignores_stale_messages_in_multiturn(self):
        """
        회귀 테스트: final_text 조립이 messages[0](윈도우에서 가장 오래된 메시지)이
        아니라 vision_results.answer를 그대로 써야 한다 — 이전엔 messages[0].content를
        읽어 멀티턴에서 몇 턴 전 메시지를 기준으로 답을 만드는 버그가 있었다.
        """
        llm_payload = {"reasoning": "answering from vision results", "plan": [], "next_agent": "__end__"}
        state = _make_state(
            context_data={
                "vision_results": {
                    "status": "success",
                    "description": "도로에 속도제한 50 표지판이 보입니다.",
                    "hazards": [],
                    "related_hazard": None,
                    "answer": "전방 표지판은 속도제한 50 표지판입니다.",
                }
            },
            next_agent="perception",
            messages=[
                HumanMessage(content="와이퍼 켜줘"),
                AIMessage(content="와이퍼를 켰습니다."),
                HumanMessage(content="전방 경고 표시판이 뭐야?"),
            ],
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
        assert final_text == "전방 표지판은 속도제한 50 표지판입니다."
        assert "와이퍼" not in final_text
