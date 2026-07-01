"""
tests/test_supervisor.py

Supervisor Agent 단위 테스트.

테스트 구성:
    Mock LLM — app.agents.supervisor.ChatOpenAI 클래스를 patch 해
               astream() 이 미리 정해둔 JSON 청크를 흘려보내도록 한다.
    Mock A2A 발견 — _a2a_client.fetch_all_cards 를 patch.

실행 방법:
    pytest tests/test_supervisor.py -v
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage


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
    """astream() 이 json_payload 를 한 청크로 흘려보내는 ChatOpenAI Mock 클래스를 patch."""

    async def _astream(_messages):
        chunk = MagicMock()
        chunk.content = json.dumps(json_payload)
        yield chunk

    mock_instance = MagicMock()
    mock_instance.astream = _astream

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
