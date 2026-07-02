"""tests/test_supervisor_entity.py"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage

from app.agents.supervisor import supervisor_node


async def _stream_json(content: str):
    yield MagicMock(content=content)


def _base_state(context_data: dict | None = None) -> dict:
    return {
        "messages": [HumanMessage(content="테스트")],
        "route_type": "",
        "plan": [],
        "next_agent": "",
        "tool_calls": [],
        "context_data": context_data if context_data is not None else {"retrieved_experience": []},
        "error_count": {"timeout": 0, "parameter": 0, "invalid_tool": 0, "sql": 0},
        "feedback": "",
    }


@pytest.mark.asyncio
async def test_profile_injected_in_context():
    """profile에 값이 있을 때 eval_prompt에 포함되는지 확인"""
    profile = {"preferred_temp": "22"}
    mock_em = MagicMock()
    mock_em.load.return_value = profile

    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(
        return_value=_stream_json('{"reasoning":"ok","plan":[],"next_agent":"__end__"}')
    )

    with patch("app.agents.supervisor.ChatOpenAI", return_value=mock_llm), \
         patch("app.agents.supervisor._get_entity_memory", return_value=mock_em), \
         patch("app.agents.supervisor._ws.websocket_manager.send_status", new=AsyncMock()):
        await supervisor_node(_base_state())

    messages_sent = mock_llm.astream.call_args[0][0]
    eval_prompt = messages_sent[-1].content
    assert "preferred_temp" in eval_prompt


@pytest.mark.asyncio
async def test_empty_profile_injected_in_context():
    """profile이 {} 일 때 eval_prompt에 {} 그대로 포함되는지 확인"""
    mock_em = MagicMock()
    mock_em.load.return_value = {}

    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(
        return_value=_stream_json('{"reasoning":"ok","plan":[],"next_agent":"__end__"}')
    )

    with patch("app.agents.supervisor.ChatOpenAI", return_value=mock_llm), \
         patch("app.agents.supervisor._get_entity_memory", return_value=mock_em), \
         patch("app.agents.supervisor._ws.websocket_manager.send_status", new=AsyncMock()):
        await supervisor_node(_base_state())

    messages_sent = mock_llm.astream.call_args[0][0]
    eval_prompt = messages_sent[-1].content
    assert "User Profile (preferences): {}" in eval_prompt
