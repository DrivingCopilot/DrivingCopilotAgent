"""tests/test_finalize.py"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agents.finalize import finalize_node


@pytest.mark.asyncio
async def test_normal_utterance():
    """정상 발화 → extract_preferences 호출 → update 호출"""
    state = {
        "messages": [HumanMessage(content="에어컨 24도로 설정해줘")],
    }
    mock_em = MagicMock()
    prefs = {"climate_temp": "24"}

    with patch("app.agents.finalize.extract_preferences", new=AsyncMock(return_value=prefs)), \
         patch("app.agents.finalize._get_entity_memory", return_value=mock_em), \
         patch("app.agents.finalize._get_extractor_llm", return_value=MagicMock()):
        result = await finalize_node(state)

    assert result == {}
    mock_em.update.assert_called_once_with(prefs)


@pytest.mark.asyncio
async def test_empty_utterance():
    """HumanMessage 없음 → extract_preferences 결과 {} → update 미호출"""
    state = {
        "messages": [],
    }
    mock_em = MagicMock()

    with patch("app.agents.finalize.extract_preferences", new=AsyncMock(return_value={})), \
         patch("app.agents.finalize._get_entity_memory", return_value=mock_em), \
         patch("app.agents.finalize._get_extractor_llm", return_value=MagicMock()):
        result = await finalize_node(state)

    assert result == {}
    mock_em.update.assert_not_called()


@pytest.mark.asyncio
async def test_no_prefs_extracted():
    """extract_preferences 결과 {} → update 미호출 (no-op)"""
    state = {
        "messages": [HumanMessage(content="안녕")],
    }
    mock_em = MagicMock()

    with patch("app.agents.finalize.extract_preferences", new=AsyncMock(return_value={})), \
         patch("app.agents.finalize._get_entity_memory", return_value=mock_em), \
         patch("app.agents.finalize._get_extractor_llm", return_value=MagicMock()):
        result = await finalize_node(state)

    assert result == {}
    mock_em.update.assert_not_called()
