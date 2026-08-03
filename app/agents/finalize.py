"""
app/agents/finalize.py

매 턴 종료 시 사용자 발화에서 선호도를 추출하고 EntityMemory에 저장 (계획서 2.5).
WebSocket done 전송은 여기서 하지 않음 — 기존 supervisor/reflect에서 그대로.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from app.graph.state import AgentState
from app.memory.entity import extract_preferences

logger = logging.getLogger(__name__)

_entity_memory = None
_extractor_llm = None


def _get_entity_memory():
    global _entity_memory
    if _entity_memory is None:
        from app.memory.entity import EntityMemory
        _entity_memory = EntityMemory()
    return _entity_memory


def _get_extractor_llm():
    global _extractor_llm
    if _extractor_llm is None:
        from langchain_openai import ChatOpenAI
        from app.core.config import MODEL_SERVER_URL, QWEN_TEXT_MODEL_NAME
        _extractor_llm = ChatOpenAI(
            model=QWEN_TEXT_MODEL_NAME, temperature=0.0, base_url=MODEL_SERVER_URL,
        )
    return _extractor_llm


async def finalize_node(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    user_message = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    try:
        llm = _get_extractor_llm()
        prefs = await extract_preferences(user_message, llm)
    except Exception as e:
        logger.warning("finalize: 선호도 추출 실패 (%s) — skip", e)
        return {}

    if not prefs:
        logger.info("finalize: 추출된 선호도 없음 — skip")
        return {}

    try:
        await asyncio.to_thread(_get_entity_memory().update, prefs)
        logger.info("finalize: 선호도 저장 완료 — %s", prefs)
    except Exception as e:
        logger.error("finalize: EntityMemory 저장 실패 — %s", e)

    return {}
