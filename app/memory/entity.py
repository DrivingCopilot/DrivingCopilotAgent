"""
app/memory/entity.py

사용자 차량 선호도를 로컬 JSON 파일에 저장·검색하는 엔티티 메모리 모듈 (계획서 2.5).
last-write-wins 병합 방식의 KV Store. 그래프 연결(finalize/supervisor)은 별도 단계.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import ENTITY_PROFILE_PATH

logger = logging.getLogger(__name__)

_EXTRACTOR_SYSTEM_PROMPT = (
    "You are a vehicle preference extractor.\n"
    "Extract ONLY persistent vehicle preferences from the user message.\n"
    "Focus on these categories: temperature, music_genre, navigation_voice, "
    "seat_position, driving_mode, and similar vehicle settings.\n"
    "IMPORTANT: Exclude any temporary or transient preferences that include time "
    "references such as 'today', 'now', 'currently', 'right now', 'for today', "
    "'지금', '오늘', '현재', or similar expressions.\n"
    "Return ONLY a valid JSON object with no explanation, no markdown, no code fences.\n"
    "If no persistent preferences are found, return {}."
)


class EntityMemory:
    """사용자 차량 선호도를 로컬 JSON 파일에 저장하는 KV Store."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else Path(ENTITY_PROFILE_PATH)

    def load(self) -> dict[str, Any]:
        """저장된 선호도를 반환. 파일 없음·손상·비-dict → {}."""
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("entity_memory 파일 로드 실패: %s", self._path)
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def update(self, prefs: dict[str, Any]) -> None:
        """선호도를 last-write-wins 방식으로 병합 저장. 빈 dict → no-op."""
        if not prefs:
            return
        existing = self.load()
        existing.update(prefs)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("entity_memory 업데이트: %s", list(prefs.keys()))


async def extract_preferences(user_message: str, llm: BaseChatModel) -> dict[str, Any]:
    """
    사용자 메시지에서 차량 선호도를 추출한다.

    Args:
        user_message: 사용자 입력 텍스트
        llm: LangChain BaseChatModel

    Returns:
        추출된 선호도 dict. 빈 메시지·추출 실패 → {}.
    """
    if not user_message.strip():
        return {}

    response = await llm.ainvoke([
        SystemMessage(content=_EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    content = response.content.strip()

    # 코드펜스 제거: ```json ... ``` 또는 ``` ... ```
    if content.startswith("```"):
        lines = content.splitlines()
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        content = "\n".join(lines[start:end]).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("extract_preferences: LLM 응답이 JSON이 아님")
        return {}

    if not isinstance(data, dict):
        return {}

    return data
