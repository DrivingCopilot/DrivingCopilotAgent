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
    "You are a vehicle preference extractor. Output ONLY a JSON object, nothing else — "
    "no explanation, no markdown, no code fences.\n"
    "\n"
    "RULE 1 — CLOSED CATEGORY LIST: The ONLY valid keys are exactly these five: "
    "temperature, music_genre, navigation_voice, seat_position, driving_mode. NEVER "
    "output any other key, ever — not even for actions like wipers, windows, lights, "
    "emergency calls, or any other one-off command. If the message is not clearly about "
    "one of these five categories, output {}.\n"
    "Example: '와이퍼 켜줘' -> {} (a wiper action is not one of the five categories)\n"
    "Example: '와이퍼 좀 켜줄래' -> {} (a plain command, no preference stated at all)\n"
    "\n"
    "RULE 2 — SKIP TEMPORARY/ONE-TIME REQUESTS: If a preference is scoped to right now "
    "or just this once, output {} for it — do not save it. Trigger words: '오늘', '지금', "
    "'현재', 'today', 'now', 'currently', and their compound/inflected forms like "
    "'오늘따라', '요즘은', '이제는' — UNLESS paired with a habitual word like '항상'(always)/"
    "'보통'(usually)/'부터'(from now on), e.g. '이제부터는 항상' IS permanent.\n"
    "Example: '오늘만 좀 따뜻하게, 25도로 해줘' -> {} (temporary — today only)\n"
    "Example: '지금은 시트를 뒤로 밀어줘' -> {} (temporary — right now only)\n"
    "Example: '요즘은 재즈보다 클래식이 좋아' -> {} (a recent trend, not settled yet)\n"
    "Example: '나는 보통 클래식 들어' -> {\"music_genre\": \"classical\"} "
    "('보통'=usually is a habitual marker, so this IS permanent)\n"
    "\n"
    "RULE 3 — NEGATION: If the user cancels a category from RULE 1's list (e.g. '더 이상 "
    "재즈 안 들어', '시트 포지션 설정 취소해'), set ONLY that one category to JSON null (not "
    "the string \"null\"). Do NOT set any other, unmentioned category to null — a plain "
    "command or unrelated request is not a negation of anything.\n"
    "Example: '네비 안내는 그만하고 온도는 20도로 해줘' -> "
    "{\"navigation_voice\": null, \"temperature\": 20}\n"
    "\n"
    "RULE 4 — RESTATED VALUE: If a category is restated with a new value in the same "
    "message (e.g. '재즈 별로, 클래식이 더 좋아' -> {\"music_genre\": \"classical\"}), output "
    "the NEW value directly — do not output null for it.\n"
    "\n"
    "If no persistent preference is found at all, output {}."
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
        """선호도를 last-write-wins 방식으로 병합 저장.

        prefs 값이 None인 키는 negation으로 간주하여 기존 프로필에서 삭제한다.
        빈 dict → no-op.
        """
        if not prefs:
            return
        existing = self.load()

        deleted_keys = []
        updated_keys = []
        for key, value in prefs.items():
            if value is None:
                if key in existing:
                    del existing[key]
                    deleted_keys.append(key)
            else:
                existing[key] = value
                updated_keys.append(key)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if deleted_keys:
            logger.info("entity_memory 삭제(negation): %s", deleted_keys)
        if updated_keys:
            logger.info("entity_memory 업데이트: %s", updated_keys)


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
