"""
tests/test_entity.py

EntityMemory 및 extract_preferences 단위 테스트.
tmp_path fixture로 파일 격리. LLM은 AsyncMock.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.memory.entity import EntityMemory, extract_preferences

# ---------------------------------------------------------------------------
# EntityMemory 테스트 (6)
# ---------------------------------------------------------------------------

def test_load_no_file(tmp_path):
    mem = EntityMemory(path=tmp_path / "profile.json")
    assert mem.load() == {}


def test_load_corrupted_json(tmp_path):
    p = tmp_path / "profile.json"
    p.write_text("not-json!!!", encoding="utf-8")
    mem = EntityMemory(path=p)
    assert mem.load() == {}


def test_load_non_dict_json(tmp_path):
    p = tmp_path / "profile.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    mem = EntityMemory(path=p)
    assert mem.load() == {}


def test_update_merge(tmp_path):
    p = tmp_path / "profile.json"
    mem = EntityMemory(path=p)
    mem.update({"temperature": 22})
    mem.update({"music_genre": "jazz", "temperature": 24})  # last-write-wins
    result = mem.load()
    assert result["temperature"] == 24
    assert result["music_genre"] == "jazz"


def test_update_empty_noop(tmp_path):
    p = tmp_path / "profile.json"
    mem = EntityMemory(path=p)
    mem.update({})
    assert not p.exists(), "빈 dict update는 파일을 생성하지 않아야 한다"


def test_load_update_roundtrip(tmp_path):
    p = tmp_path / "profile.json"
    mem = EntityMemory(path=p)

    assert mem.load() == {}
    mem.update({"driving_mode": "eco"})
    assert mem.load() == {"driving_mode": "eco"}
    mem.update({"seat_position": 3})
    assert mem.load() == {"driving_mode": "eco", "seat_position": 3}


# ---------------------------------------------------------------------------
# extract_preferences 테스트 (6)
# ---------------------------------------------------------------------------

def _make_llm(content: str) -> AsyncMock:
    mock = AsyncMock()
    mock.ainvoke.return_value = MagicMock(content=content)
    return mock


@pytest.mark.asyncio
async def test_extract_normal():
    llm = _make_llm('{"temperature": 22, "music_genre": "jazz"}')
    result = await extract_preferences("온도는 22도로, 음악은 재즈로 설정해줘", llm)
    assert result == {"temperature": 22, "music_genre": "jazz"}
    llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_extract_empty_message():
    llm = _make_llm("{}")
    result = await extract_preferences("   ", llm)
    assert result == {}
    llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_extract_transient_expression():
    """일시성 표현 → LLM이 {} 반환. 시스템 프롬프트 일시성 필터 지시 포함 검증."""
    llm = _make_llm("{}")
    result = await extract_preferences("오늘만 따뜻하게 해줘. 지금은 22도로", llm)
    assert result == {}
    # 시스템 프롬프트에 일시성 필터 지시가 있는지 확인
    call_args = llm.ainvoke.call_args[0][0]
    system_content = call_args[0].content.lower()
    assert "today" in system_content or "transient" in system_content or "temporary" in system_content


@pytest.mark.asyncio
async def test_extract_codefence_response():
    llm = _make_llm('```json\n{"navigation_voice": "female"}\n```')
    result = await extract_preferences("내비 음성을 여성으로 해줘", llm)
    assert result == {"navigation_voice": "female"}


@pytest.mark.asyncio
async def test_extract_non_dict_response():
    llm = _make_llm("[1, 2, 3]")
    result = await extract_preferences("창문 열어줘", llm)
    assert result == {}


@pytest.mark.asyncio
async def test_extract_non_json_response():
    llm = _make_llm("죄송합니다, 선호도를 추출할 수 없습니다.")
    result = await extract_preferences("뭔가 말해줘", llm)
    assert result == {}
