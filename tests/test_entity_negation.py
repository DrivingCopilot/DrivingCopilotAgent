"""
tests/test_entity_negation.py

EntityMemory.update() 의 negation(None → 키 삭제) 동작 단위 테스트.
tmp_path fixture로 파일 격리.
"""

from app.memory.entity import EntityMemory


def test_delete_existing_key(tmp_path):
    p = tmp_path / "profile.json"
    mem = EntityMemory(path=p)
    mem.update({"music_genre": "jazz", "temperature": 22})
    mem.update({"music_genre": None})
    result = mem.load()
    assert "music_genre" not in result
    assert result["temperature"] == 22


def test_delete_nonexistent_key_is_noop(tmp_path):
    p = tmp_path / "profile.json"
    mem = EntityMemory(path=p)
    mem.update({"temperature": 22})
    mem.update({"music_genre": None})  # 존재한 적 없는 키
    result = mem.load()
    assert result == {"temperature": 22}
    assert "music_genre" not in result


def test_mixed_add_and_delete(tmp_path):
    p = tmp_path / "profile.json"
    mem = EntityMemory(path=p)
    mem.update({"music_genre": "jazz", "temperature": 22})
    mem.update({"music_genre": None, "seat_position": 3})
    result = mem.load()
    assert result == {"temperature": 22, "seat_position": 3}


def test_delete_all_keys_results_in_empty_dict(tmp_path):
    p = tmp_path / "profile.json"
    mem = EntityMemory(path=p)
    mem.update({"music_genre": "jazz", "temperature": 22})
    mem.update({"music_genre": None, "temperature": None})
    result = mem.load()
    assert result == {}


def test_empty_prefs_is_noop(tmp_path):
    p = tmp_path / "profile.json"
    mem = EntityMemory(path=p)
    mem.update({"temperature": 22})
    mem.update({})
    result = mem.load()
    assert result == {"temperature": 22}
