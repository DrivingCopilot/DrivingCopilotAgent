"""
tests/test_experience.py

app/memory/experience.py 단위 테스트.

get_experience_memory() 가 프로세스당 ExperienceMemory 를 단 한 번만
생성하는지(=임베딩 모델 1회 로드) 검증한다.
"""

from __future__ import annotations


def test_experience_memory_created_once(monkeypatch):
    """get_experience_memory()는 ExperienceMemory를 프로세스당 한 번만 생성한다."""
    import app.memory.experience as exp

    monkeypatch.setattr(exp, "_instance", None)  # 테스트 격리

    call_count = {"n": 0}

    def fake_init(self):
        call_count["n"] += 1

    monkeypatch.setattr(exp.ExperienceMemory, "__init__", fake_init)

    first = exp.get_experience_memory()
    second = exp.get_experience_memory()

    assert call_count["n"] == 1
    assert first is second
