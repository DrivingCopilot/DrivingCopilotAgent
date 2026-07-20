"""
tests/test_experience_dedup.py

ExperienceMemory 저장 dedup(_has_near_duplicate, save) 단위 테스트.
실제 임베딩/Qdrant 연결 없이 _vectorstore를 mock으로 대체한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.memory.experience import ExperienceMemory


def _make_memory(vectorstore: MagicMock) -> ExperienceMemory:
    """__init__을 거치지 않고 _vectorstore만 주입한 인스턴스를 만든다."""
    memory = ExperienceMemory.__new__(ExperienceMemory)
    memory._vectorstore = vectorstore
    return memory


def test_has_near_duplicate_true_for_similar_situation():
    vectorstore = MagicMock()
    vectorstore.similarity_search_with_score.return_value = [
        (Document(page_content="query: 창문 열어줘 | route_type: tool | vehicle_state: {}"), 0.97)
    ]
    memory = _make_memory(vectorstore)

    assert memory._has_near_duplicate("query: 창문 열어줘 | route_type: tool | vehicle_state: {}", "tool") is True


def test_has_near_duplicate_false_for_different_situation():
    vectorstore = MagicMock()
    vectorstore.similarity_search_with_score.return_value = [
        (Document(page_content="query: 목적지까지 얼마나 남았어 | route_type: rag | vehicle_state: {}"), 0.3)
    ]
    memory = _make_memory(vectorstore)

    assert memory._has_near_duplicate("query: 창문 열어줘 | route_type: tool | vehicle_state: {}", "tool") is False


def test_has_near_duplicate_false_when_no_results():
    vectorstore = MagicMock()
    vectorstore.similarity_search_with_score.return_value = []
    memory = _make_memory(vectorstore)

    assert memory._has_near_duplicate("query: 창문 열어줘 | route_type: tool | vehicle_state: {}", "tool") is False


def test_has_near_duplicate_false_on_search_exception():
    """검색 자체가 실패하면 dedup을 스킵(False)해 저장 진행을 막지 않는다."""
    vectorstore = MagicMock()
    vectorstore.similarity_search_with_score.side_effect = RuntimeError("qdrant down")
    memory = _make_memory(vectorstore)

    assert memory._has_near_duplicate("query: 창문 열어줘 | route_type: tool | vehicle_state: {}", "tool") is False


def test_save_skips_when_near_duplicate(monkeypatch):
    vectorstore = MagicMock()
    memory = _make_memory(vectorstore)
    monkeypatch.setattr(memory, "_has_near_duplicate", lambda situation, route_type: True)

    result = memory.save(situation="query: 창문 열어줘", lesson="재시도 전략", route_type="tool")

    assert result is False
    vectorstore.add_documents.assert_not_called()


def test_save_stores_when_not_duplicate(monkeypatch):
    vectorstore = MagicMock()
    memory = _make_memory(vectorstore)
    monkeypatch.setattr(memory, "_has_near_duplicate", lambda situation, route_type: False)

    result = memory.save(situation="query: 창문 열어줘", lesson="재시도 전략", route_type="tool")

    assert result is True
    vectorstore.add_documents.assert_called_once()
    saved_doc = vectorstore.add_documents.call_args[0][0][0]
    assert saved_doc.page_content == "query: 창문 열어줘"
    assert saved_doc.metadata == {"lesson": "재시도 전략", "route_type": "tool"}


def test_save_proceeds_when_dedup_search_fails(monkeypatch):
    """근접 중복 검색이 예외를 던져도 save()는 정상적으로 저장을 진행한다."""
    vectorstore = MagicMock()
    vectorstore.similarity_search_with_score.side_effect = RuntimeError("qdrant down")
    memory = _make_memory(vectorstore)

    result = memory.save(situation="query: 창문 열어줘", lesson="재시도 전략", route_type="tool")

    assert result is True
    vectorstore.add_documents.assert_called_once()
