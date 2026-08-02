"""
tests/test_experience_dedup.py

ExperienceMemory 저장 dedup(_has_near_duplicate, save) 단위 테스트.
실제 임베딩/Qdrant 연결 없이 _vectorstore를 mock으로 대체한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import VECTOR_SIZE
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


class _FixedEmbeddings(Embeddings):
    """필터 매칭 검증용 — 벡터값 자체는 무관하므로 고정 벡터 반환."""

    def embed_documents(self, texts):
        return [[0.1] * VECTOR_SIZE for _ in texts]

    def embed_query(self, text):
        return [0.1] * VECTOR_SIZE


def _make_real_memory_with_inmemory_qdrant() -> ExperienceMemory:
    """실제 로컬 Qdrant 백엔드(:memory:)로 _vectorstore를 구성한다.
    MagicMock과 달리 LangChain이 실제로 만드는 중첩 payload
    ({'page_content':..., 'metadata': {...}})에 대해 진짜 Filter가 매칭되는지 검증 가능."""
    client = QdrantClient(location=":memory:")
    collection_name = "test_experience_regression"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=_FixedEmbeddings()
    )
    memory = ExperienceMemory.__new__(ExperienceMemory)
    memory._vectorstore = vectorstore
    return memory


def test_has_near_duplicate_matches_real_nested_payload_route_type_filter():
    """회귀 테스트: LangChain이 metadata를 payload에 중첩 저장하므로
    FieldCondition key는 'metadata.route_type'이어야 매칭된다.
    (수정 전 key='route_type'이면 이 테스트는 실패한다 — 항상 매칭 실패 → dedup 무력화)"""
    memory = _make_real_memory_with_inmemory_qdrant()
    situation = "query: 창문 열어줘 | route_type: tool | vehicle_state: {}"
    memory._vectorstore.add_documents(
        [Document(page_content=situation, metadata={"lesson": "재시도 전략", "route_type": "tool"})]
    )

    assert memory._has_near_duplicate(situation, "tool") is True


def test_search_matches_real_nested_payload_route_type_filter():
    """회귀 테스트: search()의 route_type 필터도 동일한 중첩 구조를 매칭해야 한다."""
    memory = _make_real_memory_with_inmemory_qdrant()
    situation = "query: 창문 열어줘 | route_type: tool | vehicle_state: {}"
    memory._vectorstore.add_documents(
        [Document(page_content=situation, metadata={"lesson": "재시도 전략", "route_type": "tool"})]
    )

    results = memory.search(situation, route_type="tool", top_k=5)

    assert len(results) == 1
    assert results[0].metadata["lesson"] == "재시도 전략"
