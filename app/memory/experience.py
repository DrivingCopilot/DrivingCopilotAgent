"""
app/memory/experience.py

ReAct Reflect 단계의 실패 경험을 Qdrant에 저장하고 검색하는 모듈.
embedder.py와 동일한 패턴(LangChain QdrantVectorStore + HuggingFaceEmbeddings).
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from app.core.config import (
    MODEL_NAME,
    VECTOR_SIZE,
    EXPERIENCE_COLLECTION_NAME,
    EXPERIENCE_DEDUP_THRESHOLD,
)
from app.services.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)


def build_situation(user_query: str, route_type: str, vehicle_state: dict | None = None) -> str:
    """save와 search가 동일한 임베딩 텍스트를 쓰도록 situation 문자열을 구성한다."""
    vehicle_state_summary = str(vehicle_state or {})[:200]
    return (
        f"query: {user_query} | route_type: {route_type} | "
        f"vehicle_state: {vehicle_state_summary}"
    )


class ExperienceMemory:
    """
    실패 경험을 Qdrant에 누적하고 유사 상황을 검색한다.

    Args:
        embeddings: 외부 주입 임베딩 모델. None이면 MODEL_NAME으로 새로 로드.
                    supervisor나 embedder와 임베딩 모델을 공유할 때 사용.
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings | None = None) -> None:
        self._embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._client = get_qdrant_client()
        self._ensure_collection()

        self._vectorstore = QdrantVectorStore(
            client=self._client,
            collection_name=EXPERIENCE_COLLECTION_NAME,
            embedding=self._embeddings,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        situation: str,
        route_type: str | None = None,
        top_k: int = 5,
    ) -> list[Document]:
        """
        유사 실패 경험을 검색한다.

        Args:
            situation: 현재 상황 텍스트 (user query + vehicle_state 요약)
            route_type: 필터링할 route_type (rag|tool|vision|chat). None이면 전체 검색.
            top_k: 반환할 최대 결과 수

        Returns:
            유사도 순으로 정렬된 list[Document]
        """
        if route_type:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.route_type",
                        match=MatchValue(value=route_type),
                    )
                ]
            )
            return self._vectorstore.similarity_search(
                situation, k=top_k, filter=search_filter
            )

        return self._vectorstore.similarity_search(situation, k=top_k)

    def save(
        self,
        situation: str,
        lesson: str,
        route_type: str,
    ) -> bool:
        """
        실패 경험을 Qdrant에 저장한다. 실패 케이스에서만 호출.

        근접 중복(같은 route_type 내 코사인 유사도 > EXPERIENCE_DEDUP_THRESHOLD)이면
        저장을 스킵한다.

        payload 3필드:
            situation (page_content): 임베딩 대상. user query + route_type 등 요약
            lesson (metadata): 개선 전략 1~2문장 (supervisor가 검색 후 활용)
            route_type (metadata): rag|tool|vision|chat (필터 검색용)

        Args:
            situation: user query + vehicle_state 요약 + route_type
            lesson: 개선 전략 1~2문장 (LLM 생성)
            route_type: 분류값 (rag|tool|vision|chat)

        Returns:
            실제로 저장했으면 True, 근접 중복으로 스킵했으면 False.
        """
        if self._has_near_duplicate(situation, route_type):
            logger.info("experience_memory 저장 스킵(근접 중복): route_type=%s", route_type)
            return False

        doc = Document(
            page_content=situation,
            metadata={
                "lesson": lesson,
                "route_type": route_type,
            },
        )
        self._vectorstore.add_documents([doc])
        logger.info("experience_memory 저장: route_type=%s", route_type)
        return True

    # ------------------------------------------------------------------
    # Qdrant 내부 처리
    # ------------------------------------------------------------------

    def _has_near_duplicate(self, situation: str, route_type: str) -> bool:
        """
        같은 route_type 내에 근접 중복 situation이 있는지 확인한다.

        similarity_search_with_score의 score는 코사인 유사도(1에 가까울수록 유사).
        검색 자체가 실패하면 저장을 막지 않도록 False를 반환한다
        (dedup 스킵보다 실패 경험 유실 방지가 우선).
        """
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.route_type",
                    match=MatchValue(value=route_type),
                )
            ]
        )

        try:
            results = self._vectorstore.similarity_search_with_score(
                situation, k=1, filter=search_filter
            )
        except Exception:
            logger.warning("근접 중복 검색 실패, dedup 스킵하고 저장 진행", exc_info=True)
            return False

        if not results:
            return False

        _, score = results[0]
        return score > EXPERIENCE_DEDUP_THRESHOLD

    def _ensure_collection(self) -> None:
        """
        experience_memory 컬렉션이 없으면 생성한다.
        embedder.py와 동일한 int8 스칼라 양자화 적용.
        route_type 필터 검색 속도를 위한 payload 인덱스 추가.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if EXPERIENCE_COLLECTION_NAME in existing:
            return

        self._client.create_collection(
            collection_name=EXPERIENCE_COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=qmodels.Distance.COSINE,
            ),
            quantization_config=qmodels.ScalarQuantization(
                scalar=qmodels.ScalarQuantizationConfig(
                    type=qmodels.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
        )

        self._client.create_payload_index(
            collection_name=EXPERIENCE_COLLECTION_NAME,
            field_name="route_type",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )

        logger.info("Qdrant 컬렉션 생성: %s", EXPERIENCE_COLLECTION_NAME)

 # 파일 맨 아래에 추가
_instance: ExperienceMemory | None = None


def get_experience_memory() -> ExperienceMemory:
    """프로세스 전체에서 ExperienceMemory 단일 인스턴스를 반환한다."""
    global _instance
    if _instance is None:
        _instance = ExperienceMemory()
    return _instance
