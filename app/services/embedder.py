"""
app/services/embedder.py

LangChain Qdrant 벡터스토어에 Document를 임베딩하고 저장한다.
1회성 인덱싱 파이프라인의 마지막 단계.

vLLM 교체 시 HuggingFaceEmbeddings → vLLM 임베딩으로 교체.
A6000 서버 통합 시 qdrant_path → qdrant_url로 전환.
"""

from __future__ import annotations  # Python 3.9 이하에서도 타입 힌트가 동작하도록 함

from app.core.config import (  # 전역 설정 상수 import
    MODEL_NAME,
    VECTOR_SIZE,
    QDRANT_PATH,
    COLLECTION_NAME,
)

import logging  # 진행 상황 로깅용
from pathlib import Path  # 파일 경로를 객체로 다루기 위한 모듈

from langchain_core.documents import Document        # LangChain 기본 문서 단위
from langchain_huggingface import HuggingFaceEmbeddings  # LangChain 기반 HuggingFace 임베딩 래퍼
from langchain_qdrant import QdrantVectorStore       # LangChain Qdrant 벡터스토어 래퍼
from qdrant_client import QdrantClient               # Qdrant 클라이언트 (컬렉션 생성 등 직접 제어용)
from qdrant_client.http import models as qmodels     # Qdrant 설정 모델 (VectorParams, Distance 등)

logger = logging.getLogger(__name__)  # 현재 모듈 이름으로 로거 생성

class VehicleEmbedder:
    """
    LangChain Qdrant 기반 임베딩 및 벡터스토어 관리 클래스.

    Args:
        embeddings : 외부에서 임베딩 모델 주입 가능.
                     SemanticChunker와 모델을 공유할 때 사용.
                     None이면 MODEL_NAME으로 새로 로드.
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings | None = None,  # 외부 주입 임베딩 모델
    ) -> None:

        # 외부에서 임베딩 모델이 주입되면 재사용, 없으면 새로 로드
        # SemanticChunker와 모델을 공유한다 생각하면 됨.
        self._embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=MODEL_NAME,                        # 사용할 모델 이름
            model_kwargs={"device": "cpu"},               # CPU 환경에서 실행
            encode_kwargs={"normalize_embeddings": True}, # 코사인 유사도 계산을 위한 벡터 정규화
        )

        # Qdrant 클라이언트 초기화 (로컬 파일 모드)
        # path 인자를 사용하면 Docker 없이 로컬 파일로 저장됨
        # A6000 서버 통합 시: QdrantClient(url="http://서버주소:6333")으로 교체
        self._client = QdrantClient(path=QDRANT_PATH)

        # 컬렉션이 없으면 자동 생성
        self._ensure_collection()

        # LangChain Qdrant 벡터스토어 초기화
        # 임베딩 생성과 Qdrant 저장을 한 번에 처리해주는 래퍼
        self._vectorstore = QdrantVectorStore(
            client=self._client,
            collection_name=COLLECTION_NAME,
            embedding=self._embeddings,  # embeddings → embedding 주의
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_and_store(self, documents: list[Document], batch_size: int = 32) -> None:
        """
        Document 리스트를 임베딩하고 Qdrant에 저장한다.
        LangChain Qdrant가 임베딩 생성과 저장을 한 번에 처리한다.

        Args:
            documents: SemanticChunker가 반환한 list[Document]
        """
        if not documents:  # 빈 리스트면 바로 반환
            return

        # LangChain Qdrant의 add_documents가 임베딩 + Qdrant upsert를 한 번에 처리
        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]  # 배치 단위로 슬라이싱
            self._vectorstore.add_documents(batch)
            logger.info("임베딩 진행: %d / %d", min(i + batch_size, len(documents)), len(documents))

        logger.info("완료: %d개 Document Qdrant 저장", len(documents))

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Qdrant에서 쿼리와 유사한 Document를 검색한다. (계획서 기준 Top-5)

        Args:
            query : 검색 쿼리 텍스트
            top_k : 반환할 최대 결과 수

        Returns:
            유사도 순으로 정렬된 list[Document]
        """
        # LangChain Qdrant의 similarity_search가 쿼리 임베딩 + 검색을 한 번에 처리
        return self._vectorstore.similarity_search(query, k=top_k)

    def search_with_score(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """
        점수와 함께 검색한다. VectorRAG의 CRAG 품질 평가에 사용.

        Args:
            query : 검색 쿼리 텍스트
            top_k : 반환할 최대 결과 수

        Returns:
            (Document, score) 튜플 리스트. score는 코사인 유사도 (0~1).
        """
        return self._vectorstore.similarity_search_with_score(query, k=top_k)

    # ------------------------------------------------------------------
    # Qdrant 내부 처리
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """
        Qdrant 컬렉션이 없으면 생성하고 payload 인덱스를 추가한다.
        int8 스칼라 양자화를 적용하여 메모리 사용량을 약 75% 줄인다.
        양자화 벡터로 1차 검색 후 원본 벡터로 재정렬(rescore)하여 정확도를 보완한다.
        payload 인덱스는 source, section, content_type 기반, 필터 검색 속도를 높여준다.
        """
        # 현재 존재하는 컬렉션 이름 목록 조회
        existing = [c.name for c in self._client.get_collections().collections]

        if COLLECTION_NAME in existing:  # 이미 있으면 생성 스킵
            return

        # 컬렉션 생성 (bge-m3 1024차원, Cosine 유사도) - 데이터베이스의 테이블 같은거. 벡터저장공간
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,                  # 벡터 차원 수
                distance=qmodels.Distance.COSINE,  # 유사도 측정 방식
            ),
            # int8 스칼라 양자화: float32 대비 메모리 75% 절감
            # always_ram=True: 양자화 벡터를 RAM에 유지하여 검색 속도 확보
            # quantile=0.99: 상위 1% 이상치 제외하고 양자화 범위 설정
            quantization_config=qmodels.ScalarQuantization(
                scalar=qmodels.ScalarQuantizationConfig(
                    type=qmodels.ScalarType.INT8,  # float32 → int8 변환
                    quantile=0.99,                 # 상위 1% 이상치 제외
                    always_ram=True,               # 양자화 벡터 RAM 상주
                )
            ),
        )

        # 필터 검색 속도를 높이기 위한 payload 인덱스 생성
        for field_name in ("source", "section", "content_type"):
            self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,                           # 인덱스를 생성할 필드명
                field_schema=qmodels.PayloadSchemaType.KEYWORD,  # 키워드 타입으로 인덱싱
            )

        logger.info("Qdrant 컬렉션 생성: %s", COLLECTION_NAME)

# 기능 시험용 코드
if __name__ == "__main__":
    import time
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "경고등"

    e = VehicleEmbedder()

    t0 = time.time()
    results = e.search(query, top_k=5)
    elapsed = time.time() - t0

    print(f"검색 시간: {elapsed:.3f}s")
    print(f"결과: {len(results)}건\n")

    for i, doc in enumerate(results, 1):
        print(f"[{i}] page={doc.metadata.get('page_num')} | section={doc.metadata.get('section')!r}")
        print(f"     {doc.page_content[:150]!r}")
        print()