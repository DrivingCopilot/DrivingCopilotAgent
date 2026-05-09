"""
app/services/semantic_chunker.py

LangChain SemanticChunker를 사용해 Document 리스트를 청크로 분할한다.

흐름:
    list[Document] → LangChain SemanticChunker → list[Document]

시맨틱 경계 감지:
    인접 문장 쌍의 코사인 유사도가 급락하는 지점을 경계로 판단.
    breakpoint_threshold_type으로 감지 방식 선택 가능.

메타데이터:
    분할 후에도 원본 Document의 메타데이터(source, page_num, section, content_type)가 유지됨.
"""

from __future__ import annotations  # Python 3.9 이하에서도 타입 힌트가 동작하도록 함

from langchain_core.documents import Document  # LangChain의 기본 문서 단위
from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker  # LangChain 시맨틱 청커
from langchain_huggingface import HuggingFaceEmbeddings  # LangChain 기반 HuggingFace 임베딩 래퍼


class SemanticChunker:
    """
    LangChain SemanticChunker 기반 청커.
    시맨틱 경계를 감지하여 Document를 의미 단위로 분할한다.

    Args:
        breakpoint_threshold_type: 경계 감지 방식.
            - "standard_deviation" : 평균 - 1σ 이하를 경계로 판단 (기본값)
            - "percentile"         : 상위 X% 유사도 급락 지점을 경계로 판단
            - "interquartile"      : IQR 기반 경계 감지
        embeddings               : 외부에서 임베딩 모델 주입 가능. embedder와 공유 시 사용.
                                   None이면 bge-m3를 새로 로드.
    """

    def __init__(
        self,
        breakpoint_threshold_type: str = "standard_deviation",  # 경계 감지 방식
        embeddings: HuggingFaceEmbeddings | None = None,        # 외부 주입 임베딩 모델
    ) -> None:

        # 외부에서 임베딩 모델이 주입되면 재사용, 없으면 bge-m3 새로 로드
        # index_manuals.py에서 embedder와 모델을 공유할 때 외부 주입 사용
        self._embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",                        # 임베딩 모델
            model_kwargs={"device": "cpu"},                  # CPU 환경에서 실행
            encode_kwargs={"normalize_embeddings": True},    # 코사인 유사도 계산을 위한 벡터 정규화
        )

        # LangChain SemanticChunker 초기화
        # 임베딩 모델과 경계 감지 방식을 주입하여 생성
        self._chunker = LangChainSemanticChunker(
            embeddings=self._embeddings,                         # 시맨틱 경계 감지용 임베딩 모델
            breakpoint_threshold_type=breakpoint_threshold_type, # 경계 판단 기준
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Document 리스트를 시맨틱 경계에서 분할한다.
        LangChain SemanticChunker가 인접 문장 쌍의 코사인 유사도를 계산하여
        유사도가 급락하는 지점을 경계로 분할한다.
        분할 후에도 원본 메타데이터(source, page_num, section, content_type)가 유지된다.

        Args:
            documents: VehiclePDFParser가 반환한 list[Document]

        Returns:
            시맨틱 경계에서 분할된 list[Document]
        """
        # LangChain SemanticChunker로 시맨틱 경계 기준 분할
        chunks = self._chunker.split_documents(documents)

        # 분할 후 내용이 비어있는 청크 제거
        chunks = [c for c in chunks if c.page_content.strip()]
        chunks = self._merge_small(chunks, min_tokens=64)  # 병합 추가
        return chunks

    def _merge_small(self, chunks: list[Document], min_tokens: int = 64) -> list[Document]:
        """min_tokens 미만 청크를 앞 청크에 병합."""
        if not chunks:
            return chunks
        merged = [chunks[0]]
        for cur in chunks[1:]:
            prev = merged[-1]
            prev_tokens = len(prev.page_content.split())
            # 같은 source이고 앞 청크가 너무 작으면 병합
            if (prev.metadata.get("source") == cur.metadata.get("source")
                    and prev_tokens < min_tokens):
                prev.page_content += " " + cur.page_content
            else:
                merged.append(cur)
        return merged