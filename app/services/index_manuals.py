"""
app/services/index_manuals.py

차량 매뉴얼 PDF를 파싱 → 청킹 → 임베딩 → Qdrant 저장하는 1회성 인덱싱 스크립트.
SemanticChunker에서 로드한 bge-m3 모델을 VehicleEmbedder와 공유하여
모델을 한 번만 로드한다.

Usage:
    # 청크 결과만 확인 (Qdrant 불필요)
    python index_manuals.py ../../manuals/매뉴얼.pdf --step chunk

    # 전체 파이프라인 (파싱 + 청킹 + 임베딩 + Qdrant 저장)
    python index_manuals.py ../../manuals/매뉴얼.pdf --step embed
"""

import argparse  # 커맨드라인 인자 파싱용
import logging  # 진행 상황 로깅용
import sys  # 프로그램 종료용
import time  # 실행 시간 측정용
from pathlib import Path  # 파일 경로 존재 여부 확인용

logging.basicConfig( 
    # 진행상황 로그 확인하고 싶으면 이거 쓰면 된다고 함
    # print보다 좋다고 해서 넣긴 했는데 그냥 print가 여기선 더 나을듯
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_chunk(pdf_path: str):
    """
    파싱 + 청킹 결과를 확인한다. Qdrant 없이도 실행 가능.
    청크 품질 확인 후 임베딩으로 넘어가기 위한 중간 검증 단계.
    """
    from pdf_parser import VehiclePDFParser  # PDF 파싱 클래스
    from semantic_chunker import SemanticChunker  # 시맨틱 청킹 클래스

    # STEP 1: PDF 파싱
    logger.info("=== STEP 1: PDF 파싱 ===")
    t0 = time.time()
    documents = VehiclePDFParser().parse(pdf_path)  # PDF → list[Document]
    logger.info("파싱 완료: %.2fs | 페이지 수: %d", time.time() - t0, len(documents))

    # STEP 2: 시맨틱 청킹
    logger.info("=== STEP 2: 시맨틱 청킹 ===")
    t0 = time.time()
    chunker = SemanticChunker()               # bge-m3 모델 로드
    chunks = chunker.chunk(documents)         # list[Document] → 청크 list[Document]
    logger.info("청킹 완료: %.2fs | 청크 수: %d", time.time() - t0, len(chunks))

    # 토큰 수 통계 출력 (청크 품질 확인용)
    token_counts = [len(c.page_content.split()) for c in chunks]
    logger.info(
        "토큰 수 — min: %d / avg: %.1f / max: %d",
        min(token_counts),
        sum(token_counts) / len(token_counts),
        max(token_counts),
    )

    # 샘플 청크 출력
    print("\n--- 샘플 청크 (5개) ---")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n[{i+1}] page={chunk.metadata.get('page_num')} | section={chunk.metadata.get('section')!r}")
        print(f"     tokens={len(chunk.page_content.split())}")
        print(f"     {chunk.page_content[:150]!r}")

    return chunks, chunker  # embedder와 모델 공유를 위해 chunker도 반환


def run_embed(pdf_path: str):
    """
    파싱 + 청킹 + 임베딩 + Qdrant 저장 전체 파이프라인 실행.
    chunker._embeddings를 embedder와 공유하여 bge-m3를 한 번만 로드한다.
    """
    from embedder import VehicleEmbedder  # Qdrant 임베딩 및 저장 클래스

    # STEP 1~2: 파싱 + 청킹
    chunks, chunker = run_chunk(pdf_path)

    # STEP 3: 임베딩 + Qdrant 저장
    logger.info("=== STEP 3: 임베딩 + Qdrant 저장 ===")

    # chunker._embeddings를 주입하여 bge-m3를 새로 로드하지 않고 재사용
    embedder = VehicleEmbedder(embeddings=chunker._embeddings)
    t0 = time.time()
    embedder.embed_and_store(chunks)  # 임베딩 생성 + Qdrant upsert
    logger.info("임베딩 완료: %.2fs", time.time() - t0)


    # 검색 테스트 (Qdrant 저장 확인용)
    logger.info("=== 검색 테스트 ===")
    for query in ["경고등", "엔진 오일 교체", "안전벨트"]:
        results = embedder.search(query, top_k=3)
        logger.info("쿼리: %r → %d건", query, len(results))
        for doc in results:
            print(f"  page={doc.metadata.get('page_num')} | {doc.page_content[:60]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="PDF 파일 경로")
    parser.add_argument(
        "--step",
        choices=["chunk", "embed"],
        default="chunk",
        help="chunk: 청크 확인만 / embed: 전체 파이프라인 (기본: chunk)",
    )
    args = parser.parse_args()

    if not Path(args.pdf_path).exists():
        logger.error("파일 없음: %s", args.pdf_path)
        sys.exit(1)

    if args.step == "chunk":
        run_chunk(args.pdf_path)
    else:
        run_embed(args.pdf_path)


if __name__ == "__main__":
    main()