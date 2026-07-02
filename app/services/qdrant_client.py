"""
app/services/qdrant_client.py

싱글톤 Qdrant 클라이언트.
로컬 파일 모드에서 QdrantClient 인스턴스가 여럿 생성되면 파일 락 충돌이 발생한다.
lru_cache로 프로세스 내 단 하나의 인스턴스만 유지한다.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient

from app.core.config import QDRANT_PATH


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """프로세스 내 싱글톤 QdrantClient를 반환한다."""
    return QdrantClient(path=QDRANT_PATH)
