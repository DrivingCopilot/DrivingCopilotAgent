# app/core/config.py
#
# 프로젝트 전역 설정 상수 관리 모듈.
# 모델, Qdrant, 경로 등 여러 서비스에서 공유하는 설정값을 한 곳에서 관리한다.
# 설정 변경 시 이 파일만 수정하면 된다.

# ---------------------------------------------------------------------------
# 임베딩 모델
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"   # 임베딩 모델. A6000 통합 시 vLLM으로 교체
VECTOR_SIZE = 1024             # bge-m3 dense 벡터 차원

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

QDRANT_PATH = "./qdrant_storage"       # 로컬 파일 모드 경로. Docker 전환 시 QDRANT_URL 사용
QDRANT_URL = "http://localhost:6333"   # Docker/A6000 서버 모드 URL
COLLECTION_NAME = "vehicle_manuals"    # Qdrant 컬렉션 이름

## 파서와 청커는 일단 넣긴 했는데 다른 코드에서 쓸거 같지 않은 지엽적인 값이라 안넣었어
# ---------------------------------------------------------------------------
# 파서
# ---------------------------------------------------------------------------

MIN_TEXT_LENGTH = 20   # 이 길이 미만 페이지는 노이즈로 제거 (VehiclePDFParser)

# ---------------------------------------------------------------------------
# 청커
# ---------------------------------------------------------------------------

CHUNK_MIN_TOKENS = 64                          # 이 토큰 수 미만 청크는 앞 청크에 병합
BREAKPOINT_THRESHOLD_TYPE = "standard_deviation"  # 시맨틱 경계 감지 방식

# ---------------------------------------------------------------------------
# 서버 (Agent supervisor / Backend)
# ---------------------------------------------------------------------------

import os

AGENT_HOST = os.getenv("AGENT_HOST", "0.0.0.0")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8001"))                # supervisor FastAPI 포트
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")  # DrivingCopilotBackend
ALLOWED_ORIGINS = [
    "http://localhost:3000",   # React frontend
    "http://localhost:8000",   # FastAPI backend (server-to-server 호출용)
]