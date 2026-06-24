
import os

# ---------------------------------------------------------------------------
# 임베딩 모델
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"   # 임베딩 모델. A6000 통합 시 vLLM으로 교체
VECTOR_SIZE = 1024             # bge-m3 dense 벡터 차원

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

QDRANT_PATH = "./data/qdrant_storage"       # 로컬 파일 모드 경로. Docker 전환 시 QDRANT_URL 사용
QDRANT_URL = "http://localhost:6333"   # Docker/A6000 서버 모드 URL
COLLECTION_NAME = "vehicle_manuals"    # Qdrant 컬렉션 이름

# ---------------------------------------------------------------------------
# Neo4j (Graph RAG)
# ---------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


DB_PATH = os.getenv("VEHICLE_DB_PATH", "./data/vehicle_data.db")


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../../DrivingCopilotBackend"))

# MCP 서버를 실행할 Python 인터프리터
# Backend venv 가 있으면 그걸 사용, 없으면 시스템 python3 폴백
_BACKEND_VENV_PYTHON = os.path.join(_BACKEND_ROOT, "venv", "bin", "python")
MCP_SERVER_PYTHON: str = os.getenv(
    "MCP_SERVER_PYTHON",
    _BACKEND_VENV_PYTHON if os.path.exists(_BACKEND_VENV_PYTHON) else "python3",
)

# MCP 서버 스크립트 절대 경로
MCP_SERVER_SCRIPT: str = os.getenv(
    "MCP_SERVER_SCRIPT",
    os.path.join(_BACKEND_ROOT, "mcp_server.py"),
)

# MCP tool 호출 타임아웃 (초)
MCP_TOOL_TIMEOUT: float = float(os.getenv("MCP_TOOL_TIMEOUT", "10.0"))


MIN_TEXT_LENGTH = 20   # 이 길이 미만 페이지는 노이즈로 제거 (VehiclePDFParser)


CHUNK_MIN_TOKENS = 64                          # 이 토큰 수 미만 청크는 앞 청크에 병합
BREAKPOINT_THRESHOLD_TYPE = "standard_deviation"  # 시맨틱 경계 감지 방식