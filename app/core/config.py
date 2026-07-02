
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


# MCP 서버 엔드포인트 (streamable-http transport)
# Backend mcp_server.py 가 streamable-http 로 기동하는 상주 서버.
# Backend REST(FastAPI)가 8000을 쓰므로 MCP 서버 기본 포트는 9000으로 분리돼 있다
# (mcp_server.py: MCP_HOST/MCP_PORT 환경변수, 기본 http://{host}:{port}/mcp).
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:9000/mcp")

# MCP tool 호출 타임아웃 (초)
MCP_TOOL_TIMEOUT: float = float(os.getenv("MCP_TOOL_TIMEOUT", "10.0"))


MIN_TEXT_LENGTH = 20   # 이 길이 미만 페이지는 노이즈로 제거 (VehiclePDFParser)


CHUNK_MIN_TOKENS = 64                          # 이 토큰 수 미만 청크는 앞 청크에 병합
BREAKPOINT_THRESHOLD_TYPE = "standard_deviation"  # 시맨틱 경계 감지 방식

# ---------------------------------------------------------------------------
# 서버 (Agent supervisor / Backend)
# ---------------------------------------------------------------------------

AGENT_HOST = os.getenv("AGENT_HOST", "0.0.0.0")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8001"))                # supervisor FastAPI 포트
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")  # DrivingCopilotBackend
ALLOWED_ORIGINS = [
    "http://localhost:3000",   # React frontend
    "http://localhost:8000",   # FastAPI backend (server-to-server 호출용)
]

# ---------------------------------------------------------------------------
# Vision 모델 (Perception Agent — Qwen2-VL 7B FP16)
# ---------------------------------------------------------------------------
# 계획서 2.4절: Planner/Executor(Qwen2-VL INT4)와 별도로 FP16 모델을 사용한다.
# 로컬 vLLM 서버가 아직 없는 경우 base_url 이 비어 OpenAI 정식 API로 요청이
# 나가니, 실제 서버 기동 후 PERCEPTION_VLM_BASE_URL 을 채워야 한다.
PERCEPTION_VLM_BASE_URL: str = os.getenv("PERCEPTION_VLM_BASE_URL", "")
PERCEPTION_VLM_MODEL: str = os.getenv("PERCEPTION_VLM_MODEL", "qwen2-vl-7b-fp16")