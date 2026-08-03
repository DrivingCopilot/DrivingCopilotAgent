
import os

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

QDRANT_PATH = "./data/qdrant_storage"       # 로컬 파일 모드 경로. Docker 전환 시 QDRANT_URL 사용
QDRANT_URL = "http://localhost:6333"   # Docker/A6000 서버 모드 URL
COLLECTION_NAME = "vehicle_manuals"    # Qdrant 컬렉션 이름
EXPERIENCE_COLLECTION_NAME = "experience_memory"  # ReAct Reflect 실패 경험 컬렉션
EXPERIENCE_TOP_K = 5                   # supervisor experience 검색 상위 결과 수
EXPERIENCE_DEDUP_THRESHOLD: float = 0.95  # 코사인 유사도 임계값. 이상이면 근접 중복으로 간주해 저장 스킵
WINDOW_SIZE = 5  # ConversationBufferWindow 단기 메모리 윈도우 크기 (계획서 2.5)

# 엔티티 메모리 (사용자 선호도 KV Store, 계획서 2.5)
ENTITY_PROFILE_PATH = "./data/user_profile.json"

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

# MCP tool 호출 타임아웃 (초). vector_rag_search(임베딩+Qdrant)가 ~9~10초로 기존
# 10초 경계에 걸쳐 간헐적 타임아웃이 나 빈 검색 결과를 유발했다 — 여유를 둔다.
MCP_TOOL_TIMEOUT: float = float(os.getenv("MCP_TOOL_TIMEOUT", "20.0"))

# A2A 태스크(POST /tasks/send) 타임아웃 (초).
# knowledge/execution/perception 노드는 내부에서 LLM/VLM 추론을 거치므로
# Agent Card 조회(GET, 5초 기본값)보다 훨씬 여유 있게 잡는다.
# knowledge 실경로는 ReAct(1.5B ~11s) + MCP 검색(~4s) + 7B fusion(~23~60s)로
# 실측 ~39~73초라, 기존 30초로는 노드가 답을 다 만들어도 호출측이 먼저 포기해
# 매번 error_type='timeout'으로 찍혔다(노드는 HTTP 200 성공). HuggingFace
# transformers 직접 로딩(app/model_server) 기준 Ollama 대비 훨씬 느려진 실경로를
# 수용하도록 상향한다. fusion max_tokens 축소(knowledge.py)와 함께 근본 지연도 줄인다.
A2A_TASK_TIMEOUT: float = float(os.getenv("A2A_TASK_TIMEOUT", "90.0"))

# 로컬 HuggingFace 모델 서버 (app/model_server) — OpenAI 호환 엔드포인트.
# 7B/1.5B 모델을 프로세스당 한 번씩만 로드해 4개 A2A 에이전트 프로세스가 공유한다
# (기존 Ollama가 하던 역할을 대체 — 토폴로지는 동일, 백엔드만 HuggingFace transformers).
MODEL_SERVER_URL: str = os.getenv("MODEL_SERVER_URL", "http://localhost:11500/v1")

# 계획/추론/비전에 쓰는 7B 모델. Qwen2-VL은 공식 1.5B 체크포인트가 없어 텍스트
# 전용 자리는 Qwen2.5-1.5B-Instruct를 쓴다(QWEN_TEXT_MODEL_NAME).
# 비양자화 체크포인트를 로딩 시점에 bitsandbytes 4bit로 즉석 양자화한다(app/model_server/backend.py) —
# GPTQ-Int4 사전양자화 체크포인트는 Marlin 커널 JIT 컴파일이 필요해 Colab 무료 T4(Turing)에서
# 컴파일 후 로딩이 멈추는 문제가 있어 피한다.
QWEN_VL_MODEL_NAME: str = os.getenv("QWEN_VL_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
QWEN_TEXT_MODEL_NAME: str = os.getenv("QWEN_TEXT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

# ---------------------------------------------------------------------------
# Observe 노드 재시도 정책 (계획서: 타임아웃 2회, 파라미터 오류 2회, 잘못된 Tool 1회, SQL 오류 3회)
# ---------------------------------------------------------------------------
MAX_RETRY: dict[str, int] = {
    "timeout": 2,
    "parameter": 2,
    "invalid_tool": 1,
    "sql": 3,
}


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
