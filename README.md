## 📂 Project Structure

`DrivingCopilotAgent` 레포지토리는 On-Device 환경에서 동작하는 Cognitive Architecture 기반의 멀티 에이전트 시스템을 관리합니다.

```text
DrivingCopilotAgent/
├── app/
│   ├── core/               # 시스템 핵심 설정 및 공유 상태 정의
│   │   ├── config.py       # 모델(7B/1.5B) 및 인프라 설정
│   │   └── state.py        # LangGraph AgentState (공유 메모리) 정의
│   ├── agents/             # 독립된 A2A Server 역할을 하는 에이전트 로직
│   │   ├── supervisor.py   # Supervisor (7B): 계획 수립 및 ReAct 루프 제어
│   │   ├── perception.py   # Perception: 영상 분석 및 멀티모달 트리거
│   │   ├── knowledge.py    # Knowledge: RAG, Graph, SQL 지식 통합
│   │   └── execution.py    # Execution (1.5B): MCP 도구 실행 및 실패 처리
│   ├── graph/              # LangGraph 오케스트레이션 구현 계층
│   │   ├── nodes.py        # 에이전트별 상태 전이 노드 함수
│   │   ├── edges.py        # 조건부 라우팅 및 리플렉션(Reflexion) 로직
│   │   └── builder.py      # StateGraph 컴파일 및 워크플로우 정의
│   ├── server/             # 에이전트 인터페이스 계층 (A2A/WebSocket)
│   │   ├── endpoints.py    # 에이전트 간 통신을 위한 API 엔드포인트
│   │   └── websocket.py    # 실시간 토큰 스트리밍 및 AG-UI 연동
│   ├── services/           # 지식 베이스 연동 서비스 모듈
│   │   ├── vector_rag.py   # Qdrant 기반 벡터 검색
│   │   ├── graph_rag.py    # Neo4j Knowledge Graph 탐색
│   │   └── text2sql.py     # 차량 데이터베이스 SQL 생성 및 실행
│   └── tools/              # MCP(Model Context Protocol) 기반 차량 제어 도구
├── evaluation/             # 에이전트 품질 평가 체계 (Phase 4)
│   ├── gold_set/           # 평가용 Gold Set 200건 데이터
│   └── metrics.py          # RAGAS 및 Hallucination Detection 로직
├── tests/                  # 단위 테스트 및 8종 시나리오 검증
├── main.py                 # 서비스 실행 진입점 (FastAPI/Uvicorn)
├── requirements.txt        # 의존성 패키지 목록
└── README.md               # 프로젝트 매뉴얼 및 구조 가이드
```

## 🚀 실행 방법 (E2E)

Supervisor는 knowledge/execution/perception과 A2A(HTTP)로 통신하는 독립 프로세스이고,
전체 E2E를 위해서는 이 레포 + `DrivingCopilotBackend` + `DrivingCopilotFrontend` +
로컬 모델 서버까지 총 4곳이 같이 떠 있어야 합니다.

```
Frontend(3000) --WS--> Supervisor(8001) --A2A HTTP--> knowledge/execution/perception(8002~8004)
Frontend(3000) --REST--> Backend(8000, vehicle-state/camera-frame)
knowledge/execution/perception --MCP--> Backend mcp_server.py(9000)
전체 LLM 호출 --> 로컬 모델 서버(11500, app/model_server — HuggingFace transformers)
```

로컬 모델 서버는 Ollama를 대체하는 이 레포 자체 프로세스입니다(`app/model_server`).
`Qwen2-VL-7B-Instruct`(7B, 로딩 시점에 bitsandbytes 4bit로 즉석 양자화)와
`Qwen2.5-1.5B-Instruct`(1.5B) 두 모델을 프로세스 시작 시 한 번만 GPU에 로드하고,
OpenAI 호환 API로 Supervisor/Knowledge/Execution/Perception 4곳에 서빙합니다 —
GPU 1개(VRAM 8GB 기준)에 모델을 1벌씩만 올리기 위한 구조라 4개 에이전트 프로세스가
각자 모델을 로드하지 않습니다. (사전 양자화된 GPTQ 체크포인트 대신 bnb 4bit를 쓰는
이유: GPTQ는 Marlin 커널을 JIT 컴파일해야 하는데, Ampere 이전 세대 GPU(Colab 무료
T4 등)에서 컴파일 후 로딩이 멈추는 문제가 있어 피한다.)

### 0. 사전 준비 (한 번만)

**.env 파일 준비**

```bash
# Mac/Linux
cp .env.example .env
# Windows
copy .env.example .env
```

`.env.example`에 필요한 값과 설명이 이미 있습니다 — 기본값 그대로 둬도 됩니다.
모델 가중치는 별도로 받을 필요 없이, 아래 3번 단계에서 모델 서버가 처음 뜰 때
HuggingFace Hub에서 자동으로 다운로드됩니다(최초 1회, 7B(비양자화 체크포인트,
로딩 시 4bit 변환) + 1.5B 합쳐 ~18GB — `~/.cache/huggingface`에 캐시).

### 1. 가상환경 + 패키지 설치

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. `DrivingCopilotBackend` 먼저 실행 (별도 레포)

REST API와 MCP 서버는 **별도 프로세스**라 터미널 2개가 필요합니다. Backend 레포의
README도 함께 참고하세요.

```bash
# 터미널 1 — REST API (8000, vehicle-state/camera-frame)
uvicorn main:app --reload

# 터미널 2 — MCP 서버 (9000, 차량 제어 12종 Tool + 카메라 프레임)
python mcp_server.py
```

### 3. 이 레포(Agent) 실행

**Mac / Linux** — 스크립트 하나로 모델 서버(11500) + 4개 에이전트 프로세스(8001~8004) 동시 기동
```bash
bash scripts/run_agents.sh
```
모델 서버가 `/health`로 `{"status":"ok"}`를 반환할 때까지 자동으로 기다린 뒤 나머지
4개 프로세스를 띄웁니다. 최초 실행 시 HuggingFace 다운로드가 겹치면 몇 분 걸릴 수
있습니다.

**Windows** — `scripts/run_agents.sh`는 bash 전용이라 그대로 못 돌립니다. 터미널을
5개 열어 각각 실행하세요 (WSL이나 Git Bash를 쓴다면 위 스크립트를 그대로 써도 됩니다).
```powershell
# 터미널 1 — 로컬 모델 서버 (11500) — 이 서버가 "ok" 뜰 때까지 기다린 뒤 아래를 실행
python -m app.model_server.server

# 터미널 2 — Supervisor (8001)
python main.py

# 터미널 3 — Knowledge (8002)
python -m app.a2a.server knowledge

# 터미널 4 — Execution (8003)
python -m app.a2a.server execution

# 터미널 5 — Perception (8004)
python -m app.a2a.server perception
```

### 4. `DrivingCopilotFrontend` 실행 (별도 레포)

```bash
npm install
npm run dev
```
브라우저에서 http://localhost:3000 접속 후 채팅으로 테스트하면 됩니다.

### 5. 다 떴는지 확인

| 포트 | 서비스 | 레포 |
|---|---|---|
| 3000 | Frontend | DrivingCopilotFrontend |
| 8000 | Backend REST | DrivingCopilotBackend |
| 8001 | Supervisor | 본 레포 |
| 8002 | Knowledge | 본 레포 |
| 8003 | Execution | 본 레포 |
| 8004 | Perception | 본 레포 |
| 9000 | MCP 서버 | DrivingCopilotBackend |
| 11500 | 로컬 모델 서버 | 본 레포 (`app/model_server`) |

```bash
# Mac/Linux
lsof -iTCP -sTCP:LISTEN -P | grep -E ":3000|:8000|:8001|:8002|:8003|:8004|:9000|:11500"
```
```powershell
# Windows
netstat -ano | findstr "3000 8000 8001 8002 8003 8004 9000 11500"
```

### 자주 겪는 문제

- **답변이 계속 "실패했습니다"로만 나옴**: 9000번(MCP 서버)이 안 떠 있을 확률이 높습니다.
  `python mcp_server.py`를 REST API(8000)와 별개로 반드시 띄워야 합니다.
- **`api_key` 관련 에러**: `.env`가 비어있거나, `.env`를 고친 뒤 서버를 재시작 안 한 경우입니다.
  값 수정 후에는 해당 프로세스를 반드시 재시작하세요.
- **CUDA out of memory**: VRAM 8GB 기준으로 7B(bnb 4bit, ~5~6GB) + 1.5B(~3GB)가
  타이트합니다. 다른 GPU 프로세스를 먼저 종료하거나, `QWEN_TEXT_MODEL_NAME`을 더 작은
  양자화 모델로 바꿔보세요.
- **모델 서버가 "loading"에서 안 넘어감**: 최초 실행 시 `Qwen2-VL-7B-Instruct`(비양자화,
  ~15GB) + `Qwen2.5-1.5B-Instruct`를 HuggingFace에서 내려받습니다. 네트워크 상태에
  따라 꽤 걸릴 수 있고, 이후엔 `~/.cache/huggingface`에서 바로 로드됩니다. (참고: GPTQ
  사전양자화 체크포인트를 쓰던 이전 방식은 Marlin 커널 JIT 컴파일이 Ampere 이전 세대
  GPU에서 멈추는 문제가 있어 bnb 4bit 즉석 양자화로 바꿨다 — 커널 컴파일이 없다.)
- **첫 질문에서 몇십 초씩 멈춤**: `BAAI/bge-m3` 임베딩 모델(2GB+)도 최초 1회 별도로
  HuggingFace에서 내려받습니다. 프로세스당 한 번만 겪는 정상 동작입니다.

### 서비스 포트 매핑

| 서비스 | 포트 | 비고 |
|---|---|---|
| `DrivingCopilotFrontend` (React) | 3000 | |
| `DrivingCopilotBackend` (FastAPI) | 8000 | 프론트엔드의 진입점, supervisor 를 호출 |
| `DrivingCopilotAgent` (Supervisor) | 8001 | LangGraph supervisor 노드, 본 레포 |
| Qdrant | 6333 | Vector RAG |

### 엔드포인트

- `GET  /` — 서비스 메타정보
- `GET  /health` — 헬스체크
- `POST /invoke` — 단발성 supervisor 호출 (백엔드 → supervisor)
- `WS   /ws` — 실시간 토큰 스트리밍 (supervisor 내부 상태/토큰을 클라이언트에 전송)

### 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `AGENT_HOST` | `0.0.0.0` | supervisor 바인딩 호스트 |
| `AGENT_PORT` | `8001` | supervisor 포트 |
| `BACKEND_URL` | `http://localhost:8000` | 백엔드 위치 (server-to-server 호출용) |
| `MODEL_SERVER_URL` | `http://localhost:11500/v1` | 로컬 모델 서버(OpenAI 호환) 위치 |
| `QWEN_VL_MODEL_NAME` | `Qwen/Qwen2-VL-7B-Instruct` | 7B 모델(계획/추론/비전, 로딩 시 bnb 4bit 양자화) |
| `QWEN_TEXT_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B 모델(Knowledge/Text2SQL/선호도) |
