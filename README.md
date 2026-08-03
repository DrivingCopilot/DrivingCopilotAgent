# 🚗 DrivingCopilotAgent

차량 내부 임베디드 환경을 겨냥한 **On-Device Multimodal Driving Copilot**의 에이전트(두뇌) 레포입니다.
카메라 영상 · 운전자 텍스트 · 차량 매뉴얼 · 센서 데이터를 통합해 실시간 운전 보조 응답을 생성하는
Cognitive Architecture 기반 Multi-Agent 시스템으로, 아래 4개의 특화 에이전트가
**A2A(수평) + MCP(수직) + AG-UI(프론트)** 3층 프로토콜 위에서 협업합니다.

| Agent | 포트 | 모델 | 역할 |
|---|---|---|---|
| **Supervisor** | 8001 | 7B | ReAct 루프(Perceive→Reason→Act→Observe→Reflect) 제어, Plan-and-Execute, Reflexion |
| **Knowledge** | 8002 | 1.5B/7B | Vector RAG · Graph RAG(Neo4j) · Text2SQL 결과를 융합해 매뉴얼/관계/데이터 질의 응답 |
| **Execution** | 8003 | 1.5B | 12종 Vehicle Tool 실행, Multi-Turn, 실패 재시도 |
| **Perception** | 8004 | 7B(Vision) | 카메라 영상 분석, 비/터널/경고등 감지 → 멀티모달 트리거 |

> **구현 범위:** 이 레포는 계획서의 **Phase 1(VLM 배포) 이후 전 계층** — RAG/Graph/SQL,
> Cognitive Architecture, A2A Multi-Agent, MCP Tool-Use, 평가 체계(RAGAS/CRAG)를 담당합니다.
> 실제 오케스트레이션은 `app/graph/builder.py`의 LangGraph `StateGraph` 하나로 조립됩니다.

## 🏗️ 런타임 구성 요소 (어느 레포가 무엇을 실행하나)

전체 데모는 **4개의 프로세스 그룹**이 함께 떠 있어야 동작합니다.

- **본 레포(Agent)** — Supervisor(8001) + Knowledge/Execution/Perception A2A 서버(8002~8004) + 로컬 모델 서버(11500)
- **`DrivingCopilotBackend`** — REST API(8000) + **MCP 서버(9000)**. Qdrant(Vector) · Neo4j(Graph) · Vehicle DB(SQL) · 12종 Vehicle Tool의 **실행 주체는 Backend**이고, 이 레포의 Knowledge/Execution 노드는 MCP로 그 도구를 호출만 합니다.
- **`DrivingCopilotFrontend`** — React Automotive UI(3000), WebSocket으로 Supervisor와 스트리밍
- **로컬 모델 서버** — Ollama를 대체하는 이 레포 자체 프로세스(`app/model_server`)

> 이 레포 안의 `app/services/`(pdf_parser·semantic_chunker·embedder·qdrant_client·index_manuals)는
> **런타임 검색기가 아니라 매뉴얼을 Qdrant에 넣는 1회성 오프라인 인덱싱 파이프라인**입니다.
> 질의 시점의 검색·그래프 탐색·SQL은 위 MCP 서버(9000)를 통해 실행됩니다.

## 📂 Project Structure

```text
DrivingCopilotAgent/
├── main.py                     # Supervisor 서비스 진입점 (FastAPI/Uvicorn, :8001)
├── app/
│   ├── core/                   # 공통 설정·유틸
│   │   ├── config.py           # 모델(7B/1.5B)·엔드포인트·인프라 환경변수
│   │   ├── mcp_client.py       # Backend MCP 서버(9000) streamable-http 클라이언트
│   │   ├── json_utils.py       # LLM 출력 JSON 파싱 헬퍼
│   │   └── trace.py            # 실행 트레이스 로깅
│   ├── agents/                 # ReAct 그래프의 노드 로직 (에이전트별)
│   │   ├── supervisor.py       # 계획 수립 + 위임 라우팅 (ReAct 제어)
│   │   ├── knowledge.py        # vector/graph/sql MCP 도구 호출 + 근거 융합
│   │   ├── execution.py        # 12종 Vehicle Tool 실행 + 재시도
│   │   ├── perception.py       # 영상 분석 + 멀티모달 트리거
│   │   ├── crag.py             # Corrective RAG: 검색 품질 평가→쿼리 재작성→재검색
│   │   ├── observe.py          # 결과 관찰/성공·실패 판정
│   │   ├── reflect.py          # Reflexion 자기평가 → 경험 메모리 적재
│   │   └── finalize.py         # 최종 응답 조립
│   ├── graph/                  # LangGraph 오케스트레이션
│   │   ├── builder.py          # StateGraph 노드/엣지 조립 + compile
│   │   ├── a2a_nodes.py        # knowledge/execution/perception A2A 호출 노드
│   │   ├── state.py            # AgentState (공유 메모리) 정의
│   │   └── ws.py               # 노드→WebSocket 스트리밍 브리지
│   ├── a2a/                    # A2A 프로토콜 계층 (수평 통신)
│   │   ├── server.py           # A2A 서버 진입점 (knowledge/execution/perception)
│   │   ├── registry.py         # Agent Card·에이전트별 URL 레지스트리
│   │   ├── dispatch.py         # 위임 대상 에이전트로 태스크 분배
│   │   ├── client.py           # A2A HTTP 클라이언트
│   │   ├── router.py           # Supervisor측 A2A 라우트
│   │   ├── models.py / serde.py# 요청/응답 스키마·직렬화
│   ├── api/                    # Supervisor 외부 인터페이스
│   │   ├── http.py             # POST /invoke 등 단발성 호출 (Backend→Supervisor)
│   │   └── websocket.py        # WS /ws 세션별 격리 토큰 스트리밍 (AG-UI)
│   ├── memory/                 # 메모리 3계층 중 장기/엔티티
│   │   ├── experience.py       # 장기: Reflexion 실패 경험 Vector Store
│   │   └── entity.py           # 엔티티: 사용자 차량 선호도 KV Store
│   ├── services/               # ⚙️ 오프라인 인덱싱 파이프라인 (런타임 아님)
│   │   ├── pdf_parser.py       # 매뉴얼 PDF 텍스트 추출
│   │   ├── semantic_chunker.py # 의미 단위 청킹 (bge-m3)
│   │   ├── embedder.py         # 청크 임베딩 → Qdrant 저장
│   │   ├── qdrant_client.py    # 싱글톤 Qdrant 클라이언트
│   │   ├── index_manuals.py    # 파싱→청킹→임베딩→인덱싱 실행 스크립트
│   │   └── text2sql.py         # Text2SQL 보조 로직
│   └── model_server/           # 로컬 모델 서버 (Ollama 대체, :11500)
│       ├── server.py           # OpenAI 호환 API 진입점
│       └── backend.py          # 7B/1.5B transformers 로드·추론
├── evaluation/                 # 평가 체계 (Phase 4)
│   ├── _gold.py                # Gold Set 로더
│   ├── run_ragas_eval.py       # RAGAS(Faithfulness 등) 채점 — local/openai/gemini judge
│   ├── run_knowledge_eval.py   # Knowledge 노드 품질 평가
│   ├── run_tool_eval.py        # Tool-Call 정확도 평가
│   └── run_exception_sweep.py  # 예외/실패 경로 스윕
├── scripts/
│   ├── unix/run_agents.sh      # (Mac/Linux) 모델서버+4프로세스 일괄 기동
│   └── windows/*.ps1           # (Windows) 기동/중지 스크립트
├── requirements.txt
└── README.md
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
로딩 시 4bit 변환) + 1.5B 합쳐 ~18GB — `~/.cache/huggingface`에 캐시). GPU(CUDA)
환경을 전제로 하며, 처음 기동 시 다운로드+로딩에 몇 분 걸릴 수 있습니다.

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
bash scripts/unix/run_agents.sh
```
모델 서버가 `/health`로 `{"status":"ok"}`를 반환할 때까지 자동으로 기다린 뒤 나머지
4개 프로세스를 띄웁니다. 최초 실행 시 HuggingFace 다운로드가 겹치면 몇 분 걸릴 수
있습니다.

**Windows** — `scripts/unix/run_agents.sh`는 bash 전용이라 그대로 못 돌립니다. PowerShell
스크립트(`scripts/windows/run_agents.ps1`)로 한 번에 띄우거나, 아래처럼 터미널을 5개 열어
각각 실행하세요 (WSL이나 Git Bash를 쓴다면 위 sh 스크립트를 그대로 써도 됩니다).
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
  양자화 모델로 바꿔보세요. CUDA 없는 환경에서는 `bitsandbytes` 4bit 양자화가 동작하지
  않으니 반드시 GPU 서버에서 띄우세요.
- **모델 서버가 "loading"에서 안 넘어감**: `curl http://localhost:11500/health`로 상태를
  확인하세요. 최초 실행 시 `Qwen2-VL-7B-Instruct`(비양자화, ~15GB) + `Qwen2.5-1.5B-Instruct`를
  HuggingFace에서 내려받습니다. 네트워크 상태에 따라 꽤 걸릴 수 있고, 이후엔
  `~/.cache/huggingface`에서 바로 로드됩니다. (참고: GPTQ 사전양자화 체크포인트를
  쓰던 이전 방식은 Marlin 커널 JIT 컴파일이 Ampere 이전 세대 GPU에서 멈추는 문제가
  있어 bnb 4bit 즉석 양자화로 바꿨다 — 커널 컴파일이 없다.)
- **첫 질문에서 몇십 초씩 멈춤**: `BAAI/bge-m3` 임베딩 모델(2GB+)도 최초 1회 별도로
  HuggingFace에서 내려받습니다. 프로세스당 한 번만 겪는 정상 동작이고, 이후엔
  캐시(`~/.cache/huggingface`)에서 바로 로드됩니다.

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
