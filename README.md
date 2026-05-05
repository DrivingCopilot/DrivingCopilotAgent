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
