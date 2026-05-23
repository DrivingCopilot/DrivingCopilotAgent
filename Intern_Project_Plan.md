# [cite_start]개발 인턴 프로젝트 계획서 [cite: 110]
[cite_start]**On-Device Multimodal Driving Copilot** [cite: 111]
[cite_start]**RAG + Graph RAG + Text2SQL + Cognitive Architecture + A2A + Tool-Use + Frontend** [cite: 112]
* [cite_start]**기간:** 2026.04.27~2026.07.31 (14주) [cite: 113]
* [cite_start]**범위:** VLM 배포(Phase 1) 제외 전체 시스템 [cite: 113]

## [cite_start]1. 주차별 플랜 (14주) [cite: 115]
### [cite_start]Phase 1: 기반 구축 (1~3주, 4/27~5/17) [cite: 116]

| 주차 | 할 일 | 완료 기준 |
| :--- | :--- | :--- |
| 1주 4/27 | [cite_start]개발 환경 세팅, 차량 매뉴얼 PDF 수집, PyMuPDF 파싱 + Semantic Chunking, FastAPI 보일러플레이트 + VehicleState 모델, React+Tailwind 셋업 + 3패널 레이아웃 [cite: 117] | [cite_start]청크 JSON 생성, FastAPI 서버 기동, 빈 레이아웃 렌더링 [cite: 117] |
| 2주 5/4 | [cite_start]MiniLM 벡터화 + Qdrant 인덱싱 + /rag/search API, Vehicle Tool 12종 구현 + MCP 서버, Chat/Dashboard/Camera 컴포넌트 (Mock 데이터) [cite: 117] | [cite_start]검색 API 동작, 12종 Tool 동작, 3대 컴포넌트 렌더링 [cite: 117] |
| 3주 5/11 | [cite_start]Query Router 4방향 분류 + /rag/route API + Reranker stub, WebSocket 서버 + Function Calling 스키마 + 기본 오케스트레이터, useWebSocket hook + 토큰 스트리밍 + Tool 결과 카드 UI [cite: 117] | [cite_start]라우팅 분류 확인, WS 송수신 확인, Mock WS 스트리밍 동작 [cite: 117] |

### [cite_start]Phase 2: 심화 기능 (4~7주, 5/18~6/14) [cite: 118]

| 주차 | 할 일 | 완료 기준 |
| :--- | :--- | :--- |
| 4주 5/18 | [cite_start]Neo4j 설치 + KG 스키마 + 관계 데이터 입력, ReAct 루프 (Perceive Reason Act Observe→Reflect) + CoT, Dashboard 상태 변경 애니메이션 + 이미지 업로드 [cite: 119] | [cite_start]Neo4j 관계 조회 확인, ReAct 1회 루프 동작, Tool→대시보드 반영 [cite: 119] |
| 5주 5/25 | [cite_start]Entity Extraction + Graph Traversal + Context Fusion + /rag/graph API, Reflexion 자기반성 + 메모리 3계층 (단기/장기/엔티티), 멀티모달 트리거 UI(오버레이) + 시나리오 테스트 설계 시작 [cite: 119] | [cite_start]Graph 쿼리 API 동작, Reflexion 루프 동작, 트리거 오버레이 렌더링 [cite: 119] |
| 6주 6/1 | [cite_start]SQLite DB 설계 (4테이블) + Mock 데이터 + 기본 Text2SQL, A2A 서버/클라이언트 + Agent Card + LangGraph StateGraph, 평가 대시보드 기초 + 시나리오 S1~S3 테스트 케이스 [cite: 119] | [cite_start]단순 SQL 동작, Agent Card 발견 확인, 대시보드 차트 렌더링 [cite: 119] |
| 7주 6/8 | [cite_start]Few-shot + Self-Consistency + SQL 검증 + /rag/sql API, Multi-Turn Tool Calling + 실패 처리/재시도 로직, 시나리오 S4~S8 테스트 케이스 + 복합 대화 UI [cite: 119] | [cite_start]복합 SQL 동작, Multi-Turn 연쇄 동작, 8종 테스트 준비 완료 [cite: 119] |

### [cite_start]Phase 3: 통합 + 평가 (8~10주, 6/15~7/5) [cite: 122]

| 주차 | 할 일 | 완료 기준 |
| :--- | :--- | :--- |
| 8주 6/15 | [cite_start]CRAG 모듈 + Vector/Graph/SQL 통합 라우팅, 오케스트레이터 ReAct+A2A 구조 업그레이드, Backend 실제 연동 (Mock→실제 WS) + S1~S3 테스트 [cite: 123] | [cite_start]통합 라우팅 동작, E2E 오케스트레이션 동작, S1~S3 통과 [cite: 123] |
| 9주 6/22<br>10주 6/29 | [cite_start]Gold Set 200건 구축 + evaluate.py + RAGAS 연동, 멀티모달 트리거 연결 (Vision→자동 Tool) + 에러 핸들링/로깅, S4~S5 테스트 + 대시보드 실시간 데이터 연동<br>Hallucination Detection + LLM-as-a-Judge + CRAG 효과 측정, 전체 시스템 안정화 + Agent 통신 디버깅, S6~S8 테스트 + 대시보드 완성 [cite: 123] | [cite_start]Gold Set 완성, 트리거 동작, S4~S5 통과<br>전체 평가 파이프라인 동작, 안정화 완료, 대시보드 완성 [cite: 123] |

### [cite_start]Phase 4: PM 서버 통합 + 최적화 (11~12주, 7/6~7/19) [cite: 124]

| 주차 | 할 일 | 완료 기준 |
| :--- | :--- | :--- |
| 11주 7/6 | [cite_start]PM 서버(A6000) 통합: bge-m3 교체 + Qdrant 재인덱싱, Reranker 적용, VLM 실제 연동, A2A Agent 전체 연결, Frontend→실제 Backend WS [cite: 125] | [cite_start]전체 시스템 PM 서버에서 E2E 동작 [cite: 125] |
| 12주 7/13 | [cite_start]Gold Set 전체 평가, 프롬프트 튜닝, CRAG 파라미터 조정, 성능 병목 분석, 버그 수정 [cite: 125] | [cite_start]목표 메트릭 근접 (Faithfulness > 0.85, Recall@5 > 90%, SQL > 75%) [cite: 125] |

### [cite_start]Phase 5: 마무리 (13~14주, 7/20~7/31) [cite: 126]

| 주차 | 할 일 | 완료 기준 |
| :--- | :--- | :--- |
| 13주 7/20 | [cite_start]최종 성능 최적화 + 엣지케이스 처리, UI 폴리싱, 데모 영상 촬영 [cite: 127] | [cite_start]최종 성능 확정, 데모 영상 완성 [cite: 127] |
| 14주 7/27 | [cite_start]기술 리포트 작성, API 문서화, 코드 정리 [cite: 127] | |

## [cite_start]구현 목록 [cite: 130]
### [cite_start]2. Vehicle RAG [cite: 131]
* [cite_start]차량 매뉴얼(사용자 매뉴얼, 정비 가이드, 사양서)에서 사용자 질문에 대한 답을 찾아주는 검색 시스템을 만든다[cite: 132].
* [cite_start]**구현할 것:** [cite: 133]
    * [cite_start]PDF 파싱 파이프라인: 차량 매뉴얼 PDF를 수집하고 PyMuPDF로 텍스트를 추출[cite: 135].
    * [cite_start]Semantic Chunking: 추출된 텍스트를 의미 단위로 분할 (평균 512 토큰, 메타데이터: 출처, 페이지, 섹션, 내용 유형)[cite: 136].
    * [cite_start]벡터 인덱싱: 청크를 벡터로 변환하여 Qdrant에 저장 (개발시 MiniLM, 통합시 bge-m3로 교체)[cite: 137].
    * [cite_start]`/rag/search` API: 사용자 질문에 대해 Top-5 관련 청크를 검색[cite: 138].
    * [cite_start]Reranker: 검색 결과 재정렬 (개발시 stub, 통합시 bge-reranker-v2-m3)[cite: 139].
    * [cite_start]CRAG: 검색 품질이 낮으면 쿼리 재작성 → 재검색[cite: 140].

### [cite_start]3. Knowledge Graph + Graph RAG [cite: 141]
* [cite_start]차량 부품, 경고등, 증상, 정비, 진단 코드 간의 관계를 그래프로 모델링하고, 이 관계를 따라가며 답변을 보강하는 시스템[cite: 142].
* [cite_start]Vector RAG만으로는 "경고등 관련 부품 → 정비 이력→ 다음 교환" 같은 다단계 관계 추론이 어렵기 때문에 필요[cite: 143].
* [cite_start]**구현할 것:** [cite: 144]
    * [cite_start]Neo4j KG 스키마 + 데이터 입력: Node(Component, WarningLight, Symptom, Maintenance, DTC) / Edge (HAS_PART, CAUSED_BY, RESOLVED_BY, HAS_INTERVAL 등)[cite: 146, 147, 148].
    * [cite_start]관계 데이터 추출: 매뉴얼과 정비 가이드에서 관계를 추출하여 그래프에 입력[cite: 149].
    * [cite_start]Entity Extraction: 검색 결과에서 부품명, 경고등, 증상 식별[cite: 150].
    * [cite_start]Graph Traversal: Neo4j에서 1~2 hop 탐색으로 관계 정보 확보[cite: 151].
    * [cite_start]Context Fusion: Vector RAG + Graph Traversal 결과를 통합하여 LLM 컨텍스트로 제공[cite: 152].
    * [cite_start]`/rag/graph` API[cite: 153].

### [cite_start]4. Text2SQL Agent [cite: 156]
* [cite_start]차량 주행/정비/진단 데이터가 저장된 RDB에서 자연어 질문을 SQL로 변환하여 답변하는 시스템[cite: 157].
* [cite_start]**구현할 것:** [cite: 158]
    * [cite_start]SQLite DB 설계: 4테이블 (vehicle_telemetry, maintenance_log, dtc_codes, trip_history) + Mock 데이터 생성[cite: 159].
    * [cite_start]Schema Linking: 질문과 관련된 테이블만 LLM에 제공[cite: 160].
    * [cite_start]Few-shot SQL: 유사 질문-SQL 예시를 벡터 검색으로 찾아 프롬프트에 포함[cite: 162].
    * [cite_start]Self-Consistency: N개 SQL 생성 → 다수결 투표로 최종 선택[cite: 163].
    * [cite_start]SQL 검증: 구문+EXPLAIN 분석 후 실행[cite: 165].
    * [cite_start]`/rag/sql` API[cite: 167].

### [cite_start]5. Query Router [cite: 168]
* [cite_start]사용자 입력을 4방향(rag, tool, vision, chat)으로 분류[cite: 169]. [cite_start]키워드 규칙(1차) + TF-IDF 분류기(2차) 2단계 구조[cite: 169]. [cite_start]GPU 불필요[cite: 169]. [cite_start]`/rag/route` API로 제공[cite: 169].

### [cite_start]6. Cognitive Architecture + A2A + Tool-Use [cite: 172]
* [cite_start]ReAct 인지 루프 기반 Multi-Agent 시스템과 차량 제어 도구를 구축한다[cite: 173].
* [cite_start]**구현할 것:** [cite: 174]
    * [cite_start]ReAct 루프: Perceive Reason/Plan Act Observe Reflect 반복[cite: 175]. [cite_start]LangGraph StateGraph로 구현[cite: 175].
    * [cite_start]Reflexion: 실패 시 LLM이 원인 분석 전략 수정 → 장기 메모리(Vector Store)에 저장[cite: 178].
    * [cite_start]Multi-Agent (A2A): 4개 Agent (Supervisor, Perception, Knowledge, Execution)를 A2A Server로 배포, Agent Card 기반 동적 발견[cite: 179].
    * [cite_start]MCP 서버 + 12종 Tool: climate, navigation, media, vehicle_status, window, lighting, seat, parking, emergency, driving_mode, wiper, dashboard_query[cite: 180].
    * [cite_start]Multi-Turn Tool Calling: 복수 Tool 순차/병렬 호출, 각 결과 컨텍스트 축적 실패 처리: 타임아웃(2회), 파라미터 오류(2회), 잘못된 Tool(1회), SQL 오류(3회) 재시도 메모리 3계층: 단기(ConversationBuffer) + 장기(Vector Experience) + 엔티티(KV Profile)[cite: 182].
    * [cite_start]멀티모달 트리거: Perception Agent가 카메라에서 비/터널/경고등 감지 → 자동 Tool 호출 오케스트레이터: Query Router → Agent 분기 → RAG/Graph/SQL/Tool 실행 → 결과 조합 → WebSocket 스트리밍[cite: 183].
    * [cite_start]WebSocket 서버: text/tool_start / tool_result / done 타입 구분 토큰 스트리밍[cite: 184].

### [cite_start]7. Frontend + 평가 대시보드 [cite: 185]
* [cite_start]React 기반 Automotive UI와 평가 시각화 대시보드를 구축한다[cite: 186].
* [cite_start]**구현할 것:** [cite: 187]
    * [cite_start]Automotive UI: React + Tailwind 다크모드 (배경 #0A0A0F, Primary #00B4D8)[cite: 188].
    * [cite_start]3패널: 좌측(Vehicle Dashboard) + 중앙(Chat Interface) + 우측(Camera Feed)[cite: 189].
    * [cite_start]Dashboard: 차량 상태 실시간 표시, Tool 실행 시 해당 값 하이라이트 애니메이션 Chat: 메시지 버블, 토큰 스트리밍, Tool 호출 결과 카드, 텍스트 입력 + 이미지 업로드[cite: 190].
    * [cite_start]Camera: Mock 이미지 또는 웹캔, 멀티모달 트리거 오버레이[cite: 190].
    * [cite_start]WebSocket 연동: useWebSocket hook, text/tool_start/tool_result/done 타입별 처리[cite: 191].
    * [cite_start]평가 대시보드: Faithfulness, Recall, Precision 등 메트릭 차트 + 평가 이력 추적[cite: 192].
    * [cite_start]시나리오 테스트 8종: 매뉴얼QA, 차량제어, Vision, 멀티모달트리거, Graph추론, Text2SQL, Multi-Turn복합, 일반대화[cite: 193].
    * [cite_start]데모 영상 촬영/편집[cite: 195].

### [cite_start]8. 평가 체계 [cite: 198]
* [cite_start]전체 시스템 품질을 자동 측정하는 평가 파이프라인을 구축한다[cite: 199].
* [cite_start]**구현할 것:** [cite: 200]
    * [cite_start]Gold Set 200건: (query, expected_answer, relevant_chunk_ids, expected_sql, route_type)[cite: 201]. [cite_start]RAG 80 + Tool 60 + SQL 40+ 복합 20[cite: 202].
    * [cite_start]RAGAS 메트릭: Faithfulness, Answer Relevancy, Context Recall, Context Precision 자동 계산[cite: 203].
    * [cite_start]Hallucination Detection: NLI 모델 또는 LLM 판단으로 소스 대비 응답 정합성 검증[cite: 204].
    * [cite_start]CRAG 효과 측정: 보정 전후 Recall@5 비교[cite: 205].
    * [cite_start]LLM-as-a-Judge: GPT-40가 응답을 1~10점 자동 채점 + Pairwise Ranking[cite: 206].
    * [cite_start]평가 대시보드: React으로 메트릭 시각화 + 평가 이력 추적[cite: 207].

### [cite_start]9. API 목록 [cite: 210]

| 엔드포인트 | 입력 | 출력 |
| :--- | :--- | :--- |
| [cite_start]POST `/rag/search` | query | chunks[], scores[] [cite: 211] |
| [cite_start]POST `/rag/graph` | query, entities[] | nodes[], edges[], context [cite: 211] |
| [cite_start]POST `/rag/sql` | query | sql, result, answer [cite: 211] |
| [cite_start]POST `/rag/route` | query, has_image | route_type (rag\|tool\|vision\|chat) [cite: 211] |
| [cite_start]POST `/rag/evaluate` | test_set_id | metrics, details[] [cite: 211] |
| [cite_start]POST `/tools/execute` | tool_name, params | result, status [cite: 211] |
| GET `/vehicle/state` | | [cite_start]Vehicle State JSON [cite: 211] |
| [cite_start]WS `/ws/chat` | message stream | token stream (text\|tool_start\|tool_result\|done) [cite: 211] |

[cite_start]**필수 패키지** [cite: 212]
* [cite_start]`pip install pymupdf langchain langchain-community qdrant-client sentence-transformers scikit-learn fastapi uvicorn pydantic websockets httpx pytest ragas` [cite: 213, 214]
* [cite_start]`pip install neo4j` # Neo4j Python 드라이버 [cite: 215]
* [cite_start]`pip install mcp` # Model Context Protocol SDK [cite: 216]
* [cite_start]`npm install react tailwindcss @headlessui/react lucide-react recharts reconnecting-websocket` [cite: 217]