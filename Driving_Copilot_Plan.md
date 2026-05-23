# [cite_start]프로젝트 계획서: On-Device Multimodal Driving Copilot [cite: 1, 2]
[cite_start]**Cognitive Architecture 기반 Multi-Agent 차량용 MLLM 시스템** [cite: 3]
* [cite_start]**작성일:** 2026. 04. 18 [cite: 4]
* **프로젝트 기간:** 2026.05~2026. [cite_start]07 (12주) [cite: 5]

## [cite_start]1. 프로젝트 개요 [cite: 7]
### [cite_start]1.1 프로젝트 정의 [cite: 8]
[cite_start]On-Device Multimodal Driving Copilot은 차량 내부 임베디드 환경에서 카메라 영상, 운전자 텍스트/음성, 차량 매뉴얼, 차량 센서 데이터를 통합 처리하여 실시간 운전 보조 응답을 생성하는 On-Device MLLM 시스템이다[cite: 9]. [cite_start]핵심 기술로는 TensorRT Edge-LLM, Qwen2-VL, ReAct, Reflexion, Plan-and-Execute, Multi-Agent(A2A), LangGraph, Agentic RAG, Graph RAG (Neo4j), Text2SQL, MCP, CRAG, LLM-as-a-Judge, AG-UI가 포함된다[cite: 10].

### [cite_start]1.2 핵심 목표 [cite: 11]
[cite_start]**G1-On-Device VLM 배포 + 비용 최적화** [cite: 12]
* [cite_start]TensorRT Edge-LLM으로 Qwen2-VL 7B와 1.5B를 각각 INT4(AWQ) 양자화하여 A6000에서 로컬 추론 환경을 구축한다[cite: 13].
* [cite_start]모든 쿼리를 7B로 처리하면 추론 비용이 높으므로, Plan-and-Execute 패턴을 적용하여 Supervisor Agent (7B)가 계획과 복잡한 판단만 담당하고, 단순한 Tool 호출이나 검색 실행은 1.5B Executor가 처리하는 구조로 평균 토큰 비용을 절감한다[cite: 14].
* [cite_start]카메라 영상 분석이 필요한 경우에만 Vision 모델을 호출하는 이중 라우팅도 포함한다[cite: 15].

| 지표 | 목표 | 측정 방법 |
| :--- | :--- | :--- |
| First Token Latency | < 2초 | [cite_start]TensorRT 벤치마크 (7B INT4, A6000) [cite: 16] |
| 토큰 생성 속도 | > 15 tok/s | [cite_start]VLLM 프로파일러 [cite: 16] |
| 양자화 모델 크기 | [cite_start]< 5GB VRAM | nvidia-smi 측정 [cite: 16] |
| 7B 호출 비율 | < 40% | [cite_start]100건 테스트 쿼리 라우팅 로그 집계 [cite: 16] |

[cite_start]**G2 - 차량 도메인 지식 시스템** [cite: 17]
* [cite_start]사용자 질문의 성격에 따라 세 가지 지식 경로를 자동 분기한다[cite: 18].
* [cite_start]"경고등이 뭐야" 같은 매뉴얼 질문은 Vector RAG(bge-m3 + Qdrant)로 관련 문서를 검색하여 답변하고, "엔진 경고등과 관련된 부품과 정비 주기" 같은 관계 추론 질문은 Graph RAG가 Neo4j Knowledge Graph에서 부품 경고등→ 정비 관계를 탐색하여 컨텍스트를 보강한다[cite: 19].
* [cite_start]"이번 달 총 주행 거리" 같은 데이터 조회 질문은 Text2SQL Agent가 차량 텔레매트리/정비 이력 DB에서 SQL을 생성·실행하여 답변한다[cite: 20].
* [cite_start]검색 품질이 낮다고 판단되면 CRAG가 쿼리를 재작성하여 자동 재검색한다[cite: 21].

| 지표 | 목표 | 측정 방법 |
| :--- | :--- | :--- |
| 매뉴얼 QA 정확도 | > 85% | [cite_start]Gold Set 100문항, 정답 포함 여부 [cite: 22] |
| Retrieval Recall@5 | > 90% | [cite_start]정답 청크 Top-5 포함 비율 [cite: 22] |
| Text2SQL 실행 정확도 | > 75% | [cite_start]40문항, SQL 결과 Exact Match [cite: 25] |
| Graph RAG 경로 정확도 | > 80% | [cite_start]30문항, 탐색 경로 일치율 [cite: 25] |

[cite_start]**G3-Cognitive Agent 시스템** [cite: 26]
* [cite_start]4개의 특화 Agent(Supervisor, Perception, Knowledge, Execution)가 A2A 프로토콜로 통신하며 역할을 분담한다[cite: 27].
* [cite_start]Supervisor Agent가 ReAct 루프(Perceive→Reason Act Observe→Reflect)를 제어하며, 상황에 따라 하위 Agent에 작업을 위임한다[cite: 28].
* [cite_start]Perception Agent는 카메라 영상에서 비, 터널 진입, 경고등 같은 환경 조건을 감지하여 자동으로 관련 Tool을 트리거한다[cite: 29].
* [cite_start]"타이어 압력 확인하고, 비정상이면 정비소 찾아줘"처럼 하나의 요청이 여러 Tool을 순차적으로 필요로 하는 경우 Execution Agent가 Multi-Turn으로 처리하며, 중간에 실패하면 재시도하거나 Supervisor에게 대안을 요청한다[cite: 30].

| 지표 | 목표 | 측정 방법 |
| :--- | :--- | :--- |
| 단일 Tool-Call 정확도 | > 90% | [cite_start]50건, Tool명+파라미터 Exact Match [cite: 31] |
| 멀티모달 트리거 Precision | > 80% | [cite_start]비/터널/경고등 영상 30건 [cite: 31] |
| Multi-Turn 완수율 | > 80% | [cite_start]20건, 최종 VehicleState 일치 [cite: 31] |

[cite_start]**G4-Al Agent 품질 평가 체계** [cite: 32]
* [cite_start]Agent가 생성한 응답의 품질을 사람 없이 자동으로 측정하는 평가 파이프라인을 구축한다[cite: 33].
* [cite_start]별도 LLM(GPT-40)이 Judge로서 응답을 1~10점으로 채점하고, 두 모델의 응답을 나란히 비교하는 Pairwise Ranking도 지원한다[cite: 34].
* [cite_start]Hallucination Detection 모듈은 NLI 모델로 응답의 각 클레임이 검색된 소스 문서에 근거하는지 자동 검증하여, 소스에 없는 내용을 만들어낸 경우를 탐지한다[cite: 35].
* [cite_start]200건 규모의 Gold Set(RAG 80건 + Tool 60건 + SQL 40건 + 복합 20건)을 구축하고, Faithfulness, Answer Relevancy, Context Recall 등 RAG 품질 메트릭을 실시간 대시보드로 시각화한다[cite: 36].

| 지표 | 목표 | 측정 방법 |
| :--- | :--- | :--- |
| Faithfulness | > 0.85 | [cite_start]RAGAS 라이브러리 자동 계산 [cite: 37] |
| Hallucination 탐지율 | > [cite_start]85% | adversarial 세트 50건 [cite: 37] |
| CRAG 보정 후 Recall 향상 | > 10%p | [cite_start]보정 전후 Recall@5 비교 [cite: 37] |

[cite_start]**G5-End-to-End 통합 데모** [cite: 38]
* [cite_start]카메라 입력 → VLM 추론→ Agent 협업 → RAG/Graph/SQL/Tool 실행 → React UI 응답까지 전체 파이프라인이 하나의 흐름으로 동작하는 데모 시스템을 구축한다[cite: 39].
* [cite_start]매뉴얼 QA, 차량 제어, Vision 장면 이해, 멀티모달 트리거, Graph 관계 추론, Text2SQL 데이터 조회, Multi-Turn 복합 요청, 일반 대화 총 8개 시나리오를 커버한다[cite: 40].

| 지표 | 목표 | 측정 방법 |
| :--- | :--- | :--- |
| E2E 시나리오 통과 | 8개 전체 | [cite_start]시나리오별 녹화 + 로그 [cite: 43] |
| 단순 시나리오 응답 | < 3초 | [cite_start]타임스탬프 로그 [cite: 43] |
| 복합 시나리오 응답 | < 8초 | [cite_start]타임스탬프 로그 [cite: 43] |

## [cite_start]2. 시스템 아키텍처 [cite: 46]
### [cite_start]2.1 프로토콜 3층 구조 [cite: 47]

| 프로토콜 | 역할 | 적용 범위 |
| :--- | :--- | :--- |
| MCP (수직) | Agent Tool/Data 연결 | [cite_start]12종 Vehicle Tool, Qdrant, Neo4j, Vehicle DB [cite: 48] |
| A2A (수평) | Agent Agent 통신. Agent Card 기반 동적 발견 | [cite_start]Supervisor Perception / Knowledge / Execution [cite: 48] |
| AG-UI (프론트) | Agent Frontend 스트리밍 | [cite_start]WebSocket 토큰, Tool 결과 카드, 대시보드 [cite: 48] |

### [cite_start]2.2 ReAct Cognitive Architecture [cite: 49]
* [cite_start]Perceive → Reason/Plan → Act Observe → Reflect 반복 루프[cite: 50].
* [cite_start]Reflexion으로 실패 경험을 장기 메모리에 축적하고 유사 상황에서 개선된 전략을 적용한다[cite: 51].

| 단계 | 설명 | 구현 | 예시 |
| :--- | :--- | :--- | :--- |
| Perceive | 입력 + 카메라 + 차량 상태 통합 인지 | Query Router + VLM + VehicleState | [cite_start]"비 오는데 창문 열려있어" + 카메라 [cite: 52] |
| Reason | CoT 추론, 액션 계획 수립 | LLM CoT + 멀티 스텝 플래닝 | [cite_start]"비+창문열림→ 닫기+와이퍼" [cite: 52] |
| Act | 계획된 액션 실행 | [cite_start]MCP Tool + RAG + Text2SQL + Graph | window_control → wiper_control [cite: 52] |
| Observe | 결과 확인, 성공/실패 판단 | ToolResult 검증 + State 확인 | [cite_start]창문닫힘, 와이퍼실패→ 재시도 [cite: 52] |
| Reflect | Reflexion: 평가→ 전략 수정→ 메모리 | LLM 자기평가 + Experience Memory | [cite_start]"와이퍼실패시 speed확인" → 저장 [cite: 52] |

### [cite_start]2.3 A2A Multi-Agent [cite: 53]
* [cite_start]각 Agent를 A2A Server로 배포, Supervisor가 A2A Client로 Agent Card 통해 동적 발견·호출[cite: 54].
* [cite_start]LangGraph StateGraph로 오케스트레이션[cite: 55].

| Agent | 역할 | A2A Skill | MCP Tool |
| :--- | :--- | :--- | :--- |
| Supervisor | ReAct 루프, Plan-and-Execute, Reflexion | plan, delegate, evaluate | - (위임만) [cite_start][cite: 56] |
| Perception | 영상 분석, 환경 감지, 멀티모달 트리거 | analyze_scene, detect | [cite_start]VLM, camera_feed [cite: 56] |
| Knowledge | RAG, Graph RAG, Text2SQL, 매뉴얼 QA | search, query_db, traverse | [cite_start]Qdrant, Neo4j, DB [cite: 56, 59] |
| Execution | Tool 실행, Multi-Turn, 재시도, 검증 | execute, verify | [cite_start]12종 Vehicle Tool [cite: 59] |

### [cite_start]2.4 Plan-and-Execute 모델 라우팅 [cite: 60]

| 역할 | 모델 | 담당 | 특성 |
| :--- | :--- | :--- | :--- |
| Planner | Qwen2-VL 7B INT4 | ReAct, 복잡 판단, 계획, Reflexion | [cite_start]느리지만 정확 [cite: 61] |
| Executor | Qwen2-VL 1.5B INT4 | Tool 호출, 단순 RAG, SQL, 요약 | [cite_start]빠르고 저렴 [cite: 61] |
| Vision | Qwen2-VL 7B FP16 | 카메라 분석, 환경 감지 | [cite_start]Vision 시에만 호출 [cite: 61] |

### [cite_start]2.5 메모리 시스템 [cite: 62]

| 유형 | 구현 | 용도 |
| :--- | :--- | :--- |
| 단기 | ConversationBufferWindow | [cite_start]대화 컨텍스트, Multi-Turn 상태 [cite: 63] |
| 장기 | Vector Store Experience Memory | [cite_start]Reflexion 경험 축적, 유사 상황 검색 [cite: 63] |
| 엔티티 | 사용자 선호도 KV Store | [cite_start]개인화 응답 [cite: 63] |

## [cite_start]3. Knowledge Graph + Graph RAG [cite: 66]
### [cite_start]3.1 차량 Knowledge Graph [cite: 67]
* [cite_start]부품, 경고등, 증상, 정비, 진단코드 간 관계를 Neo4j로 모델링한다[cite: 68].

| Node | 예시 | Edge | 연결 |
| :--- | :--- | :--- | :--- |
| Component | 엔진오일, 브레이크, 타이어 | HAS PART, MAINTAINED BY | [cite_start]System, Maintenance [cite: 69] |
| WarningLight | 엔진, ABS, 배터리 | INDICATES, CAUSED BY | [cite_start]Symptom, Component [cite: 69] |
| Symptom | 소음, 진동, 누유 | SYMPTOM OF, RESOLVED BY | [cite_start]Component, Action [cite: 69] |
| Maintenance | 오일교환, 타이어교체 | APPLIES TO, HAS INTERVAL | [cite_start]Component, Schedule [cite: 69] |
| DTC Code | P0301, P0420 | TRIGGERS, MAPS_TO | [cite_start]WarningLight, Component [cite: 69] |

### [cite_start]3.2 Graph RAG 파이프라인 [cite: 70]
* [cite_start]1. Vector RAG: bge-m3 + Qdrant로 관련 문서 추출 [cite: 71]
* [cite_start]2. Entity Extraction: 검색 결과에서 부품/경고등/증상 엔티티 추출 [cite: 72]
* [cite_start]3. Graph Traversal: Neo4j 1~2 hop 탐색으로 관계 정보 확보 [cite: 73]
* [cite_start]4. Context Fusion: 벡터+그래프 결과 통합하여 LLM 컨텍스트로 제공 [cite: 74]

## [cite_start]4. Text2SQL Agent [cite: 77]
### [cite_start]4.1 차량 DB [cite: 78]

| 테이블 | 컬럼 | 설명 | 쿼리 예시 |
| :--- | :--- | :--- | :--- |
| vehicle_telemetry | ts, speed, rpm, fuel, battery, tire * | 실시간 센서 | [cite_start]"어제 평균 속도?" [cite: 79] |
| maintenance_log | date, mileage, type, parts, cost | 정비 이력 | [cite_start]"마지막 오일교환?" [cite: 79] |
| dtc_codes | code, description, severity, resolved | OBD-II 코드 | [cite_start]"미해결 경고 목록" [cite: 79] |
| trip_history | start, end, distance, avg_speed, fuel | 운행 기록 | [cite_start]"이번 달 총 거리?" [cite: 79] |

### [cite_start]4.2 Text2SQL 파이프라인 [cite: 80]
* [cite_start]Schema Linking: 쿼리 관련 테이블만 LLM에 제공 [cite: 81]
* [cite_start]Few-shot SQL: 유사 쿼리-SQL 예시를 벡터 검색으로 찾아 제공 [cite: 82]
* [cite_start]Self-Consistency: N개 SQL 생성 다수결 투표로 최종 선택 [cite: 83]
* [cite_start]SQL 검증: 구문+EXPLAIN 분석 후 실행 [cite: 84]

## [cite_start]5. Multi-Turn Tool Calling + 실패 처리 [cite: 85]
* [cite_start]하나의 요청이 복수 Tool을 필요로 할 때, Execution Agent가 순차/병렬 호출하고 각 결과를 컨텍스트에 축적한다[cite: 86].

| 실패 유형 | 감지 | 재시도 | 최대 |
| :--- | :--- | :--- | :--- |
| Tool 타임아웃 | asyncio.wait_for | 동일 Tool 재호출 | [cite_start]2회 [cite: 87] |
| 파라미터 오류 | ToolResult.success=false | LLM에 오류전달 수정 | [cite_start]2회 [cite: 87] |
| 잘못된 Tool | 결과 검증 실패 | Supervisor 대안 제안 | [cite_start]1회 [cite: 87] |
| SQL 오류 | DB exception | 스키마 재전달→ SQL 재생성 | [cite_start]3회 [cite: 87] |

## [cite_start]6. 프로젝트 일정 (12주) [cite: 90]

| Phase | 기간 | 주요 활동 | 산출물 |
| :--- | :--- | :--- | :--- |
| Phase 1 VLM 배포 | 1~2주 | TensorRT Edge-LLM 7B+1.5B INT4 배포, vLLM AWQ 병행, Plan-and-Execute 라우팅, 벤치마크 | [cite_start]TensorRT 엔진, vLLM, 라우팅 로직, 비용 리포트 [cite: 91] |
| Phase 2 RAG+Graph+SQL | 3~5주 | Vehicle RAG(bge-m3+Qdrant+Reranker+CRAG), Knowledge Graph (Neo4j), Graph RAG, Text2SQL(4테이블), Query Router | [cite_start]RAG+Graph RAG, KG, Text2SQL, Vehicle DB [cite: 91] |
| Phase 3 Cog.Arch+A2A+Tool | 6~8주 | ReAct+Reflexion, Multi-Agent(LangGraph), A2A Server/Client, MCP 12종 Tool, Multi-Turn+Retry, 메모리 3계층, 멀티모달 트리거 | [cite_start]Cognitive Arch., A2A Multi-Agent, MCP, 메모리 [cite: 91] |
| Phase 4 평가 체계 | 9~10주 | LLM-as-a-Judge, Hallucination Detection, CRAG, Gold Set 200건, 품질 대시보드 | [cite_start]Gold Set, 평가 파이프라인, 대시보드 [cite: 91] |
| Phase 5 통합+ 데모 | 11~12주 | E2E 통합, React Automotive UI, AG-UI, 시나리오 8종, 기술 리포트, 포트폴리오 | [cite_start]통합 시스템, 데모, 리포트, 포트폴리오 [cite: 91] |

## [cite_start]7. 기술 스택 [cite: 92]

| 영역 | 기술 |
| :--- | :--- |
| VLM | [cite_start]TensorRT Edge-LLM Qwen2-VL 7B+1.5B(INT4) /vLLM+AutoAWQ [cite: 93] |
| Agent | [cite_start]LangGraph StateGraph + A2A Protocol + MCP [cite: 93] |
| Cognitive | [cite_start]ReAct + Reflexion + Plan-and-Execute + CoT [cite: 93] |
| [cite_start]RAG | bge-m3 + bge-reranker-v2-m3 + Qdrant + CRAG [cite: 93] |
| Graph RAG | [cite_start]Neo4j + Entity Extraction + Traversal + Context Fusion [cite: 93] |
| Text2SQL | [cite_start]Few-shot + Self-Consistency + Schema Linking + SQLite [cite: 93] |
| 메모리 | [cite_start]단기(Buffer) + 장기(Vector Experience) + 엔티티(KV) [cite: 93] |
| 평가 | [cite_start]LLM-as-a-Judge + Hallucination(NLI) + CRAG + Gold Set 200+ 대시보드 [cite: 93] |
| Frontend | [cite_start]React + Tailwind (Automotive Dark) + AG-UI + 평가 대시보드 [cite: 93] |
| Infra | [cite_start]A6000 48GB, Ubuntu 22.04, Docker, Neo4j, Git, W&B [cite: 93] |

## [cite_start]8. 참고 문헌 [cite: 96]
* [cite_start][1] NVIDIA TensorRT Edge-LLM (GitHub) - EAGLE-3, NVFP4, CES 2026 [cite: 97]
* [cite_start][2] A2A Protocol v1.0 (Linux Foundation, Apr 2026) - 150+ 기업, Azure/AWS 프로덕션 [cite: 98]
* [cite_start][3] Google: Agent Protocols Guide (Mar 2026) MCP+A2A+AG-UI 통합 [cite: 99]
* [cite_start][4] MLMastery: 7 Agentic Trends 2026 Plan-and-Execute, 비용 최적화 [cite: 100]
* [cite_start][5] IBM Think: Al Trends 2026 Cooperative Model Routing [cite: 101]
* [cite_start][6] ReAct (Yao 2023) / Reflexion (Shinn 2023) - Cognitive Architecture [cite: 102]
* [cite_start][7] CRAG (Yan 2024) Corrective RAG [cite: 104]
* [cite_start][8] Arm On-Device Assistant (Sep 2025) - VLM+LLM, MCP, KV Cache [cite: 106]
* [cite_start][9] Deloitte: Agentic Al Strategy (Feb 2026) - Toyota Agent, 스케일링 [cite: 108]