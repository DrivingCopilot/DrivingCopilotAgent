from typing import TypedDict, Annotated, List, Dict, Any
import operator

class AgentState(TypedDict):
    # 1. 단기 메모리 및 기본 컨텍스트
    messages: Annotated[list, operator.add]  # 사용자와 에이전트 간의 대화 기록 (ConversationBufferWindow 역할)
    route_type: str                          # Query Router 결과 (rag, tool, vision, chat)
    
    # 2. Plan-and-Execute 및 A2A 상태
    plan: list                               # 수퍼바이저(7B)가 세운 다단계 실행 계획
    next_agent: str                          # A2A 위임을 위해 다음에 호출할 에이전트 이름 (knowledge, execution 등)
    
    # 3. Tool 및 외부 시스템 실행 결과
    tool_calls: list                         # 실행할 MCP Tool 이름과 파라미터
    context_data: dict                       # Vector RAG, Graph RAG, Text2SQL 결과 병합 컨텍스트
    
    # 4. 실패 처리 (Retry) 로직
    error_count: Dict[str, int]              # 실패 유형별 카운터 (timeout, param_error, sql_error 등)
    feedback: str                            # Reflexion 단계에서 생성된 피드백 (수퍼바이저의 자기반성)