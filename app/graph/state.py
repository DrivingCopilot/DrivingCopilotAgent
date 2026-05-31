from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 1. 단기 메모리 및 기본 컨텍스트
    messages: Annotated[list, add_messages]  
    diagnostic_text: str
    route_type: str                          
    
    # 2. Plan-and-Execute 및 A2A 상태
    plan: list                             
    next_agent: str                        
    
    # 3. Tool 및 외부 시스템 실행 결과
    tool_calls: list                       
    context_data: dict                      
    
    # 4. 실패 처리 (Retry) 로직
    error_count: Dict[str, int]              
    feedback: str             


    # 지식 추출 및 통합
    extracted_entities: Annotated[List[str], operator.add]
    extracted_relationships: Annotated[List[Dict], operator.add]
    natural_language_context: str