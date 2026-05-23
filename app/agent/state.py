from typing import TypedDict, Annotated, Dict
import operator
from langgraph.graph.message import add_messages


def merge_context(left: dict, right: dict) -> dict:
    """context_data 얕은 병합 reducer. 같은 키는 right 값으로 덮어쓰기."""
    return {**left, **right}


class AgentState(TypedDict):
    # 1. 단기 메모리 및 기본 컨텍스트
    messages: Annotated[list, add_messages]  
    route_type: str                          
    
    # 2. Plan-and-Execute 및 A2A 상태
    plan: list                             
    next_agent: str                        
    
    # 3. Tool 및 외부 시스템 실행 결과
    tool_calls: Annotated[list, operator.add]                  
    context_data: Annotated[dict, merge_context]
    
    # 4. 실패 처리 (Retry) 로직
    error_count: Dict[str, int]              
    feedback: str                           