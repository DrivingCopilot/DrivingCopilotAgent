import json
import logging
from typing import Any, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI 
from .state import AgentState

logger = logging.getLogger(__name__)

# 수퍼바이저 에이전트의 페르소나 및 하위 에이전트(Agent Card) 정의
AGENT_CARDS = """You are a Supervisor Agent orchestrating a Multi-Agent System for a Driving Copilot.
Your role is to analyze the user's request, evaluate current context, and coordinate sub-agents using a 'Plan-and-Execute' pattern.
You will communicate with agents via A2A delegation and MCP (Model Context Protocol).

[Available Sub-Agents (Agent Cards)]
1. perception: Handles visual/spatial reasoning, analyzing camera/image data, and understanding the physical environment of the vehicle.
2. knowledge: Handles bge-m3 based RAG searches for vehicle manuals, FAQs, and domain knowledge.
3. execution: Handles actual vehicle control commands, API interactions, and SQL queries to modify or retrieve structured vehicle states.

[Rules]
- Always output your response in strictly valid JSON format.
- Keys must be "plan" (a list of step-by-step strings) and "next_agent" (one of "perception", "knowledge", "execution", or "__end__").
- If the request is fulfilled, set "next_agent" to "__end__" and provide an empty plan.
"""

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph 기반 Supervisor Node (Qwen2-VL 7B 활용)
    1. 실패 처리 로직(에러 카운트) 검사
    2. Context Data (RAG, VehicleState 등) 평가 (Evaluate)
    3. Plan 수정 및 다음 에이전트 라우팅 (Reflexion)
    """
    # 1. WebSocket 스트리밍을 위한 노드 실행 상태 로깅
    logger.info("tool_start: supervisor_node - evaluating state and planning")
    
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    context_data = state.get("context_data", {})
    error_count = state.get("error_count", {})
    
    # 2. 실패 처리 로직 반영 (타임아웃 2회, 파라미터 2회, SQL 3회)
    timeout_err = error_count.get("timeout", 0)
    param_err = error_count.get("parameter", 0)
    sql_err = error_count.get("sql", 0)
    
    if timeout_err >= 2 or param_err >= 2 or sql_err >= 3:
        logger.warning(f"Error thresholds exceeded (timeout: {timeout_err}, param: {param_err}, sql: {sql_err}). Terminating execution.")
        fallback_msg = AIMessage(content="시스템 오류가 반복 발생하여 안전을 위해 작업을 종료합니다. (수퍼바이저 대안 개입)")
        logger.info("tool_end: supervisor_node - fallback triggered")
        return {
            "messages": [fallback_msg],
            "next_agent": "__end__",
            "plan": []
        }
    
    # 3. 모델 설정 (Qwen2-VL 7B Planner/Supervisor 역할 - OpenAI 호환 API 가정)
    # LLM Provider의 설정에 따라 base_url 등을 변경 가능합니다.
    llm = ChatOpenAI(model="qwen2-vl-7b-instruct", temperature=0.1) 
    
    # 4. 수퍼바이저 핵심 로직: Context Data 판단(Evaluate) 및 Reflexion
    context_str = json.dumps(context_data, ensure_ascii=False) if context_data else "None"
    
    eval_prompt = f"""Current Context Data (bge-m3 RAG results & VehicleState):
{context_str}

Current Plan:
{plan}

[Instructions]
1. Evaluate if the 'Current Context Data' provides enough information to answer the user's latest request.
2. If sufficient, or if the user's request is completely fulfilled:
   - "plan": []
   - "next_agent": "__end__"
3. If insufficient (Reflexion needed), revise the 'Current Plan' and choose the 'next_agent' to execute the next step.
   - "plan": ["step 1", "step 2", ...]
   - "next_agent": "perception" or "knowledge" or "execution"

Respond ONLY in JSON.
"""

    messages_to_send = [SystemMessage(content=AGENT_CARDS)]
    messages_to_send.extend(messages)
    messages_to_send.append(HumanMessage(content=eval_prompt))
    
    logger.info("planning: Generating next steps via Qwen2-VL...")
    
    try:
        # Qwen2-VL 7B를 활용한 라우팅 및 계획 수립
        response = llm.invoke(messages_to_send)
        
        # JSON 결과 파싱 (향후 with_structured_output 사용 권장)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        parsed_result = json.loads(content)
        
        new_plan = parsed_result.get("plan", [])
        next_agent = parsed_result.get("next_agent", "__end__")
        
    except Exception as e:
        logger.error(f"Failed to parse Qwen2-VL response: {e}")
        # 파싱 오류 시 안전하게 종료 (또는 에러 카운트 증가 로직 추가 가능)
        return {"next_agent": "__end__"}
        
    logger.info(f"tool_end: supervisor_node - Next agent selected: {next_agent}")
    
    return {
        "plan": new_plan,
        "next_agent": next_agent
    }
