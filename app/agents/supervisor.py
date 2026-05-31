import json
import logging
from typing import Any, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI 
from app.graph.state import AgentState
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


logger = logging.getLogger(__name__)

# --- Mock external dependencies ---
# In a real scenario, these would be imported from other modules in the project.

class MockWebsocketManager:
    async def send_status(self, message: str):
        logger.info(f"WS_MOCK_SEND: {message}")
        
    def send_status_sync(self, message: str):
        # Synchronous fallback if the node is not async
        logger.info(f"WS_MOCK_SEND: {message}")

websocket_manager = MockWebsocketManager()

def get_agent_registry() -> list[dict]:
    """Mock registry to fetch available agents dynamically."""
    return [
        { # Perception agent의 MCP Tool: VLM, camera_feed
            "name": "perception",
            "skill": "Visual/spatial reasoning and camera/image analysis.",
            "mcp_tools": ["analyze_camera_feed", "detect_objects"]
        },
        { # knowledge agent의 MCP Tool: Qdrant, Neo4j, DB
            "name": "knowledge",
            "skill": "Retrieves vehicle manuals, FAQs, and domain knowledge.",
            "mcp_tools": ["vector_rag_search", "graph_rag_search"]
        },
        { # execution agent의 MCP Tool: 12종 Vehicle Tool
            "name": "execution",
            "skill": "Vehicle control commands, API interactions, and structured state modification.",
            "mcp_tools": [
                "climate", "navigation", "media", "vehicle_status", 
                "window", "lighting", "seat", "parking", 
                "emergency", "driving_mode", "wiper", "dashboard_query"
            ]
        }
    ]

# -----------------------------------

def _build_dynamic_agent_cards() -> str:
    registry = get_agent_registry()
    cards_str = ""
    for idx, agent in enumerate(registry, 1):
        cards_str += f"{idx}. {agent['name']}:\n"
        cards_str += f"   - Skill: {agent['skill']}\n"
        cards_str += f"   - MCP Tools: {', '.join(agent['mcp_tools'])}\n"
    return cards_str

SUPERVISOR_SYSTEM_PROMPT = """You are a highly capable Supervisor Agent orchestrating a Multi-Agent System for a Driving Copilot.
Your role is to analyze the user's request, evaluate the current context, and coordinate sub-agents using a 'Plan-and-Execute' pattern.
You rely on Chain-of-Thought reasoning to make decisions.

[Available Sub-Agents (Dynamic Agent Cards)]
{agent_cards}

[Rules & Protocol]
1. Use A2A delegation by selecting the appropriate agent from the list above.
2. If the user's request requires understanding the physical environment, delegate to 'perception'.
3. If the user's request requires manuals or relational knowledge, delegate to 'knowledge'.
   - IMPORTANT Context Fusion: Evaluate if the current context has adequate 'Vector RAG' and 'Graph RAG' data. If entities and relationships are unclear, explicitly instruct the 'knowledge' agent to use Graph RAG.
4. If the user requests an action or structured data retrieval, delegate to 'execution'.
5. Always output your response in strictly valid JSON format.
6. The JSON must contain three keys:
   - "reasoning": A brief explanation of your thought process (Chain-of-Thought).
   - "plan": A list of step-by-step strings for the execution plan.
   - "next_agent": One of the agent names from the registry, or "__end__" if the task is complete.
"""

async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph 기반 Supervisor Node 고도화 (Qwen2-VL 7B 활용)
    1. WebSocket 실시간 스트리밍 연동
    2. 동적 Agent Registry 주입
    3. Knowledge Fusion 및 CoT 기반 추론
    4. Reflexion 및 파라미터 에러 Self-Loop 복구 로직
    """
    
    # 1. WebSocket 실시간 스트리밍 연동
    await websocket_manager.send_status(json.dumps({"type": "status", "data": "Planning next steps..."}))
    
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    context_data = state.get("context_data", {})
    error_count = state.get("error_count", {})
    feedback = state.get("feedback", "")
    route_type = state.get("route_type", "")
    current_next_agent = state.get("next_agent", "")
    
    # 2. 에러 횟수 초과에 따른 하드 Fallback
    timeout_err = error_count.get("timeout", 0)
    param_err = error_count.get("parameter", 0)
    sql_err = error_count.get("sql", 0)
    
    if timeout_err >= 2 or param_err >= 2 or sql_err >= 3:
        fallback_msg = AIMessage(content="시스템 오류가 반복 발생하여 안전을 위해 작업을 종료합니다. (수퍼바이저 대안 개입)")
        await websocket_manager.send_status(json.dumps({"type": "status", "data": "System error limit reached. Terminating."}))
        return {
            "messages": [fallback_msg],
            "next_agent": "__end__",
            "plan": []
        }
    
    # 수퍼바이저 전용 모델 고정 (G1 비용 최적화 준수)
    llm = ChatOpenAI(model="qwen2-vl-7b-instruct-int4", temperature=0.1) 
    
    # 3. Context Fusion 고도화
    vector_rag = context_data.get("vector_results", [])
    graph_rag = context_data.get("graph_results", [])
    vehicle_state = context_data.get("vehicle_state", {})
    
    context_str = f"""[Context Data]
- Route Type Hint: {route_type}
- Previous Agent Hint: {current_next_agent}
- Vector RAG Results: {vector_rag}
- Graph RAG Results: {graph_rag}
- Vehicle State: {vehicle_state}
"""
    
    # 4. 강화된 Reflexion 로직: feedback이 존재하면 프롬프트에 반영
    reflexion_str = ""
    if feedback:
        reflexion_str = f"\n[Reflexion Feedback from Previous Attempt]\n{feedback}\nPlease adjust your plan based on this feedback."

    eval_prompt = f"""{context_str}
[Current Plan]
{plan}
{reflexion_str}

[Instructions]
1. Think step-by-step (Chain-of-Thought). First, evaluate if the current Context Data is sufficient to answer the user's request.
2. Pay special attention to whether you need more 'entities and relationships' (Graph RAG) vs 'semantic similarity' (Vector RAG).
3. If sufficient, output an empty plan and "__end__" for next_agent.
4. If insufficient, formulate the next steps in the 'plan' array and choose the 'next_agent'.
"""

    # 5. A2A 프로토콜 동적 연동 
    dynamic_cards = _build_dynamic_agent_cards()
    system_msg_content = SUPERVISOR_SYSTEM_PROMPT.format(agent_cards=dynamic_cards)

    messages_to_send = [SystemMessage(content=system_msg_content)]
    messages_to_send.extend(messages)
    messages_to_send.append(HumanMessage(content=eval_prompt))
    
    try:
        content = ""
        # 실시간 토큰 스트리밍 구현
        async for chunk in llm.astream(messages_to_send):
            if chunk.content:
                await websocket_manager.send_status(json.dumps({"type": "text", "data": chunk.content}))
                content += chunk.content
        
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        parsed_result = json.loads(content)
        
        new_plan = parsed_result.get("plan", [])
        next_agent = parsed_result.get("next_agent", "__end__")
        reasoning = parsed_result.get("reasoning", "")
        
        logger.info(f"Supervisor Reasoning: {reasoning}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Qwen2-VL response as JSON: {e}. Raw content: {content}")
        updated_error_count = dict(error_count) 
        updated_error_count["parameter"] = updated_error_count.get("parameter", 0) + 1
        
        await websocket_manager.send_status(json.dumps({"type": "status", "data": "Output parsing failed. Attempting self-recovery..."}))
        return {
            "error_count": updated_error_count,
            "feedback": f"Failed to parse your last response as valid JSON. Ensure strictly valid JSON format. Error: {str(e)}",
            "next_agent": "supervisor" 
        }
    except Exception as e:
         logger.error(f"Unexpected error during LLM invocation: {e}")
         return {"next_agent": "__end__"}
        
    await websocket_manager.send_status(json.dumps({"type": "status", "data": f"Delegating task to {next_agent}"}))
    
    # AgentState 업데이트 시 messages 리스트에 수퍼바이저의 판단 결과(reasoning)를 AIMessage 형태로 추가
    result = {
        "plan": new_plan,
        "next_agent": next_agent,
        "feedback": ""
    }
    if reasoning:
        result["messages"] = [AIMessage(content=reasoning)]
        
    return result



async def knowledge_node(state: AgentState) -> Dict[str, Any]:
    await websocket_manager.send_status(json.dumps({"type": "status", "data": "Knowledge agent processing..."}))
    
