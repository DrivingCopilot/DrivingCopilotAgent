# app/agents/supervisor.py
#
# Supervisor Agent — LangGraph 노드로 동작하며 하위 Agent(Knowledge/Execution/Perception)를 조율한다.
# Agent Card 기반 동적 레지스트리를 참조해 Plan-and-Execute + CoT + Reflexion 루프를 수행한다.
# ExperienceMemory에서 과거 실패 lesson을 검색해 프롬프트에 주입한다.

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.graph import ws as _ws
from app.graph.state import AgentState
from app.core.config import MAX_RETRY, EXPERIENCE_TOP_K

logger = logging.getLogger(__name__)

# ExperienceMemory lazy-init 싱글톤
_experience_memory = None


def _get_experience_memory():
    global _experience_memory
    if _experience_memory is None:
        from app.memory.experience import ExperienceMemory
        _experience_memory = ExperienceMemory()
    return _experience_memory


def get_agent_registry() -> list[dict]:
    """사용 가능한 하위 Agent 목록을 반환한다. 추후 A2A HTTP 발견으로 교체 예정."""
    return [
        {
            "name": "perception",
            "skill": "Visual/spatial reasoning and camera/image analysis.",
            "mcp_tools": ["analyze_camera_feed", "detect_objects"],
        },
        {
            "name": "knowledge",
            "skill": "Retrieves vehicle manuals, FAQs, and domain knowledge.",
            "mcp_tools": ["vector_rag_search", "graph_rag_search"],
        },
        {
            "name": "execution",
            "skill": "Vehicle control commands, API interactions, and structured state modification.",
            "mcp_tools": [
                "control_climate", "set_navigation", "control_media", "get_vehicle_status",
                "control_window", "control_lighting", "control_seat", "control_parking",
                "trigger_emergency", "set_driving_mode", "control_wiper", "query_dashboard",
            ],
        },
    ]


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

[Error Recovery Protocol]
When a 'Reflexion Feedback' indicates a tool failure, choose the recovery strategy based on the error_type:

- error_type='timeout': The tool call timed out. Retry with the SAME tool and SAME parameters. The failure is likely transient.
- error_type='parameter': The tool was correct but parameters were invalid. Retry with the SAME tool but FIX the parameters based on the error message.
- error_type='invalid_tool': The tool itself was wrong for this task. Choose a DIFFERENT tool. Do not call the same tool again.
- error_type='sql': SQL generation failed against the database. Re-examine the schema and regenerate a corrected SQL query. Use the same 'knowledge' agent with a corrected SQL.

Always include your error_recovery reasoning in the 'reasoning' field of your JSON output when responding to a failure.
"""


async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph 기반 Supervisor Node (Qwen2-VL 7B 활용)
    1. WebSocket 실시간 스트리밍 연동
    2. 동적 Agent Registry 주입
    3. ExperienceMemory 검색 + Knowledge Fusion 및 CoT 기반 추론
    4. Reflexion 및 파라미터 에러 Self-Loop 복구 로직
    """

    # 1. WebSocket 실시간 스트리밍 연동
    await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": "Planning next steps..."}))

    messages = state.get("messages", [])
    plan = state.get("plan", [])
    context_data = state.get("context_data", {})
    error_count = state.get("error_count", {})
    feedback = state.get("feedback", "")
    route_type = state.get("route_type", "")
    current_next_agent = state.get("next_agent", "")

    # 수퍼바이저 전용 모델 고정 (G1 비용 최적화 준수)
    llm = ChatOpenAI(model="qwen2-vl-7b-instruct-int4", temperature=0.1)

    # 첫 진입 시에만 experience 검색 (캐시 분기: run_graph 한 번에 재사용)
    if "retrieved_experience" not in context_data:
        user_query = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        try:
            retrieved = await asyncio.to_thread(
                _get_experience_memory().search,
                user_query,
                route_type or None,
                EXPERIENCE_TOP_K,
            )
            context_data = {
                **context_data,
                "retrieved_experience": [d.metadata["lesson"] for d in retrieved],
            }
        except Exception as e:
            logger.warning("supervisor: experience 검색 실패 — %s", e)
            context_data = {**context_data, "retrieved_experience": []}

    # Context Fusion
    vector_rag = context_data.get("vector_results", [])
    graph_rag = context_data.get("graph_results", [])
    vehicle_state = context_data.get("vehicle_state", {})
    retrieved_experience = context_data.get("retrieved_experience", [])

    context_str = f"""[Context Data]
- Route Type Hint: {route_type}
- Previous Agent Hint: {current_next_agent}
- Vector RAG Results: {vector_rag}
- Graph RAG Results: {graph_rag}
- Vehicle State: {vehicle_state}
- Retrieved Experience (past lessons): {retrieved_experience}
"""

    # 강화된 Reflexion 로직: feedback이 존재하면 프롬프트에 반영
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

    # A2A 프로토콜 동적 연동
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
                await _ws.websocket_manager.send_status(json.dumps({"type": "text", "data": chunk.content}))
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
        count = updated_error_count["parameter"]
        limit = MAX_RETRY.get("parameter", 2)

        if count >= limit:
            logger.error(f"Supervisor JSON parse retry limit exceeded: {count}/{limit}")
            user_msg = (
                f"JSON 생성에 {count}회 실패했습니다. 요청을 처리할 수 없습니다."
            )
            await _ws.websocket_manager.send_status(json.dumps({"type": "text", "data": user_msg}))
            await _ws.websocket_manager.send_status(json.dumps({"type": "done", "reason": "parse_error_limit"}))
            return {
                "error_count": updated_error_count,
                "feedback": f"Supervisor JSON parse failed {count}/{limit} times. Stopping.",
                "next_agent": "__end__",
                "messages": [AIMessage(content=user_msg)]
            }

        await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": f"JSON 생성 재시도 중... ({count}/{limit})"}))
        return {
            "error_count": updated_error_count,
            "feedback": f"Failed to parse response as valid JSON (attempt {count}/{limit}). Error: {str(e)}",
            "next_agent": "supervisor"
        }
    except Exception as e:
        logger.error(f"Unexpected error during LLM invocation: {e}")
        await _ws.websocket_manager.send_status(json.dumps({"type": "done", "reason": "llm_error"}))
        return {"next_agent": "__end__"}

    await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": f"Delegating task to {next_agent}"}))
    if next_agent == "__end__":
        await _ws.websocket_manager.send_status(json.dumps({"type": "done"}))

    # AgentState 업데이트 시 messages 리스트에 수퍼바이저의 판단 결과(reasoning)를 AIMessage 형태로 추가
    result = {
        "plan": new_plan,
        "next_agent": next_agent,
        "feedback": ""
    }
    if reasoning:
        result["messages"] = [AIMessage(content=reasoning)]

    return result
