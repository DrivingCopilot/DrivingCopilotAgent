import json
import logging
import os
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from app.core.config import MODEL_SERVER_URL, QWEN_TEXT_MODEL_NAME
from app.graph.state import AgentState
from app.graph import ws as _ws
from app.core.mcp_client import call_mcp_tool_once

logger = logging.getLogger(__name__)


async def _call_knowledge_tool(tool_name: str, params: Dict[str, Any]) -> str:
    """
    공용 MCP 서버의 knowledge tool 을 호출하고 결과 텍스트만 반환한다.
    실패 시 ReAct 에이전트가 읽고 재시도/대안 판단할 수 있는 오류 문자열을 돌려준다.
    """
    result_text, status, error_type, error_msg = await call_mcp_tool_once(tool_name, params)
    if status != "success":
        logger.warning("MCP knowledge tool 실패: %s (%s) — %s", tool_name, error_type, error_msg)
        return f"[{tool_name} 실패: {error_type or 'error'}] {error_msg}"
    return result_text


@tool
async def vector_rag_search(query: str) -> str:
    """
    Search for vehicle manual information using Vector RAG (Qdrant).
    Best for: General questions about the manual, usage instructions, or FAQs.
    """
    return await _call_knowledge_tool("vector_rag_search", {"query": query})


@tool
async def graph_rag_search(query: str, entities: List[str] = None) -> str:
    """
    Search for relational information using Graph RAG (Neo4j).
    Best for: Multi-hop reasoning like "What components are related to this warning light?" or "Maintenance interval for a part".
    """
    return await _call_knowledge_tool(
        "graph_rag_search", {"query": query, "entities": entities or []}
    )


@tool
async def text_to_sql_query(query: str) -> str:
    """
    Query structured vehicle telemetry or maintenance history from the database (Text2SQL).
    Best for: Data retrieval like "What was my average speed?" or "Last oil change date".
    """
    return await _call_knowledge_tool("text_to_sql_query", {"query": query})

KNOWLEDGE_SYSTEM_PROMPT = """You are the Knowledge Agent in the On-Device Multimodal Driving Copilot system.
Your role is to act as the primary knowledge hub, retrieving information from various sources to answer user queries or fulfill plans delegated by the Supervisor.

Available Tools:
1. vector_rag_search: Use for standard manual queries (e.g., "What does this button do?").
2. graph_rag_search: Use for relational queries (e.g., "Engine warning light components and maintenance").
3. text_to_sql_query: Use for database lookups (e.g., "Total mileage this month", "Recent DTC codes").

IMPORTANT — Source of truth:
- The tool results ARE excerpts parsed directly from this vehicle's own official owner's manual and
  internal knowledge base (Vector RAG / Graph RAG indices are built from the manual itself). You DO
  have direct access to this material through these tools — never claim you lack access to the
  manual, internal documents, or manufacturer data, and never tell the user to check the manual
  themselves or contact the manufacturer/customer service when a tool already returned relevant
  content.
- Base your final answer strictly on the tool results returned to you. Only if a tool result is
  genuinely empty or irrelevant after reformulating the query should you say the information was not
  found in the manual — do not fall back to a generic apology/disclaimer instead.

Workflow:
- Read the current plan and the user's request.
- Decide which tool(s) are needed. You may need to call multiple tools if the query is complex (e.g., Context Fusion of Vector + Graph).
- If the search result is poor, try reformulating the query (CRAG approach).
- Synthesize the retrieved context into a clear, concise, and helpful response.

IMPORTANT — Output language: Always write your FINAL answer to the user in Korean (한국어),
regardless of the language of the retrieved tool results or your own intermediate reasoning.
"""

async def knowledge_node(state: AgentState) -> Dict[str, Any]:
    await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": "Knowledge agent retrieving context..."}))
    
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    context_data = state.get("context_data", {})

    # CRAG 재검색 여부: transform_query_node 가 채운 refined_query 가 있으면
    # 동일 plan 스텝을 재작성된 쿼리로 재검색하는 중이다.
    refined_query = context_data.get("refined_query", "")
    is_reretrieval = bool(refined_query)

    # 1. Prepare Instructions based on Plan-and-Execute pattern
    instruction_text = ""
    if is_reretrieval:
        instruction_text = refined_query
    elif plan and len(plan) > 0:
        instruction_text = f"Execute the next step in the plan: {plan[0]}"
    else:
        instruction_text = "Answer the user's latest query based on your domain knowledge tools."

    instruction_msg = HumanMessage(content=f"[Supervisor Instruction] {instruction_text}")
    
    # 2. Setup LLM & Tools (G1: Executor uses Qwen2.5 1.5B)
    llm = ChatOpenAI(model=QWEN_TEXT_MODEL_NAME, temperature=0.1, base_url=MODEL_SERVER_URL)
    tools = [vector_rag_search, graph_rag_search, text_to_sql_query]
    
    agent = create_react_agent(llm, tools, prompt=KNOWLEDGE_SYSTEM_PROMPT)
    
    messages_to_pass = messages + [instruction_msg]
    
    try:
        # 3. Invoke the ReAct Agent
        response = await agent.ainvoke({"messages": messages_to_pass})
        
        # Extract new messages generated by the Knowledge Agent
        new_messages = response["messages"][len(messages_to_pass):]
        
        # Extract the final answer (the last AI message)
        final_answer = new_messages[-1].content if new_messages else "Knowledge retrieval completed."
        
        # Context data update (accumulate findings)
        context_data["last_knowledge_result"] = final_answer
        # refined_query 는 이번 재검색으로 소비됐으므로 초기화(빈 문자열=없음).
        context_data["refined_query"] = ""
        # 이전 시도의 knowledge_failed=True 가 얕은 병합(merge_context)으로
        # 남아있지 않도록 성공 시 명시적으로 False 로 되돌린다(CRAG 노드들이
        # 이 플래그로 재작성/정제 LLM 호출을 건너뛰는 기준이므로, 낡은 True가
        # 남으면 정상 검색 결과도 실패로 오인해 정제를 건너뛰게 된다).
        context_data["knowledge_failed"] = False

        if is_reretrieval:
            # CRAG 재검색: 동일 스텝을 다시 검색한 것이므로 plan 을 재-pop 하지 않고,
            # grade/transform 의 평가 기준(crag_query)도 원본 스텝 그대로 유지한다.
            new_plan = plan
        else:
            # 신규 스텝 실행: 완료 스텝 pop + CRAG 재검색 예산 리셋.
            # grade_retrieval/transform_query 가 "실제로 실행된 쿼리"를 평가하도록
            # crag_query 를 기록한다 — pop 이후 plan[0] 이 다음 스텝으로 바뀌어
            # grader 가 엉뚱한 스텝을 평가하는 문제를 막는다.
            user_query = next(
                (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
            )
            context_data["crag_query"] = plan[0] if plan else user_query
            new_plan = plan[1:] if plan else []
            context_data["crag_attempts"] = 0
        
        await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": "Knowledge retrieval complete. Returning to Supervisor."}))

        return {
            "messages": new_messages,
            "plan": new_plan,
            "context_data": context_data,
            # execution.py와 동일한 계약으로 tool_calls에 append한다 — observe_node는
            # tool_calls[-1]만 보고 성공/실패를 판정하는데, knowledge가 여기 아무것도
            # 안 쓰면 observe가 몇 턴 전 다른 agent의 결과를 재관측하게 된다.
            "tool_calls": [{
                "tool": "knowledge",
                "params": {"instruction": instruction_text},
                "result": final_answer,
                "status": "success",
            }],
            "next_agent": "supervisor", # Always return control to supervisor
            # error_count 반환 없음 — execution.py와 동일하게 observe_node가 단일
            # 권위자다. 여기서 같이 건드리면 observe가 tool_calls로 다시 세면서
            # 중복 카운트된다.
        }
        
    except Exception as e:
        logger.error(f"Error in Knowledge Agent Node: {e}")

        await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": f"Knowledge agent failed: {str(e)[:50]}..."}))

        return {
            "messages": [AIMessage(content=f"Knowledge Agent encountered an error: {e}")],
            "context_data": {
                # 재검색 중 실패해도 refined_query 를 반드시 초기화한다 — 남겨두면
                # 다음 knowledge 위임이 낡은 refined_query 를 재검색으로 오인해
                # 엉뚱한 instruction 을 쓰고 plan 스텝 pop 을 건너뛴다.
                "refined_query": "",
                # 실패를 last_knowledge_result 에 명시적으로 기록한다 — 비워두면
                # (a) 이번이 첫 knowledge 호출인 경우 supervisor 의 재호출 차단
                # 가드(if next_agent=="knowledge" and last_knowledge_result)가
                # 발동하지 않아 knowledge 가 계속 재위임되고, (b) 이전에 knowledge
                # 가 성공한 적이 있으면 그 낡은 결과가 그대로 남아 이번 실패를
                # 가리고 supervisor 가 무관한 답으로 턴을 종료한다.
                "last_knowledge_result": f"[knowledge 실패] {e}",
                # CRAG(grade_retrieval/route_after_grade/refine_knowledge)가 이
                # 플래그를 보고 재작성·정제 LLM 호출을 생략하고 곧장 observe로
                # 보낸다 — "검색 결과가 부실하다"는 CRAG의 전제 자체가 예외
                # 상황(agent 실행 실패)에는 맞지 않는다(재검색해도 같은 예외가
                # 다시 날 뿐이다).
                "knowledge_failed": True,
            },
            "tool_calls": [{
                "tool": "knowledge",
                "params": {"instruction": instruction_text},
                "result": str(e),
                "status": "error",
                "error_type": "parameter",
                "error_msg": str(e),
            }],
            "next_agent": "supervisor",
            # error_count 반환 없음(성공 경로와 동일 이유) — observe_node가
            # tool_calls를 보고 유일하게 카운트한다. 여기서 같이 올리면 이번 한
            # 번의 실패가 observe 카운트 + 이 카운트로 중복 집계돼, MAX_RETRY
            # 한도(parameter=2)를 첫 실패만으로 소진해버린다.
            "feedback": f"Knowledge Agent failed to execute the plan step due to: {e}"
        }
