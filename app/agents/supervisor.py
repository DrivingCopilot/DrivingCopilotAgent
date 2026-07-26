# app/agents/supervisor.py
#
# Supervisor Agent — LangGraph 노드로 동작하며 하위 Agent(Knowledge/Execution/Perception)를 조율한다.
# Agent Card 기반 동적 레지스트리를 참조해 Plan-and-Execute + CoT + Reflexion 루프를 수행한다.
# ExperienceMemory에서 과거 실패 lesson을 검색해 프롬프트에 주입한다.

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.a2a.client import A2AClient
from app.a2a.registry import list_cards
from app.core.config import AGENT_PORT, OLLAMA_BASE_URL
from app.graph import ws as _ws
from app.graph.state import AgentState
from app.core.config import MAX_RETRY, EXPERIENCE_TOP_K, WINDOW_SIZE
from app.memory.experience import build_situation, get_experience_memory

logger = logging.getLogger(__name__)

# EntityMemory lazy-init 싱글톤
_entity_memory = None


def _get_entity_memory():
    global _entity_memory
    if _entity_memory is None:
        from app.memory.entity import EntityMemory
        _entity_memory = EntityMemory()
    return _entity_memory


# 자기 자신 서버에서 Agent Card를 발견하는 클라이언트 (self-discovery)
_a2a_client = A2AClient(base_url=f"http://localhost:{AGENT_PORT}")

# perception.py의 HAZARD_PLAN_STEPS와 동일한 controlled vocabulary에 대한
# 사용자 안내용 한국어 라벨. _compose_vision_summary 가 최종 답변 생성에 사용한다.
# "사용자 질문이 어떤 hazard에 대한 것인가"는 더 이상 여기서 텍스트 키워드로
# 추측하지 않는다 — perception.py의 VLM이 이미지와 질문을 함께 보고
# vision_results["related_hazard"]로 직접 판단해서 넘겨준다.
HAZARD_LABELS: Dict[str, str] = {
    "rain": "비",
    "tunnel": "터널",
    "warning_light": "경고등",
}

# reasoning 텍스트가 특정 sub-agent 위임을 언급하는지 감지하기 위한 힌트.
# 7B 모델이 reasoning에서는 "execution agent가 처리해야 한다"고 결론 내리고도
# next_agent 필드는 "__end__"로 내보내는 instruction-following 불일치를
# 코드 레벨에서 바로잡는 데 쓴다 (perception 재호출 차단과 반대 방향의 보정).
_AGENT_NAME_HINTS: Dict[str, List[str]] = {
    "execution": ["execution agent", "delegate to 'execution'", "delegate to execution"],
    "knowledge": ["knowledge agent", "delegate to 'knowledge'", "delegate to knowledge"],
    "perception": ["perception agent", "delegate to 'perception'", "delegate to perception"],
}


class SupervisorDecision(BaseModel):
    """
    next_agent는 app/graph/builder.py의 StateGraph 노드/엣지 토폴로지에 고정된 값이다.
    app/a2a/registry.py의 Agent Card 목록(4개, "supervisor" 포함)과는 의도적으로
    다르다 — "supervisor"를 LLM이 next_agent로 직접 선택하게 허용하면 안 되고(자기
    루프는 JSON 파싱 실패 시에만 코드가 넣는 값), builder.py에 새 노드를 추가하지
    않는 한 이 목록을 동적으로 바꿔도 route_next가 받아주지 않는다.
    """

    reasoning: str = Field(..., description="Brief chain-of-thought explanation of the decision.")
    plan: List[str] = Field(..., description="Step-by-step execution plan as strings.")
    next_agent: Literal["knowledge", "execution", "perception", "__end__"] = Field(
        ..., description="Exactly one of the available sub-agent names, or '__end__' if the task is complete."
    )


def _infer_intended_agent(reasoning: str) -> str | None:
    """reasoning에서 정확히 하나의 sub-agent만 언급됐을 때만 그 이름을 반환한다."""
    lowered = reasoning.lower()
    mentioned = [
        agent for agent, hints in _AGENT_NAME_HINTS.items()
        if any(hint in lowered for hint in hints)
    ]
    return mentioned[0] if len(mentioned) == 1 else None


# LangGraph 내부 라우팅 예약어. SupervisorDecision.plan은 List[str]일 뿐 값
# 자체엔 제약이 없어(217~227줄 grammar-constrained decoding은 스키마 이탈만
# 막는다), 모델이 next_agent에 쓰는 값을 plan 스텝으로 착각해 그대로
# hallucinate할 수 있다(예: plan=["__end__"]) — WS로 내보내기 전에 걸러낸다.
_INTERNAL_ROUTING_TOKENS = {"__end__", "end", "__start__", "start"}


def _filter_internal_plan_steps(plan: List[str]) -> List[str]:
    """plan 배열에서 LangGraph 내부 라우팅 예약어만 제거한다(사용자 노출용)."""
    return [step for step in plan if step.strip().lower() not in _INTERNAL_ROUTING_TOKENS]


def _compose_vision_summary(vision_results: Dict[str, Any]) -> str:
    """
    vision_results 로부터 사용자 질문에 직접 답하는 한 줄 요약을 만든다.
    "사용자 질문이 rain/tunnel/warning_light 중 하나에 대한 것인가"는 이미지와
    질문을 함께 본 perception.py의 VLM이 related_hazard로 이미 판단해 넘겨준다 —
    여기서 사용자 질문 텍스트를 다시 키워드로 매칭하지 않는다(그 방식은 "경고
    표시판"처럼 정해둔 키워드 밖의 표현을 놓치는 문제가 있었음).
    related_hazard가 없는 질문(hazard 어휘 밖의 질문)은 VLM이 이미지+질문을 보고
    직접 작성한 answer를 그대로 신뢰한다.
    """
    if vision_results.get("status") != "success":
        return f"카메라 분석에 실패했습니다: {vision_results.get('error_msg', '알 수 없는 오류')}"

    hazards = vision_results.get("hazards", [])
    description = vision_results.get("description", "")
    related_hazard = vision_results.get("related_hazard")
    answer = vision_results.get("answer", "")

    if related_hazard:
        label = HAZARD_LABELS[related_hazard]
        if related_hazard in hazards:
            return f"네, {label}가 감지되었습니다. (카메라 상황: {description})"
        return f"아니요, {label}는 감지되지 않았습니다. (카메라 상황: {description})"

    if answer:
        return answer

    if hazards:
        labels = ", ".join(HAZARD_LABELS.get(h, h) for h in hazards)
        return f"{labels}가 감지되었습니다. (카메라 상황: {description})"

    return f"비/터널/경고등 등 특별한 위험 요인은 감지되지 않았습니다. (카메라 상황: {description})"


def _compose_tool_result_summary(last_tool_call: Dict[str, Any]) -> str:
    """
    execution 위임 완료 후 최종 답변을 last_tool_call로부터 결정적으로 구성한다.
    _compose_vision_summary와 동일한 원칙 — reasoning(CoT)은 instruction-following
    불안정으로 신뢰할 수 없으므로, 이미 자연어인 tool 실행 결과(result)를 그대로
    노출한다. 키 구조는 execution.py/a2a_nodes.py 양쪽에서 동일하게
    {"tool", "params", "result", "status", (error 시) "error_type", "error_msg"}.
    """
    tool = last_tool_call.get("tool", "")
    if last_tool_call.get("status") == "success":
        return last_tool_call.get("result") or f"{tool} 실행을 완료했습니다."
    error_msg = last_tool_call.get("error_msg") or last_tool_call.get("result") or "알 수 없는 오류"
    return f"{tool} 실행에 실패했습니다: {error_msg}"


async def _build_dynamic_agent_cards() -> str:
    """
    A2A HTTP 발견으로 Agent Card 목록을 가져온다.
    서버 미기동 등 HTTP 실패 시 로컬 registry로 폴백한다.
    """
    cards = await _a2a_client.fetch_all_cards()
    if not cards:
        logger.warning("A2A HTTP 발견 실패 — 로컬 registry로 폴백")
        cards = list_cards()

    cards_str = ""
    for idx, card in enumerate(cards, 1):
        cards_str += f"{idx}. {card.name}:\n"
        cards_str += f"   - Skill: {card.description}\n"
        cards_str += f"   - MCP Tools: {', '.join(card.capabilities.mcp_tools)}\n"
    return cards_str


SUPERVISOR_SYSTEM_PROMPT = """You are a highly capable Supervisor Agent orchestrating a Multi-Agent System for a Driving Copilot.
Your role is to analyze the user's request, evaluate the current context, and coordinate sub-agents using a 'Plan-and-Execute' pattern.
You rely on Chain-of-Thought reasoning to make decisions.

[Available Sub-Agents (Dynamic Agent Cards)]
{agent_cards}

[Rules & Protocol]
1. Use A2A delegation by selecting the appropriate agent from the list above.
2. If the user's request requires understanding the physical environment (e.g. weather, road, obstacles, warning lights) AND 'Vision/Perception Results' below is EMPTY, delegate to 'perception'. Never delegate to 'perception' twice in a row — if it is already populated, you have your answer (see Rule 6).
3. If the user's request requires manuals or relational knowledge, delegate to 'knowledge'.
   - IMPORTANT Context Fusion: Evaluate if the current context has adequate 'Vector RAG' and 'Graph RAG' data. If entities and relationships are unclear, explicitly instruct the 'knowledge' agent to use Graph RAG.
4. If the user requests an action or structured data retrieval, delegate to 'execution'. The "plan" MUST
   name the EXACT MCP tool that literally matches the user's request, chosen from execution's MCP Tools
   list above — never substitute an unrelated tool just because it appears in the list. If the user's
   words map directly to a tool name (e.g. "wiper"/"와이퍼" -> control_wiper), use that tool. Do NOT default
   to "trigger_emergency" or "set_driving_mode" unless the user explicitly asks for an emergency action or
   a driving-mode change.
   Examples:
   - User: "와이퍼 켜줘" -> plan: ["control_wiper on=true"], next_agent: "execution"
   - User: "에어컨 22도로 켜줘" -> plan: ["control_climate temperature=22 on=true"], next_agent: "execution"
   - User: "긴급 상황이야 신고해줘" -> plan: ["trigger_emergency kind=call"], next_agent: "execution"
5. An EMPTY 'Vector RAG'/'Graph RAG' result is normal and expected for requests that are about vision/physical-environment or vehicle actions — it does NOT mean the request is unanswerable. Only treat it as missing information when the request actually needs manual/relational knowledge (Rule 3).
6. If 'Vision/Perception Results' or 'Last Tool Call Result' already contains a relevant result for the request — whether it succeeded or failed — that IS sufficient: output "__end__" and summarize it (including any failure) for the user in 'reasoning'.
7. If 'Vision/Perception Results' shows detected hazards (e.g. rain, tunnel, warning_light) AND 'Last Tool Call Result' shows a related action was already taken, explicitly mention BOTH the detected condition and the action taken in your summary — the action was triggered automatically by the Perception agent, not requested by the user.
8. Always output your response in strictly valid JSON format.
9. The JSON must contain three keys:
   - "reasoning": A brief explanation of your thought process (Chain-of-Thought).
   - "plan": A list of step-by-step strings for the execution plan.
   - "next_agent": One of the agent names from the registry, or "__end__" if the task is complete.
10. "next_agent" MUST be consistent with your own "reasoning". If your reasoning concludes that a
    specific sub-agent (knowledge/execution/perception) needs to act, "next_agent" MUST be that
    agent's name — never output "__end__" while your reasoning says a sub-agent should handle the
    request. Only output "__end__" when your reasoning concludes the request is already answered
    or cannot be delegated further.

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

    messages = state.get("messages", [])[-WINDOW_SIZE * 2:]
    plan = state.get("plan", [])
    context_data = state.get("context_data", {})
    error_count = state.get("error_count", {})
    feedback = state.get("feedback", "")
    route_type = state.get("route_type", "")
    current_next_agent = state.get("next_agent", "")

    logger.info(
        "supervisor_node 시작: route_type=%s prev_next_agent=%s plan=%s error_count=%s",
        route_type, current_next_agent, plan, error_count,
    )

    # Ollama 네이티브 grammar-constrained decoding으로 JSON 스키마를 강제한다 —
    # next_agent는 반드시 유효한 4개 값 중 하나, plan은 반드시 문자열 배열로만
    # 나오게 되어(SupervisorDecision), 깨진 JSON이나 스키마 이탈 자체가 구조적으로
    # 불가능해진다. 단, "그 값이 사용자 의도와 의미적으로 맞는가"는 스키마가
    # 보장 못 하므로 아래 _infer_intended_agent 등 기존 안전장치는 그대로 둔다.
    structured_llm = ChatOllama(
        model="qwen2.5vl:7b",
        temperature=0.0,
        num_predict=300,
        base_url=OLLAMA_BASE_URL,
    ).with_structured_output(SupervisorDecision, method="json_schema", include_raw=True)

    # 첫 진입 시에만 experience 검색 (캐시 분기: run_graph 한 번에 재사용)
    if "retrieved_experience" not in context_data:
        user_query = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        vehicle_state = context_data.get("vehicle_state", {})
        situation_query = build_situation(user_query, route_type or "", vehicle_state)
        try:
            retrieved = await asyncio.to_thread(
                get_experience_memory().search,
                situation_query,
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
    profile = _get_entity_memory().load()
    vision_results = context_data.get("vision_results", {})
    tool_calls = state.get("tool_calls", [])
    last_tool_call = tool_calls[-1] if tool_calls else {}
    last_knowledge_result = context_data.get("last_knowledge_result", "")

    context_str = f"""[Context Data]
- Route Type Hint: {route_type}
- Previous Agent Hint: {current_next_agent}
- Vector RAG Results: {vector_rag}
- Graph RAG Results: {graph_rag}
- Vehicle State: {vehicle_state}
- Retrieved Experience (past lessons): {retrieved_experience}
- User Profile (preferences): {profile}
- Vision/Perception Results: {vision_results}
- Last Tool Call Result: {last_tool_call}
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
Follow the Rules & Protocol above (especially Rules 2, 5, 6, 7) using the Context Data. Think step-by-step (Chain-of-Thought) about which rule applies, then output next_agent accordingly.
"""

    dynamic_cards = await _build_dynamic_agent_cards()
    system_msg_content = SUPERVISOR_SYSTEM_PROMPT.format(agent_cards=dynamic_cards)

    messages_to_send = [SystemMessage(content=system_msg_content)]
    messages_to_send.extend(messages)
    messages_to_send.append(HumanMessage(content=eval_prompt))

    try:
        llm_result = await structured_llm.ainvoke(messages_to_send)
    except Exception as e:
        logger.error(f"supervisor_node 완료 (LLM 호출 예외): {e}")
        await _ws.websocket_manager.send_status(json.dumps({"type": "done", "reason": "llm_error"}))
        return {"next_agent": "__end__"}

    parsing_error = llm_result.get("parsing_error")
    if parsing_error is not None:
        logger.error(f"supervisor_node 완료 (구조화 출력 파싱 실패): {parsing_error}")
        updated_error_count = dict(error_count)
        updated_error_count["parameter"] = updated_error_count.get("parameter", 0) + 1
        count = updated_error_count["parameter"]
        limit = MAX_RETRY.get("parameter", 2)

        if count >= limit:
            logger.error(f"Supervisor 구조화 출력 파싱 retry limit exceeded: {count}/{limit}")
            user_msg = (
                f"응답 생성에 {count}회 실패했습니다. 요청을 처리할 수 없습니다."
            )
            await _ws.websocket_manager.send_status(json.dumps({"type": "text", "data": user_msg}))
            await _ws.websocket_manager.send_status(json.dumps({"type": "done", "reason": "parse_error_limit"}))
            return {
                "error_count": updated_error_count,
                "feedback": f"Supervisor structured output parsing failed {count}/{limit} times. Stopping.",
                "next_agent": "__end__",
                "messages": [AIMessage(content=user_msg)]
            }

        await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": f"응답 생성 재시도 중... ({count}/{limit})"}))
        return {
            "error_count": updated_error_count,
            "feedback": f"Structured output parsing failed (attempt {count}/{limit}). Error: {parsing_error}",
            "next_agent": "supervisor"
        }

    parsed: SupervisorDecision = llm_result["parsed"]
    new_plan = parsed.plan
    next_agent = parsed.next_agent
    reasoning = parsed.reasoning

    logger.info(f"Supervisor Reasoning: {reasoning}")

    # 안전장치: vision_results 가 이미 채워져 있는데도 LLM 이 시스템 프롬프트 Rule 2
    # ("Never delegate to 'perception' twice in a row")를 어기고 perception 을 다시
    # 호출하려 하면 — 7B 모델의 알려진 instruction-following 불안정 — 코드 레벨에서
    # 강제로 종료 처리해 무한 루프를 막는다 (멀티모달 트리거와 동일한 결정적 라우팅 원칙).
    if next_agent == "perception" and vision_results:
        logger.warning(
            "supervisor_node: Rule 2 위반 감지(perception 재호출 차단, vision_results=%s) "
            "— __end__ 로 강제 전환",
            vision_results,
        )
        next_agent = "__end__"
        new_plan = []

    # 안전장치(반대 방향): reasoning은 특정 sub-agent에게 위임해야 한다고 결론
    # 내렸는데 next_agent가 "__end__"로 나오는 instruction-following 불일치를
    # 바로잡는다. 단, 그 agent가 이번 턴에 이미 결과를 낸 상태(재호출이면
    # 무한루프 위험)라면 모델의 __end__ 판단을 신뢰하고 덮어쓰지 않는다.
    _agent_has_fresh_result = {
        "execution": bool(last_tool_call),
        "knowledge": bool(vector_rag or graph_rag),
        "perception": bool(vision_results),
    }
    if next_agent == "__end__":
        inferred_agent = _infer_intended_agent(reasoning)
        if inferred_agent and not _agent_has_fresh_result.get(inferred_agent, False):
            logger.warning(
                "supervisor_node: reasoning-next_agent 불일치 감지"
                "(reasoning에 '%s' 위임 언급되었으나 next_agent=__end__) — '%s'로 강제 전환",
                inferred_agent, inferred_agent,
            )
            next_agent = inferred_agent
            if not new_plan:
                new_plan = [f"delegate_to_{inferred_agent}"]

    # 사용자에게 보일 최종 답변(final_text)은 reasoning과 분리한다. vision_results
    # 가 있는 상태로 턴이 끝나면 — Rule 2 위반으로 강제 종료됐든 모델이 스스로
    # __end__ 를 택했든 동일하게 — reasoning(모델의 CoT, 신뢰 불가)이 아니라
    # vision_results 로부터 결정적으로 답변을 구성한다.
    # 우선순위: vision_results > last_tool_call > last_knowledge_result > reasoning.
    # 앞 세 경우는 결정적(코드 레벨)으로 답변을 구성할 수 있는 자연어 데이터가
    # 이미 있으므로 reasoning(CoT)에 기대지 않는다. 셋 다 없는 순수 대화 종료
    # 케이스(예: 델리게이션 없이 바로 답하는 잡담)만 reasoning 폴백에 남는다 —
    # 이 잔여 케이스는 이번 수정 범위 밖.
    final_text = reasoning
    if next_agent == "__end__" and vision_results:
        final_text = _compose_vision_summary(vision_results)
    elif next_agent == "__end__" and last_tool_call:
        final_text = _compose_tool_result_summary(last_tool_call)
    elif next_agent == "__end__" and last_knowledge_result:
        final_text = last_knowledge_result

    # frontend AG-UI 프로토콜: reasoning(사고 과정 accordion), plan(실행 계획 카드)을
    # 각각의 프레임으로 전달한다. WS로 나가는 plan은 라우팅에 쓰이는 new_plan과
    # 별개로 필터링한다 — route_next 등이 참조하는 반환값(new_plan)은 그대로 둔다.
    if reasoning:
        await _ws.websocket_manager.send_status(json.dumps({"type": "reasoning", "data": reasoning}))
    plan_for_ws = _filter_internal_plan_steps(new_plan)
    if plan_for_ws:
        await _ws.websocket_manager.send_status(json.dumps({"type": "plan", "data": plan_for_ws}))

    await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": f"Delegating task to {next_agent}"}))

    if next_agent == "__end__":
        # 턴이 종료될 때만 최종 답변을 type:"text" 로 노출한다 — 진행 중인 위임
        # 단계에서는 reasoning이 CoT일 뿐 사용자에게 보일 답변이 아니다.
        if final_text:
            await _ws.websocket_manager.send_status(json.dumps({"type": "text", "data": final_text}))
        await _ws.websocket_manager.send_status(json.dumps({"type": "done"}))

    logger.info(
        "supervisor_node 완료: next_agent=%s plan=%s reasoning=%r final_text=%r",
        next_agent, new_plan, reasoning, final_text,
    )

    result = {
        "plan": new_plan,
        "next_agent": next_agent,
        "feedback": ""
    }
    if final_text:
        result["messages"] = [AIMessage(content=final_text)]

    return result