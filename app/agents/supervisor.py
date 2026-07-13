# app/agents/supervisor.py
#
# Supervisor Agent — LangGraph 노드로 동작하며 하위 Agent(Knowledge/Execution/Perception)를 조율한다.
# Agent Card 기반 동적 레지스트리를 참조해 Plan-and-Execute + CoT + Reflexion 루프를 수행한다.
# ExperienceMemory에서 과거 실패 lesson을 검색해 프롬프트에 주입한다.

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.a2a.client import A2AClient
from app.a2a.registry import list_cards
from app.core.config import AGENT_PORT
from app.core.json_utils import extract_first_json_object
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
# 사용자 안내용 한국어 라벨과, 사용자 질문에서 해당 항목을 묻고 있는지 판별할
# 키워드. _compose_vision_summary 가 최종 답변 생성에 사용한다.
HAZARD_LABELS: Dict[str, str] = {
    "rain": "비",
    "tunnel": "터널",
    "warning_light": "경고등",
}

HAZARD_KEYWORDS: Dict[str, List[str]] = {
    "rain": ["비", "rain", "우산"],
    "tunnel": ["터널", "tunnel"],
    "warning_light": ["경고등", "warning"],
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


def _infer_intended_agent(reasoning: str) -> str | None:
    """reasoning에서 정확히 하나의 sub-agent만 언급됐을 때만 그 이름을 반환한다."""
    lowered = reasoning.lower()
    mentioned = [
        agent for agent, hints in _AGENT_NAME_HINTS.items()
        if any(hint in lowered for hint in hints)
    ]
    return mentioned[0] if len(mentioned) == 1 else None


def _compose_vision_summary(vision_results: Dict[str, Any], user_query: str = "") -> str:
    """
    vision_results 로부터 사용자 질문에 직접 답하는 한 줄 요약을 만든다.
    LLM의 reasoning은 모델의 instruction-following 불안정으로 신뢰할 수 없으므로
    (CoT일 뿐 실제 답변이 아닐 수 있음), 최종 답변은 이 결정적(코드 레벨) 생성기가
    담당한다 — vision_results.description을 그대로 노출하지 않고, 사용자가
    "비와?" 처럼 특정 항목을 물었으면 그 항목에 대해서만 명확히 답한다.
    """
    if vision_results.get("status") != "success":
        return f"카메라 분석에 실패했습니다: {vision_results.get('error_msg', '알 수 없는 오류')}"

    hazards = vision_results.get("hazards", [])
    description = vision_results.get("description", "")

    asked_hazard = next(
        (h for h, keywords in HAZARD_KEYWORDS.items() if any(kw in user_query for kw in keywords)),
        None,
    )
    if asked_hazard:
        label = HAZARD_LABELS[asked_hazard]
        if asked_hazard in hazards:
            return f"네, {label}가 감지되었습니다. (카메라 상황: {description})"
        return f"아니요, {label}는 감지되지 않았습니다. (카메라 상황: {description})"

    if hazards:
        labels = ", ".join(HAZARD_LABELS.get(h, h) for h in hazards)
        return f"{labels}가 감지되었습니다. (카메라 상황: {description})"

    return f"비/터널/경고등 등 특별한 위험 요인은 감지되지 않았습니다. (카메라 상황: {description})"


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

    # JSON 모드 + 페널티는 모델이 반복하는 걸 "줄여줄" 뿐 보장하지는 않는다 —
    # 진짜 보장은 파싱 단계의 extract_first_json_object 가 한다 (아래 참고).
    llm = ChatOpenAI(
        model="qwen2-vl-7b-instruct-int4",
        temperature=0.0,
        max_tokens=300,
        frequency_penalty=1.2,
        presence_penalty=0.6,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

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
        content = ""
        # 모델 출력은 reasoning/plan/next_agent 를 담은 구조화 JSON 한 덩어리이므로,
        # 토큰 단위로 그대로 WS에 흘려보내면 frontend(type:"text")가 그걸 최종 자연어
        # 답변으로 오인해 깨진 JSON을 채팅창에 그대로 노출한다. 따라서 전체를 모아
        # 파싱한 뒤, frontend AG-UI 프로토콜에 맞춰 reasoning/plan/text 프레임을
        # 각각 따로 보낸다 (아래 참고).
        async for chunk in llm.astream(messages_to_send):
            if chunk.content:
                content += chunk.content

        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        # 모델이 같은/다른 JSON 객체를 반복해서 이어붙여도 첫 블록만 사용한다.
        content = extract_first_json_object(content)

        parsed_result = json.loads(content)

        # 모델이 프롬프트(Rule 9: "plan"은 문자열 리스트)를 어기고 각 스텝을
        # {"tool": ...} 같은 객체로 출력할 때가 있다 — 이를 그대로 WS "plan"
        # 프레임에 실어 보내면 frontend PlanCard가 문자열을 기대하고 렌더링하다
        # React 크래시("Objects are not valid as a React child")를 낸다.
        raw_plan = parsed_result.get("plan", [])
        new_plan = [
            step if isinstance(step, str) else json.dumps(step, ensure_ascii=False)
            for step in raw_plan
        ]
        next_agent = parsed_result.get("next_agent", "__end__")
        reasoning = parsed_result.get("reasoning", "")

        logger.info(f"Supervisor Reasoning: {reasoning}")

    except json.JSONDecodeError as e:
        logger.error(f"supervisor_node 완료 (JSON 파싱 실패): {e}. Raw content: {content}")
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
        logger.error(f"supervisor_node 완료 (LLM 호출 예외): {e}")
        await _ws.websocket_manager.send_status(json.dumps({"type": "done", "reason": "llm_error"}))
        return {"next_agent": "__end__"}

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
    final_text = reasoning
    if next_agent == "__end__" and vision_results:
        user_query = messages[0].content if messages else ""
        final_text = _compose_vision_summary(vision_results, user_query)

    # frontend AG-UI 프로토콜: reasoning(사고 과정 accordion), plan(실행 계획 카드)을
    # 각각의 프레임으로 전달한다.
    if reasoning:
        await _ws.websocket_manager.send_status(json.dumps({"type": "reasoning", "data": reasoning}))
    if new_plan:
        await _ws.websocket_manager.send_status(json.dumps({"type": "plan", "data": new_plan}))

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