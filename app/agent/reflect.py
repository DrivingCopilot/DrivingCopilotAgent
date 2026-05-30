"""
app/agent/reflect.py

ReAct 5단계 중 Reflect 노드 (계획서 2.2).
observe → reflect → [supervisor | END] 흐름으로 매 사이클 진입.

분기:
  1. failure 케이스 (next_agent=="reflect"): LLM으로 lesson 생성 → ExperienceMemory 저장 → END
  2. 정상/feedback 없음: LLM으로 done/continue 평가 → [END | supervisor]
  3. 정상/feedback 있음: LLM 호출 생략 → supervisor 패스 (G1 비용 절감)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.state import AgentState
from app.agent.nodes import websocket_manager

logger = logging.getLogger(__name__)

# ExperienceMemory lazy-init 싱글톤
_experience_memory = None


def _get_experience_memory():
    global _experience_memory
    if _experience_memory is None:
        from app.memory.experience import ExperienceMemory
        _experience_memory = ExperienceMemory()
    return _experience_memory


REFLECT_FAILURE_PROMPT = """You are reflecting on a failed tool execution in a Driving Copilot system.
Analyze the failure and generate a concise lesson for future attempts.

Output strictly valid JSON with two keys:
- "reasoning": brief chain-of-thought about why the failure occurred
- "lesson": 1-2 sentences on how to avoid this failure next time
"""

REFLECT_EVAL_PROMPT = """You are evaluating whether the Driving Copilot has successfully fulfilled the user's intent.
Review the conversation history and context, then decide if the task is complete.

Output strictly valid JSON with two keys:
- "verdict": either "done" (task complete, no further action needed) or "continue" (more steps required)
- "reasoning": brief chain-of-thought explaining your verdict
"""


async def reflect_node(state: AgentState) -> Dict[str, Any]:
    """
    매 사이클 진입하는 Reflect 노드.
    observe의 next_agent 값과 feedback 유무로 3가지 분기 처리.
    """
    next_agent: str = state.get("next_agent", "supervisor")
    feedback: str = state.get("feedback", "")
    messages = state.get("messages", [])
    context_data = state.get("context_data", {})
    route_type: str = state.get("route_type", "")
    error_count: Dict[str, int] = state.get("error_count", {})

    llm = ChatOpenAI(model="qwen2-vl-7b-instruct-int4", temperature=0.1)

    # ------------------------------------------------------------------
    # 분기 1: failure 케이스 (한도 초과, observe가 next_agent="reflect"로 라우팅)
    # ------------------------------------------------------------------
    if next_agent == "reflect":
        logger.info("reflect: failure 케이스 진입 — lesson 생성 및 experience 저장")

        # feedback에서 error_type 추출 (observe가 기록한 형식 파싱)
        error_type = "parameter"
        for et in ("timeout", "parameter", "invalid_tool", "sql"):
            if et in feedback:
                error_type = et
                break

        # situation 구성: user query + route_type + vehicle_state 간략 요약
        user_query = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        vehicle_state_summary = str(context_data.get("vehicle_state", {}))[:200]
        situation = (
            f"query: {user_query} | route_type: {route_type} | "
            f"vehicle_state: {vehicle_state_summary}"
        )

        # LLM 호출: lesson 생성
        lesson = ""
        try:
            reflect_messages = [
                SystemMessage(content=REFLECT_FAILURE_PROMPT),
                HumanMessage(
                    content=(
                        f"Failure feedback: {feedback}\n"
                        f"Error type: {error_type}\n"
                        f"Situation: {situation}"
                    )
                ),
            ]
            content = ""
            async for chunk in llm.astream(reflect_messages):
                if chunk.content:
                    content += chunk.content

            content = content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            parsed = json.loads(content)
            lesson = parsed.get("lesson", feedback)
            logger.info("reflect: lesson 생성 완료 — %s", lesson)

        except (json.JSONDecodeError, Exception) as e:
            logger.warning("reflect: lesson LLM 파싱 실패 (%s) — feedback 사용", e)
            lesson = feedback

        # ExperienceMemory 저장 (동기 함수 → asyncio.to_thread)
        try:
            await asyncio.to_thread(
                _get_experience_memory().save,
                situation,
                error_type,
                lesson,
                route_type or "unknown",
            )
        except Exception as e:
            logger.error("reflect: ExperienceMemory 저장 실패 — %s", e)

        # WebSocket 사용자 메시지 + done 신호 송신
        tool_calls = state.get("tool_calls", [])
        tool_name = tool_calls[-1].get("tool", "unknown") if tool_calls else "unknown"
        error_count_val = error_count.get(error_type, 0)
        from app.core.config import MAX_RETRY
        limit_val = MAX_RETRY.get(error_type, 1)
        user_msg = (
            f"'{tool_name}' 도구가 '{error_type}' 오류로 {error_count_val}회 실패하여 "
            f"최대 재시도 횟수({limit_val}회)에 도달했습니다. 요청을 처리할 수 없습니다."
        )

        await websocket_manager.send_status(
            json.dumps({"type": "text", "data": user_msg})
        )
        await websocket_manager.send_status(
            json.dumps({"type": "done", "reason": "error_limit"})
        )

        return {
            "next_agent": "__end__",
            "messages": [AIMessage(content=user_msg)],
        }

    # ------------------------------------------------------------------
    # 분기 3: 정상 케이스 + feedback 있음 (한도 미달 재시도)
    # LLM 호출 없이 supervisor로 바로 패스 (G1 비용 절감)
    # ------------------------------------------------------------------
    if feedback:
        logger.info("reflect: 정상 케이스 (feedback 있음) — LLM 생략, supervisor 패스")
        return {"next_agent": "supervisor"}

    # ------------------------------------------------------------------
    # 분기 2: 정상 케이스 + feedback 없음 → done/continue 평가
    # ------------------------------------------------------------------
    logger.info("reflect: 정상 케이스 (feedback 없음) — done/continue LLM 평가")

    try:
        eval_messages = [SystemMessage(content=REFLECT_EVAL_PROMPT)]
        eval_messages.extend(messages)
        eval_messages.append(
            HumanMessage(
                content=(
                    f"Context: {context_data}\n"
                    f"Route type: {route_type}\n"
                    "Has the user's intent been fully satisfied?"
                )
            )
        )

        content = ""
        async for chunk in llm.astream(eval_messages):
            if chunk.content:
                content += chunk.content

        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        parsed = json.loads(content)
        verdict: str = parsed.get("verdict", "done")
        reasoning: str = parsed.get("reasoning", "")
        logger.info("reflect: verdict=%s reasoning=%s", verdict, reasoning)

        if verdict == "continue":
            return {"next_agent": "supervisor"}
        return {"next_agent": "__end__"}

    except (json.JSONDecodeError, Exception) as e:
        logger.warning("reflect: done/continue LLM 파싱 실패 (%s) — END로 처리", e)
        return {"next_agent": "__end__"}
