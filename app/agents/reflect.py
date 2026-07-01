"""
app/agents/reflect.py

ReAct 5단계 중 Reflect 노드 (계획서 2.2).
observe → reflect → [supervisor | END] 흐름으로 매 사이클 진입.

분기:
  1. failure 케이스 (next_agent=="__end__", 한도 초과): LLM으로 lesson 생성 → ExperienceMemory 저장 → END
  2. 그 외 (정상 + 한도 미달 fail): LLM 호출 없이 supervisor 패스
     - 의도 충족 평가는 supervisor의 Reason 단계 책임
     - 인턴 계획서 6절 "Reflexion: 실패 시 LLM" 정합
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.graph import ws as _ws
from app.graph.state import AgentState
from app.memory.experience import build_situation, get_experience_memory

logger = logging.getLogger(__name__)

REFLECT_FAILURE_PROMPT = """You are reflecting on a failed tool execution in a Driving Copilot system.
Analyze the failure and generate a concise lesson for future attempts.

Output strictly valid JSON with two keys:
- "reasoning": brief chain-of-thought about why the failure occurred
- "lesson": 1-2 sentences on how to avoid this failure next time
"""


async def reflect_node(state: AgentState) -> Dict[str, Any]:
    """
    매 사이클 진입하는 Reflect 노드.
    observe의 next_agent 값으로 2가지 분기 처리.
    """
    next_agent: str = state.get("next_agent", "supervisor")
    feedback: str = state.get("feedback", "")
    messages = state.get("messages", [])
    context_data = state.get("context_data", {})
    route_type: str = state.get("route_type", "")
    error_count: Dict[str, int] = state.get("error_count", {})

    # ------------------------------------------------------------------
    # 분기 1: failure 케이스 (한도 초과, observe가 next_agent="__end__"로 라우팅)
    # ------------------------------------------------------------------
    if next_agent == "__end__":
        logger.info("reflect: failure 케이스 진입 — lesson 생성 및 experience 저장")

        llm = ChatOpenAI(model="qwen2-vl-7b-instruct-int4", temperature=0.1)

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
        vehicle_state = context_data.get("vehicle_state", {})
        situation = build_situation(user_query, route_type, vehicle_state)

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

        except json.JSONDecodeError as e:
            logger.warning("reflect: lesson JSON 파싱 실패 (%s) — feedback 사용", e)
            lesson = feedback
        except Exception as e:
            logger.warning("reflect: lesson LLM 호출 실패 (%s) — feedback 사용", e)
            lesson = feedback

        # ExperienceMemory 저장 (동기 함수 → asyncio.to_thread)
        try:
            await asyncio.to_thread(
                get_experience_memory().save,
                situation,
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

        await _ws.websocket_manager.send_status(
            json.dumps({"type": "text", "data": user_msg})
        )
        await _ws.websocket_manager.send_status(
            json.dumps({"type": "done", "reason": "error_limit"})
        )

        return {
            "next_agent": "__end__",
            "messages": [AIMessage(content=user_msg)],
        }

    # ------------------------------------------------------------------
    # 분기 2: 그 외 (정상 + 한도 미달 fail) → supervisor 패스
    # ------------------------------------------------------------------
    logger.info("reflect: 정상/retry 케이스 — supervisor 패스 (LLM 호출 생략)")
    return {"next_agent": "supervisor"}
