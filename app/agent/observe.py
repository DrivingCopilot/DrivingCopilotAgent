from __future__ import annotations

import copy
import logging
from typing import Any, Dict
import json

from langchain_core.messages import AIMessage

from app.agent.state import AgentState
from app.agent.nodes import websocket_manager
from app.core.config import MAX_RETRY

logger = logging.getLogger(__name__)


def _classify_last_tool(tool_calls: list[dict]) -> tuple[str, str | None]:
    if not tool_calls:
        return ("empty", None)

    last = tool_calls[-1]

    if "status" not in last:
        logger.warning("observe: last tool_call missing 'status' field: %s", last)
        return ("unknown", None)

    if last["status"] == "success":
        return ("success", None)

    if last["status"] == "fail":
        error_type = last.get("error_type", "")
        if error_type not in MAX_RETRY:
            logger.warning(
                "observe: unknown error_type '%s' — falling back to 'parameter'", error_type
            )
            error_type = "parameter"
        return ("fail", error_type)

    logger.warning("observe: unrecognised status value '%s'", last.get("status"))
    return ("unknown", None)


async def observe_node(state: AgentState) -> Dict[str, Any]:
    tool_calls: list[dict] = state.get("tool_calls", [])
    error_count: Dict[str, int] = copy.copy(state.get("error_count", {}))

    classification, error_type = _classify_last_tool(tool_calls)
    logger.debug("observe: classification=%s error_type=%s", classification, error_type)

    if classification in ("empty", "success"):
        return {"next_agent": "supervisor", "feedback": ""} 

    if classification == "unknown":
        return {
            "next_agent": "supervisor",
            "feedback": "Observe: unknown tool_call status — forwarding to supervisor.",
        }

    # 마지막으로 실행한 툴 call의 결과
    last = tool_calls[-1]
    tool_name: str = last.get("tool", "unknown")
    error_msg: str = last.get("error_msg", "")
    limit: int = MAX_RETRY[error_type]

    # 오류 카운트 증가
    error_count[error_type] = error_count.get(error_type, 0) + 1
    count: int = error_count[error_type] 
    logger.info(
        "observe: tool='%s' error_type='%s' count=%d/%d", tool_name, error_type, count, limit
    )

    # 해당 오류의 카운트(증가한 후)가 기준을 넘어섰는지
    if count >= limit:
        logger.warning(
            "observe: retry limit reached — tool='%s' error_type='%s' %d/%d",
            tool_name, error_type, count, limit,
        )
        user_msg = (
            f"'{tool_name}' 도구가 '{error_type}' 오류로 {count}회 실패하여 "
            f"최대 재시도 횟수({limit}회)에 도달했습니다. 요청을 처리할 수 없습니다."
        )

        await websocket_manager.send_status(
            json.dumps({"type": "text", "data": user_msg})
        )
        await websocket_manager.send_status(
            json.dumps({"type": "done", "reason": "error_limit"})
        )

        return {
            "error_count": error_count,
            "next_agent": "__end__",
            "plan": [],
            "feedback": f"Tool '{tool_name}' failed {error_type} {count}/{limit} times. Stopping.",
            "messages": [AIMessage(content=user_msg)],
        }

    return {
        "error_count": error_count,
        "next_agent": "supervisor",
        "feedback": (
            f"Tool '{tool_name}' failed with error_type='{error_type}' ({count}/{limit}). "
            f"Error: {error_msg}. Please adjust your plan and try again."
        ), 
    }
