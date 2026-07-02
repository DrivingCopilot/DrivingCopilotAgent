from __future__ import annotations

import copy
import logging
from typing import Any, Dict

from app.graph.state import AgentState
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

    if last["status"] == "error":
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

    # success / empty → supervisor로 패스 (feedback 초기화)
    if classification in ("empty", "success"):
        return {"next_agent": "supervisor", "feedback": ""}

    if classification == "unknown":
        return {
            "next_agent": "supervisor",
            "feedback": "Observe: unknown tool_call status — forwarding to supervisor.",
        }

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

    # 한도 초과 → __end__ (reflect failure 분기에서 WebSocket 송신 및 experience 저장 처리)
    if count >= limit:
        logger.warning(
            "observe: retry limit reached — tool='%s' error_type='%s' %d/%d",
            tool_name, error_type, count, limit,
        )
        return {
            "error_count": error_count,
            "next_agent": "__end__",
            "plan": [],
            "feedback": (
                f"Tool '{tool_name}' failed {error_type} {count}/{limit} times. Stopping."
            ),
        }

    # 한도 미달 → supervisor로 재시도 (feedback에 에러 내용 전달)
    return {
        "error_count": error_count,
        "next_agent": "supervisor",
        "feedback": (
            f"Tool '{tool_name}' failed with error_type='{error_type}' ({count}/{limit}). "
            f"Error: {error_msg}. Please adjust your plan and try again."
        ),
    }
