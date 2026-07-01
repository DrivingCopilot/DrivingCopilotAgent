# app/core/mcp_client.py
#
# MCP streamable-http 클라이언트 공용 헬퍼.
# Backend mcp_server.py(transport="streamable-http")에 HTTP 로 연결한다.
# execution.py(Vehicle Tool 12종)와 knowledge.py(text2sql/vector/graph)가 함께 사용한다.

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Tuple

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.core.config import MCP_SERVER_URL, MCP_TOOL_TIMEOUT

logger = logging.getLogger(__name__)


async def call_mcp_tool_raw(tool_name: str, params: Dict[str, Any]) -> Tuple[str, str]:
    """
    streamable-http transport 로 MCP 서버에 연결해 tool 을 호출한다.

    Returns:
        (result_text, status) — status: "success" | "error"
    """
    # streamablehttp_client 는 (read, write, get_session_id) 3-튜플을 yield 한다.
    async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, params)

            if result.isError:
                err_text = str(result.content)
                logger.error("MCP tool error: %s → %s", tool_name, err_text)
                return err_text, "error"

            content = result.content
            if content and hasattr(content[0], "text"):
                return content[0].text, "success"
            return str(content), "success"


async def call_mcp_tool_once(
    tool_name: str,
    params: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    """
    단일 MCP tool 호출. retry 없음 — 재시도 결정은 호출자(observe 단계) 책임.

    Returns:
        (result_text, status, error_type, error_msg)
        - status    : "success" | "error"
        - error_type: "" | "timeout" | "parameter"  
        - error_msg : 원본 에러 메시지              
    """
    try:
        result_text, raw_status = await asyncio.wait_for(
            call_mcp_tool_raw(tool_name, params),
            timeout=MCP_TOOL_TIMEOUT,
        )
        # MCP 서버가 isError 응답을 보낸 경우 → parameter 오류로 분류
        if raw_status == "error":
            logger.warning("MCP tool returned error: %s — %s", tool_name, result_text)
            return result_text, "fail", "parameter", result_text

        return result_text, "success", "", ""

    except asyncio.TimeoutError:
        error_msg = f"'{tool_name}' 호출 타임아웃 ({MCP_TOOL_TIMEOUT}초 초과)"
        logger.warning("MCP timeout: %s", tool_name)
        return error_msg, "fail", "timeout", error_msg

    except Exception as exc:
        err_str = str(exc)
        logger.error("MCP unexpected error: %s — %s", tool_name, exc)
        return f"[error] {err_str}", "fail", "parameter", err_str