"""
app/agents/execution.py

Execution Agent — supervisor 의 plan 을 받아 MCP 12종 Tool 을 호출한다.

흐름:
    supervisor (next_agent=execution) → execution_node(state)
        ├ supervisor 의 plan 에서 호출할 MCP tool/parameter 추출  (LLM 보조)
        ├ MCP 서버(stdio) 호출 — 12종 tool
        ├ 결과를 state.tool_calls 에 누적, vehicle_state 갱신
        └ supervisor 로 복귀

실패 처리 (계획서 6장):
    - TimeoutError  : 2회 retry → error_count["timeout"]  누적
    - 파라미터 오류 : 2회 retry → error_count["parameter"] 누적
    - 잘못된 Tool   : 1회 재추출 재시도

WS 토큰 (계획서 표준):
    {"type": "tool_start",  "data": {"tool_name": "...", "params": {...}}}
    {"type": "tool_result", "data": {"tool_name": "...", "result": "...", "status": "success"|"error"}}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.agent.state import AgentState
from app.core.config import MCP_SERVER_PYTHON, MCP_SERVER_SCRIPT, MCP_TOOL_TIMEOUT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

# Backend mcp_server.py 에 정의된 12종 tool
MCP_TOOLS: List[str] = [
    "control_climate",
    "set_navigation",
    "control_media",
    "get_vehicle_status",
    "control_window",
    "control_lighting",
    "control_seat",
    "control_parking",
    "trigger_emergency",
    "set_driving_mode",
    "control_wiper",
    "query_dashboard",
]

# LLM 이 tool 선택·파라미터 추출에 쓸 시그니처 레퍼런스
_TOOL_SIGNATURES = """\
- control_climate(temperature: int [16~32], on: bool = True)
- set_navigation(destination: str)
- control_media(action: "play"|"pause"|"next"|"prev")
- get_vehicle_status()
- control_window(is_open: bool)
- control_lighting(on: bool)
- control_seat(direction: "forward"|"backward"|"up"|"down", recline_angle: int|null [90~120])
- control_parking(enable: bool)
- trigger_emergency(kind: "call"|"alert")
- set_driving_mode(mode: "normal"|"eco"|"sport")
- control_wiper(on: bool)
- query_dashboard(metric: "speed"|"rpm"|"fuel"|"battery"|"tire_pressure"|"warning_lights")
"""

_EXTRACTION_SYSTEM_PROMPT = f"""\
You are a tool-call extractor for a vehicle control system.
Given a task description, extract which MCP tool to call and what parameters to use.

Available tools:
{_TOOL_SIGNATURES}

Output ONLY valid JSON in this exact format:
{{"tool_name": "<tool_name>", "params": {{<key>: <value>}}}}

Rules:
- For tools with no parameters (e.g. get_vehicle_status), output: {{"tool_name": "get_vehicle_status", "params": {{}}}}
- If the task does not map to any listed tool, output: {{"tool_name": null, "params": {{}}}}
- Do NOT output anything outside the JSON object.
"""

# ---------------------------------------------------------------------------
# MCP 클라이언트 헬퍼
# ---------------------------------------------------------------------------

async def _call_mcp_tool_raw(tool_name: str, params: Dict[str, Any]) -> Tuple[str, str]:
    """
    stdio transport 로 MCP 서버에 연결해 tool 을 호출한다.

    Returns:
        (result_text, status) — status: "success" | "error"
    """
    server_params = StdioServerParameters(
        command=MCP_SERVER_PYTHON,
        args=[MCP_SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
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


# WsSend 타입 — websocket_manager.send_status 와 동일 시그니처
_WsSend = Callable[[str], Coroutine[Any, Any, None]]


async def _call_with_retry(
    tool_name: str,
    params: Dict[str, Any],
    error_count: Dict[str, int],
    ws_send: _WsSend,
) -> Tuple[str, str]:
    """
    MCP tool 호출 + 계획서 6장 실패 처리.

    - TimeoutError  : MCP_TOOL_TIMEOUT 초 초과 시 최대 2회 retry
                      → error_count["timeout"] 누적
    - 파라미터 오류 : validation/field/parameter 키워드 포함 예외 시 최대 2회 retry
                      → error_count["parameter"] 누적
    - 기타 예외     : 1회 즉시 반환 (error 상태)

    Returns:
        (result_text, status)
    """
    MAX_TIMEOUT_RETRY = 2
    MAX_PARAM_RETRY = 2

    timeout_tries = 0
    param_tries = 0

    while True:
        try:
            result_text, status = await asyncio.wait_for(
                _call_mcp_tool_raw(tool_name, params),
                timeout=MCP_TOOL_TIMEOUT,
            )
            return result_text, status

        except asyncio.TimeoutError:
            timeout_tries += 1
            error_count["timeout"] = error_count.get("timeout", 0) + 1
            logger.warning(
                "MCP timeout (%d/%d): %s", timeout_tries, MAX_TIMEOUT_RETRY, tool_name
            )
            await ws_send(
                json.dumps({
                    "type": "status",
                    "data": f"Tool '{tool_name}' 타임아웃 ({timeout_tries}/{MAX_TIMEOUT_RETRY})",
                })
            )
            if timeout_tries >= MAX_TIMEOUT_RETRY:
                return (
                    f"[timeout] '{tool_name}' 호출 실패 ({MAX_TIMEOUT_RETRY}회 초과)",
                    "error",
                )
            await asyncio.sleep(0.5)

        except Exception as exc:
            err_msg = str(exc).lower()
            is_param_error = any(
                kw in err_msg
                for kw in ("validation", "parameter", "invalid", "field", "required")
            )

            if is_param_error:
                param_tries += 1
                error_count["parameter"] = error_count.get("parameter", 0) + 1
                logger.warning(
                    "MCP param error (%d/%d): %s — %s",
                    param_tries, MAX_PARAM_RETRY, tool_name, exc,
                )
                await ws_send(
                    json.dumps({
                        "type": "status",
                        "data": (
                            f"Tool '{tool_name}' 파라미터 오류 "
                            f"({param_tries}/{MAX_PARAM_RETRY}): {exc}"
                        ),
                    })
                )
                if param_tries >= MAX_PARAM_RETRY:
                    return f"[param_error] '{tool_name}': {exc}", "error"
                await asyncio.sleep(0.3)
            else:
                logger.error("MCP unexpected error: %s — %s", tool_name, exc)
                return f"[error] '{tool_name}': {exc}", "error"


# ---------------------------------------------------------------------------
# Plan → Tool Call 추출 (LLM 보조)
# ---------------------------------------------------------------------------

async def _extract_tool_call(plan_step: str) -> Optional[Dict[str, Any]]:
    """
    supervisor 가 만든 plan 의 단일 스텝 문자열에서
    MCP tool 이름과 파라미터를 추출한다.

    LLM(qwen2-vl-7b) 에게 구조화된 JSON 출력을 요청한다.
    파싱에 실패하면 None 반환.
    """
    llm = ChatOpenAI(model="qwen2-vl-7b-instruct-int4", temperature=0.0)

    messages = [
        SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Task: {plan_step}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # ``` 코드 블록 제거
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].strip()

        parsed = json.loads(raw)

        if parsed.get("tool_name") is None:
            return None

        return parsed

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Tool extraction failed for %r: %s", plan_step, exc)
        return None


# ---------------------------------------------------------------------------
# Execution Agent 메인 엔트리
# ---------------------------------------------------------------------------

async def run_execution(state: AgentState) -> Dict[str, Any]:
    """
    Execution Agent 실 구현.

    1. supervisor 의 plan 에서 MCP tool/parameter 추출
    2. MCP 서버 호출 (12종 tool, retry 포함)
    3. tool_calls 누적 + vehicle_state 갱신
    4. tool_start / tool_result WS 토큰 송출
    5. error_count 를 state 에 반영해 supervisor 의 하드 Fallback 판단에 활용

    Args:
        state: 현재 AgentState

    Returns:
        state 에 병합할 딕셔너리 (tool_calls, context_data, error_count)
    """
    # nodes.py 의 websocket_manager(또는 _StreamerProxy) 참조
    from app.agent.nodes import websocket_manager

    plan: List[str] = state.get("plan", [])
    error_count: Dict[str, int] = dict(state.get("error_count", {}))
    tool_calls_acc: List[Dict[str, Any]] = list(state.get("tool_calls", []))
    context_data: Dict[str, Any] = dict(state.get("context_data", {}))
    vehicle_state: Dict[str, Any] = dict(context_data.get("vehicle_state", {}))

    new_tool_calls: List[Dict[str, Any]] = []
    new_vehicle_state: Dict[str, Any] = dict(vehicle_state)

    logger.info("execution_node 시작: plan 스텝 %d 개", len(plan))

    for step in plan:
        # ── 1. Plan step → tool call 추출 ──────────────────────────────────
        tool_info = await _extract_tool_call(step)

        if tool_info is None:
            logger.info("tool 추출 불가 (skip): %r", step)
            continue

        tool_name: str = tool_info["tool_name"]
        tool_params: Dict[str, Any] = tool_info.get("params", {})

        # ── 2. 잘못된 Tool — 1회 재추출 재시도 ─────────────────────────────
        if tool_name not in MCP_TOOLS:
            logger.warning("알 수 없는 tool '%s' (step=%r), 재추출 시도", tool_name, step)
            await websocket_manager.send_status(
                json.dumps({
                    "type": "status",
                    "data": f"알 수 없는 Tool '{tool_name}', 재추출 시도",
                })
            )
            tool_info = await _extract_tool_call(f"[RETRY] {step}")
            if tool_info is None or tool_info.get("tool_name") not in MCP_TOOLS:
                logger.error("재추출 실패 (skip): %r", step)
                continue
            tool_name = tool_info["tool_name"]
            tool_params = tool_info.get("params", {})

        # ── 3. tool_start WS 토큰 ───────────────────────────────────────────
        await websocket_manager.send_status(
            json.dumps({
                "type": "tool_start",
                "data": {"tool_name": tool_name, "params": tool_params},
            })
        )
        logger.info("tool_start: %s params=%s", tool_name, tool_params)

        # ── 4. MCP tool 호출 (retry 포함) ───────────────────────────────────
        result_text, status = await _call_with_retry(
            tool_name,
            tool_params,
            error_count,
            ws_send=websocket_manager.send_status,
        )

        # ── 5. tool_result WS 토큰 ──────────────────────────────────────────
        await websocket_manager.send_status(
            json.dumps({
                "type": "tool_result",
                "data": {
                    "tool_name": tool_name,
                    "result": result_text,
                    "status": status,
                },
            })
        )
        logger.info("tool_result: %s status=%s", tool_name, status)

        new_tool_calls.append({
            "tool": tool_name,
            "params": tool_params,
            "result": result_text,
            "status": status,
        })

        # ── 6. vehicle_state 갱신 ────────────────────────────────────────────
        if status == "success":
            if tool_name == "get_vehicle_status":
                new_vehicle_state["last_status_report"] = result_text
            else:
                new_vehicle_state[f"last_{tool_name}"] = result_text

    logger.info("execution_node 완료: 신규 tool_calls %d 개", len(new_tool_calls))

    # 이슈 지정 state 병합 패턴 준수
    return {
        "tool_calls": [*tool_calls_acc, *new_tool_calls],
        "context_data": {
            **context_data,
            "vehicle_state": new_vehicle_state,
        },
        "error_count": error_count,
    }
