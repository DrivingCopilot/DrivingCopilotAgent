"""
app/agents/execution.py

Execution Agent — supervisor 의 plan 을 받아 MCP 12종 Tool 을 호출한다.

흐름:
    supervisor (next_agent=execution) → execution_node(state)
        ├ supervisor 의 plan 에서 호출할 MCP tool/parameter 추출  (LLM 보조)
        ├ MCP 서버(stdio) 단일 호출 — 12종 tool
        ├ 결과를 state.tool_calls 에 누적, vehicle_state 갱신
        └ observe_node 로 복귀

실패 처리 — observe_node 에게 위임 (ReAct Observe 단계 책임 분리):
    실패 시 error_type / error_msg 를 tool_calls 에 담아 반환.
    재시도 횟수 누적 · 종료 판단은 observe_node 의 단독 책임.

WS 토큰 (계획서 표준):
    {"type": "tool_start",  "data": {"tool_name": "...", "params": {...}}}
    {"type": "tool_result", "data": {"tool_name": "...", "result": "...", "status": "success"|"fail"}}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.graph import ws as _ws
from app.graph.state import AgentState
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

# 모듈 레벨 싱글턴 — plan step마다 새 인스턴스를 만들지 않는다.
# None 으로 시작하는 lazy init: import 시점에 API key 검증을 하지 않는다.
# TODO: 로컬 qwen2-vl 서버 기동 후 base_url 추가
_EXTRACTION_LLM: Optional[ChatOpenAI] = None


def _get_extraction_llm() -> ChatOpenAI:
    """_EXTRACTION_LLM 싱글턴을 반환한다. 최초 호출 시 생성된다."""
    global _EXTRACTION_LLM
    if _EXTRACTION_LLM is None:
        _EXTRACTION_LLM = ChatOpenAI(model="qwen2-vl-7b-instruct-int4", temperature=0.0)
    return _EXTRACTION_LLM


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


async def _call_mcp_tool_once(
    tool_name: str,
    params: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    """
    단일 MCP tool 호출. retry 없음 — 재시도 결정은 observe_node 책임.

    예외를 잡아 error_type 으로 분류한 뒤 반환한다.
    observe_node 가 error_type 을 읽어 재시도 횟수를 관리한다.

    Returns:
        (result_text, status, error_type, error_msg)
        - status    : "success" | "fail"
        - error_type: "" | "timeout" | "parameter"  (fail 시에만 의미 있음)
        - error_msg : 원본 에러 메시지               (fail 시에만 의미 있음)
    """
    try:
        result_text, raw_status = await asyncio.wait_for(
            _call_mcp_tool_raw(tool_name, params),
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
        is_param_error = any(
            kw in err_str.lower()
            for kw in ("validation", "parameter", "invalid", "field", "required")
        )
        error_type = "parameter" if is_param_error else "parameter"
        logger.error("MCP unexpected error: %s — %s", tool_name, exc)
        return f"[error] {err_str}", "fail", error_type, err_str


# ---------------------------------------------------------------------------
# Plan → Tool Call 추출 (LLM 보조)
# ---------------------------------------------------------------------------

async def _extract_tool_call(plan_step: str) -> Optional[Dict[str, Any]]:
    """
    supervisor 가 만든 plan 의 단일 스텝 문자열에서
    MCP tool 이름과 파라미터를 추출한다.

    LLM 에게 구조화된 JSON 출력을 요청한다.
    파싱에 실패하면 None 반환.
    """
    messages = [
        SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Task: {plan_step}"),
    ]

    try:
        response = await _get_extraction_llm().ainvoke(messages)
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
    2. MCP 서버 단일 호출 (12종 tool)
    3. tool_calls 누적 + vehicle_state 갱신
    4. tool_start / tool_result WS 토큰 송출

    실패 시 error_type / error_msg 를 tool_calls 에 담아 반환.
    error_count 관리 및 재시도 결정은 observe_node 의 책임이므로
    이 함수는 error_count 를 읽거나 쓰지 않는다.

    Args:
        state: 현재 AgentState

    Returns:
        state 에 병합할 딕셔너리 (tool_calls, context_data)
    """
    plan: List[str] = state.get("plan", [])
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
            await _ws.websocket_manager.send_status(
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
        await _ws.websocket_manager.send_status(
            json.dumps({
                "type": "tool_start",
                "data": {"tool_name": tool_name, "params": tool_params},
            })
        )
        logger.info("tool_start: %s params=%s", tool_name, tool_params)

        # ── 4. MCP tool 단일 호출 ────────────────────────────────────────────
        #   retry 없음 — 재시도는 observe → supervisor 루프가 담당
        result_text, status, error_type, error_msg = await _call_mcp_tool_once(
            tool_name, tool_params
        )

        # ── 5. tool_result WS 토큰 ──────────────────────────────────────────
        await _ws.websocket_manager.send_status(
            json.dumps({
                "type": "tool_result",
                "data": {
                    "tool_name": tool_name,
                    "result": result_text,
                    "status": status,
                },
            })
        )
        logger.info(
            "tool_result: %s status=%s error_type=%s",
            tool_name, status, error_type or "-",
        )

        # ── 6. tool_calls 누적 ──────────────────────────────────────────────
        tool_call: Dict[str, Any] = {
            "tool": tool_name,
            "params": tool_params,
            "result": result_text,
            "status": status,
        }
        if status == "fail":
            tool_call["error_type"] = error_type
            tool_call["error_msg"] = error_msg

        new_tool_calls.append(tool_call)

        # ── 7. vehicle_state 갱신 (성공 시에만) ─────────────────────────────
        if status == "success":
            if tool_name == "get_vehicle_status":
                new_vehicle_state["last_status_report"] = result_text
            else:
                new_vehicle_state[f"last_{tool_name}"] = result_text

    logger.info("execution_node 완료: 신규 tool_calls %d 개", len(new_tool_calls))

    # error_count 반환 없음 — observe_node 가 단일 권위자
    return {
        "tool_calls": [*tool_calls_acc, *new_tool_calls],
        "context_data": {
            **context_data,
            "vehicle_state": new_vehicle_state,
        },
    }
