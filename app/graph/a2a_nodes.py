# app/graph/a2a_nodes.py
#
# Supervisor의 StateGraph가 등록하는 knowledge/execution/perception 노드.
# 로컬 함수 호출 대신 A2AClient.send_task()로 각 Agent 서버(8002/8003/8004)에
# HTTP 위임하고, 응답(A2ATaskResponse)을 기존 노드가 반환하던 partial-state
# dict 형태로 되돌린다.
#
# WS 스트리밍은 sub-agent에서 되쏘지 않고 여기(A2A 호출 경계)에서 합성한다 —
# tool_start는 send_task 호출 직전, tool_result는 응답 수신 직후.

import json
import logging
import uuid
from typing import Any

from app.a2a.client import A2AClient
from app.a2a.models import A2ATaskRequest
from app.a2a.registry import get_card
from app.a2a.serde import deserialize_messages, serialize_messages
from app.core.config import A2A_TASK_TIMEOUT
from app.graph import ws as _ws
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


async def _run_remote_agent(agent_name: str, state: AgentState) -> dict[str, Any]:
    card = get_card(agent_name)
    context = {
        "messages": serialize_messages(state.get("messages", [])),
        "route_type": state.get("route_type", ""),
        "plan": state.get("plan", []),
        "context_data": state.get("context_data", {}),
        "error_count": state.get("error_count", {}),
    }
    request = A2ATaskRequest(
        task_id=str(uuid.uuid4()),
        agent_name=agent_name,
        instruction=f"delegate to {agent_name}",
        context=context,
    )

    await _ws.websocket_manager.send_status(
        json.dumps({"type": "tool_start", "data": {"tool_name": agent_name, "params": {}}})
    )

    # tool_start를 보낸 이상 tool_result는 반드시 한 번 나가야 한다 — 프론트
    # ToolCallCard가 tool_result 미수신 시 "실행 중…" 상태로 영구 고착되기
    # 때문에, send_task 자체는 예외를 던지지 않도록 설계돼 있어도(A2AClient
    # 참고) A2AClient 생성 등 그 전후 어디서든 예기치 못한 예외가 나면
    # except에서 잡아 최소한의 에러 tool_result를 대신 내보낸다.
    response = None
    try:
        response = await A2AClient(base_url=card.url, timeout=A2A_TASK_TIMEOUT).send_task(request)
    except Exception as exc:
        logger.error("A2A %s 호출 중 미포착 예외: %s", agent_name, exc)

    if response is not None:
        # response.result는 다음 노드로 넘길 partial-state dict(tool_calls/
        # context_data/messages 등)라 사람이 읽을 요약이 아니다 — WS로는
        # 상태만 알리는 고정 문구로 충분하다(과설계 방지).
        tool_result_data = {
            "tool_name": agent_name,
            "status": response.status,
            "result": (response.error or "실패") if response.status == "error" else "완료",
        }
    else:
        tool_result_data = {"tool_name": agent_name, "status": "error", "result": "A2A 호출 중 예외 발생"}
    await _ws.websocket_manager.send_status(
        json.dumps({"type": "tool_result", "data": tool_result_data})
    )

    if response is None:
        return {
            "plan": [],
            "tool_calls": [{
                "tool": agent_name,
                "params": {},
                "result": "A2A 호출 중 예외 발생",
                "status": "error",
                "error_type": "parameter",
                "error_msg": "A2A 호출 중 예외 발생",
            }]
        }

    if response.status == "error":
        # A2A 통신 자체가 실패한 경우(타임아웃/연결 실패/미포착 예외) — sub-agent
        # 프로세스가 죽어도 observe_node가 감지할 수 있도록 tool_calls를 합성한다.
        logger.warning("A2A %s 태스크 실패: %s (%s)", agent_name, response.error, response.error_type)
        return {
            # plan을 비우지 않으면 supervisor가 넣어둔 낡은 plan이 그대로 남아,
            # perception 실패 시 route_after_perception이 이를 "hazard 감지"로
            # 오인해 execution으로 잘못 튄다(멀티모달 트리거 오탐).
            "plan": [],
            "tool_calls": [{
                "tool": agent_name,
                "params": {},
                "result": response.error or "",
                "status": "error",
                "error_type": response.error_type or "parameter",
                "error_msg": response.error or "A2A 통신 실패",
            }]
        }

    result = dict(response.result or {})
    if "messages" in result:
        result["messages"] = deserialize_messages(result["messages"])
    # 성공 시에도 실패 시(위 90~103줄)와 동일하게 plan을 비운다 — 안 비우면
    # supervisor가 넣어둔 낡은 plan이 남아 다음 supervisor_node 호출의
    # [Current Plan] 컨텍스트에 그대로 다시 들어가고, 모델이 "아직 할 일이
    # 남았다"고 오인해 같은 plan/next_agent를 무한 반복하는 루프가 생긴다.
    result.setdefault("plan", [])
    return result


async def knowledge_a2a_node(state: AgentState) -> dict[str, Any]:
    return await _run_remote_agent("knowledge", state)


async def execution_a2a_node(state: AgentState) -> dict[str, Any]:
    return await _run_remote_agent("execution", state)


async def perception_a2a_node(state: AgentState) -> dict[str, Any]:
    return await _run_remote_agent("perception", state)
