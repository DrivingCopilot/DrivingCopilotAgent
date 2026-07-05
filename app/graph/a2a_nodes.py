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
from typing import Any, Dict

from app.a2a.client import A2AClient
from app.a2a.models import A2ATaskRequest
from app.a2a.registry import get_card
from app.a2a.serde import deserialize_messages, serialize_messages
from app.core.config import A2A_TASK_TIMEOUT
from app.graph import ws as _ws
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


async def _run_remote_agent(agent_name: str, state: AgentState) -> Dict[str, Any]:
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
    response = await A2AClient(base_url=card.url, timeout=A2A_TASK_TIMEOUT).send_task(request)
    await _ws.websocket_manager.send_status(
        json.dumps({"type": "tool_result", "data": {"tool_name": agent_name, "status": response.status}})
    )

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
    return result


async def knowledge_a2a_node(state: AgentState) -> Dict[str, Any]:
    return await _run_remote_agent("knowledge", state)


async def execution_a2a_node(state: AgentState) -> Dict[str, Any]:
    return await _run_remote_agent("execution", state)


async def perception_a2a_node(state: AgentState) -> Dict[str, Any]:
    return await _run_remote_agent("perception", state)
