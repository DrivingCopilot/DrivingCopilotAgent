# app/a2a/dispatch.py
#
# A2A 태스크 위임 디스패처.
# A2ATaskRequest를 받아 해당 Agent 노드(knowledge_node/run_execution/perception_node)를
# 그대로 호출하고, 결과를 A2ATaskResponse로 감싸 반환한다.
#
# knowledge_node/run_execution/perception_node의 시그니처(state: AgentState)는
# 건드리지 않는다 — 이 파일이 AgentState ↔ A2ATaskRequest/Response 변환을 전담하는
# 얇은 경계층이다.

import logging
from typing import Any

from app.a2a.models import A2ATaskRequest, A2ATaskResponse
from app.a2a.serde import deserialize_messages, serialize_messages
from app.agents.execution import run_execution
from app.agents.knowledge import knowledge_node
from app.agents.perception import perception_node
from app.graph.state import AgentState

logger = logging.getLogger(__name__)

_DISPATCH_TABLE = {
    "knowledge": knowledge_node,
    "execution": run_execution,
    "perception": perception_node,
}

DELEGATABLE_AGENTS = tuple(_DISPATCH_TABLE.keys())


def _build_state(req: A2ATaskRequest) -> AgentState:
    """A2ATaskRequest.context를 노드 실행용 최소 AgentState로 재구성한다."""
    ctx = req.context or {}
    return {
        "messages": deserialize_messages(ctx.get("messages", [])),
        "route_type": ctx.get("route_type", ""),
        "plan": ctx.get("plan", []),
        "next_agent": "",
        "tool_calls": [],
        "context_data": ctx.get("context_data", {}),
        "error_count": ctx.get("error_count", {}),
        "feedback": "",
    }


async def dispatch_task(req: A2ATaskRequest) -> A2ATaskResponse:
    """A2ATaskRequest를 해당 Agent 노드로 위임하고 결과를 A2ATaskResponse로 감싼다."""
    node_fn = _DISPATCH_TABLE.get(req.agent_name)
    if node_fn is None:
        return A2ATaskResponse(
            task_id=req.task_id,
            status="error",
            error=f"알 수 없는 agent: {req.agent_name!r}",
            error_type="invalid_tool",
        )

    state = _build_state(req)
    try:
        result: dict[str, Any] = await node_fn(state)
    except Exception as exc:
        logger.exception("dispatch_task: %s 노드 실행 실패", req.agent_name)
        return A2ATaskResponse(
            task_id=req.task_id, status="error", error=str(exc), error_type="parameter"
        )

    if "messages" in result:
        result = {**result, "messages": serialize_messages(result["messages"])}

    return A2ATaskResponse(task_id=req.task_id, status="success", result=result)
