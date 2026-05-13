# app/server/endpoints.py
#
# Supervisor 에이전트의 HTTP 인터페이스.
# DrivingCopilotBackend (port 8000) 에서 단발성으로 supervisor 를 호출할 때 사용한다.

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent import nodes
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

router = APIRouter()


class InvokeRequest(BaseModel):
    query: str = Field(..., description="사용자 발화 또는 백엔드가 전달하는 작업 지시")
    route_type: str = Field("", description="rag/tool/sql/vision/chat 등 백엔드 라우터가 사전 분류한 결과")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="vector_results, graph_results, vehicle_state 등")


class InvokeResponse(BaseModel):
    plan: List[Any]
    next_agent: str
    messages: List[str]
    feedback: str = ""


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "supervisor-agent"}


@router.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest) -> InvokeResponse:
    """
    단발성 supervisor 호출. WebSocket 없이 결과만 받고 싶을 때 사용.
    스트리밍이 필요하면 /ws 엔드포인트를 사용.
    """
    initial_state: AgentState = {
        "messages": [HumanMessage(content=req.query)],
        "route_type": req.route_type,
        "plan": [],
        "next_agent": "",
        "tool_calls": [],
        "context_data": req.context_data,
        "error_count": {},
        "feedback": "",
    }
    result = await nodes.supervisor_node(initial_state)

    messages = result.get("messages", [])
    return InvokeResponse(
        plan=result.get("plan", []),
        next_agent=result.get("next_agent", "__end__"),
        messages=[getattr(m, "content", str(m)) for m in messages],
        feedback=result.get("feedback", "") or "",
    )
