# app/server/endpoints.py
#
# Supervisor 에이전트의 HTTP 인터페이스.
# DrivingCopilotBackend (port 8000) 에서 단발성으로 supervisor 를 호출할 때 사용한다.

import logging
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.graph.builder import run_graph

logger = logging.getLogger(__name__)

router = APIRouter()


class InvokeRequest(BaseModel):
    query: str = Field(..., description="사용자 발화 또는 백엔드가 전달하는 작업 지시")
    route_type: str = Field("", description="rag/tool/sql/vision/chat 등 백엔드 라우터가 사전 분류한 결과")


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
    단발성 그래프 실행. 디버그 / 스모크 테스트용.
    스트리밍이 필요하면 /ws 엔드포인트를 사용.
    """
    result = await run_graph(req.query, req.route_type)

    messages = result.get("messages", [])
    return InvokeResponse(
        plan=result.get("plan", []),
        next_agent=result.get("next_agent", "__end__"),
        messages=[getattr(m, "content", str(m)) for m in messages],
        feedback=result.get("feedback", "") or "",
    )
