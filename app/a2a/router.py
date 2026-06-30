# app/a2a/router.py
#
# A2A 프로토콜 HTTP 엔드포인트.
#
# GET  /.well-known/agent.json   → Supervisor Agent Card 반환 (A2A 표준 발견 경로)
# GET  /a2a/agents               → 전체 Agent Card 목록 반환
# GET  /a2a/agents/{name}        → 특정 Agent Card 반환

from fastapi import APIRouter, HTTPException

from app.a2a.models import AgentCard
from app.a2a.registry import get_card, list_cards

router = APIRouter()


@router.get("/.well-known/agent.json", response_model=AgentCard)
async def well_known_agent():
    """A2A 표준 발견 경로 — Supervisor의 Agent Card를 반환한다."""
    return get_card("supervisor")


@router.get("/a2a/agents", response_model=list[AgentCard])
async def list_agents():
    """등록된 전체 Agent Card 목록을 반환한다."""
    return list_cards()


@router.get("/a2a/agents/{agent_name}", response_model=AgentCard)
async def get_agent(agent_name: str):
    """특정 Agent의 Card를 반환한다."""
    card = get_card(agent_name)
    if card is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return card
