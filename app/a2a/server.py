# app/a2a/server.py
#
# Agent별 독립 A2A 서버.
#
# registry에 정의된 에이전트별 URL(knowledge 8002 / execution 8003 / perception 8004)로
# 각 Agent를 별도 FastAPI 프로세스로 띄운다. 각 서버는:
#   GET  /.well-known/agent.json  → 자신의 Agent Card
#   GET  /a2a/agents[/{name}]     → 전체/특정 Agent Card (발견용)
#   POST /tasks/send              → 자신에게 온 태스크만 처리 (다른 agent면 error)
#
# Supervisor는 A2AClient(base_url=card.url).send_task(...)로 각 URL에 원격 위임한다.
#
# 실행:
#     python -m app.a2a.server knowledge      # → http://localhost:8002
#     python -m app.a2a.server execution      # → http://localhost:8003
#     python -m app.a2a.server perception     # → http://localhost:8004

import logging

from dotenv import load_dotenv

load_dotenv()  # app.core.config가 import 시점에 os.getenv를 읽으므로 가장 먼저 실행

from fastapi import FastAPI, HTTPException

from app.a2a.dispatch import DELEGATABLE_AGENTS, dispatch_task
from app.a2a.models import A2ATaskRequest, A2ATaskResponse, AgentCard
from app.a2a.registry import get_card, list_cards

logger = logging.getLogger(__name__)


def create_agent_app(agent_name: str) -> FastAPI:
    """단일 Agent를 위한 독립 A2A 서버 앱을 생성한다.

    POST /tasks/send는 자신의 agent_name으로 온 태스크만 처리하고,
    다른 Agent로의 요청은 status="error" 봉투로 거절한다.
    """
    card = get_card(agent_name)
    if card is None:
        raise ValueError(f"알 수 없는 Agent: {agent_name!r}")
    if agent_name not in DELEGATABLE_AGENTS:
        raise ValueError(
            f"'{agent_name}'은(는) 독립 서버로 띄울 수 없습니다. "
            f"가능: {', '.join(DELEGATABLE_AGENTS)}"
        )

    app = FastAPI(
        title=f"A2A Agent Server — {agent_name}",
        description=card.description,
        version=card.version,
    )

    @app.get("/.well-known/agent.json", response_model=AgentCard)
    async def well_known_agent():
        """이 서버가 대표하는 Agent의 Card를 반환한다 (A2A 표준 발견 경로)."""
        return card

    @app.get("/a2a/agents", response_model=list[AgentCard])
    async def list_agents():
        return list_cards()

    @app.get("/a2a/agents/{name}", response_model=AgentCard)
    async def get_agent(name: str):
        found = get_card(name)
        if found is None:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
        return found

    @app.post("/tasks/send", response_model=A2ATaskResponse)
    async def tasks_send(request: A2ATaskRequest) -> A2ATaskResponse:
        if request.agent_name != agent_name:
            return A2ATaskResponse(
                task_id=request.task_id,
                status="error",
                error=f"이 서버는 '{agent_name}' 전용입니다.",
                error_type="invalid_tool",
            )
        return await dispatch_task(request)

    return app


if __name__ == "__main__":
    import argparse
    from urllib.parse import urlparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="A2A Agent 독립 서버 실행")
    parser.add_argument("agent_name", choices=DELEGATABLE_AGENTS)
    args = parser.parse_args()

    agent_card = get_card(args.agent_name)
    agent_port = urlparse(agent_card.url).port

    uvicorn.run(
        create_agent_app(args.agent_name),
        host="0.0.0.0",
        port=agent_port,
    )
