# app/a2a/client.py
#
# A2A HTTP 클라이언트.
# Agent Card URL에서 카드를 가져오고, supervisor.py의 get_agent_registry()를
# 하드코딩 대신 HTTP 발견 기반으로 동작하게 한다.
#
# 현재는 같은 서버(8001)에서 /a2a/agents 를 호출하는 self-discovery 방식.
# 추후 Agent별 독립 서버로 분리 시 각 URL만 추가하면 된다.

import logging
from typing import Optional

import httpx

from app.a2a.models import A2ATaskRequest, A2ATaskResponse, AgentCard

logger = logging.getLogger(__name__)


class A2AClient:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def fetch_card(self, agent_name: str) -> Optional[AgentCard]:
        """특정 Agent의 Card를 HTTP로 가져온다."""
        url = f"{self._base_url}/a2a/agents/{agent_name}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return AgentCard(**resp.json())
        except Exception as exc:
            logger.warning("Agent Card 조회 실패 (%s): %s", url, exc)
            return None

    async def fetch_all_cards(self) -> list[AgentCard]:
        """등록된 전체 Agent Card 목록을 HTTP로 가져온다."""
        url = f"{self._base_url}/a2a/agents"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return [AgentCard(**item) for item in resp.json()]
        except Exception as exc:
            logger.warning("Agent Card 목록 조회 실패 (%s): %s", url, exc)
            return []

    async def send_task(self, request: A2ATaskRequest) -> A2ATaskResponse:
        """
        A2ATaskRequest를 대상 서버(POST /tasks/send)로 전송한다.

        fetch_card/fetch_all_cards와 달리 실패해도 None을 반환하지 않고
        항상 유효한 A2ATaskResponse(status="error", error_type=...)를 반환한다.
        호출부(app/graph/a2a_nodes.py)가 널 체크 없이 status/error_type만
        보고 observe_node 재시도 로직에 바로 흘려보낼 수 있어야 하기 때문이다.
        """
        url = f"{self._base_url}/tasks/send"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=request.model_dump())
                resp.raise_for_status()
                return A2ATaskResponse(**resp.json())
        except httpx.TimeoutException as exc:
            logger.warning("A2A send_task 타임아웃 (%s): %s", url, exc)
            return A2ATaskResponse(
                task_id=request.task_id, status="error", error=str(exc), error_type="timeout"
            )
        except Exception as exc:
            logger.warning("A2A send_task 실패 (%s): %s", url, exc)
            return A2ATaskResponse(
                task_id=request.task_id, status="error", error=str(exc), error_type="parameter"
            )

    async def discover_registry(self) -> list[dict]:
        """
        전체 Agent Card를 가져와 supervisor.py의 get_agent_registry() 형식으로 변환한다.
        HTTP 연결 실패 시 빈 리스트를 반환한다.
        """
        cards = await self.fetch_all_cards()
        return [
            {
                "name": card.name,
                "skill": card.description,
                "mcp_tools": card.capabilities.mcp_tools,
            }
            for card in cards
        ]
