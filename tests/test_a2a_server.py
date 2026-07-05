"""
tests/test_a2a_server.py

app/a2a/server.py의 create_agent_app 라우팅 테스트.
실제 소켓 없이 httpx.ASGITransport로 FastAPI 앱에 직접 요청을 보낸다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.a2a.models import A2ATaskResponse
from app.a2a.server import create_agent_app


@pytest.fixture
def knowledge_app():
    return create_agent_app("knowledge")


async def _post(app, path: str, json: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, json=json)


class TestCreateAgentAppDiscovery:
    async def test_well_known_agent_returns_own_card(self, knowledge_app):
        transport = httpx.ASGITransport(app=knowledge_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/.well-known/agent.json")

        assert resp.status_code == 200
        assert resp.json()["name"] == "knowledge"

    def test_unknown_agent_name_raises(self):
        with pytest.raises(ValueError):
            create_agent_app("supervisor")  # DELEGATABLE_AGENTS에 없음

        with pytest.raises(ValueError):
            create_agent_app("no_such_agent")


class TestTasksSendRouting:
    async def test_matching_agent_name_dispatches(self, knowledge_app):
        with patch(
            "app.a2a.server.dispatch_task",
            new_callable=AsyncMock,
            return_value=A2ATaskResponse(task_id="t-1", status="success", result={"answer": "ok"}),
        ) as mock_dispatch:
            resp = await _post(
                knowledge_app,
                "/tasks/send",
                {"task_id": "t-1", "agent_name": "knowledge", "instruction": "hi", "context": {}},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["result"] == {"answer": "ok"}
        mock_dispatch.assert_awaited_once()

    async def test_mismatched_agent_name_rejected_without_dispatch(self, knowledge_app):
        with patch("app.a2a.server.dispatch_task", new_callable=AsyncMock) as mock_dispatch:
            resp = await _post(
                knowledge_app,
                "/tasks/send",
                {"task_id": "t-1", "agent_name": "execution", "instruction": "hi", "context": {}},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert body["error_type"] == "invalid_tool"
        mock_dispatch.assert_not_awaited()
