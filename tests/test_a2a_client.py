"""
tests/test_a2a_client.py

A2AClient.send_task 단위 테스트.
fetch_card/fetch_all_cards와 달리 실패 시에도 None이 아니라 항상 유효한
A2ATaskResponse(status="error", error_type=...)를 반환하는지 검증한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.a2a.client import A2AClient
from app.a2a.models import A2ATaskRequest


def _make_request() -> A2ATaskRequest:
    return A2ATaskRequest(task_id="t-1", agent_name="knowledge", instruction="do it", context={})


def _mock_async_client(mock_client: AsyncMock):
    mock_client_cls = MagicMock()
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return patch("app.a2a.client.httpx.AsyncClient", mock_client_cls)


class TestSendTaskSuccess:
    async def test_success_returns_parsed_response(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "task_id": "t-1",
            "status": "success",
            "result": {"answer": "42"},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with _mock_async_client(mock_client):
            resp = await A2AClient(base_url="http://localhost:8002").send_task(_make_request())

        assert resp.status == "success"
        assert resp.result == {"answer": "42"}
        # POST 경로가 이슈에서 명시한 /tasks/send 인지 확인
        called_url = mock_client.post.call_args.args[0]
        assert called_url == "http://localhost:8002/tasks/send"


class TestSendTaskFailure:
    async def test_timeout_returns_error_response_with_timeout_type(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with _mock_async_client(mock_client):
            resp = await A2AClient(base_url="http://localhost:8002").send_task(_make_request())

        assert resp.status == "error"
        assert resp.error_type == "timeout"
        assert resp.task_id == "t-1"

    async def test_connect_error_returns_error_response_with_timeout_type(self):
        """연결 거부(ConnectError)는 파라미터를 고쳐도 복구되지 않으므로 timeout으로 분류돼야 한다."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        with _mock_async_client(mock_client):
            resp = await A2AClient(base_url="http://localhost:8002").send_task(_make_request())

        assert resp.status == "error"
        assert resp.error_type == "timeout"

    async def test_never_returns_none(self):
        """fetch_card와 달리 실패해도 None이 아닌 A2ATaskResponse 객체를 반환해야 한다."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=RuntimeError("unexpected"))

        with _mock_async_client(mock_client):
            resp = await A2AClient(base_url="http://localhost:8002").send_task(_make_request())

        assert resp is not None
        assert resp.status == "error"
        # 연결 실패류(TransportError)가 아닌 진짜 알 수 없는 예외는 여전히 parameter로 분류돼야 한다.
        assert resp.error_type == "parameter"
