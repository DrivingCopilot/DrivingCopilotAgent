# app/server/websocket.py
#
# Supervisor 에이전트의 WebSocket 인터페이스.
# - 세션별 격리: 각 WS 연결마다 별도의 Streamer 를 contextvars 로 묶어
#   supervisor_node 안의 await websocket_manager.send_status(...) 호출이
#   다른 사용자 세션으로 새지 않도록 보장한다.
# - 메시지 타입 표준: text / tool_start / tool_result / done (+ status 보조)

import contextvars
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.graph.builder import run_graph

logger = logging.getLogger(__name__)

router = APIRouter()


_current_streamer: contextvars.ContextVar[Optional["Streamer"]] = contextvars.ContextVar(
    "current_streamer", default=None
)


class Streamer:
    """단일 WebSocket 연결을 감싸는 세션 로컬 전송 객체."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._done_sent = False

    async def send_status(self, message: str) -> None:
        """nodes.MockWebsocketManager 와 동일한 시그니처. supervisor_node 가 호출."""
        try:
            await self.ws.send_text(message)
        except Exception as e:
            logger.warning(f"WS send failed: {e}")

    def send_status_sync(self, message: str) -> None:
        logger.info(f"WS_SEND_SYNC: {message}")

    async def send_done(self, reason: Optional[str] = None) -> None:
        if self._done_sent:
            return
        self._done_sent = True
        payload = {"type": "done"}
        if reason:
            payload["reason"] = reason
        try:
            await self.ws.send_text(json.dumps(payload))
        except Exception as e:
            logger.warning(f"WS done send failed: {e}")


class _StreamerProxy:
    """
    nodes.websocket_manager 가 참조할 모듈 레벨 프록시.
    실제 전송은 contextvars 에 바인딩된 현재 세션의 Streamer 로 위임한다.
    바인딩이 없으면 (e.g. HTTP /invoke 단발 호출) 로그로만 출력.
    """

    async def send_status(self, message: str) -> None:
        streamer = _current_streamer.get()
        if streamer is None:
            logger.info(f"WS_SEND (no session): {message}")
            return
        await streamer.send_status(message)

    def send_status_sync(self, message: str) -> None:
        logger.info(f"WS_SEND_SYNC: {message}")


streamer_proxy = _StreamerProxy()


@router.websocket("/ws")
async def chat_ws(ws: WebSocket) -> None:
    """
    클라이언트가 query 를 JSON 으로 보내면 LangGraph 그래프(run_graph) 를 실행하고,
    내부 스트리밍(text/status 등)은 세션 로컬 Streamer 가 전송한다.
    턴이 끝나면 done 프레임을 보내고 다음 입력을 기다린다.

    기대 입력: {"query": "에어컨 켜줘", "route_type": "tool"}  (route_type 은 선택)
    """
    await ws.accept()
    while True:
        try:
            raw = await ws.receive_text()
        except WebSocketDisconnect:
            return

        try:
            payload = json.loads(raw)
            query = payload.get("query", "")
            route_type = payload.get("route_type", "")
        except json.JSONDecodeError:
            query = raw
            route_type = ""

        if not query:
            await ws.send_text(json.dumps({"type": "error", "data": "empty query"}))
            continue

        streamer = Streamer(ws)
        token = _current_streamer.set(streamer)
        try:
            try:
                await run_graph(query, route_type)
            except Exception:
                logger.exception("graph run failed inside WS handler")
                await streamer.send_done(reason="exception")
            else:
                await streamer.send_done()
        finally:
            _current_streamer.reset(token)
