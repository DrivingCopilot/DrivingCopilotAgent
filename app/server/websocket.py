# app/server/websocket.py
#
# Supervisor 에이전트의 WebSocket 인터페이스.
# - ConnectionManager: 연결 중인 WS 클라이언트에게 supervisor의 스트리밍 메시지를 전달
# - /ws 엔드포인트: 클라이언트가 query를 보내면 supervisor_node를 호출

import json
import logging
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

from app.agent import nodes
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """
    nodes.MockWebsocketManager 와 동일한 인터페이스(send_status async/sync)를 제공한다.
    main.py 의 startup 훅에서 nodes.websocket_manager 를 이 매니저 인스턴스로 교체하면
    supervisor_node 안의 await websocket_manager.send_status(...) 호출이 자동으로
    연결된 모든 클라이언트로 broadcast 된다.
    """

    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def send_status(self, message: str) -> None:
        if not self._connections:
            logger.info(f"WS_SEND (no client): {message}")
            return
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"WS send failed, will drop connection: {e}")
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    def send_status_sync(self, message: str) -> None:
        logger.info(f"WS_SEND_SYNC: {message}")


@router.websocket("/ws")
async def supervisor_ws(ws: WebSocket) -> None:
    """
    클라이언트(백엔드 또는 프론트엔드)가 query 를 JSON 으로 보내면
    supervisor_node 를 호출하고, 내부 스트리밍은 ConnectionManager 가 자동 전송한다.

    기대 입력: {"query": "에어컨 켜줘"}
    """
    manager: ConnectionManager = ws.app.state.ws_manager
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
                query = payload.get("query", "")
            except json.JSONDecodeError:
                query = raw

            if not query:
                await ws.send_text(json.dumps({"type": "error", "data": "empty query"}))
                continue

            initial_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "route_type": "",
                "plan": [],
                "next_agent": "",
                "tool_calls": [],
                "context_data": {},
                "error_count": {},
                "feedback": "",
            }
            result = await nodes.supervisor_node(initial_state)
            await ws.send_text(json.dumps({"type": "result", "data": _serialize(result)}))
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        logger.exception("Unexpected error in /ws handler")
        manager.disconnect(ws)
        raise


def _serialize(result: dict) -> dict:
    """LangChain 메시지 객체가 섞인 supervisor 결과를 JSON-안전한 dict 로 변환."""
    out: dict = {}
    for k, v in result.items():
        if k == "messages":
            out[k] = [getattr(m, "content", str(m)) for m in v]
        else:
            out[k] = v
    return out
