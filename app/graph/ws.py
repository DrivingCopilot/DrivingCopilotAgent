# app/graph/ws.py
#
# 모든 Agent가 공유하는 WebSocket 전송 인터페이스.
# 기동 시 main.py의 lifespan에서 streamer_proxy로 교체된다.

import logging

logger = logging.getLogger(__name__)


class MockWebsocketManager:
    async def send_status(self, message: str):
        logger.info(f"WS_MOCK_SEND: {message}")

    def send_status_sync(self, message: str):
        logger.info(f"WS_MOCK_SEND: {message}")


websocket_manager = MockWebsocketManager()
