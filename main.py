# main.py
#
# DrivingCopilotAgent (Supervisor) 서비스 진입점.
# 같은 머신에서 DrivingCopilotBackend(8000) 와 분리된 별도 프로세스로 동작한다.
#
# 실행:
#     uvicorn main:app --reload --port 8001
#     # 또는
#     python main.py

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent import nodes
from app.core import config
from app.server.endpoints import router as http_router
from app.server.websocket import ConnectionManager, router as ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # supervisor_node 내부의 websocket_manager 를 실제 ConnectionManager 로 교체.
    # 이렇게 하면 supervisor 의 토큰 스트리밍이 /ws 로 연결된 클라이언트에 자동 전달된다.
    manager = ConnectionManager()
    app.state.ws_manager = manager
    nodes.websocket_manager = manager
    logger.info("Supervisor WebSocket manager attached")
    yield


app = FastAPI(
    title="DrivingCopilot Supervisor Agent",
    description="LangGraph 기반 supervisor 에이전트 (port 8001)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(http_router, tags=["Supervisor"])
app.include_router(ws_router, tags=["Supervisor-WS"])


@app.get("/")
def root():
    return {
        "service": "DrivingCopilot Supervisor Agent",
        "port": config.AGENT_PORT,
        "backend_url": config.BACKEND_URL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.AGENT_HOST,
        port=config.AGENT_PORT,
        reload=True,
    )
