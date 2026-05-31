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

from app.graph import ws as graph_ws
from app.core import config
from app.api.http import router as http_router
from app.api.websocket import router as ws_router, streamer_proxy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # supervisor_node 내부의 websocket_manager 를 contextvars 기반 프록시로 교체.
    # 실제 전송은 현재 WS 세션의 Streamer 로 위임되므로 세션 간 broadcast 누출이 없다.
    # HTTP /invoke 같이 WS 세션 바깥에서 호출되는 경로는 로그로만 출력된다.
    graph_ws.websocket_manager = streamer_proxy
    logger.info("Supervisor streamer proxy attached")
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
