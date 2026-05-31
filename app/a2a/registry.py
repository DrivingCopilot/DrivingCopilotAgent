# app/a2a/registry.py
#
# 4개 Agent의 Agent Card를 정의한다.
# 현재는 모두 동일한 Supervisor 서버(8001)에서 동작한다.
# 추후 Agent별 독립 서버로 분리 시 url 필드만 수정하면 된다.

import os
from app.a2a.models import AgentCard, AgentCapabilities

_BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:8001")

AGENT_CARDS: dict[str, AgentCard] = {
    "supervisor": AgentCard(
        name="supervisor",
        description="하위 Agent(Knowledge/Execution/Perception)를 조율하는 오케스트레이터. Plan-and-Execute + CoT + Reflexion 루프를 수행한다.",
        url=_BASE_URL,
        capabilities=AgentCapabilities(mcp_tools=[]),
    ),
    "knowledge": AgentCard(
        name="knowledge",
        description="차량 매뉴얼 Vector RAG, Graph RAG, Text2SQL을 담당한다.",
        url=_BASE_URL,
        capabilities=AgentCapabilities(mcp_tools=["vector_rag_search", "graph_rag_search"]),
    ),
    "execution": AgentCard(
        name="execution",
        description="MCP 12종 Vehicle Tool을 실행한다. 차량 제어 및 상태 조회 담당.",
        url=_BASE_URL,
        capabilities=AgentCapabilities(mcp_tools=[
            "control_climate", "set_navigation", "control_media", "get_vehicle_status",
            "control_window", "control_lighting", "control_seat", "control_parking",
            "trigger_emergency", "set_driving_mode", "control_wiper", "query_dashboard",
        ]),
    ),
    "perception": AgentCard(
        name="perception",
        description="Qwen2-VL Vision 분석을 담당한다. 카메라 피드에서 위험 상황을 감지한다.",
        url=_BASE_URL,
        capabilities=AgentCapabilities(mcp_tools=["analyze_camera_feed", "detect_objects"]),
    ),
}


def get_card(agent_name: str) -> AgentCard | None:
    return AGENT_CARDS.get(agent_name)


def list_cards() -> list[AgentCard]:
    return list(AGENT_CARDS.values())
