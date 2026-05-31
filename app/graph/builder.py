"""
app/graph/builder.py

LangGraph StateGraph 기반 Multi-Agent 오케스트레이션.
각 Agent 구현은 app/agents/ 에 위치하며, 이 파일은 StateGraph 배선만 담당한다.

흐름:
    START → supervisor → [knowledge | execution | perception | supervisor | END]
                ↑               |           |            |
                └───────────────┴───────────┴────────────┘
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agents.execution import run_execution
from app.agents.knowledge import knowledge_node
from app.agents.perception import perception_node
from app.agents.supervisor import supervisor_node
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 조건부 엣지 함수
# ---------------------------------------------------------------------------

def route_next(state: AgentState) -> str:
    """supervisor_node가 반환한 next_agent 값으로 다음 노드를 결정한다."""
    next_agent = state.get("next_agent", "__end__")

    if next_agent in ("knowledge", "execution", "perception", "supervisor"):
        logger.info(f"route_next: → {next_agent}")
        return next_agent

    logger.info("route_next: → __end__")
    return "__end__"


# ---------------------------------------------------------------------------
# StateGraph 조립
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_graph():
    """StateGraph를 조립하고 컴파일하여 반환한다. 최초 호출 후 캐시된다."""
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("execution", run_execution)
    graph.add_node("perception", perception_node)

    # TODO: Query Router 구현 후 START → query_router → supervisor 로 교체
    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "knowledge": "knowledge",
            "execution": "execution",
            "perception": "perception",
            "supervisor": "supervisor",  # JSON 파싱 실패 시 자기 복구 루프
            "__end__": END,
        },
    )

    graph.add_edge("knowledge", "supervisor")
    graph.add_edge("execution", "supervisor")
    graph.add_edge("perception", "supervisor")

    return graph.compile()


# ---------------------------------------------------------------------------
# 외부 호출 함수
# ---------------------------------------------------------------------------

async def run_graph(user_message: str, route_type: str = "") -> AgentState:
    """
    그래프를 실행하고 최종 AgentState를 반환한다.
    main.py 또는 API 엔드포인트에서 호출.

    Args:
        user_message : 사용자 입력 텍스트
        route_type   : Query Router가 결정한 분류값 (rag/tool/vision/chat)
    """
    app = build_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "route_type": route_type,
        "plan": [],
        "next_agent": "",
        "tool_calls": [],
        "context_data": {},
        "error_count": {},
        "feedback": "",
    }

    logger.info(f"run_graph 시작: {user_message!r} | route_type={route_type!r}")
    result = await app.ainvoke(initial_state)
    logger.info("run_graph 완료")

    return result
