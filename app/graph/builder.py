"""
app/graph/builder.py

LangGraph StateGraph 기반 Multi-Agent 오케스트레이션.
Mock 노드, 조건부 엣지, StateGraph 조립, 외부 호출 함수를 한 파일에서 관리.
추후 규모가 커지면 nodes.py / edges.py로 분리.

흐름:
    START → supervisor → [knowledge | execution | perception | supervisor | END]
                ↑               ↓             ↓              ↓
             observe ←──────────┴─────────────┴──────────────┘
                ↓ (무조건)
            reflect
                ↓
          [supervisor | END]
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agents.supervisor import supervisor_node
from app.agents.observe import observe_node
from app.agents.reflect import reflect_node
from app.agents.finalize import finalize_node
from app.agents.perception import perception_node
from app.agents.knowledge import knowledge_node
from app.agents.execution import run_execution
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 조건부 엣지 함수
# ---------------------------------------------------------------------------

def route_next(state: AgentState) -> str:
    """
    supervisor_node가 반환한 next_agent 값을 보고 다음 노드를 결정한다.

    Returns:
        "knowledge" | "execution" | "perception" | "supervisor" | "__end__"
    """
    next_agent = state.get("next_agent", "__end__")

    if next_agent in ("knowledge", "execution", "perception", "supervisor"):
        logger.info("route_next: → %s", next_agent)
        return next_agent

    logger.info("route_next: → __end__")
    return "__end__"


def route_after_reflect(state: AgentState) -> str:
    """
    reflect_node가 반환한 next_agent 값을 보고 다음을 결정.

    Returns:
        "supervisor" | "__end__"
    """
    next_agent = state.get("next_agent", "__end__")

    if next_agent == "supervisor":
        logger.info("route_after_reflect: → supervisor")
        return "supervisor"

    logger.info("route_after_reflect: → __end__")
    return "__end__"


# ---------------------------------------------------------------------------
# StateGraph 조립
# ---------------------------------------------------------------------------

# @lru_cache가 중간에서 알아서 캐시된 걸 돌려줌. build_graph()할때마다 그래프 다시 컴파일 안해도 되게 함
# 한번 컴파일해놓고 계속 사용.

@lru_cache(maxsize=1)
def build_graph():
    """
    StateGraph를 조립하고 컴파일하여 반환한다.

    Returns:
        컴파일된 LangGraph CompiledGraph
    """
    graph = StateGraph(AgentState)

    # 1. 노드 등록
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("execution", run_execution)
    graph.add_node("perception", perception_node)
    graph.add_node("observe", observe_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)

    # 2. 시작 엣지
    # TODO: Query Router 구현 후 START → query_router → supervisor 로 교체
    graph.add_edge(START, "supervisor")

    # 3. supervisor 조건부 분기
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "knowledge": "knowledge",
            "execution": "execution",
            "perception": "perception",
            "supervisor": "supervisor",  # JSON 파싱 실패 시 자기 복구 루프
            "__end__": "finalize",
        },
    )

    # 4. 각 Agent → observe (실행 결과 검증)
    graph.add_edge("knowledge", "observe")
    graph.add_edge("execution", "observe")
    graph.add_edge("perception", "observe")

    # 5. observe → reflect (무조건)
    graph.add_edge("observe", "reflect")

    # 6. reflect → [supervisor | END] 조건부 분기
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "supervisor": "supervisor",
            "__end__": "finalize",
        },
    )

    graph.add_edge("finalize", END)

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
                       Query Router 구현 전까지는 빈 문자열로 호출

    Returns:
        최종 AgentState
    """
    app = build_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "route_type": route_type,
        "plan": [],
        "next_agent": "",
        "tool_calls": [],
        "context_data": {},
        "error_count": {"timeout": 0, "parameter": 0, "invalid_tool": 0, "sql": 0},
        "feedback": "",
    }

    logger.info("run_graph 시작: %r | route_type=%r", user_message, route_type)
    # recursion_limit: 무한 루프 안전망 (ReAct 루프 + observe + reflect 고려)
    result = await app.ainvoke(initial_state, config={"recursion_limit": 25})
    logger.info("run_graph 완료")

    return result
