"""
app/graph/builder.py

LangGraph StateGraph 기반 Multi-Agent 오케스트레이션.
Mock 노드, 조건부 엣지, StateGraph 조립, 외부 호출 함수를 한 파일에서 관리.
추후 규모가 커지면 nodes.py / edges.py로 분리.

흐름:
    START → supervisor → [knowledge | execution | perception | supervisor | END]
                ↑               ↓             ↓              ↓
             observe ←──────────┴─────────────┴──────────────┘
                ↓
          [supervisor | END]
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import supervisor_node
from app.agent.observe import observe_node
from app.agent.perception import perception_node
from app.agent.state import AgentState
from app.agents.execution import run_execution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock 노드 (추후 app/agents/ 실제 로직으로 교체)
# ---------------------------------------------------------------------------

async def knowledge_node(state: AgentState) -> Dict[str, Any]:
    """
    Knowledge Agent Mock.
    추후 app/agents/knowledge.py 실제 로직으로 교체.
    (Vector RAG / Graph RAG / Text2SQL)
    """
    logger.info("knowledge_node: Mock 실행")

    return {
        "tool_calls": [
            {
                "tool": "mock_knowledge_search",
                "params": {},
                "status": "success",
                "result": "[Mock] 검색 완료",
            },
        ],
        "context_data": {
            "vector_results": ["[Mock] 매뉴얼 청크 1", "[Mock] 매뉴얼 청크 2"],
            "graph_results": ["[Mock] 엔진경고등 → 점화플러그 → 교체주기"],
        },
    }

async def execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Execution Agent — app/agents/execution.py 실 구현 호출.
    supervisor 가 plan 과 함께 next_agent="execution" 을 반환하면 이 노드가 실행된다.

    흐름: plan 파싱 → MCP 12종 tool 호출 → tool_calls/vehicle_state 병합 반환
    """
    logger.info("execution_node: 실 구현 호출")
    return await run_execution(state)



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
        logger.info(f"route_next: → {next_agent}")
        return next_agent

    logger.info("route_next: → __end__")
    return "__end__"


def route_after_observe(state: AgentState) -> str:
    """
    observe_node 가 반환한 next_agent 값을 보고 다음을 결정.

    Returns:
        "supervisor" | "__end__"
    """
    next_agent = state.get("next_agent", "__end__")

    if next_agent == "supervisor":
        logger.info("route_after_observe: → supervisor")
        return "supervisor"

    logger.info("route_after_observe: → __end__")
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
    graph.add_node("execution", execution_node)
    graph.add_node("perception", perception_node)
    graph.add_node("observe", observe_node)

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
            "__end__": END,
        },
    )

    # 4. 각 Agent → observe (실행 결과 검증)
    graph.add_edge("knowledge", "observe")
    graph.add_edge("execution", "observe")
    graph.add_edge("perception", "observe")

    # 5. observe → [supervisor | __end__] 조건부 분기
    graph.add_conditional_edges(
        "observe",
        route_after_observe,
        {
            "supervisor": "supervisor",
            "__end__": END,
        },
    )

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

    logger.info(f"run_graph 시작: {user_message!r} | route_type={route_type!r}")
    # recursion_limit: 무한 루프 안전망 (ReAct 루프 + observe 재시도 고려)
    result = await app.ainvoke(initial_state, config={"recursion_limit": 25})
    logger.info("run_graph 완료")

    return result
