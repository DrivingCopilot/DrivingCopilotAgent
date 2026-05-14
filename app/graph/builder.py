"""
app/graph/builder.py

LangGraph StateGraph 기반 Multi-Agent 오케스트레이션.
Mock 노드, 조건부 엣지, StateGraph 조립, 외부 호출 함수를 한 파일에서 관리.
추후 규모가 커지면 nodes.py / edges.py로 분리.

흐름:
    START → supervisor → [knowledge | execution | perception | supervisor | END]
                ↑               |           |            |
                └───────────────┴───────────┴────────────┘
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import supervisor_node
from app.agent.state import AgentState

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

    # from app.agents.knowledge import run_knowledge
    # return await run_knowledge(state)
    # 나중에 state 반환해서 넣을때도 밑의 작업 해줘야 할듯
    # return {
    # "context_data": {
    #     **state.get("context_data", {}),
    #     **await run_knowledge(state),  # 바로 펼칠 수 있음
    # }

    return {
        # 이거 보니까 잘못 덮어쓰면 vehicle_state채우다가 vector_results까지 덮어씌워질 수 있음.
        "context_data": {
            **state.get("context_data", {}), # 기존 context_data에다가 나머지를 병합하는 형태로 코드를 짜야함.
            # **a = 딕셔너리를 그 자리에 풀어 펼치는거임.
            "vector_results": ["[Mock] 매뉴얼 청크 1", "[Mock] 매뉴얼 청크 2"],
            "graph_results": ["[Mock] 엔진경고등 → 점화플러그 → 교체주기"],
        }
    }

async def execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Execution Agent Mock.
    추후 app/agents/execution.py 실제 로직으로 교체.
    (MCP Tool 12종)
    """
    logger.info("execution_node: Mock 실행")

    # from app.agents.execution import run_execution
    # execution_result =  await run_execution(state)
    # return {
    #     "context_data": {
    #         **state.get("context_data", {}),
    #         **execution_result["context_data"],
    #     },
    #     "tool_calls": [
    #         *state.get("tool_calls", []),
    #         *execution_result["tool_calls"],
    #     ]
    # }

    return {
        "tool_calls": [{"tool": "mock_tool", "status": "success", "result": "[Mock] Tool 실행 완료"}],
        "context_data": {
            **state.get("context_data", {}),
            "vehicle_state": {"mock_key": "mock_value"},
        },
    }


async def perception_node(state: AgentState) -> Dict[str, Any]:
    """
    Perception Agent Mock.
    추후 app/agents/perception.py 실제 로직으로 교체.
    (Qwen2-VL Vision 분석)
    """
    logger.info("perception_node: Mock 실행")

    # from app.agents.perception import run_perception
    # return await run_perception(state)
    # 동일하게 하면 됨

    return {
        "context_data": {
            **state.get("context_data", {}),
            "vision_results": ["[Mock] 비 감지됨", "[Mock] 창문 열림 감지"],
        }
    }


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


# ---------------------------------------------------------------------------
# StateGraph 조립
# ---------------------------------------------------------------------------

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

    # 4. 각 Agent → supervisor 복귀 (루프)
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
        "error_count": {},
        "feedback": "",
    }

    logger.info(f"run_graph 시작: {user_message!r} | route_type={route_type!r}")
    result = await app.ainvoke(initial_state)
    logger.info("run_graph 완료")

    return result
