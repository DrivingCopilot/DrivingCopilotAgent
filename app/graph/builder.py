"""
app/graph/builder.py

LangGraph StateGraph 기반 Multi-Agent 오케스트레이션.
Mock 노드, 조건부 엣지, StateGraph 조립, 외부 호출 함수를 한 파일에서 관리.
추후 규모가 커지면 nodes.py / edges.py로 분리.

흐름:
    START → supervisor → [knowledge | execution | perception | supervisor | finalize(__end__)]
                                  ↓           ↓            ↓
              ┌── observe ←───────┼───────────┘            │
              │       ↓           │  hazard 감지 → execution 직행 → observe
              │   reflect         │  hazard 없음 → observe
              │       ↓           │  (route_after_perception)
              └─[supervisor | finalize(__end__)]
                        ↓
                    finalize → END

    CRAG(Corrective RAG) 서브그래프 — knowledge 결과 교정 루프:
    knowledge → grade_retrieval →
        correct / 재검색 캡 소진 → refine_knowledge → observe
        incorrect | ambiguous    → transform_query  → knowledge (재검색, MAX_CRAG_ATTEMPTS 캡)
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from app.agents.crag import (
    grade_retrieval_node,
    refine_knowledge_node,
    route_after_grade,
    transform_query_node,
)
from app.agents.finalize import finalize_node
from app.agents.observe import observe_node
from app.agents.reflect import reflect_node
from app.agents.supervisor import supervisor_node
from app.graph import ws as _ws
from app.graph.a2a_nodes import (
    execution_a2a_node,
    knowledge_a2a_node,
    perception_a2a_node,
)
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


def route_after_perception(state: AgentState) -> str:
    """
    perception_node가 hazard(비/터널/경고등)를 감지해 plan을 채웠으면
    execution으로 직행한다 — 멀티모달 트리거(계획서 6번 항목).
    supervisor의 LLM 판단을 거치지 않는 이유는 app/agents/perception.py 상단
    주석 참고. hazard가 없으면(plan 비어있음) observe로 보내 결과 검증 흐름에
    합류시킨다 — 모든 agent는 observe를 거친다는 원칙 유지.
    """
    if state.get("plan"):
        logger.info("route_after_perception: → execution (멀티모달 자동 트리거)")
        return "execution"

    logger.info("route_after_perception: → observe")
    return "observe"


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
    graph.add_node("knowledge", knowledge_a2a_node)
    graph.add_node("execution", execution_a2a_node)
    graph.add_node("perception", perception_a2a_node)
    graph.add_node("observe", observe_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)
    # CRAG(Corrective RAG) 노드
    graph.add_node("grade_retrieval", grade_retrieval_node)
    graph.add_node("transform_query", transform_query_node)
    graph.add_node("refine_knowledge", refine_knowledge_node)

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

    # 4. execution → observe (실행 결과 검증)
    graph.add_edge("execution", "observe")

    # 4-0. knowledge → CRAG 교정 루프 → observe
    #   knowledge 검색 결과를 grade_retrieval 이 평가하고,
    #   부실하면 transform_query 로 쿼리를 재작성해 knowledge 로 재검색시킨다.
    #   충분(correct)하거나 재검색 캡 소진 시 refine_knowledge 로 정제 후 observe 합류.
    graph.add_edge("knowledge", "grade_retrieval")
    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_grade,
        {
            "refine": "refine_knowledge",
            "transform": "transform_query",
        },
    )
    graph.add_edge("transform_query", "knowledge")   # 재검색 루프
    graph.add_edge("refine_knowledge", "observe")    # 기존 검증 흐름 합류

    # 4-1. perception → 멀티모달 트리거 분기
    #   hazard 감지(plan 있음) → execution 직행, 없으면 observe로 합류
    graph.add_conditional_edges(
        "perception",
        route_after_perception,
        {
            "execution": "execution",
            "observe": "observe",
        },
    )

    # 5. observe → reflect (무조건)
    graph.add_edge("observe", "reflect")

    # 6. reflect → [supervisor | finalize] 조건부 분기
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "supervisor": "supervisor",
            "__end__": "finalize",
        },
    )

    # 7. finalize → END (엔티티 메모리 저장 후 종료)
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
    # recursion_limit: 무한 루프 안전망.
    # supervisor 루프 + observe + reflect 에 더해, CRAG 는 knowledge 위임 1회당
    # grade_retrieval→(transform_query→knowledge→grade_retrieval)→refine 노드를
    # 추가하므로(위임당 최대 ~6 노드) 여유를 둔다.
    try:
        result = await app.ainvoke(initial_state, config={"recursion_limit": 40})
    except GraphRecursionError:
        # 정상적으로는 각 노드의 재시도/재호출 차단 가드(observe의 MAX_RETRY,
        # supervisor의 perception/knowledge 재호출 차단 등)가 먼저 걸려야 하지만,
        # 그 가드를 우회하는 루프가 생기면 여기서 최후 방어선으로 걸린다. 안
        # 잡으면 websocket.py 바깥의 except가 done(reason="exception")만 보내고
        # 사용자에게 보일 text는 전혀 안 나가 "응답 없음"으로 보인다.
        logger.error("run_graph: recursion_limit 도달 — 강제 종료")
        user_msg = "요청 처리가 예상보다 길어져 중단했습니다. 다시 시도해 주세요."
        await _ws.websocket_manager.send_status(json.dumps({"type": "text", "data": user_msg}))
        await _ws.websocket_manager.send_status(
            json.dumps({"type": "done", "reason": "recursion_limit"})
        )
        result = {**initial_state, "messages": [AIMessage(content=user_msg)], "next_agent": "__end__"}
    logger.info("run_graph 완료")

    return result