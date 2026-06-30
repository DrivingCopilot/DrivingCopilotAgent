# app/agents/knowledge.py
#
# Knowledge Agent — Vector RAG / Graph RAG / Text2SQL 담당.
# 현재 Mock 구현. 추후 Qdrant, Neo4j, SQLite 실 연동으로 교체 예정.

import logging
from typing import Any, Dict

from app.graph.state import AgentState

logger = logging.getLogger(__name__)


async def knowledge_node(state: AgentState) -> Dict[str, Any]:
    logger.info(
        "knowledge_node 시작: route_type=%s plan=%s",
        state.get("route_type"), state.get("plan"),
    )

    result = {
        "context_data": {
            **state.get("context_data", {}),
            "vector_results": ["[Mock] 매뉴얼 청크 1", "[Mock] 매뉴얼 청크 2"],
            "graph_results": ["[Mock] 엔진경고등 → 점화플러그 → 교체주기"],
        }
    }

    logger.info(
        "knowledge_node 완료 (Mock): vector_results=%d개 graph_results=%d개",
        len(result["context_data"]["vector_results"]),
        len(result["context_data"]["graph_results"]),
    )
    return result
