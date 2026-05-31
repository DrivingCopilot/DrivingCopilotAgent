# app/agents/knowledge.py
#
# Knowledge Agent — Vector RAG / Graph RAG / Text2SQL 담당.
# 현재 Mock 구현. 추후 Qdrant, Neo4j, SQLite 실 연동으로 교체 예정.

import logging
from typing import Any, Dict

from app.graph.state import AgentState

logger = logging.getLogger(__name__)


async def knowledge_node(state: AgentState) -> Dict[str, Any]:
    logger.info("knowledge_node: Mock 실행")
    return {
        "context_data": {
            **state.get("context_data", {}),
            "vector_results": ["[Mock] 매뉴얼 청크 1", "[Mock] 매뉴얼 청크 2"],
            "graph_results": ["[Mock] 엔진경고등 → 점화플러그 → 교체주기"],
        }
    }
