# app/agents/perception.py
#
# Perception Agent — Qwen2-VL Vision 분석 담당.
# 현재 Mock 구현. 추후 VLM 카메라 피드 연동으로 교체 예정.

import logging
from typing import Any, Dict

from app.graph.state import AgentState

logger = logging.getLogger(__name__)


async def perception_node(state: AgentState) -> Dict[str, Any]:
    logger.info("perception_node: Mock 실행")
    return {
        "context_data": {
            **state.get("context_data", {}),
            "vision_results": ["[Mock] 비 감지됨", "[Mock] 창문 열림 감지"],
        }
    }
