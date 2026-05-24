"""
app/agent/perception.py

Perception Agent (Mock VLM).

계획서 명시:
- 역할: 영상 분석, 환경 감지, 멀티모달 트리거 (풀버전 2.3절)
- 모델: Qwen2-VL 7B FP16 (풀버전 2.4절, Phase 1 배포)
- MCP Tool: analyze_camera_feed, detect_objects

책임:
- Vision 분석 결과만 반환 (감지)
- 액션 결정(어떤 Tool 호출할지)은 supervisor 책임
- 실제 Tool 호출은 execution 책임

Mock 한정: 항상 "비 감지" 시나리오 반환.
실제 Qwen2-VL 호출 구현 시 _MOCK_RAIN_SCENARIO 및 관련 처리 제거.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.agent.state import AgentState

logger = logging.getLogger(__name__)


# ============================================================
# Mock 데이터 (실제 VLM 호출 구현 시 제거)
# ============================================================
_MOCK_RAIN_SCENARIO = {
    "scene": {
        "scene_description": "젖은 도로, 비가 내리는 상황. 시야 약간 제한.",
        "weather": "rain",
        "visibility": "medium",
        "confidence": 0.91,
    },
    "objects": [
        {"label": "rain", "confidence": 0.94, "bbox": [0, 0, 1920, 1080]},
        {"label": "wet_road", "confidence": 0.87, "bbox": [200, 600, 1700, 1080]},
    ],
}
# ============================================================


async def perception_node(state: AgentState) -> Dict[str, Any]:
    """
    Perception Agent 노드.

    Mock 한정: VLM 호출 대신 _MOCK_RAIN_SCENARIO 반환.
    실제 구현 시 아래 mock 호출 부분만 교체:
        # from app.vision.qwen_vl import qwen_vl_client
        # scene = await qwen_vl_client.analyze_camera_feed(camera_input)
        # objects = await qwen_vl_client.detect_objects(camera_input)
    """
    logger.info("perception_node: VLM mock 분석 시작")

    scene = _MOCK_RAIN_SCENARIO["scene"]
    objects = _MOCK_RAIN_SCENARIO["objects"]

    logger.info(
        "perception_node: scene='%s' objects=%d",
        scene["scene_description"][:30],
        len(objects),
    )

    return {
        "tool_calls": [
            {
                "tool": "analyze_camera_feed",
                "params": {"camera_id": "front", "mode": "scene"},
                "status": "success",
                "result": scene,
            },
            {
                "tool": "detect_objects",
                "params": {"camera_id": "front", "categories": ["weather", "vehicle", "warning"]},
                "status": "success",
                "result": objects,
            },
        ],
        "context_data": {
            "vision_results": {
                "scene": scene,
                "objects": objects,
            },
        },
    }
