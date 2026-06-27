# app/agents/perception.py
#
# Perception Agent — Qwen2-VL Vision 분석 담당.
#
# 흐름:
#     supervisor (next_agent=perception) → perception_node(state)
#         ├ Backend MCP 의 get_camera_frame 으로 카메라 프레임(base64 JPEG) 조회
#         ├ 프레임을 멀티모달 메시지로 감싸 Vision LLM(Qwen2-VL 7B FP16) 호출
#         └ 분석 결과를 context_data.vision_results 에 담아 supervisor 로 복귀
#
# 모델 라우팅(계획서 2.4절): Planner/Executor(Qwen2-VL INT4)와 별도로
# Vision 전용 FP16 모델을 사용한다 — PERCEPTION_VLM_MODEL/BASE_URL 로 설정.
#
# 실패 처리: execution.py 와 동일하게 error_count 는 직접 쓰지 않고
# context_data 에 상태만 담아 반환한다 (재시도 판단은 supervisor 책임).

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import PERCEPTION_VLM_BASE_URL, PERCEPTION_VLM_MODEL
from app.core.mcp_client import call_mcp_tool_once
from app.graph import ws as _ws
from app.graph.state import AgentState

logger = logging.getLogger(__name__)

_VISION_PROMPT = (
    "당신은 차량 카메라 영상을 분석하는 비전 어시스턴트입니다. "
    "이 영상에서 날씨, 도로 상태, 위험 상황(보행자, 장애물, 경고등 등)을 "
    "한국어로 간결하게 설명하세요."
)

# 모듈 레벨 싱글턴 — execution.py 의 _EXTRACTION_LLM 과 동일한 lazy init 패턴.
_VISION_LLM: Optional[ChatOpenAI] = None


def _get_vision_llm() -> ChatOpenAI:
    """_VISION_LLM 싱글턴을 반환한다. 최초 호출 시 생성된다."""
    global _VISION_LLM
    if _VISION_LLM is None:
        kwargs: Dict[str, Any] = {"model": PERCEPTION_VLM_MODEL, "temperature": 0.1}
        if PERCEPTION_VLM_BASE_URL:
            kwargs["base_url"] = PERCEPTION_VLM_BASE_URL
        _VISION_LLM = ChatOpenAI(**kwargs)
    return _VISION_LLM


async def perception_node(state: AgentState) -> Dict[str, Any]:
    """Perception Agent 노드. 카메라 프레임을 조회해 Vision LLM 으로 분석한다."""
    context_data: Dict[str, Any] = dict(state.get("context_data", {}))

    await _ws.websocket_manager.send_status(
        json.dumps({"type": "status", "data": "Perception agent 카메라 분석 중..."})
    )

    # ── 1. camera_feed MCP tool 호출 ────────────────────────────────────────
    await _ws.websocket_manager.send_status(
        json.dumps({
            "type": "tool_start",
            "data": {"tool_name": "get_camera_frame", "params": {"camera_id": "front"}},
        })
    )
    frame_b64, status, error_type, error_msg = await call_mcp_tool_once(
        "get_camera_frame", {"camera_id": "front"}
    )
    await _ws.websocket_manager.send_status(
        json.dumps({
            "type": "tool_result",
            "data": {"tool_name": "get_camera_frame", "result": status, "status": status},
        })
    )

    if status != "success":
        logger.warning("camera_feed 조회 실패: %s", error_msg)
        return {
            "context_data": {
                **context_data,
                "vision_results": {
                    "status": "fail",
                    "error_type": error_type,
                    "error_msg": error_msg,
                },
            },
        }

    # ── 2. Vision LLM 호출 ───────────────────────────────────────────────────
    message = HumanMessage(content=[
        {"type": "text", "text": _VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
    ])

    try:
        response = await _get_vision_llm().ainvoke([message])
        description = response.content
    except Exception as exc:
        logger.error("Vision LLM 호출 실패: %s", exc)
        return {
            "context_data": {
                **context_data,
                "vision_results": {
                    "status": "fail",
                    "error_type": "vlm",
                    "error_msg": str(exc),
                },
            },
        }

    logger.info("perception_node: 분석 완료 (%d chars)", len(description))

    return {
        "context_data": {
            **context_data,
            "vision_results": {"status": "success", "description": description},
        },
    }
