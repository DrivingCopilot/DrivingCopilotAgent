# app/agents/perception.py
#
# Perception Agent — Qwen2-VL Vision 분석 담당.
#
# 흐름:
#     supervisor (next_agent=perception) → perception_node(state)
#         ├ Backend MCP 의 get_camera_frame 으로 카메라 프레임(base64 JPEG) 조회
#         ├ 프레임을 멀티모달 메시지로 감싸 Vision LLM(Qwen2-VL 7B FP16) 호출
#         ├ 응답을 description/hazards 로 구조화 파싱
#         └ 분석 결과를 context_data.vision_results 에 담아 supervisor 로 복귀
#             (hazard 감지 시 plan 에 자동 대응 조치를 채워 execution 으로 직행 — 아래 참고)
#
# 모델 라우팅(계획서 2.4절): Planner/Executor(Qwen2-VL INT4)와 별도로
# Vision 전용 FP16 모델을 사용한다 — PERCEPTION_VLM_MODEL/BASE_URL 로 설정.
#
# 멀티모달 트리거(계획서 6번 항목): 비/터널/경고등 감지 시 execution agent의
# tool을 자동 호출해야 한다. supervisor(7B 모델)의 JSON 판단에 이 안전 트리거를
# 맡기면 그동안 겪은 반복 출력/판단 불안정 문제가 그대로 재현되므로, 이 판단은
# 여기서 코드로 결정적으로 처리하고 graph 레벨에서 execution 으로 직행시킨다
# (app/graph/builder.py 의 route_after_perception 참고). supervisor 는 그 결과를
# 받아서 사용자에게 요약만 한다.

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError, field_validator

from app.core.config import PERCEPTION_VLM_BASE_URL, PERCEPTION_VLM_MODEL
from app.core.json_utils import extract_first_json_object
from app.core.mcp_client import call_mcp_tool_once
from app.graph import ws as _ws
from app.graph.state import AgentState

logger = logging.getLogger(__name__)

# hazard → execution agent 가 수행할 plan step. execution._extract_tool_call 이
# 이 문자열을 보고 MCP tool/파라미터를 추출하므로, execution.py의 _TOOL_SIGNATURES
# 표기와 맞춰서 작성한다 (app/agents/execution.py 참고).
HAZARD_PLAN_STEPS: Dict[str, str] = {
    "rain": "비가 감지되었습니다. 와이퍼를 켜세요. (control_wiper on=true)",
    "tunnel": "터널 진입이 감지되었습니다. 전조등을 켜세요. (control_lighting on=true)",
    "warning_light": "경고등이 감지되었습니다. 대시보드 경고등 상태를 조회하세요. (query_dashboard metric=warning_lights)",
}

_KNOWN_HAZARDS = tuple(HAZARD_PLAN_STEPS.keys())


class PerceptionVisionResult(BaseModel):
    """
    Vision LLM의 JSON 응답을 검증한다. hazards/related_hazard는 프롬프트로만
    강제되므로(백엔드가 Ollama든 Colab HF-transformers 서버든 동일하게 동작해야
    해 grammar-constrained decoding은 쓰지 않음), 모델이 어휘 밖의 값을 내도
    예외 대신 조용히 걸러내도록 validator로 검증한다.
    """

    description: str = ""
    hazards: List[str] = []
    related_hazard: Optional[str] = None
    answer: str = ""

    @field_validator("hazards")
    @classmethod
    def _filter_unknown_hazards(cls, v: List[str]) -> List[str]:
        return [h for h in v if h in _KNOWN_HAZARDS]

    @field_validator("related_hazard")
    @classmethod
    def _validate_related_hazard(cls, v: Optional[str]) -> Optional[str]:
        return v if v in _KNOWN_HAZARDS else None


def _build_vision_prompt(user_question: str) -> str:
    """
    Vision LLM 프롬프트를 조립한다. 사용자 질문이 있으면 그 질문을 실제로 VLM에
    전달하고, 3종 hazard 어휘(rain/tunnel/warning_light) 중 하나에 대한 질문인지
    VLM 스스로 판단(related_hazard)하게 한다 — supervisor.py가 사용자 질문 텍스트를
    별도로 키워드 매칭하지 않도록 하기 위함.
    """
    base = (
        "당신은 차량 카메라 영상을 분석하는 비전 어시스턴트입니다. "
        "이 영상에서 날씨, 도로 상태, 위험 상황(보행자, 장애물, 경고등 등)을 분석하세요.\n\n"
    )
    question_part = (
        f'운전자가 다음과 같이 질문했습니다: "{user_question}"\n'
        "이 질문이 rain(비)/tunnel(터널)/warning_light(대시보드 경고등) 중 하나에 "
        "대한 것이면 related_hazard에 그 값을, 아니면 null을 넣으세요. "
        "질문 내용과 무관하게 영상을 근거로 answer 필드에 직접 답변하세요.\n\n"
        if user_question else ""
    )
    schema_part = (
        "다음 JSON 형식으로만 답하세요(다른 텍스트 없이):\n"
        '{"description": "<한국어로 간결한 설명>", '
        '"hazards": [<감지된 항목, "rain"|"tunnel"|"warning_light" 중에서만 선택. 없으면 빈 배열>], '
        '"related_hazard": <"rain"|"tunnel"|"warning_light"|null>, '
        '"answer": "<운전자 질문에 대한 한국어 직접 답변. 질문이 없으면 빈 문자열>"}'
    )
    return base + question_part + schema_part


def _parse_vision_response(raw: str) -> PerceptionVisionResult:
    """
    Vision LLM 응답에서 description/hazards/related_hazard/answer 를 추출한다.
    구조화 JSON 파싱/검증에 실패하면 원본 텍스트를 description으로, 나머지는
    빈 값으로 폴백한다 (기존 자유 텍스트 응답과의 하위호환).
    """
    try:
        parsed = json.loads(extract_first_json_object(raw.strip()))
        result = PerceptionVisionResult.model_validate(parsed)
        if not result.description:
            result.description = raw
        result.hazards = _sanity_check_hazards(result.description, result.hazards)
        return result
    except (json.JSONDecodeError, AttributeError, TypeError, ValidationError):
        return PerceptionVisionResult(description=raw)


# description(자유 텍스트)에 이 키워드가 있으면 VLM이 "위험 없음"이라고 서술한
# 것으로 간주해, hazards에 그와 모순되는 항목이 남아있으면 제거한다. VLM
# 환각(hallucination)에 대한 완벽한 해결책이 아니라 description과 hazards가
# 서로 명백히 모순되는 케이스만 걸러내는 규칙 기반 필터다.
_CLEAR_WEATHER_KEYWORDS = ("맑", "화창", "clear", "비 안", "비가 안", "눈 안", "눈이 안")


def _sanity_check_hazards(description: str, hazards: List[str]) -> List[str]:
    """description이 명시적으로 맑은 날씨를 서술하면 hazards의 'rain'을 제거한다."""
    if "rain" in hazards and any(kw in description for kw in _CLEAR_WEATHER_KEYWORDS):
        return [h for h in hazards if h != "rain"]
    return hazards


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
    messages = state.get("messages", [])
    user_question = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    logger.info(
        "perception_node 시작: route_type=%s vlm_model=%s",
        state.get("route_type"), PERCEPTION_VLM_MODEL,
    )

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

    logger.info("perception_node: get_camera_frame status=%s", status)

    if status != "success":
        logger.warning(
            "perception_node 완료 (camera 실패): error_type=%s error_msg=%s",
            error_type, error_msg,
        )
        return {
            "context_data": {
                **context_data,
                "vision_results": {
                    "status": "fail",
                    "error_type": error_type,
                    "error_msg": error_msg,
                },
            },
            "plan": [],
        }

    # ── 2. Vision LLM 호출 ───────────────────────────────────────────────────
    message = HumanMessage(content=[
        {"type": "text", "text": _build_vision_prompt(user_question)},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
    ])

    try:
        response = await _get_vision_llm().ainvoke([message])
    except Exception as exc:
        logger.error("perception_node 완료 (VLM 실패): %s", exc)
        return {
            "context_data": {
                **context_data,
                "vision_results": {
                    "status": "fail",
                    "error_type": "vlm",
                    "error_msg": str(exc),
                },
            },
            "plan": [],
        }

    logger.debug("perception_node: VLM 원본 응답=%r", response.content)

    result = _parse_vision_response(response.content)
    plan_steps = [HAZARD_PLAN_STEPS[h] for h in result.hazards]

    logger.info(
        "perception_node 완료: description=%r hazards=%s related_hazard=%s answer=%r plan=%s",
        result.description, result.hazards, result.related_hazard, result.answer, plan_steps,
    )

    return {
        "context_data": {
            **context_data,
            "vision_results": {
                "status": "success",
                "description": result.description,
                "hazards": result.hazards,
                "related_hazard": result.related_hazard,
                "answer": result.answer,
            },
        },
        # hazard 감지 시 execution agent 가 바로 실행할 plan (route_after_perception 참고).
        # 없으면 빈 리스트로 명시 — 이전에 perception 위임용으로 쓰였던 stale plan을 정리한다.
        "plan": plan_steps,
    }
