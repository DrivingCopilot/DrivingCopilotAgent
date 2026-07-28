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
    Vision LLM의 JSON 응답을 검증한다.

    설계 노트: 이전엔 "사용자 질문이 rain/tunnel/warning_light 중 하나에 대한
    것인가"를 VLM 스스로 판단하는 related_hazard 필드가 있었으나, 작은 VLM이
    이 조건부 분류를 신뢰성 있게 못 해서(few-shot을 넣어도 개선 안 됨 — 실측
    확인) 제거했다. 지금은 역할을 둘로 완전히 분리한다:
      - hazards: 화면에 실제로 뭐가 보이는지만 판단(질문과 무관, 자동 안전
        트리거 입력) — 이 부분은 원래도 신뢰도가 괜찮았다.
      - answer: 질문이 있으면 무조건 직접 답변(조건부 분류 없이 항상 채움).
    supervisor.py는 hazards로 매칭 여부를 재판단하지 않고 answer를 그대로
    신뢰한다.
    """

    answer: str = ""
    description: str = ""
    hazards: List[str] = []

    @field_validator("hazards")
    @classmethod
    def _filter_unknown_hazards(cls, v: List[str]) -> List[str]:
        return [h for h in v if h in _KNOWN_HAZARDS]


def _build_vision_prompt(user_question: str) -> str:
    """
    Vision LLM 프롬프트를 조립한다. 사용자 질문이 있으면 그 질문을 실제로 VLM에
    전달해 answer 필드에 조건 없이 직접 답변하게 한다 — "이 질문이 어떤 hazard
    범주에 속하는가" 같은 메타 분류는 더 이상 요구하지 않는다(모델이 그 판단을
    못 해서 화면의 hazard를 질문과 무관하게 반사적으로 확답해버리는 문제가
    있었음). answer를 JSON 첫 필드로 둔 것도 의도적 — 모델이 판단/묘사부터
    하고 답변을 뒷전으로 미루는 경향이 있어, 질문에 먼저 답하도록 순서로 유도.
    """
    base = (
        "당신은 차량 카메라 영상을 분석하는 비전 어시스턴트입니다. "
        "이 영상에서 날씨, 도로 상태, 위험 상황(보행자, 장애물, 경고등 등)을 분석하세요.\n\n"
    )
    question_part = (
        f'운전자가 다음과 같이 질문했습니다: "{user_question}"\n'
        "이 질문이 무엇에 관한 것이든 상관없이(날씨, 표지판, 사람, 잡담 등) "
        "영상을 근거로 answer 필드에 반드시 직접 답변하세요. 판단을 미루거나 "
        "비워두지 마세요.\n\n"
        if user_question else ""
    )
    schema_part = (
        "다음 JSON 형식으로만 답하세요(다른 텍스트 없이):\n"
        '{"answer": "<운전자 질문에 대한 한국어 직접 답변. 질문이 없으면 빈 문자열>", '
        '"description": "<한국어로 간결한 설명>", '
        '"hazards": [<감지된 항목, "rain"|"tunnel"|"warning_light" 중에서만 선택. 없으면 빈 배열>]}'
    )
    return base + question_part + schema_part


def _parse_vision_response(raw: str) -> PerceptionVisionResult:
    """
    Vision LLM 응답에서 answer/description/hazards 를 추출한다.
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
        "perception_node 완료: description=%r hazards=%s answer=%r plan=%s",
        result.description, result.hazards, result.answer, plan_steps,
    )

    return {
        "context_data": {
            **context_data,
            "vision_results": {
                "status": "success",
                "description": result.description,
                "hazards": result.hazards,
                "answer": result.answer,
            },
        },
        # hazard 감지 시 execution agent 가 바로 실행할 plan (route_after_perception 참고).
        # 없으면 빈 리스트로 명시 — 이전에 perception 위임용으로 쓰였던 stale plan을 정리한다.
        "plan": plan_steps,
    }
