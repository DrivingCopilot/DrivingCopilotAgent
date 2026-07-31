# app/agents/perception.py
#
# Perception Agent — Qwen2-VL Vision 분석 담당.
#
# 흐름:
#     supervisor (next_agent=perception) → perception_node(state)
#         ├ Backend MCP 의 get_camera_frame 으로 카메라 프레임(base64 JPEG) 조회
#         ├ 프레임을 멀티모달 메시지로 감싸 Vision LLM(Qwen2-VL 7B FP16) 호출
#         ├ 응답을 answer/hazards 로 구조화 파싱
#         └ 분석 결과를 context_data.vision_results 에 담아 supervisor 로 복귀
#             (hazard 감지 시 plan 에 자동 대응 조치를 채워 execution 으로 직행 — 아래 참고)
#
# 모델 라우팅: Supervisor/Execution과 동일한 로컬 모델 서버(app/model_server)의
# 7B VL 모델(QWEN_VL_MODEL_NAME)을 공유해서 쓴다.
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

from app.core.config import MODEL_SERVER_URL, QWEN_VL_MODEL_NAME
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

    설계 노트: 원래는 description(항상 채우는 일반 장면 묘사)과 answer(질문이
    있을 때만 채우는 직접 답변)를 분리했었는데, 그러다 보니 모델이 부정적인
    답("표지판이 없습니다" 등)을 description에만 쓰고 answer는 비워버리는
    문제가 실측으로 확인됐다(두 필드가 겹치니 모델이 하나만 쓰고 만 것으로
    보임). 지금은 필드를 answer 하나로 합쳐서 이 중복 자체를 없앴다 — perception
    노드는 항상 사용자 질문에 응답해서 호출되므로(질문 없이 호출되는 경로는
    없음) answer는 항상 채워지는 것을 전제로 한다.
      - answer: 질문에 대한 직접 답변(항상 채움).
      - hazards: 화면에 실제로 뭐가 보이는지만 판단(질문과 무관, 자동 안전
        트리거 입력) — 원래도 신뢰도가 괜찮았던 부분이라 그대로 유지.
    """

    answer: str = ""
    hazards: List[str] = []

    @field_validator("hazards")
    @classmethod
    def _filter_unknown_hazards(cls, v: List[str]) -> List[str]:
        return [h for h in v if h in _KNOWN_HAZARDS]


def _build_vision_prompt(user_question: str) -> str:
    """
    Vision LLM 프롬프트를 조립한다. perception은 항상 사용자 질문에 응답해서
    호출되므로 answer는 항상 채우는 것을 전제로 지시한다. "완전한 문장으로"를
    넣은 건 부정 답변("없습니다"처럼 한 단어로만 끝내는 경향)을 조금이라도
    완화하기 위함 — 실측 결과 완전히 없어지진 않았지만(그래도 문법적으론
    완결된 문장), 최소한 답이 아예 비는 문제는 이 필드 통합으로 해결됨.
    """
    return (
        "당신은 차량 카메라 영상을 분석하는 비전 어시스턴트입니다. "
        "이 영상에서 날씨, 도로 상태, 위험 상황(보행자, 장애물, 경고등 등)을 분석하세요.\n\n"
        f'운전자가 다음과 같이 질문했습니다: "{user_question}"\n'
        "이 질문이 무엇에 관한 것이든 상관없이(날씨, 표지판, 사람, 잡담 등) "
        "영상을 근거로 answer 필드에 완전한 문장으로 직접 답변하세요. "
        "'없습니다'/'아닙니다' 같은 부정적인 답이어도 반드시 완전한 문장으로 "
        "채우세요. 판단을 미루거나 비워두지 마세요.\n\n"
        "다음 JSON 형식으로만 답하세요(다른 텍스트 없이):\n"
        '{"answer": "<운전자 질문에 대한 완전한 한국어 문장 답변>", '
        '"hazards": [<감지된 항목, "rain"|"tunnel"|"warning_light" 중에서만 선택. 없으면 빈 배열>]}'
    )


def _parse_vision_response(raw: str) -> PerceptionVisionResult:
    """
    Vision LLM 응답에서 answer/hazards 를 추출한다.
    구조화 JSON 파싱/검증에 실패하면 원본 텍스트를 answer로, hazards는 빈
    리스트로 폴백한다 (기존 자유 텍스트 응답과의 하위호환).
    """
    try:
        parsed = json.loads(extract_first_json_object(raw.strip()))
        result = PerceptionVisionResult.model_validate(parsed)
        result.hazards = _sanity_check_hazards(result.answer, result.hazards)
        return result
    except (json.JSONDecodeError, AttributeError, TypeError, ValidationError):
        return PerceptionVisionResult(answer=raw)


# answer(자유 텍스트)에 이 키워드가 있으면 VLM이 "위험 없음"이라고 서술한
# 것으로 간주해, hazards에 그와 모순되는 항목이 남아있으면 제거한다. VLM
# 환각(hallucination)에 대한 완벽한 해결책이 아니라 answer와 hazards가
# 서로 명백히 모순되는 케이스만 걸러내는 규칙 기반 필터다.
_CLEAR_WEATHER_KEYWORDS = ("맑", "화창", "clear", "비 안", "비가 안", "눈 안", "눈이 안")


def _sanity_check_hazards(text: str, hazards: List[str]) -> List[str]:
    """answer가 명시적으로 맑은 날씨를 서술하면 hazards의 'rain'을 제거한다."""
    if "rain" in hazards and any(kw in text for kw in _CLEAR_WEATHER_KEYWORDS):
        return [h for h in hazards if h != "rain"]
    return hazards


# 모듈 레벨 싱글턴 — execution.py 의 _EXTRACTION_LLM 과 동일한 lazy init 패턴.
_VISION_LLM: Optional[ChatOpenAI] = None


def _get_vision_llm() -> ChatOpenAI:
    """_VISION_LLM 싱글턴을 반환한다. 최초 호출 시 생성된다."""
    global _VISION_LLM
    if _VISION_LLM is None:
        _VISION_LLM = ChatOpenAI(
            model=QWEN_VL_MODEL_NAME, temperature=0.1, base_url=MODEL_SERVER_URL,
        )
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
        state.get("route_type"), QWEN_VL_MODEL_NAME,
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
        "perception_node 완료: hazards=%s answer=%r plan=%s",
        result.hazards, result.answer, plan_steps,
    )

    return {
        "context_data": {
            **context_data,
            "vision_results": {
                "status": "success",
                "hazards": result.hazards,
                "answer": result.answer,
            },
        },
        # hazard 감지 시 execution agent 가 바로 실행할 plan (route_after_perception 참고).
        # 없으면 빈 리스트로 명시 — 이전에 perception 위임용으로 쓰였던 stale plan을 정리한다.
        "plan": plan_steps,
    }
