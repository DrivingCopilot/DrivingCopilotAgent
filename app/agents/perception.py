# app/agents/perception.py
#
# Perception Agent — Qwen2-VL Vision 분석 담당.
#
# 흐름:
#     supervisor (next_agent=perception) → perception_node(state)
#         ├ Backend MCP 의 get_camera_frame 으로 카메라 프레임(base64 JPEG) 조회
#         ├ 프레임을 멀티모달 메시지로 감싸 Vision LLM(Qwen2-VL-7B-Instruct-GPTQ-Int4) 호출
#         ├ 응답을 description/hazards 로 구조화 파싱
#         └ 분석 결과를 context_data.vision_results 에 담아 supervisor 로 복귀
#             (hazard 감지 시 plan 에 자동 대응 조치를 채워 execution 으로 직행 — 아래 참고)
#
# 모델 라우팅: Supervisor/Execution/Reflect 와 동일한 7B 모델(QWEN_VL_MODEL_NAME)을
# 공유한다 — app/model_server 가 프로세스당 한 벌만 로드해 서빙하므로, 텍스트/비전
# 요청 모두 이 한 모델로 처리된다(app/core/config.py, app/model_server/server.py).
#
# 멀티모달 트리거(계획서 6번 항목): 비/터널/경고등 감지 시 execution agent의
# tool을 자동 호출해야 한다. supervisor(7B 모델)의 JSON 판단에 이 안전 트리거를
# 맡기면 그동안 겪은 반복 출력/판단 불안정 문제가 그대로 재현되므로, 이 판단은
# 여기서 코드로 결정적으로 처리하고 graph 레벨에서 execution 으로 직행시킨다
# (app/graph/builder.py 의 route_after_perception 참고). supervisor 는 그 결과를
# 받아서 사용자에게 요약만 한다.

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import MODEL_SERVER_URL, QWEN_VL_MODEL_NAME
from app.core.json_utils import extract_first_json_object
from app.core.mcp_client import call_mcp_tool_once
from app.graph import ws as _ws
from app.graph.state import AgentState

logger = logging.getLogger(__name__)

_VISION_PROMPT = (
    "당신은 차량 카메라 영상을 분석하는 비전 어시스턴트입니다. "
    "이 영상에서 날씨, 도로 상태, 위험 상황(보행자, 장애물, 경고등 등)을 분석하세요.\n\n"
    "다음 JSON 형식으로만 답하세요(다른 텍스트 없이):\n"
    '{"description": "<한국어로 간결한 설명>", "hazards": [<감지된 항목, '
    '"rain"|"tunnel"|"warning_light" 중에서만 선택. 없으면 빈 배열>]}'
)

# hazard → execution agent 가 수행할 plan step. execution._extract_tool_call 이
# 이 문자열을 보고 MCP tool/파라미터를 추출하므로, execution.py의 _TOOL_SIGNATURES
# 표기와 맞춰서 작성한다 (app/agents/execution.py 참고).
HAZARD_PLAN_STEPS: dict[str, str] = {
    "rain": "비가 감지되었습니다. 와이퍼를 켜세요. (control_wiper on=true)",
    "tunnel": "터널 진입이 감지되었습니다. 전조등을 켜세요. (control_lighting on=true)",
    "warning_light": "경고등이 감지되었습니다. 대시보드 경고등 상태를 조회하세요. (query_dashboard metric=warning_lights)",
}


def _parse_vision_response(raw: str) -> tuple[str, list[str]]:
    """
    Vision LLM 응답에서 description/hazards 를 추출한다.
    구조화 JSON 파싱에 실패하면 원본 텍스트를 description으로, hazards는 빈
    리스트로 폴백한다 (기존 자유 텍스트 응답과의 하위호환).
    """
    try:
        parsed = json.loads(extract_first_json_object(raw.strip()))
        description = parsed.get("description") or raw
        hazards = [h for h in parsed.get("hazards", []) if h in HAZARD_PLAN_STEPS]
        hazards = _sanity_check_hazards(description, hazards)
        return description, hazards
    except (json.JSONDecodeError, AttributeError, TypeError):
        return raw, []


# description(자유 텍스트)에 이 키워드가 있으면 VLM이 "위험 없음"이라고 서술한
# 것으로 간주해, hazards에 그와 모순되는 항목이 남아있으면 제거한다. VLM
# 환각(hallucination)에 대한 완벽한 해결책이 아니라 description과 hazards가
# 서로 명백히 모순되는 케이스만 걸러내는 규칙 기반 필터다.
_CLEAR_WEATHER_KEYWORDS = ("맑", "화창", "clear", "비 안", "비가 안", "눈 안", "눈이 안")


def _sanity_check_hazards(description: str, hazards: list[str]) -> list[str]:
    """description이 명시적으로 맑은 날씨를 서술하면 hazards의 'rain'을 제거한다."""
    if "rain" in hazards and any(kw in description for kw in _CLEAR_WEATHER_KEYWORDS):
        return [h for h in hazards if h != "rain"]
    return hazards


# 모듈 레벨 싱글턴 — execution.py 의 _EXTRACTION_LLM 과 동일한 lazy init 패턴.
_VISION_LLM: ChatOpenAI | None = None


def _get_vision_llm() -> ChatOpenAI:
    """_VISION_LLM 싱글턴을 반환한다. 최초 호출 시 생성된다."""
    global _VISION_LLM
    if _VISION_LLM is None:
        _VISION_LLM = ChatOpenAI(
            model=QWEN_VL_MODEL_NAME, temperature=0.1, base_url=MODEL_SERVER_URL,
        )
    return _VISION_LLM


async def perception_node(state: AgentState) -> dict[str, Any]:
    """Perception Agent 노드. 카메라 프레임을 조회해 Vision LLM 으로 분석한다."""
    context_data: dict[str, Any] = dict(state.get("context_data", {}))

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
            # execution.py와 동일한 계약으로 tool_calls에 append한다 — observe_node는
            # tool_calls[-1]만 보고 성공/실패를 판정하는데, perception이 여기 아무것도
            # 안 쓰면 observe가 몇 턴 전 다른 agent의 결과를 재관측하게 된다.
            "tool_calls": [{
                "tool": "perception",
                "params": {"camera_id": "front"},
                "result": error_msg,
                "status": "error",
                "error_type": error_type or "parameter",
                "error_msg": error_msg,
            }],
        }

    # ── 2. Vision LLM 호출 ───────────────────────────────────────────────────
    message = HumanMessage(content=[
        {"type": "text", "text": _VISION_PROMPT},
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
            "tool_calls": [{
                "tool": "perception",
                "params": {"camera_id": "front"},
                "result": str(exc),
                "status": "error",
                # observe.py의 MAX_RETRY 키(timeout/parameter/invalid_tool/sql)에
                # "vlm"은 없다 — observe가 알아서 "parameter"로 폴백하지만, 여기서
                # 명시적으로 넘겨 관측/로그의 일관성을 유지한다.
                "error_type": "parameter",
                "error_msg": str(exc),
            }],
        }

    logger.debug("perception_node: VLM 원본 응답=%r", response.content)

    description, hazards = _parse_vision_response(response.content)
    plan_steps = [HAZARD_PLAN_STEPS[h] for h in hazards]

    logger.info(
        "perception_node 완료: description=%r hazards=%s plan=%s",
        description, hazards, plan_steps,
    )

    return {
        "context_data": {
            **context_data,
            "vision_results": {
                "status": "success",
                "description": description,
                "hazards": hazards,
            },
        },
        # hazard 감지 시 execution agent 가 바로 실행할 plan (route_after_perception 참고).
        # 없으면 빈 리스트로 명시 — 이전에 perception 위임용으로 쓰였던 stale plan을 정리한다.
        "plan": plan_steps,
        # hazard 유무와 무관하게 append한다 — hazard 없으면 observe로 바로 가서
        # 이번 성공을 관측해야 하고(예전엔 여기서 안 써서 observe가 몇 턴 전
        # 다른 agent의 stale 결과를 재관측했다), hazard 있으면 execution으로
        # 직행하지만 execution이 뒤이어 자기 tool_calls를 또 append하므로 두
        # 항목이 순서대로 남는 것도 이력상 자연스럽다.
        "tool_calls": [{
            "tool": "perception",
            "params": {"camera_id": "front"},
            "result": description,
            "status": "success",
        }],
    }
