# app/model_server/server.py
#
# Ollama를 대체하는 로컬 OpenAI-호환 추론 서버.
#
# Supervisor/Knowledge/Execution/Perception 4개 A2A 프로세스가 각각
# ChatOpenAI(base_url=MODEL_SERVER_URL)로 이 서버 하나에 요청을 보낸다 — GPU에는
# 7B(Qwen2-VL-7B-Instruct-GPTQ-Int4) + 1.5B(Qwen2.5-1.5B-Instruct) 모델이 이
# 프로세스에서 한 벌씩만 로드된다(app/model_server/backend.py).
#
# 지원 범위(app/agents/*.py 8개 호출 지점 기준):
#   - 순수 텍스트 생성 (execution/reflect/crag/finalize/text2sql)
#   - response_format=json_object|json_schema (crag/supervisor) — prompt 주입 기반
#     best-effort. Ollama의 grammar-constrained decoding과 동일한 보장은 없다.
#   - tools/tool_choice (knowledge의 create_react_agent) — Qwen 표준
#     <tool_call>{...}</tool_call> 블록을 OpenAI tool_calls 형식으로 파싱하고,
#     다음 턴 요청에 실려 돌아오는 assistant tool_calls/tool 메시지는
#     _normalize_message_for_text로 원형(템플릿이 읽는 형태)을 복원한다 —
#     안 그러면 모델이 자신의 과거 tool 호출/결과를 못 보고 매 턴 재호출한다.
#   - 멀티모달 image_url 콘텐츠 블록 (perception) — VL 모델 경로로 라우팅.
#   - stream=true (reflect.py의 llm.astream()) — SSE chat.completion.chunk.
#
# 실행:
#     python -m app.model_server.server

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request

from app.core.config import MODEL_SERVER_URL, QWEN_VL_MODEL_NAME
from app.core.json_utils import extract_first_json_object
from app.model_server import backend

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 로딩은 수십 초~수 분 걸릴 수 있어 백그라운드 스레드로 돌린다 —
    # 그래야 /health가 서버 기동 즉시 "loading"으로 응답하고, run_agents.sh의
    # 헬스체크 폴링 루프가 실제로 로딩 완료를 기다릴 수 있다.
    threading.Thread(target=backend.load_models, daemon=True).start()
    yield


app = FastAPI(title="Local Qwen Model Server (OpenAI-compatible)", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok" if backend.is_ready() else "loading"}


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _normalize_content_for_vl(content: Any) -> Any:
    """OpenAI 스타일 content 블록(image_url)을 Qwen2-VL 프로세서 형태로 변환한다.
    tool_calls를 담은 assistant 메시지는 content=None으로 오므로 빈 문자열로
    취급한다 — Qwen2-VL 채팅 템플릿은 content가 문자열이 아니면 무조건 순회
    가능한 블록 리스트로 가정하고 for-loop을 돌리므로, None을 그대로 넘기면
    'NoneType' object is not iterable 로 죽는다(_normalize_content_for_text와
    동일한 이유 — supervisor는 항상 QWEN_VL_MODEL_NAME을 쓰고 대화 히스토리
    전체(messages_to_send.extend(messages))를 함께 보내므로, knowledge의
    ReAct 루프가 state["messages"]에 남긴 tool_calls 메시지가 여기로도 들어온다)."""
    if content is None:
        return ""
    if not isinstance(content, list):
        return content
    normalized = []
    for block in content:
        if block.get("type") == "image_url":
            normalized.append({"type": "image", "image": block["image_url"]["url"]})
        else:
            normalized.append(block)
    return normalized


def _normalize_content_for_text(content: Any) -> str:
    """텍스트 모델 경로 — content가 블록 리스트면 text 블록만 이어붙인다.
    tool_calls를 담은 assistant 메시지는 content=None으로 오므로 빈 문자열로
    취급한다(str(None) == "None" 문자열이 그대로 들어가는 사고 방지)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
    return str(content)


def _normalize_tool_call_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI 왕복 규약은 function.arguments를 JSON 문자열로 담는다(_parse_tool_calls
    참고). Qwen 채팅 템플릿은 `tool_call.arguments | tojson`으로 객체를 렌더링하므로,
    문자열을 그대로 넘기면 따옴표로 한 번 더 감싸져(이중 인코딩) 모델이 자신이
    과거에 만든 tool_call을 스스로 알아보지 못한다 — 여기서 dict로 되돌린다."""
    function = dict(tool_call.get("function") or {})
    raw_args = function.get("arguments", "{}")
    if isinstance(raw_args, str):
        try:
            function["arguments"] = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            function["arguments"] = {}
    return {**tool_call, "function": function}


def _normalize_message_for_text(message: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI 포맷 메시지를 텍스트 모델의 Qwen 채팅 템플릿이 기대하는 형태로
    정규화한다. role/content만 남기고 재조립하면 assistant의 tool_calls와
    tool 메시지가 사라져, ReAct 에이전트(knowledge_node)가 자신이 이미 tool을
    호출·수신했다는 사실을 다음 턴에서 볼 수 없다 — 매 턴 같은 tool을 다시
    호출하며 recursion_limit까지 수렴하지 못하는 원인이었다.
    - assistant + tool_calls: content(falsy 허용) 그대로 유지, tool_calls는
      arguments를 dict로 되돌려 템플릿의 이중 인코딩을 막는다.
    - tool: content만 전달한다 — 템플릿은 tool_call_id/name을 쓰지 않고,
      인접한 tool 메시지들을 role만으로 <tool_response> 블록에 함께 묶는다.
    """
    role = message.get("role", "user")
    tool_calls = message.get("tool_calls")

    if role == "assistant" and tool_calls:
        return {
            "role": "assistant",
            "content": _normalize_content_for_text(message.get("content")),
            "tool_calls": [_normalize_tool_call_arguments(tc) for tc in tool_calls],
        }
    return {"role": role, "content": _normalize_content_for_text(message.get("content"))}


def _has_image(messages: List[Dict[str, Any]]) -> bool:
    return any(
        isinstance(m.get("content"), list)
        and any(b.get("type") == "image_url" for b in m["content"])
        for m in messages
    )


def _response_format_instruction(response_format: Optional[Dict[str, Any]]) -> Optional[str]:
    if not response_format:
        return None
    fmt_type = response_format.get("type")
    if fmt_type == "json_schema":
        schema = response_format.get("json_schema", {}).get("schema", {})
        return (
            "You MUST respond with ONLY a single JSON object that strictly matches "
            f"this JSON Schema, with no prose or markdown fences:\n{json.dumps(schema)}"
        )
    if fmt_type == "json_object":
        return "You MUST respond with ONLY a single valid JSON object, no prose or markdown fences."
    return None


def _apply_response_format(
    messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    instruction = _response_format_instruction(response_format)
    if not instruction:
        return messages
    return [*messages, {"role": "system", "content": instruction}]


def _parse_tool_calls(raw_text: str) -> Optional[List[Dict[str, Any]]]:
    """Qwen 표준 <tool_call>{...}</tool_call> 블록을 OpenAI tool_calls 형식으로 변환한다."""
    matches = _TOOL_CALL_RE.findall(raw_text)
    if not matches:
        return None

    tool_calls = []
    for block in matches:
        try:
            parsed = json.loads(extract_first_json_object(block.strip()))
        except (json.JSONDecodeError, AttributeError):
            continue
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": parsed.get("name", ""),
                "arguments": json.dumps(parsed.get("arguments", {})),
            },
        })
    return tool_calls or None


def _strip_tool_call_blocks(raw_text: str) -> str:
    return _TOOL_CALL_RE.sub("", raw_text).strip()


def _completion_payload(
    model: str, content: Optional[str], tool_calls: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream_chunk(model: str, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> str:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name: str = body.get("model", "")
    messages: List[Dict[str, Any]] = body.get("messages", [])
    temperature: float = float(body.get("temperature") or 0.0)
    max_tokens: int = int(body.get("max_tokens") or 512)
    stream: bool = bool(body.get("stream", False))
    response_format = body.get("response_format")
    tools = body.get("tools")

    # 모델명이 VL 별칭과 정확히 일치하지 않아도, 메시지에 이미지가 섞여 있으면
    # 무조건 VL 경로로 보낸다 — perception.py가 보내는 요청은 항상 이 케이스.
    is_vl = model_name == QWEN_VL_MODEL_NAME or _has_image(messages)

    if is_vl:
        vl_messages = _apply_response_format(
            [{"role": m.get("role", "user"), "content": _normalize_content_for_vl(m.get("content"))}
             for m in messages],
            response_format,
        )
        model = backend.get_vl_model()

        if stream:
            async def _gen():
                for token in model.stream_generate(vl_messages, max_new_tokens=max_tokens, temperature=temperature):
                    yield _stream_chunk(model_name, {"content": token})
                yield _stream_chunk(model_name, {}, finish_reason="stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")

        raw = model.generate(vl_messages, max_new_tokens=max_tokens, temperature=temperature)
        return JSONResponse(_completion_payload(model_name, raw, None))

    # 텍스트 모델 경로 (tool-calling 지원)
    text_messages = _apply_response_format(
        [_normalize_message_for_text(m) for m in messages],
        response_format,
    )
    model = backend.get_text_model()

    if stream:
        async def _gen():
            for token in model.stream_generate(text_messages, max_new_tokens=max_tokens, temperature=temperature):
                yield _stream_chunk(model_name, {"content": token})
            yield _stream_chunk(model_name, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    raw = model.generate(text_messages, tools=tools, max_new_tokens=max_tokens, temperature=temperature)
    tool_calls = _parse_tool_calls(raw) if tools else None
    content = None if tool_calls else _strip_tool_call_blocks(raw)
    return JSONResponse(_completion_payload(model_name, content, tool_calls))


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    port = urlparse(MODEL_SERVER_URL).port or 11500
    uvicorn.run(app, host="0.0.0.0", port=port)
