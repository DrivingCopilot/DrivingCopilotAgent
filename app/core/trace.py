# app/core/trace.py
#
# 디버깅용 LLM/노드 실행 trace 레코더.
# LangChain BaseCallbackHandler를 상속해 on_chat_model_start / on_llm_end /
# on_llm_error 시점의 정보를 JSONL로 append한다. 노드 delta / 최종 state는
# record_node / record_final 공개 메서드로 외부(scripts/trace_query.py)에서 기록한다.

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


def _serialize_message(message: BaseMessage) -> dict[str, Any]:
    return {
        "role": getattr(message, "type", message.__class__.__name__),
        "content": message.content,
        "tool_calls": getattr(message, "tool_calls", None),
    }


class TraceRecorder(BaseCallbackHandler):
    """LangChain 콜백 이벤트와 그래프 노드 delta를 JSONL 파일에 append하는 레코더."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._lock = threading.Lock()
        self._start_times: dict[UUID, float] = {}

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self._output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.monotonic()
        invocation_params = kwargs.get("invocation_params", {}) or {}
        model = (
            invocation_params.get("model")
            or invocation_params.get("model_name")
            or (serialized.get("kwargs", {}) or {}).get("model_name")
            or (serialized.get("kwargs", {}) or {}).get("model")
        )
        self._write(
            {
                "kind": "chat_model_start",
                "ts": datetime.now().isoformat(),
                "run_id": run_id,
                "model": model,
                "messages": [
                    [_serialize_message(m) for m in batch] for batch in messages
                ],
                "tools": invocation_params.get("tools"),
            }
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._start_times.pop(run_id, None)
        duration_ms = (time.monotonic() - start) * 1000 if start is not None else None

        raw_output = [
            [
                {
                    "text": generation.text,
                    "message": _serialize_message(generation.message)
                    if hasattr(generation, "message")
                    else None,
                }
                for generation in batch
            ]
            for batch in response.generations
        ]
        tool_calls = [
            tc
            for batch in response.generations
            for generation in batch
            for tc in (getattr(getattr(generation, "message", None), "tool_calls", None) or [])
        ]

        self._write(
            {
                "kind": "llm_end",
                "ts": datetime.now().isoformat(),
                "run_id": run_id,
                "duration_ms": duration_ms,
                "raw_output": raw_output,
                "llm_output": response.llm_output,
                "tool_calls": tool_calls,
            }
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_times.pop(run_id, None)
        self._write(
            {
                "kind": "llm_error",
                "ts": datetime.now().isoformat(),
                "run_id": run_id,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    def record_node(self, node_name: str, delta: dict[str, Any]) -> None:
        """그래프 stream_mode="updates"에서 노드별 delta를 기록한다."""
        self._write(
            {
                "kind": "node",
                "ts": datetime.now().isoformat(),
                "node": node_name,
                "delta": delta,
            }
        )

    def record_final(self, state: dict[str, Any]) -> None:
        """그래프 stream_mode="values"의 마지막 스냅샷(최종 state)을 기록한다."""
        self._write(
            {
                "kind": "final",
                "ts": datetime.now().isoformat(),
                "state": state,
            }
        )
