"""
tests/test_model_server.py

app/model_server/server.py 단위 테스트 — 특히 "침묵" 버그의 근본 원인이었던
두 지점을 회귀 방지한다:

1. _strip_tool_messages_for_vl: VL 경로가 tool_calls/tool 메시지를 그대로
   흘려보내 Qwen2-VL 채팅 템플릿이 예상 못한 구조를 만나 죽는 문제.
2. chat_completions 텍스트 경로: <tool_call> 태그는 있었지만 JSON 파싱에 전부
   실패했을 때, content가 조용히 빈 문자열이 되어 "정상 종료"처럼 보이던 문제.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.model_server.server import app, _strip_tool_messages_for_vl


# ---------------------------------------------------------------------------
# _strip_tool_messages_for_vl
# ---------------------------------------------------------------------------

class TestStripToolMessagesForVL:
    def test_keeps_plain_user_assistant_messages(self):
        messages = [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답변"},
        ]
        assert _strip_tool_messages_for_vl(messages) == messages

    def test_drops_assistant_message_with_tool_calls(self):
        messages = [
            {"role": "user", "content": "질문"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "vector_rag_search", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "매뉴얼 검색 결과"},
            {"role": "assistant", "content": "최종 답변"},
        ]
        filtered = _strip_tool_messages_for_vl(messages)
        assert filtered == [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "최종 답변"},
        ]

    def test_assistant_without_tool_calls_is_kept_even_if_falsy_content(self):
        # tool_calls가 없는 일반 assistant 메시지는 content가 비어있어도 유지한다
        # (필터링 대상은 role=="tool" 이거나 tool_calls가 있는 assistant뿐).
        messages = [{"role": "assistant", "content": ""}]
        assert _strip_tool_messages_for_vl(messages) == messages


# ---------------------------------------------------------------------------
# chat_completions — tool_call 파싱 실패 시 content 처리
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


class TestToolCallParseFailureFallback:
    def test_malformed_tool_call_does_not_produce_blank_content(self, client, monkeypatch):
        """
        모델이 <tool_call> 태그는 냈지만 안의 JSON이 깨진 경우 — 예전에는
        _strip_tool_call_blocks가 그대로 적용되어 content가 빈 문자열이 되고,
        knowledge_node의 ReAct 루프가 이를 "성공한 빈 답변"으로 오인해 사용자에게
        조용히 무응답을 전달하는 원인이었다. 지금은 원본을 그대로 남겨야 한다.
        """
        broken_raw = '<tool_call>{"name": "vector_rag_search", "arguments": {broken json'

        mock_text_model = MagicMock()
        mock_text_model.generate = MagicMock(return_value=broken_raw)
        monkeypatch.setattr("app.model_server.server.backend.get_text_model", lambda: mock_text_model)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "messages": [{"role": "user", "content": "질문"}],
                "tools": [{"type": "function", "function": {"name": "vector_rag_search"}}],
            },
        )
        assert resp.status_code == 200
        message = resp.json()["choices"][0]["message"]
        assert "tool_calls" not in message or message["tool_calls"] is None
        assert message["content"], f"content must not be blank, got {message['content']!r}"

    def test_well_formed_tool_call_still_parses_normally(self, client, monkeypatch):
        """정상적으로 파싱되는 tool_call은 기존과 동일하게 content=None, tool_calls 채움."""
        good_raw = '<tool_call>{"name": "vector_rag_search", "arguments": {"query": "주행거리"}}</tool_call>'

        mock_text_model = MagicMock()
        mock_text_model.generate = MagicMock(return_value=good_raw)
        monkeypatch.setattr("app.model_server.server.backend.get_text_model", lambda: mock_text_model)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "messages": [{"role": "user", "content": "질문"}],
                "tools": [{"type": "function", "function": {"name": "vector_rag_search"}}],
            },
        )
        assert resp.status_code == 200
        message = resp.json()["choices"][0]["message"]
        assert message["content"] is None
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "vector_rag_search"

    def test_no_tool_call_tags_at_all_behaves_as_before(self, client, monkeypatch):
        """애초에 <tool_call> 태그가 없는 정상 자연어 답변은 그대로 content로 나간다."""
        plain_raw = "네, 주행거리는 계기판에서 확인할 수 있습니다."

        mock_text_model = MagicMock()
        mock_text_model.generate = MagicMock(return_value=plain_raw)
        monkeypatch.setattr("app.model_server.server.backend.get_text_model", lambda: mock_text_model)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "messages": [{"role": "user", "content": "질문"}],
                "tools": [{"type": "function", "function": {"name": "vector_rag_search"}}],
            },
        )
        assert resp.status_code == 200
        message = resp.json()["choices"][0]["message"]
        assert message["content"] == plain_raw
        assert message.get("tool_calls") is None
