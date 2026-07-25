"""
tests/test_ws_silent_end.py

"침묵" 버그 회귀 방지 테스트 — type:"done" 프레임은 나가지만 type:"text" 프레임은
한 번도 안 나가 사용자가 그 턴에 대해 아무 응답도 못 받는 버그 클래스를 겨냥한다.

이번 대화에서 실제로 재현/수정된 지점 3곳을 각각 검증한다:
  A. supervisor_node의 LLM 호출(structured_llm.ainvoke) 자체가 예외를 던지는 경우
     (app/agents/supervisor.py의 raw exception 분기)
  B. supervisor_node가 next_agent="__end__"에 도달했는데 final_text 우선순위
     체인(vision/tool/knowledge/reasoning)이 전부 비어있는 경우
     (app/agents/supervisor.py의 최종 안전망)
  C. run_graph() 자체가 던진, 그래프 내부 어디서도 못 잡은 예외가
     app/api/websocket.py의 최외곽 핸들러까지 새는 경우

+ D. 위 세 지점을 개별로 고쳐도 그래프 나머지 구간(knowledge 실패 → CRAG →
     observe → reflect → supervisor 재시도 루프)이 여전히 올바르게 동작해
     결국 사용자에게 text가 도달하는지를 실제 supervisor_node를 그대로 써서
     end-to-end로 확인한다.

새로운 노드가 추가되거나 next_agent="__end__"로 가는 새 경로가 생겨도, 이
파일의 assert_text_before_done 헬퍼로 같은 패턴의 테스트를 계속 추가해 침묵
지점을 계속 감시할 수 있다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.graph.builder as builder_module
import app.a2a.dispatch as dispatch_module
from app.a2a.dispatch import dispatch_task
from app.graph.builder import run_graph


# ---------------------------------------------------------------------------
# 공용 헬퍼
# ---------------------------------------------------------------------------

def assert_text_before_done(frames: List[Dict[str, Any]]) -> None:
    """'done'이 나갔다면 그 전에 비어있지 않은 'text'가 최소 1번은 나갔어야 한다."""
    done_frames = [f for f in frames if f.get("type") == "done"]
    assert done_frames, f"expected the turn to end with a 'done' frame, got frames={frames}"
    text_frames = [
        f for f in frames if f.get("type") == "text" and str(f.get("data") or "").strip()
    ]
    assert text_frames, (
        "turn ended with 'done' but no non-empty 'text' frame was ever sent — "
        f"the user would see complete silence. captured frames={frames}"
    )


@pytest.fixture
def ws_capture(monkeypatch):
    """app.graph.ws.websocket_manager.send_status로 나가는 모든 프레임을 기록한다."""
    frames: List[Dict[str, Any]] = []

    async def _capture(message: str) -> None:
        frames.append(json.loads(message))

    monkeypatch.setattr("app.graph.ws.websocket_manager.send_status", _capture)
    return frames


def _mock_structured_llm(*, ainvoke_result=None, ainvoke_side_effect=None):
    """app.agents.supervisor.ChatOpenAI(...).with_structured_output(...).ainvoke(...) mock."""
    mock_structured = MagicMock()
    if ainvoke_side_effect is not None:
        mock_structured.ainvoke = AsyncMock(side_effect=ainvoke_side_effect)
    else:
        mock_structured.ainvoke = AsyncMock(return_value=ainvoke_result)
    mock_instance = MagicMock()
    mock_instance.with_structured_output = MagicMock(return_value=mock_structured)
    return MagicMock(return_value=mock_instance)


def _base_state(user_query: str) -> Dict[str, Any]:
    return {
        "messages": [HumanMessage(content=user_query)],
        "route_type": "",
        "plan": [],
        "next_agent": "",
        "tool_calls": [],
        "context_data": {},
        "error_count": {},
        "feedback": "",
    }


# ---------------------------------------------------------------------------
# A. supervisor_node — LLM 호출 자체가 예외
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_llm_exception_first_time_retries(ws_capture):
    # LLM 호출이 예외(잘린 JSON 등)를 던지면 — 즉시 포기하지 않고 한도 내에서
    # 재시도해야 한다(next_agent="supervisor", parameter 카운트 +1). 재시도 경로에서는
    # done을 보내지 않으므로 침묵 불변식(text-before-done)은 적용되지 않는다.
    from app.agents.supervisor import supervisor_node

    mock_cls = _mock_structured_llm(ainvoke_side_effect=RuntimeError("Invalid JSON: EOF"))

    with (
        patch("app.agents.supervisor.ChatOpenAI", new=mock_cls),
        patch(
            "app.agents.supervisor._a2a_client.fetch_all_cards",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await supervisor_node(_base_state("고속도로 주행 보조(HDA)는 어떻게 작동해?"))

    assert result["next_agent"] == "supervisor"           # 재시도
    assert result["error_count"]["parameter"] == 1
    done = [f for f in ws_capture if f.get("type") == "done"]
    assert not done, "재시도 턴에서는 done을 보내지 않아야 한다"


@pytest.mark.asyncio
async def test_supervisor_llm_exception_at_retry_limit_sends_text_before_done(ws_capture):
    # 재시도 한도(parameter=2)에 도달하면 — 침묵 없이 사용자에게 실패 안내 text를
    # 보낸 뒤 done으로 종료해야 한다.
    from app.agents.supervisor import supervisor_node

    mock_cls = _mock_structured_llm(ainvoke_side_effect=RuntimeError("Invalid JSON: EOF"))

    state = _base_state("고속도로 주행 보조(HDA)는 어떻게 작동해?")
    state["error_count"] = {"parameter": 1}  # 이미 1회 실패 → 이번 실패로 한도 도달

    with (
        patch("app.agents.supervisor.ChatOpenAI", new=mock_cls),
        patch(
            "app.agents.supervisor._a2a_client.fetch_all_cards",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await supervisor_node(state)

    assert result["next_agent"] == "__end__"
    assert result.get("messages"), "한도 초과 시 사용자에게 보인 답변이 대화 기록에 남아야 한다"
    assert result["messages"][0].content.strip()
    assert_text_before_done(ws_capture)


# ---------------------------------------------------------------------------
# B. supervisor_node — __end__ 인데 final_text 우선순위 체인이 전부 빈 경우
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_end_with_empty_everything_sends_nonblank_text(ws_capture):
    from app.agents.supervisor import supervisor_node, SupervisorDecision

    parsed = SupervisorDecision(reasoning="", plan=[], next_agent="__end__")
    mock_cls = _mock_structured_llm(
        ainvoke_result={"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}
    )

    with (
        patch("app.agents.supervisor.ChatOpenAI", new=mock_cls),
        patch(
            "app.agents.supervisor._a2a_client.fetch_all_cards",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        # vision/tool/knowledge 결과가 전부 없는 순수 잡담 종료 케이스
        result = await supervisor_node(_base_state("안녕"))

    assert result["next_agent"] == "__end__"
    assert result["messages"][0].content.strip()
    assert_text_before_done(ws_capture)


# ---------------------------------------------------------------------------
# C. websocket.py — run_graph()가 던진 예외가 그래프 밖으로 새는 경우
# ---------------------------------------------------------------------------

def test_websocket_handler_sends_text_when_run_graph_raises(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.graph import ws as graph_ws
    from app.api.websocket import router as ws_router, streamer_proxy

    # main.py의 lifespan과 동일하게 배선한다 — 그래야 supervisor_node 등이 쓰는
    # graph_ws.websocket_manager가 실제 WS 세션(streamer_proxy)으로 향한다.
    monkeypatch.setattr(graph_ws, "websocket_manager", streamer_proxy)

    app = FastAPI()
    app.include_router(ws_router)

    with patch(
        "app.api.websocket.run_graph",
        new_callable=AsyncMock,
        side_effect=RuntimeError("unexpected crash somewhere inside the graph"),
    ):
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"query": "에어컨 바람이 안 나와"}))
            frames: List[Dict[str, Any]] = []
            for _ in range(10):
                frame = json.loads(ws.receive_text())
                frames.append(frame)
                if frame.get("type") == "done":
                    break

    assert_text_before_done(frames)


# ---------------------------------------------------------------------------
# D. End-to-end — 실제 supervisor_node를 그대로 쓰고, knowledge만 "실패"로
#    stub해 CRAG/observe/reflect/재시도 루프를 실제로 통과시킨다. 이 대화에서
#    처음 보고된 증상(knowledge 반복 호출 후 완전 무응답)이 재현되지 않는지
#    확인하는 end-to-end 회귀 테스트.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_graph_cache():
    builder_module.build_graph.cache_clear()
    yield
    builder_module.build_graph.cache_clear()


@pytest.fixture
def route_a2a_through_dispatch(monkeypatch):
    """A2AClient.send_task를 실제 HTTP 대신 in-process dispatch_task로 우회한다."""

    class _InProcessA2AClient:
        def __init__(self, *args, **kwargs):
            pass

        async def send_task(self, request):
            return await dispatch_task(request)

    monkeypatch.setattr("app.graph.a2a_nodes.A2AClient", _InProcessA2AClient)


@pytest.mark.asyncio
async def test_knowledge_repeated_failure_still_reaches_user_with_text(
    ws_capture, route_a2a_through_dispatch, monkeypatch
):
    from app.agents.supervisor import SupervisorDecision

    # knowledge는 매번 "실패"를 반환한다 — knowledge_node가 blank/malformed 답변을
    # 만났을 때 실제로 반환하는 것과 동일한 계약(app/agents/knowledge.py 참고).
    async def failing_knowledge(state):
        return {
            "context_data": {
                "knowledge_failed": True,
                "last_knowledge_result": "[knowledge 실패] empty answer",
                "refined_query": "",
            },
            "tool_calls": [{
                "tool": "knowledge",
                "params": {},
                "result": "empty answer",
                "status": "error",
                "error_type": "parameter",
                "error_msg": "empty answer",
            }],
            "next_agent": "supervisor",
        }

    monkeypatch.setitem(dispatch_module._DISPATCH_TABLE, "knowledge", failing_knowledge)
    builder_module.build_graph.cache_clear()

    # supervisor는 실제 노드를 그대로 쓰되, LLM만 mock한다: 1차 위임 + (재시도
    # 한도 미달 시) 재위임까지 최대 2번 호출될 수 있으므로 넉넉히 준비한다.
    decision = SupervisorDecision(
        reasoning="에어컨 문제 원인을 매뉴얼에서 확인합니다.",
        plan=["에어컨 송풍 불량 원인 검색"],
        next_agent="knowledge",
    )
    llm_result = {"raw": AIMessage(content=""), "parsed": decision, "parsing_error": None}
    mock_cls = _mock_structured_llm(ainvoke_result=llm_result)

    with (
        patch("app.agents.supervisor.ChatOpenAI", new=mock_cls),
        patch(
            "app.agents.supervisor._a2a_client.fetch_all_cards",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        await run_graph("에어컨 바람이 안 나오는데, 어떤게 문제야?")

    # MAX_RETRY(parameter=2) 도달 시 observe가 __end__로 강제 라우팅하고
    # reflect의 failure 분기가 text+done을 보낸다 — 침묵 없이 종료돼야 한다.
    assert_text_before_done(ws_capture)
