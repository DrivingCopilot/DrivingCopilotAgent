import json

import pytest
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage

from app.agent.observe import observe_node, _classify_last_tool, MAX_RETRY


# ---------------------------------------------------------------------------
# _classify_last_tool
# ---------------------------------------------------------------------------

def test_classify_empty():
    assert _classify_last_tool([]) == ("empty", None)


def test_classify_success():
    result = _classify_last_tool([{"tool": "x", "status": "success", "result": "ok"}])
    assert result == ("success", None)


@pytest.mark.parametrize("error_type", ["timeout", "parameter", "invalid_tool", "sql"])
def test_classify_fail_known(error_type):
    entry = {"tool": "x", "status": "fail", "error_type": error_type, "error_msg": "err"}
    result = _classify_last_tool([entry])
    assert result == ("fail", error_type)


def test_classify_fail_unknown_type():
    entry = {"tool": "x", "status": "fail", "error_type": "weird", "error_msg": "err"}
    result = _classify_last_tool([entry])
    assert result == ("fail", "parameter"), "unknown error_type should fall back to 'parameter'"


def test_classify_missing_status():
    result = _classify_last_tool([{"tool": "x"}])
    assert result == ("unknown", None)


def test_classify_unrecognised_status():
    result = _classify_last_tool([{"tool": "x", "status": "pending"}])
    assert result == ("unknown", None)


# ---------------------------------------------------------------------------
# observe_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observe_empty():
    state = {"tool_calls": [], "error_count": {}}
    result = await observe_node(state)
    assert result["next_agent"] == "supervisor"
    assert result["feedback"] == ""


@pytest.mark.asyncio
async def test_observe_success():
    state = {
        "tool_calls": [{"tool": "x", "status": "success", "result": "ok"}],
        "error_count": {},
    }
    result = await observe_node(state)
    assert result["next_agent"] == "supervisor"
    assert result["feedback"] == ""


@pytest.mark.asyncio
async def test_observe_unknown():
    state = {"tool_calls": [{"tool": "x"}], "error_count": {}}
    result = await observe_node(state)
    assert result["next_agent"] == "supervisor"
    assert result["feedback"], "feedback should not be empty for unknown classification"
    assert "unknown" in result["feedback"].lower()


@pytest.mark.asyncio
async def test_observe_fail_below_limit():
    state = {
        "tool_calls": [
            {"tool": "wiper", "status": "fail", "error_type": "parameter", "error_msg": "invalid"}
        ],
        "error_count": {},
    }
    result = await observe_node(state)
    assert result["next_agent"] == "supervisor"
    assert result["error_count"]["parameter"] == 1
    feedback = result["feedback"]
    assert "wiper" in feedback
    assert "parameter" in feedback
    assert "(1/2)" in feedback
    assert "messages" not in result


@pytest.mark.asyncio
async def test_observe_fail_at_limit_triggers_end():
    state = {
        "tool_calls": [
            {"tool": "wiper", "status": "fail", "error_type": "parameter", "error_msg": "invalid"}
        ],
        "error_count": {"parameter": 1},
    }
    result = await observe_node(state)
    assert result["next_agent"] == "__end__"
    assert result["error_count"]["parameter"] == 2
    assert result["plan"] == []
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "Stopping" in result["feedback"]


@pytest.mark.parametrize(
    "error_type, initial_count, final_count",
    [
        ("timeout", 1, 2),
        ("parameter", 1, 2),
        ("invalid_tool", 0, 1),
        ("sql", 2, 3),
    ],
)
@pytest.mark.asyncio
async def test_observe_fail_each_type_limit(error_type, initial_count, final_count):
    state = {
        "tool_calls": [
            {"tool": "mock_tool", "status": "fail", "error_type": error_type, "error_msg": "err"}
        ],
        "error_count": {error_type: initial_count} if initial_count > 0 else {},
    }
    result = await observe_node(state)
    assert result["next_agent"] == "__end__", f"{error_type}: expected __end__ at limit"
    assert result["error_count"][error_type] == final_count


@pytest.mark.asyncio
async def test_observe_fail_at_limit_sends_websocket(monkeypatch):
    import app.agent.observe as observe_module

    mock_send = AsyncMock()
    monkeypatch.setattr(observe_module.websocket_manager, "send_status", mock_send)

    state = {
        "tool_calls": [
            {"tool": "wiper", "status": "fail", "error_type": "parameter", "error_msg": "invalid"}
        ],
        "error_count": {"parameter": 1},
    }
    await observe_node(state)

    assert mock_send.call_count == 2

    first_payload = json.loads(mock_send.call_args_list[0][0][0])
    assert first_payload["type"] == "text"
    assert "최대 재시도 횟수" in first_payload["data"]

    second_payload = json.loads(mock_send.call_args_list[1][0][0])
    assert second_payload["reason"] == "error_limit"
