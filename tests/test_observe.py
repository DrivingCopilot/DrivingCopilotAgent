
import pytest

from app.agents.observe import _classify_last_tool, observe_node

# ---------------------------------------------------------------------------
# _classify_last_tool 테스트(6)
# ---------------------------------------------------------------------------

def test_classify_empty():
    assert _classify_last_tool([]) == ("empty", None)


def test_classify_success():
    result = _classify_last_tool([{"tool": "x", "status": "success", "result": "ok"}])
    assert result == ("success", None)


@pytest.mark.parametrize("error_type", ["timeout", "parameter", "invalid_tool", "sql"])
def test_classify_fail_known(error_type):
    entry = {"tool": "x", "status": "error", "error_type": error_type, "error_msg": "err"}
    result = _classify_last_tool([entry])
    assert result == ("fail", error_type)


def test_classify_fail_unknown_type():
    entry = {"tool": "x", "status": "error", "error_type": "weird", "error_msg": "err"}
    result = _classify_last_tool([entry])
    assert result == ("fail", "parameter"), "unknown error_type should fall back to 'parameter'"


def test_classify_missing_status():
    result = _classify_last_tool([{"tool": "x"}])
    assert result == ("unknown", None)


def test_classify_unrecognised_status():
    result = _classify_last_tool([{"tool": "x", "status": "pending"}])
    assert result == ("unknown", None)


# ---------------------------------------------------------------------------
# observe_node 테스트(13)
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
            {"tool": "wiper", "status": "error", "error_type": "parameter", "error_msg": "invalid"}
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
            {"tool": "wiper", "status": "error", "error_type": "parameter", "error_msg": "invalid"}
        ],
        "error_count": {"parameter": 1},
    }
    result = await observe_node(state)
    # 한도 초과 시 __end__ (reflect failure 분기에서 WebSocket 송신 처리)
    assert result["next_agent"] == "__end__"
    assert result["error_count"]["parameter"] == 2
    assert result["plan"] == []
    assert "messages" not in result
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
            {"tool": "mock_tool", "status": "error", "error_type": error_type, "error_msg": "err"}
        ],
        "error_count": {error_type: initial_count} if initial_count > 0 else {},
    }
    result = await observe_node(state)
    # 한도 초과 시 __end__
    assert result["next_agent"] == "__end__", f"{error_type}: expected __end__ at limit"
    assert result["error_count"][error_type] == final_count


@pytest.mark.asyncio
async def test_observe_fail_at_limit_sends_websocket():
    # WebSocket 송신은 reflect_node failure 분기에서 처리.
    # observe_node 자체는 WebSocket을 호출하지 않고 next_agent="__end__"만 반환.
    state = {
        "tool_calls": [
            {"tool": "wiper", "status": "error", "error_type": "parameter", "error_msg": "invalid"}
        ],
        "error_count": {"parameter": 1},
    }
    result = await observe_node(state)
    assert result["next_agent"] == "__end__"
    assert "messages" not in result
