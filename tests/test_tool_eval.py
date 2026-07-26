"""
tests/test_tool_eval.py

evaluation/run_tool_eval.py 의 score_tool_item() 순수 로직 단위 테스트.

run_execution 은 호출하지 않는다 — calls 를 직접 구성해 채점 함수 입출력만 검증한다
(mocking 도 하지 않는다: score_tool_item 은 외부 의존성이 없는 순수 함수).

calls 튜플은 extract_tool_invocations() 의 5-tuple 계약과 동일하게
(name, params, status, error_type, result_text) 형태로 구성한다.
"""

from __future__ import annotations

from evaluation.run_tool_eval import MCP_TOOLS, score_tool_item


def test_exact_match():
    """이름·파라미터가 완전히 일치하면 tool_correct/params_exact/params_effective 모두 True, failure=None."""
    calls = [("control_wiper", {"on": True}, "success", "", "와이퍼를 켰습니다.")]
    result = score_tool_item("control_wiper", {"on": True}, calls, "와이퍼를 켰습니다.")

    assert result["tool_correct"] is True
    assert result["params_exact"] is True
    assert result["params_effective"] is True
    assert result["failure"] is None


def test_no_tool_call():
    """calls 가 비어있으면 failure='no_tool_call', params_effective 도 params_exact 와 동일하게 False."""
    result = score_tool_item("control_wiper", {"on": True}, [])

    assert result["tool_correct"] is False
    assert result["params_exact"] is False
    assert result["params_effective"] is False
    assert result["result_exact"] is None
    assert result["failure"] == "no_tool_call"


def test_status_error_timeout():
    """status='error', error_type='timeout' 이면 failure='timeout' — 파라미터/결과 비교는 하지 않는다."""
    calls = [("control_wiper", {}, "error", "timeout", "")]
    result = score_tool_item("control_wiper", {"on": True}, calls, "와이퍼를 켰습니다.")

    assert result["failure"] == "timeout"
    assert result["error_type"] == "timeout"
    assert result["tool_correct"] is False
    assert result["params_exact"] is False
    assert result["params_effective"] is False
    assert result["result_exact"] is None


def test_hallucinated_tool():
    """MCP_TOOLS 밖의 이름이면 failure='hallucinated_tool'."""
    assert "teleport_car" not in MCP_TOOLS

    calls = [("teleport_car", {}, "success", "", "차를 순간이동했습니다.")]
    result = score_tool_item("control_climate", {"temperature": 22, "on": True}, calls)

    assert result["failure"] == "hallucinated_tool"
    assert result["tool_correct"] is False
    assert result["result_exact"] is None


def test_wrong_tool():
    """MCP_TOOLS 안이지만 expected_name 과 다르면 failure='wrong_tool'."""
    assert "control_window" in MCP_TOOLS

    calls = [("control_window", {"is_open": True}, "success", "", "창문을 열었습니다.")]
    result = score_tool_item("control_climate", {"temperature": 22, "on": True}, calls)

    assert result["failure"] == "wrong_tool"
    assert result["tool_correct"] is False
    assert result["result_exact"] is None


def test_control_climate_default_absorption():
    """on 이 생략돼도 mcp_server.py 기본값(on=True)과 일치하면 params_effective=True.

    params_exact 는 실제 호출 args 와 expected_params 의 리터럴 비교이므로,
    on 이 args 에 없으면(생략) expected_params 의 on:true 와 dict 자체가 다르므로 False.
    """
    calls = [("control_climate", {"temperature": 22}, "success", "", "에어컨을 켜고 온도를 22℃로 설정했어요.")]
    result = score_tool_item("control_climate", {"temperature": 22, "on": True}, calls,
                              "에어컨을 켜고 온도를 22℃로 설정했어요.")

    assert result["tool_correct"] is True
    assert result["params_exact"] is False
    assert result["params_effective"] is True
    assert result["failure"] is None


def test_param_mismatch():
    """이름은 맞지만 파라미터 값이 다르면(기본값 흡수로도 못 맞추면) failure='param_mismatch'."""
    calls = [("control_climate", {"temperature": 26, "on": True}, "success", "", "에어컨을 켜고 온도를 26℃로 설정했어요.")]
    result = score_tool_item("control_climate", {"temperature": 22, "on": True}, calls,
                              "에어컨을 켜고 온도를 22℃로 설정했어요.")

    assert result["tool_correct"] is True
    assert result["params_exact"] is False
    assert result["params_effective"] is False
    assert result["failure"] == "param_mismatch"
    assert result["result_exact"] is None


def test_result_exact_true():
    """tool/params 가 정확히 맞고 응답 문자열도 gold 와 완전히 같으면 result_exact=True."""
    calls = [("control_wiper", {"on": True}, "success", "", "와이퍼를 켰습니다.")]
    result = score_tool_item("control_wiper", {"on": True}, calls, "와이퍼를 켰습니다.")

    assert result["result_exact"] is True


def test_result_exact_false_on_content_drift():
    """tool/params 는 맞았는데 반환 문자열이 gold 와 다른 경우 — '성공으로 찍혔지만
    내용이 이상한' 케이스. tool_correct/params_effective 는 True 로 유지되면서
    result_exact 만 False 로 따로 잡혀야 한다."""
    calls = [("control_wiper", {"on": True}, "success", "", "와이퍼 속도를 2단으로 조절했습니다.")]
    result = score_tool_item("control_wiper", {"on": True}, calls, "와이퍼를 켰습니다.")

    assert result["tool_correct"] is True
    assert result["params_effective"] is True
    assert result["result_exact"] is False


def test_result_exact_none_on_error():
    """status='error' 면 이미 failure 로 분류되므로 result_exact 비교 자체를 하지 않고 None."""
    calls = [("control_wiper", {}, "error", "parameter", "")]
    result = score_tool_item("control_wiper", {"on": True}, calls, "와이퍼를 켰습니다.")

    assert result["result_exact"] is None
    assert result["failure"] == "parameter"
