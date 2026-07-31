"""
tests/test_knowledge.py

Knowledge Agent 노드(app/agents/knowledge.py) 단위 테스트.

핵심 원칙:
    knowledge_node는 실제 LLM(ChatOpenAI/vLLM), Qdrant, Neo4j, SQLite에 의존한다.
    단위 테스트에서는 이 외부 경계를 mock 처리하고, "노드의 오케스트레이션 로직"만 검증한다.
    즉, LLM 성능이 아니라 다음을 본다:
      - plan을 읽어 지시문을 만드는가
      - ReAct 에이전트를 호출하는가
      - 결과를 context_data에 올바르게 담는가
      - 완료된 plan 스텝을 pop 하는가
      - 항상 supervisor로 제어를 반환하는가
      - 실패 시 error_count / feedback 을 채우는가

실행:
    .venv/bin/python -m pytest tests/test_knowledge.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.knowledge import (
    _dedup_lines,
    _ensure_korean,
    _extract_query_terms,
    _extract_tool_context,
    _grounding_overlap,
    _is_probably_korean,
    _is_unusable_result,
    _verify_answer_grounded,
    graph_rag_search,
    knowledge_node,
    text_to_sql_query,
    vector_rag_search,
)

# ---------------------------------------------------------------------------
# 테스트용 가짜 ReAct 에이전트
# create_react_agent(...) 가 반환하는 객체를 대체한다.
# ainvoke 는 입력 messages 뒤에 (tool 출력 ToolMessage들 +) AI 최종 답변을 덧붙여
# 돌려준다(실제 ReAct 동작 모사). tool_outputs 를 주면 knowledge_node의 검색 단계가
# "실제로 tool을 호출해 grounding한" 것으로 인식된다(=결정적 검색 폴백을 타지 않음).
# ---------------------------------------------------------------------------

# 기본값: vector tool을 1번 호출해 근거 텍스트를 얻은 grounded ReAct 실행.
_DEFAULT_TOOL_OUTPUTS = [("vector_rag_search", "매뉴얼 근거 텍스트")]


class FakeReactAgent:
    def __init__(self, answer="엔진 경고등은 엔진 점검이 필요함을 의미합니다.",
                 raise_exc=None, return_empty=False, tool_outputs=None):
        self.answer = answer
        self.raise_exc = raise_exc
        self.return_empty = return_empty
        # None → 기본 grounded, [] → tool 미호출(hallucination) 모사
        self.tool_outputs = _DEFAULT_TOOL_OUTPUTS if tool_outputs is None else tool_outputs
        self.received = None  # ainvoke 에 전달된 payload 캡처

    async def ainvoke(self, payload):
        self.received = payload
        if self.raise_exc is not None:
            raise self.raise_exc
        msgs = list(payload["messages"])
        if self.return_empty:
            # 새 메시지를 생성하지 못한 경우(빈 응답) 모사
            return {"messages": msgs}
        new = [
            ToolMessage(content=content, name=name, tool_call_id=f"call_{i}")
            for i, (name, content) in enumerate(self.tool_outputs)
        ]
        new.append(AIMessage(content=self.answer))
        return {"messages": msgs + new}


def _patch_agent(fake_agent, *, fused="7B 융합 답변",
                 summarized="1.5B 요약 답변", det_ctx=None):
    """knowledge 모듈의 LLM/에이전트 생성 + 새 2단계(검색→합성) 헬퍼를 mock.

    - _deterministic_retrieve: ReAct가 tool 미호출 시의 안전망(실제 MCP 대신 stub).
    - _fuse_with_7b: 복잡 질의의 7B Graph Fusion/CoT 합성(실제 7B 호출 대신 stub).
    - _summarize_with_1_5b: 단순 질의의 1.5B 요약 fast path(실제 1.5B 호출 대신 stub).
    """
    if det_ctx is None:
        det_ctx = {"graph": "", "vector": "det-vector", "sql": "",
                   "all": "[Vector RAG]\ndet-vector"}
    return patch.multiple(
        "app.agents.knowledge",
        create_react_agent=MagicMock(return_value=fake_agent),
        ChatOpenAI=MagicMock(return_value=MagicMock()),
        _deterministic_retrieve=AsyncMock(return_value=det_ctx),
        _fuse_with_7b=AsyncMock(return_value=fused),
        _summarize_with_1_5b=AsyncMock(return_value=summarized),
        # 언어 보정은 passthrough로 stub(실제 번역 LLM 호출 방지) — 언어 판정/번역
        # 자체는 별도 단위 테스트에서 검증한다.
        _ensure_korean=AsyncMock(side_effect=lambda t: t),
        # 근거성 게이트도 기본 통과로 stub(실제 judge LLM 호출 방지) — 게이트 자체는
        # 별도 단위 테스트에서 검증한다.
        _verify_answer_grounded=AsyncMock(return_value=True),
    )


# ---------------------------------------------------------------------------
# 시나리오 1: Plan 기반 실행 (정상 경로)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_executes_plan_step():
    fake = FakeReactAgent(answer="엔진 경고등 설명입니다.")
    state = {
        "messages": [HumanMessage(content="엔진 경고등이 뭐야?")],
        "plan": ["매뉴얼에서 엔진 경고등 검색", "관련 부품 그래프 탐색"],
        "context_data": {},
        "error_count": {},
    }

    with _patch_agent(fake):
        result = await knowledge_node(state)

    # 지시문에 plan[0] 이 포함되어 에이전트로 전달되었는가
    last_instruction = fake.received["messages"][-1].content
    assert "매뉴얼에서 엔진 경고등 검색" in last_instruction

    # 완료된 첫 스텝이 pop 되어 나머지만 남는가
    assert result["plan"] == ["관련 부품 그래프 탐색"]

    # 결과가 context_data 에 저장되는가
    assert result["context_data"]["last_knowledge_result"] == "엔진 경고등 설명입니다."

    # 항상 supervisor 로 복귀
    assert result["next_agent"] == "supervisor"

    # 마지막 메시지가 에이전트의 최종 답변인가
    assert result["messages"][-1].content == "엔진 경고등 설명입니다."


# ---------------------------------------------------------------------------
# 시나리오 2: Plan 없음 (기본 경로)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_without_plan_uses_default_instruction():
    fake = FakeReactAgent(answer="기본 질의 응답입니다.")
    state = {
        "messages": [HumanMessage(content="가까운 정비소 알려줘")],
        "plan": [],
        "context_data": {},
        "error_count": {},
    }

    with _patch_agent(fake):
        result = await knowledge_node(state)

    last_instruction = fake.received["messages"][-1].content
    assert "latest query" in last_instruction  # 기본 지시문 사용
    assert result["plan"] == []                # plan 은 비어 있는 그대로
    assert result["next_agent"] == "supervisor"
    assert result["context_data"]["last_knowledge_result"] == "기본 질의 응답입니다."


# ---------------------------------------------------------------------------
# 시나리오 3: context_data 병합 안전성 (기존 결과 보존)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_preserves_existing_context():
    fake = FakeReactAgent(answer="새 검색 결과")
    state = {
        "messages": [HumanMessage(content="질문")],
        "plan": [],
        "context_data": {
            "vector_results": ["기존 벡터 결과"],
            "vehicle_state": {"speed": 60},
        },
        "error_count": {},
    }

    with _patch_agent(fake):
        result = await knowledge_node(state)

    ctx = result["context_data"]
    # 기존 키가 덮어써지지 않고 보존되어야 한다
    assert ctx["vector_results"] == ["기존 벡터 결과"]
    assert ctx["vehicle_state"] == {"speed": 60}
    # 새 결과도 함께 존재
    assert ctx["last_knowledge_result"] == "새 검색 결과"


# ---------------------------------------------------------------------------
# 시나리오 4: 예외 처리 (Retry 로직)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_handles_agent_exception():
    fake = FakeReactAgent(raise_exc=RuntimeError("vLLM endpoint down"))
    state = {
        "messages": [HumanMessage(content="질문")],
        "plan": ["검색 스텝"],
        "context_data": {},
        "error_count": {"parameter": 1},  # 이미 1회 실패한 상태
    }

    with _patch_agent(fake):
        result = await knowledge_node(state)

    # knowledge_node 는 error_count 를 직접 올리지 않는다(설계: observe_node 가
    # tool_calls 를 보고 유일하게 카운트해 중복 집계를 막는다). 대신 실패를
    # error tool_call 로 기록하고, CRAG 우회 플래그(knowledge_failed)를 세운다.
    err_tc = result["tool_calls"][-1]
    assert err_tc["status"] == "error"
    assert err_tc["error_type"] == "parameter"
    assert "vLLM endpoint down" in err_tc["error_msg"]
    assert result["context_data"]["knowledge_failed"] is True
    # Reflexion 용 feedback 이 채워졌는가
    assert "vLLM endpoint down" in result["feedback"]
    # 실패해도 제어는 supervisor 로
    assert result["next_agent"] == "supervisor"
    # 에러 안내 메시지를 반환
    assert "error" in result["messages"][0].content.lower()


# ---------------------------------------------------------------------------
# 시나리오 5: 빈 응답 fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_empty_response_falls_back_to_retrieval_and_fusion():
    # ReAct가 아무 메시지도 못 만든 경우(빈 응답) — 조용히 종료하지 않고 결정적
    # 검색(_deterministic_retrieve)으로 context를 확보한 뒤 합성해 답을 만든다.
    fake = FakeReactAgent(return_empty=True)
    state = {
        "messages": [HumanMessage(content="질문")],
        "plan": [],
        "context_data": {},
        "error_count": {},
    }

    # 단순 질의(graph 근거 없음)의 폴백 요약은 1.5B가 담당(설계 스펙: "요약은 1.5B").
    with _patch_agent(fake, summarized="폴백 요약 답변"):
        result = await knowledge_node(state)

    assert result["context_data"]["last_knowledge_result"] == "폴백 요약 답변"
    assert result["context_data"]["knowledge_failed"] is False
    assert result["next_agent"] == "supervisor"


# ---------------------------------------------------------------------------
# 시나리오 5-1: 빈/깨진 최종 답변 — "침묵" 버그 회귀 방지
#
# model_server의 tool_call JSON 파싱이 전부 실패하면 content가 조용히 빈
# 문자열로 새어나오거나(app/model_server/server.py 참고), 파싱 안 된 <tool_call>
# 태그가 그대로 노출될 수 있다. new_messages 자체는 비어있지 않으므로(시나리오
# 5의 return_empty와 다름) "Knowledge retrieval completed." 폴백을 안 타고,
# 예외 없이 "성공"으로 통과하면 observe/CRAG의 실패 감지를 모두 우회해
# 사용자에게 완전 침묵으로 이어진다 — except 분기와 동일한 실패 계약으로
# 처리돼야 한다.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_answer",
    [
        "",
        "   \n  ",
        '<tool_call>{"name": "vector_rag_search", "arguments": {broken',
    ],
    ids=["blank", "whitespace_only", "leaked_tool_call_tag"],
)
async def test_knowledge_node_blank_or_malformed_answer_is_treated_as_failure(bad_answer):
    # 검색 근거가 전혀 없는데(tool 미호출 + 결정적 검색도 빈 결과) 최종 답변까지
    # 비었거나 깨진 경우 — 합성으로도 복구 불가하므로 명시적 실패여야 한다.
    # (근거가 있으면 blank 답변은 융합으로 복구되는 것이 정상 — 별도 테스트 참조.)
    fake = FakeReactAgent(answer=bad_answer, tool_outputs=[])
    state = {
        "messages": [HumanMessage(content="질문")],
        "plan": ["검색 스텝"],
        "context_data": {},
        "error_count": {},
    }

    empty_ctx = {"graph": "", "vector": "", "sql": "", "all": ""}
    with _patch_agent(fake, det_ctx=empty_ctx):
        result = await knowledge_node(state)

    assert result["context_data"]["knowledge_failed"] is True
    assert result["tool_calls"][0]["status"] == "error"
    assert result["tool_calls"][0]["error_type"] == "parameter"
    # 침묵 방지 핵심: 실패해도 next_agent는 반드시 supervisor로 돌아가야
    # observe/reflect가 재시도·최종 실패 메시지를 처리할 수 있다.
    assert result["next_agent"] == "supervisor"


# ---------------------------------------------------------------------------
# 시나리오 6: Tool - MCP 위임 (vector / graph)
# 검색 로직은 공용 MCP 서버에 있으므로, 툴은 call_mcp_tool_once 로 위임만 한다.
# 여기서는 "올바른 tool 이름/파라미터로 호출하고 결과 텍스트를 그대로 반환하는가"를 본다.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vector_rag_delegates_to_mcp():
    mock_call = AsyncMock(return_value=("매뉴얼 검색 결과입니다.", "success", "", ""))
    with patch("app.agents.knowledge.call_mcp_tool_once", mock_call):
        result = await vector_rag_search.ainvoke({"query": "엔진 경고등"})

    # 올바른 MCP tool 이름과 파라미터로 호출했는가
    mock_call.assert_awaited_once_with("vector_rag_search", {"query": "엔진 경고등"})
    # 서버 결과를 그대로 반환
    assert result == "매뉴얼 검색 결과입니다."


# ---------------------------------------------------------------------------
# 시나리오 7: CRAG 연동 (crag_query 기록 / 재검색 / 실패 시 상태 초기화)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_node_records_crag_query_for_grader():
    # 멀티스텝 plan 에서 완료 스텝을 pop 하더라도, grade/transform 이 평가할
    # 쿼리(crag_query)에는 '실제 실행된 스텝'이 기록되어야 한다.
    fake = FakeReactAgent(answer="결과")
    state = {
        "messages": [HumanMessage(content="원본 질문")],
        "plan": ["스텝A", "스텝B"],
        "context_data": {},
        "error_count": {},
    }
    with _patch_agent(fake):
        result = await knowledge_node(state)

    assert result["context_data"]["crag_query"] == "스텝A"  # 실행된 스텝 기록
    assert result["plan"] == ["스텝B"]                       # 완료 스텝 pop


@pytest.mark.asyncio
async def test_knowledge_node_reretrieval_uses_refined_query_keeps_plan():
    fake = FakeReactAgent(answer="재검색 결과")
    state = {
        "messages": [HumanMessage(content="원본")],
        "plan": ["스텝A"],
        "context_data": {"refined_query": "재작성된 쿼리", "crag_query": "스텝A", "crag_attempts": 1},
        "error_count": {},
    }
    with _patch_agent(fake):
        result = await knowledge_node(state)

    # 재검색 지시문에 refined_query 사용
    assert "재작성된 쿼리" in fake.received["messages"][-1].content
    # plan 재-pop 안 함, refined_query 소비, crag_query 원본 유지
    assert result["plan"] == ["스텝A"]
    assert result["context_data"]["refined_query"] == ""
    assert result["context_data"]["crag_query"] == "스텝A"


@pytest.mark.asyncio
async def test_knowledge_node_exception_clears_refined_query():
    # 재검색 중 실패해도 refined_query 를 초기화해 다음 위임으로 새지 않게 한다.
    fake = FakeReactAgent(raise_exc=RuntimeError("boom"))
    state = {
        "messages": [HumanMessage(content="질문")],
        "plan": ["스텝"],
        "context_data": {"refined_query": "재작성"},
        "error_count": {},
    }
    with _patch_agent(fake):
        result = await knowledge_node(state)

    assert result["context_data"]["refined_query"] == ""


# ---------------------------------------------------------------------------
# 시나리오 8: 모델별 역할 분담 (단순검색 1.5B / 복잡 Graph Fusion·CoT 7B)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_tool_call_triggers_deterministic_retrieval():
    # 1.5B ReAct가 tool을 한 번도 안 부르고 바로 답을 내면(에어백 오답 = 검색 없는
    # hallucination) 그 답을 신뢰하지 않고 결정적 검색으로 grounding해야 한다.
    fake = FakeReactAgent(answer="이 차에는 에어백이 없습니다.", tool_outputs=[])
    state = {
        "messages": [HumanMessage(content="이 차에는 어떤 종류의 에어백이 있어?")],
        "plan": ["에어백 종류 검색"],
        "context_data": {},
        "error_count": {},
    }

    det = {"graph": "(운전석 에어백)-[:HAS_PART]-(에어백 시스템)", "vector": "",
           "sql": "", "all": "[Graph RAG]\n(운전석 에어백)-[:HAS_PART]-(에어백 시스템)"}
    with _patch_agent(fake, fused="이 차량에는 운전석/동승석/커튼 에어백이 있습니다.", det_ctx=det) as _:
        result = await knowledge_node(state)

    # hallucination 답("에어백이 없습니다")이 사용자에게 가지 않고, 검색 기반
    # 합성 답으로 대체됐는가.
    assert "에어백이 없습니다" not in result["context_data"]["last_knowledge_result"]
    assert result["context_data"]["last_knowledge_result"] == "이 차량에는 운전석/동승석/커튼 에어백이 있습니다."
    assert result["context_data"]["knowledge_failed"] is False
    assert result["next_agent"] == "supervisor"


@pytest.mark.parametrize(
    "query,expected_in",
    [
        ("시동을 껐는데도 선루프가 작동돼?", "선루프"),
        ("뒷좌석을 수동으로 접으려면 어떻게 해?", "뒷좌석"),
        ("HUD가 뭐고 어떻게 켜?", "HUD"),
        ("이 차에는 어떤 종류의 에어백이 있어?", "에어백"),
    ],
)
def test_extract_query_terms_strips_josa(query, expected_in):
    # 조사가 붙은 자연어에서 깨끗한 엔티티가 추출돼야 graph substring 매칭이 된다
    # ('선루프가'→'선루프', '뒷좌석을'→'뒷좌석', 'HUD가'→'HUD').
    terms = _extract_query_terms(query)
    assert expected_in in terms, f"{query!r} → {terms}"
    # 의문사/기능어는 엔티티로 안 뽑힌다
    assert "어떻게" not in terms and "뭐야" not in terms


@pytest.mark.parametrize(
    "text,unusable",
    [
        ("'hud가' 관련 그래프 관계를 찾지 못했습니다.", True),
        ("검색할 엔티티가 없습니다.", True),
        ("[vector_rag_search 실패: timeout] 10초 초과", True),
        ("", True),
        ("## 그래프 관계\n(선루프)-[:HAS_PART]-(선루프 시스템)", False),
    ],
)
def test_is_unusable_result(text, unusable):
    assert _is_unusable_result(text) is unusable


@pytest.mark.parametrize(
    "text,is_korean",
    [
        ("배터리 취급 시 절연장갑을 착용하세요.", True),
        ("When handling the battery, wear insulating gloves.", False),
        ("타이어 압력은 33 psi 입니다.", True),          # 한글+숫자 혼합 → 한국어
        ("33.0/33.0/32.0/33.0", True),                    # 숫자/기호만 → 판정 제외(통과)
        ("", True),                                        # 빈값 → 별도 가드가 처리
        ("HDA는 ADAS의 한 기능입니다.", True),            # 소량 영어 약어 섞여도 한국어
    ],
)
def test_is_probably_korean(text, is_korean):
    assert _is_probably_korean(text) is is_korean


@pytest.mark.asyncio
async def test_ensure_korean_translates_english_answer():
    # 최종 답변이 영어로 유출되면 결정적으로 한국어 번역해야 한다.
    english = "When handling the battery, wear insulating gloves and avoid short circuits."
    with patch(
        "app.agents.knowledge._translate_to_korean",
        new=AsyncMock(return_value="배터리를 다룰 때는 절연장갑을 착용하고 단락을 피하세요."),
    ) as translate_mock:
        result = await _ensure_korean(english)
    assert result == "배터리를 다룰 때는 절연장갑을 착용하고 단락을 피하세요."
    translate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_korean_keeps_korean_answer_without_calling_translator():
    # 이미 한국어면 번역 LLM을 호출하지 않고 그대로 반환(불필요한 지연 방지).
    korean = "배터리를 다룰 때는 절연장갑을 착용하세요."
    with patch(
        "app.agents.knowledge._translate_to_korean",
        new=AsyncMock(return_value="쓰이면 안 됨"),
    ) as translate_mock:
        result = await _ensure_korean(korean)
    assert result == korean
    translate_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_usable_context_fails_instead_of_shipping_hallucination():
    # 1.5B가 tool을 안 부르고(폴백), 결정적 검색도 graph/vector 모두 무결과면 —
    # grounding 근거가 0이므로 1.5B의 미검증 답변(환각)을 내보내지 말고 실패 처리해야
    # 한다(CRAG 재검색/observe가 처리). 그럴듯한 환각이 사용자에게 가면 안 된다.
    fake = FakeReactAgent(answer="이 차에는 그런 기능이 없습니다.", tool_outputs=[])
    state = {
        "messages": [HumanMessage(content="선루프 자동 개폐 되나?")],
        "plan": ["선루프 검색"],
        "context_data": {},
        "error_count": {},
    }
    empty_ctx = {"graph": "", "vector": "", "sql": "", "all": ""}
    with _patch_agent(fake, det_ctx=empty_ctx):
        result = await knowledge_node(state)

    assert result["context_data"]["knowledge_failed"] is True
    assert "없습니다" not in (result["context_data"]["last_knowledge_result"] or "").replace("[knowledge 실패]", "")
    assert result["next_agent"] == "supervisor"


@pytest.mark.asyncio
async def test_7b_fusion_prompt_echo_falls_back_not_leaked():
    # 7B 융합 모델이 답변 대신 입력 프롬프트('[User Query]...[Retrieved Context]...')를
    # 그대로 echo하면, 그 echo가 사용자에게 노출되면 안 된다 — 1.5B ReAct 답변으로
    # 폴백해야 한다(가드가 malformed로 잡아 실패시키거나 폴백값 사용).
    fake = FakeReactAgent(
        answer="트립 컴퓨터에서 확인할 수 있습니다.",  # 1.5B 폴백 답변
        tool_outputs=[("graph_rag_search", "(선루프)-[:HAS_PART]-(선루프 시스템)")],
    )
    state = {
        "messages": [HumanMessage(content="선루프 종류가 뭐야?")],
        "plan": ["선루프 관계 탐색"],
        "context_data": {},
        "error_count": {},
    }

    echoed = "[User Query]\n선루프 종류가 뭐야?\n\n[Retrieved Context]\n[Graph RAG]\n(선루프)..."
    with patch.multiple(
        "app.agents.knowledge",
        create_react_agent=MagicMock(return_value=fake),
        ChatOpenAI=MagicMock(return_value=MagicMock()),
        _deterministic_retrieve=AsyncMock(),
        _fuse_with_7b=AsyncMock(return_value=echoed),
    ):
        result = await knowledge_node(state)

    ans = result["context_data"]["last_knowledge_result"]
    assert "[Retrieved Context]" not in ans  # 프롬프트 echo가 노출되면 안 됨
    assert "[User Query]" not in ans
    assert ans == "트립 컴퓨터에서 확인할 수 있습니다."  # 1.5B 폴백 답변 사용
    assert result["next_agent"] == "supervisor"


@pytest.mark.asyncio
async def test_graph_present_without_keyword_skips_7b_fusion():
    # (Phase 4) graph 결과가 있어도 관계형 키워드가 없는 단순 절차 질의는 7B fusion을
    # 강제하지 않고 1.5B ReAct 답변을 그대로 쓴다(broad-match graph_present로 인한
    # fusion 남발·지연·7B 500 방지). graph 근거는 context로 들어가 grounding은 유지.
    fake = FakeReactAgent(
        answer="트립 버튼을 짧게 누르면 표시됩니다.",
        tool_outputs=[("graph_rag_search", "(트립 버튼)-[:HAS_PART]-(계기판)")],
    )
    state = {
        "messages": [HumanMessage(content="주행거리는 어디서 확인해?")],  # 관계형 키워드 없음
        "plan": ["주행거리 확인 방법"],
        "context_data": {},
        "error_count": {},
    }
    fuse_mock = AsyncMock(return_value="쓰이면 안 됨")
    with patch.multiple(
        "app.agents.knowledge",
        create_react_agent=MagicMock(return_value=fake),
        ChatOpenAI=MagicMock(return_value=MagicMock()),
        _deterministic_retrieve=AsyncMock(),
        _fuse_with_7b=fuse_mock,
        _verify_answer_grounded=AsyncMock(return_value=True),
    ):
        result = await knowledge_node(state)
    assert result["context_data"]["last_knowledge_result"] == "트립 버튼을 짧게 누르면 표시됩니다."
    fuse_mock.assert_not_awaited()  # 키워드 없으면 graph_present여도 fusion 미호출


@pytest.mark.asyncio
async def test_complex_relational_query_uses_7b_fusion():
    # graph tool 결과가 있으면(관계형 데이터) 복잡 질의로 보고 7B 융합/CoT 경로를 탄다.
    fake = FakeReactAgent(
        answer="1.5B 초안 답변",
        tool_outputs=[("graph_rag_search", "(운전석 에어백)-[:HAS_PART]-(에어백 시스템)")],
    )
    state = {
        "messages": [HumanMessage(content="에어백 종류가 뭐가 있어?")],
        "plan": ["에어백 부품 관계 탐색"],
        "context_data": {},
        "error_count": {},
    }

    fuse_mock = AsyncMock(return_value="융합된 최종 답변")
    with patch.multiple(
        "app.agents.knowledge",
        create_react_agent=MagicMock(return_value=fake),
        ChatOpenAI=MagicMock(return_value=MagicMock()),
        _deterministic_retrieve=AsyncMock(),  # 호출되면 안 됨(tool_ctx 이미 채워짐)
        _fuse_with_7b=fuse_mock,
    ):
        result = await knowledge_node(state)

    # 7B 융합 결과가 최종 답변이 되고, 결정적 검색 폴백은 타지 않았다.
    assert result["context_data"]["last_knowledge_result"] == "융합된 최종 답변"
    fuse_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_simple_query_keeps_1_5b_answer_without_fusion():
    # graph 결과가 없고 관계형 키워드도 없는 단순 질의는 1.5B ReAct 답변을 그대로
    # 쓰고 7B 융합을 호출하지 않는다.
    fake = FakeReactAgent(
        answer="트립 버튼을 짧게 누르면 주행거리가 표시됩니다.",
        tool_outputs=[("vector_rag_search", "주행거리 표시 방법 매뉴얼 발췌")],
    )
    state = {
        "messages": [HumanMessage(content="주행거리는 어디서 확인해?")],
        "plan": ["주행거리 확인 방법 검색"],
        "context_data": {},
        "error_count": {},
    }

    fuse_mock = AsyncMock(return_value="이건 쓰이면 안 됨")
    with patch.multiple(
        "app.agents.knowledge",
        create_react_agent=MagicMock(return_value=fake),
        ChatOpenAI=MagicMock(return_value=MagicMock()),
        _deterministic_retrieve=AsyncMock(),
        _fuse_with_7b=fuse_mock,
    ):
        result = await knowledge_node(state)

    assert result["context_data"]["last_knowledge_result"] == "트립 버튼을 짧게 누르면 주행거리가 표시됩니다."
    fuse_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_simple_ungrounded_fallback_summarizes_with_1_5b_not_7b():
    # 1.5B ReAct가 tool을 안 불러(ungrounded) 결정적 검색으로 grounding했지만,
    # graph 근거가 없고 관계형 키워드도 없는 단순 질의라면 — 7B fusion(느림)이
    # 아니라 1.5B 요약(fast path)으로 검색 context를 정리해야 한다(설계 스펙).
    fake = FakeReactAgent(answer="근거 없는 답", tool_outputs=[])  # ungrounded
    state = {
        "messages": [HumanMessage(content="배터리를 다룰 때 안전상 주의할 점이 뭐야?")],
        "plan": ["배터리 안전 주의사항 검색"],
        "context_data": {},
        "error_count": {},
    }

    # 결정적 검색은 vector만 확보(graph 없음) → is_complex=False → 1.5B 요약 경로.
    det = {"graph": "", "vector": "배터리 취급 시 절연장갑 착용, 단자 단락 주의",
           "sql": "", "all": "[Vector RAG]\n배터리 취급 시 절연장갑 착용, 단자 단락 주의"}
    fuse_mock = AsyncMock(return_value="이건 쓰이면 안 됨")
    summarize_mock = AsyncMock(return_value="배터리 취급 시 절연장갑을 착용하고 단자 단락에 주의하세요.")
    with patch.multiple(
        "app.agents.knowledge",
        create_react_agent=MagicMock(return_value=fake),
        ChatOpenAI=MagicMock(return_value=MagicMock()),
        _deterministic_retrieve=AsyncMock(return_value=det),
        _fuse_with_7b=fuse_mock,
        _summarize_with_1_5b=summarize_mock,
    ):
        result = await knowledge_node(state)

    assert result["context_data"]["last_knowledge_result"] == "배터리 취급 시 절연장갑을 착용하고 단자 단락에 주의하세요."
    summarize_mock.assert_awaited_once()   # 1.5B fast path 사용
    fuse_mock.assert_not_awaited()         # 7B fusion은 안 탐


@pytest.mark.asyncio
async def test_graph_rag_forwards_entities():
    mock_call = AsyncMock(return_value=("관계 정보", "success", "", ""))
    with patch("app.agents.knowledge.call_mcp_tool_once", mock_call):
        result = await graph_rag_search.ainvoke(
            {"query": "엔진 경고등 부품", "entities": ["엔진", "점화플러그"]}
        )

    mock_call.assert_awaited_once_with(
        "graph_rag_search", {"query": "엔진 경고등 부품", "entities": ["엔진", "점화플러그"]}
    )
    assert result == "관계 정보"


@pytest.mark.asyncio
async def test_graph_rag_auto_extracts_entities_when_unspecified():
    """entities 미지정 시 조사/의문사를 제거한 깨끗한 term을 자동 추출해 넘긴다
    (백엔드가 query.split() 조사포함 매칭으로 엔티티명과 안 맞는 문제 방지)."""
    mock_call = AsyncMock(return_value=("관계 정보", "success", "", ""))
    with patch("app.agents.knowledge.call_mcp_tool_once", mock_call):
        await graph_rag_search.ainvoke({"query": "타이어 공기압은 어떻게 점검해?"})

    _, params = mock_call.await_args.args
    # '어떻게'(의문사)는 제외, '타이어'/'공기압'(조사 '은' 제거)은 포함되어야 한다.
    assert "어떻게" not in params["entities"]
    assert any("타이어" in e for e in params["entities"])
    assert any("공기압" in e for e in params["entities"])


@pytest.mark.parametrize("text,expected", [
    ("a\nb\na\nc\nb", "a\nb\nc"),                       # 정확 중복 라인 제거(순서 유지)
    ("  x  \nx\ny", "  x  \ny"),                          # strip 후 동일 → 첫 등장만 유지
    ("한 줄\n\n한 줄", "한 줄\n"),                          # 빈 줄은 보존, 중복 텍스트만 제거
])
def test_dedup_lines(text, expected):
    assert _dedup_lines(text) == expected


def test_extract_tool_context_dedups_repeated_tool_output():
    """같은 tool이 동일 청크를 두 번 돌려줘도 all/vector에 한 번만 담긴다."""
    dup = "매뉴얼 근거 A"
    msgs = [
        ToolMessage(content=dup, name="vector_rag_search", tool_call_id="c1"),
        ToolMessage(content=dup, name="vector_rag_search", tool_call_id="c2"),
    ]
    ctx = _extract_tool_context(msgs)
    assert ctx["vector"].count(dup) == 1
    assert ctx["all"].count(dup) == 1


# ---------------------------------------------------------------------------
# 근거성(faithfulness) 게이트
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer,context,expected_high", [
    ("타이어 공기압을 점검하십시오", "타이어 공기압을 점검하십시오. 규정 공기압 유지.", True),
    ("완전히 무관한 환각 답변 텍스트", "타이어 공기압 매뉴얼 내용", False),
])
def test_grounding_overlap(answer, context, expected_high):
    overlap = _grounding_overlap(answer, context)
    assert (overlap >= 0.5) == expected_high


@pytest.mark.asyncio
async def test_verify_grounded_high_overlap_skips_llm_judge():
    """어휘 중첩이 높으면 LLM judge 호출 없이 즉시 grounded(지연 0)."""
    answer = "타이어 공기압을 점검하십시오"
    context = "타이어 공기압을 점검하십시오. 규정 공기압을 유지하십시오."
    with patch("app.agents.knowledge.ChatOpenAI") as mock_llm:
        assert await _verify_answer_grounded(answer, context) is True
        mock_llm.assert_not_called()  # judge 미호출


@pytest.mark.asyncio
async def test_verify_grounded_low_overlap_uses_llm_judge():
    """중첩이 낮으면 1.5B judge로 확정 — judge가 미근거 판정하면 False."""
    judge = MagicMock()
    judge.ainvoke = AsyncMock(return_value=MagicMock(content='{"grounded": false, "reasoning": "unsupported"}'))
    with patch("app.agents.knowledge.ChatOpenAI", MagicMock(return_value=judge)):
        result = await _verify_answer_grounded("전혀 무관한 환각", "타이어 공기압 매뉴얼")
    assert result is False
    judge.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_ungrounded_answer_raises_and_is_handled_as_failure():
    """근거성 게이트가 미근거로 판정하면 답을 ship하지 않고 실패 처리(knowledge_failed)한다."""
    fake = FakeReactAgent(answer="유창하지만 근거 없는 환각 답변")
    state = {"messages": [HumanMessage(content="배터리 취급 주의점")], "plan": [], "context_data": {}}
    with _patch_agent(fake) as _:
        # 기본 stub은 게이트 통과이므로, 이 테스트에서만 미근거(False)로 오버라이드.
        with patch("app.agents.knowledge._verify_answer_grounded", AsyncMock(return_value=False)):
            result = await knowledge_node(state)
    assert result["context_data"].get("knowledge_failed") is True


# ---------------------------------------------------------------------------
# 시나리오 7: Tool - MCP 실패 시 에이전트가 읽을 오류 문자열 반환
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_returns_error_string_on_mcp_failure():
    """MCP 호출이 실패하면 예외를 던지지 않고 오류 문자열을 반환한다(ReAct 재시도용)."""
    mock_call = AsyncMock(return_value=("타임아웃", "fail", "timeout", "10초 초과"))
    with patch("app.agents.knowledge.call_mcp_tool_once", mock_call):
        result = await text_to_sql_query.ainvoke({"query": "이번 달 총 주행거리"})

    assert "text_to_sql_query 실패" in result
    assert "timeout" in result
    assert "10초 초과" in result
