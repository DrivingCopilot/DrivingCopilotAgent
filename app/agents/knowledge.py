import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from app.core.config import MODEL_SERVER_URL, QWEN_TEXT_MODEL_NAME, QWEN_VL_MODEL_NAME
from app.graph.state import AgentState
from app.graph import ws as _ws
from app.core.mcp_client import call_mcp_tool_once

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 모델별 역할 분담 (Knowledge 노드)
# ---------------------------------------------------------------------------
# 단순검색(ReAct tool-calling) / query 변환 / 요약 → 1.5B(QWEN_TEXT_MODEL_NAME).
#   query 변환·요약은 crag.py(transform_query_node/refine_knowledge_node)가 이미
#   QWEN_TEXT_MODEL_NAME으로 수행한다 — 여기서는 검색 단계만 담당한다.
# 복잡한 Graph Context Fusion / CoT(다중 소스 관계 추론) → 7B(QWEN_VL_MODEL_NAME).
#   주의: model_server의 VL 경로는 tool-calling을 지원하지 않으므로(tools 무시 +
#   _strip_tool_messages_for_vl), 7B는 tool을 직접 부르지 않고 "이미 검색된"
#   context를 받아 융합/추론하는 합성 단계에만 쓴다. 검색은 항상 1.5B가 한다.
KNOWLEDGE_RETRIEVAL_MODEL = os.getenv("KNOWLEDGE_MODEL", QWEN_TEXT_MODEL_NAME)
KNOWLEDGE_FUSION_MODEL = os.getenv("KNOWLEDGE_FUSION_MODEL", QWEN_VL_MODEL_NAME)


async def _call_knowledge_tool(tool_name: str, params: Dict[str, Any]) -> str:
    """
    공용 MCP 서버의 knowledge tool 을 호출하고 결과 텍스트만 반환한다.
    실패 시 ReAct 에이전트가 읽고 재시도/대안 판단할 수 있는 오류 문자열을 돌려준다.
    """
    result_text, status, error_type, error_msg = await call_mcp_tool_once(tool_name, params)
    if status != "success":
        logger.warning("MCP knowledge tool 실패: %s (%s) — %s", tool_name, error_type, error_msg)
        return f"[{tool_name} 실패: {error_type or 'error'}] {error_msg}"
    return result_text


@tool
async def vector_rag_search(query: str) -> str:
    """
    Search for vehicle manual information using Vector RAG (Qdrant).
    Best for: General questions about the manual, usage instructions, or FAQs.
    """
    return await _call_knowledge_tool("vector_rag_search", {"query": query})


@tool
async def graph_rag_search(query: str, entities: List[str] = None) -> str:
    """
    Search for relational information using Graph RAG (Neo4j).
    Best for: Multi-hop reasoning like "What components are related to this warning light?" or "Maintenance interval for a part".
    """
    # 1.5B ReAct는 entities를 거의 안 채워 보낸다 → 백엔드가 query.split()(조사 포함)으로
    # substring 매칭해 '선루프가'/'hud가'처럼 엔티티명과 안 맞는다. 호출측이 entities를
    # 안 주면 조사·의문사를 제거한 깨끗한 term을 자동 추출해 넘긴다(_deterministic_retrieve와 동일 정책).
    clean_entities = entities or _extract_query_terms(query)
    return await _call_knowledge_tool(
        "graph_rag_search", {"query": query, "entities": clean_entities}
    )


@tool
async def text_to_sql_query(query: str) -> str:
    """
    Query structured vehicle telemetry or maintenance history from the database (Text2SQL).
    Best for: Data retrieval like "What was my average speed?" or "Last oil change date".
    """
    return await _call_knowledge_tool("text_to_sql_query", {"query": query})

KNOWLEDGE_SYSTEM_PROMPT = """You are the Knowledge Agent in the On-Device Multimodal Driving Copilot system.
Your role is to act as the primary knowledge hub, retrieving information from various sources to answer user queries or fulfill plans delegated by the Supervisor.

Available Tools:
1. vector_rag_search: Use for standard manual queries (e.g., "What does this button do?").
2. graph_rag_search: Use for relational queries (e.g., "Engine warning light components and maintenance").
3. text_to_sql_query: Use for database lookups (e.g., "Total mileage this month", "Recent DTC codes").

IMPORTANT — Source of truth:
- The tool results ARE excerpts parsed directly from this vehicle's own official owner's manual and
  internal knowledge base (Vector RAG / Graph RAG indices are built from the manual itself). You DO
  have direct access to this material through these tools — never claim you lack access to the
  manual, internal documents, or manufacturer data, and never tell the user to check the manual
  themselves or contact the manufacturer/customer service when a tool already returned relevant
  content.
- Base your final answer strictly on the tool results returned to you. Only if a tool result is
  genuinely empty or irrelevant after reformulating the query should you say the information was not
  found in the manual — do not fall back to a generic apology/disclaimer instead.
- Answer the exact quantity that was asked. When the user asks for a specific value (e.g. tire
  pressure, voltage, torque, capacity, a service interval), give that value ONLY if the context
  states that exact quantity. If the context does not contain it, say the tools did not return that
  value — NEVER substitute a different KIND of specification (a size, model/part code, or an unrelated
  number) and present it as the answer. A tire SIZE like "235/60R18" is NOT a tire pressure; a wheel
  spec is not a torque; do not equate quantities of different types. A quantity must carry the unit of
  the asked measure to qualify — a tire pressure reads in kPa/psi/bar, a voltage in V, a torque in
  N·m/kgf·m. If the context has no number in the asked measure's unit, the specific value is NOT
  present — say so instead of offering a differently-typed spec.

Workflow:
- Read the current plan and the user's request.
- You MUST call at least one retrieval tool before writing your final answer. Never answer directly
  from your own parametric knowledge or refuse/redirect the user without first calling a tool — you
  always have tools available for vehicle-manual questions, so an un-searched answer is never correct.
- Decide which tool(s) are needed. You may need to call multiple tools if the query is complex (e.g., Context Fusion of Vector + Graph).
- If the search result is poor, try reformulating the query (CRAG approach).
- If a tool call returns an error or timeout message (e.g., text starting with "[tool_name 실패:" or "[tool_name failed:"),
  do NOT treat that as "no information exists" — that is a transient tool failure, not an empty manual. Retry once,
  or call a DIFFERENT tool that can answer the same question (e.g., if vector_rag_search fails, try graph_rag_search)
  before concluding information was not found.
- Synthesize the retrieved context into a clear, concise, and helpful response.

IMPORTANT — Output language: Always write your FINAL answer to the user in Korean (한국어),
regardless of the language of the retrieved tool results or your own intermediate reasoning.
"""


# 7B(Graph Context Fusion / CoT)용 합성 프롬프트. tool 호출 없이, 이미 수집된
# Graph RAG + Vector RAG context를 융합해 다중홉 추론으로 최종 답을 만든다.
KNOWLEDGE_FUSION_PROMPT = """You are the Knowledge Fusion Reasoner (7B) in an on-device driving copilot.
You are given knowledge that was ALREADY retrieved from this vehicle's own official owner's manual and
knowledge graph (Graph RAG over Neo4j + Vector RAG). Your job is NOT to search — it is to FUSE the
Graph and Vector context and reason over it, step by step (chain-of-thought), to produce one correct,
grounded answer to the user's question.

Rules:
- The retrieved context IS the source of truth and IS from the manual — never say you lack access to
  the manual or tell the user to contact the manufacturer when relevant context is present below.
- Fuse Graph relations (e.g. "(운전석 에어백)-[:HAS_PART]-(에어백 시스템)") with Vector excerpts:
  enumerate/relate the entities the graph exposes and ground them in the manual text.
- Reason over multi-hop relations when the question needs it (types/components/causes/links) — but only
  chain hops that are ACTUALLY present in the graph/vector context. Do NOT invent an intermediate link,
  component, cause, or number to complete a chain the context does not establish.
- GROUNDING (critical): every factual claim, number, spec, and instruction in your answer must be
  traceable to the retrieved context. Do NOT add outside/world knowledge, do NOT guess, do NOT
  extrapolate beyond what the context states. If the context only partially answers the question, answer
  only that part and say the rest is not covered — a shorter fully-grounded answer beats a fuller one
  with unsupported claims.
- If the context is genuinely empty or irrelevant, say in one line that the manual does not cover it —
  do NOT produce a generic apology/disclaimer instead.

Output: a clear, concise final answer for the user. Always write in Korean (한국어)."""


# 단순 질의용 1.5B 요약 프롬프트 — 관계형 융합/CoT가 필요 없는(단일 소스, 매뉴얼
# 절차/설명형) 질의는 검색된 context를 1.5B가 그대로 요약한다(설계 스펙: "요약은
# 1.5B"). 7B fusion(~23~60s)을 안 타므로 대부분 질의가 fast path(~5s)로 끝난다.
KNOWLEDGE_SUMMARIZE_PROMPT = """You are the Knowledge Summarizer (1.5B) in an on-device driving copilot.
You are given text that was ALREADY retrieved from this vehicle's own official owner's manual
(Vector RAG) and/or knowledge graph. Your job is NOT to search and NOT to reason multi-hop — it is to
summarize the retrieved context into one concise, grounded answer to the user's question.

Rules:
- The retrieved context IS the source of truth and IS from the manual — never say you lack access to the
  manual or tell the user to contact the manufacturer when relevant context is present below.
- Answer ONLY from the context; do not add facts that are not in it.
- Answer the exact quantity asked. If the user asks for a specific value (tire pressure, voltage,
  torque, capacity, interval) and the context does not state that exact quantity, say it is not in the
  retrieved context — NEVER substitute a different KIND of spec (a size, model/part code, or unrelated
  number) as if it were the answer. A tire SIZE like "235/60R18" is NOT a tire pressure. A value only
  qualifies if it carries the asked measure's unit (pressure→kPa/psi/bar, voltage→V, torque→N·m); if no
  such value is in the context, say the specific value is not present.
- Keep it short (3~5 sentences) and directly answer the question.

Output: a clear, concise final answer for the user. Always write in Korean (한국어)."""


# 근거성(faithfulness) 검증용 판정 프롬프트 — 생성된 답변이 검색 context에 실제로
# 근거하는지(환각 아닌지) 1.5B judge로 확인한다. 어휘 중첩이 낮아 '의심'될 때만 호출된다.
GROUNDING_CHECK_PROMPT = """You are a strict grounding verifier for a vehicle manual QA system.
Given the RETRIEVED CONTEXT (excerpts from the vehicle's official manual) and a candidate ANSWER,
decide whether the ANSWER is supported by (entailed by) the context — i.e. every factual claim,
number, and instruction in the ANSWER can be traced to the context, with no invented facts.

Rules:
- "grounded": true only if the answer's substantive claims are supported by the context.
- Paraphrase/summary of the context is fine. Minor connective wording is fine.
- If the answer adds facts/numbers not present in the context, or contradicts it → grounded: false.
- A generic "정보를 찾지 못했습니다" style answer is trivially grounded → true.

Output ONLY compact JSON: {"grounded": true|false, "reasoning": "one short sentence"}"""


# 관계형/다중홉(Graph Context Fusion·CoT)이 필요한 질의를 감지하는 휴리스틱 키워드.
# supervisor Rule 3가 관계형 질의에 Graph RAG를 지시하는 것과 같은 취지 — 종류/구성/
# 부품/원인/연동/차이 등을 묻거나 여러 소스를 엮어야 하는 질의는 7B 융합/추론으로 보낸다.
_FUSION_KEYWORDS_KO = (
    "종류", "구성", "부품", "관계", "원인", "연동", "차이", "목록", "리스트",
    "어떤 것", "무엇이 있", "무엇이있", "어떤게 있", "얼마나", "구조", "연결", "포함",
)
_FUSION_KEYWORDS_EN = (
    "type", "kind", "component", "relation", "cause", "difference", "list",
    "what are", "which", "related", "structure", "consist", "include",
)


# (Phase 4 라우팅 실험) graph_present만으로 7B fusion을 강제할지 여부.
# 문제: Neo4j가 broad CONTAINS 매칭이라 거의 모든 질의에 관계가 매칭돼 graph_present=True →
# 7B fusion이 남발되고 지연(median ~23s)과 7B 서버 500(복합 질의 70% 실패)을 유발했다.
# 기본(False)은 '관계형 키워드가 있을 때만' fusion하고, 그 외에는 1.5B(요약/ReAct)로 라우팅해
# 7B 부하를 낮춘다. graph 근거는 여전히 context로 들어가 grounding은 유지된다.
# 구 동작(graph_present→fusion)은 KNOWLEDGE_FUSION_ON_GRAPH_PRESENT=1 로 복원 가능.
FUSION_ON_GRAPH_PRESENT = os.getenv("KNOWLEDGE_FUSION_ON_GRAPH_PRESENT", "0") == "1"


def _needs_graph_fusion(instruction: str, query: str, graph_present: bool) -> bool:
    """이 질의가 7B Graph Fusion/CoT 경로가 필요한 '복잡한' 질의인지 판정한다.

    - 관계형·열거형 지표 키워드(종류/구성/관계/원인 등)가 있으면 복잡으로 본다.
    - graph_present는 기본적으론 fusion을 강제하지 않는다(위 상수 주석 참고) —
      broad-match로 거의 항상 True라 실질 신호가 아니기 때문. env로만 구 동작 복원.
    """
    if graph_present and FUSION_ON_GRAPH_PRESENT:
        return True
    haystack = f"{instruction} {query}".lower()
    if any(k in haystack for k in _FUSION_KEYWORDS_KO):
        return True
    return any(k in haystack for k in _FUSION_KEYWORDS_EN)


def _dedup_lines(text: str) -> str:
    """텍스트에서 공백 제거 후 동일한 라인의 중복을 순서 유지하며 제거한다.
    같은 tool 반복 호출/겹치는 청크로 동일 excerpt가 여러 번 들어오면 fusion 입력을
    오염시키고 precision·토큰을 낮추므로 정확 중복 라인을 걷어낸다."""
    seen = set()
    out: List[str] = []
    for line in text.split("\n"):
        key = line.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    return "\n".join(out)


def _extract_tool_context(new_messages: List[Any]) -> Dict[str, str]:
    """ReAct 실행 중 호출된 tool들의 출력을 tool별로 모은다.

    Returns: {"graph": "...", "vector": "...", "sql": "...", "all": "합쳐진 원문"}
    tool이 하나도 안 불렸으면 all=""(빈 문자열).
    중복 excerpt(같은 tool 반복 호출/겹치는 청크)는 엔트리·라인 단위로 제거한다.
    """
    buckets: Dict[str, List[str]] = {"graph": [], "vector": [], "sql": []}
    seen_entries = set()  # 엔트리(전체 tool 출력) 단위 정확 중복 제거
    for m in new_messages:
        if not isinstance(m, ToolMessage):
            continue
        name = (getattr(m, "name", "") or "").lower()
        content = m.content if isinstance(m.content, str) else str(m.content)
        # 에러/결과없음 출력은 grounding에 못 쓰므로 수집하지 않는다(graph_present
        # 오인·garbage 융합 입력 방지).
        if _is_unusable_result(content):
            continue
        entry_key = content.strip()
        if entry_key in seen_entries:  # 동일 tool 재호출이 같은 청크를 또 돌려준 경우
            continue
        seen_entries.add(entry_key)
        if "graph" in name:
            buckets["graph"].append(content)
        elif "vector" in name:
            buckets["vector"].append(content)
        elif "sql" in name:
            buckets["sql"].append(content)
        else:
            buckets["vector"].append(content)  # 미상 tool은 vector 취급
    # 버킷별로 라인 단위 중복까지 제거(청크 인덱스는 다르지만 본문이 겹치는 경우 대비).
    joined = {k: _dedup_lines("\n".join(v)) for k, v in buckets.items()}
    parts = []
    for label, key in (("Graph RAG", "graph"), ("Vector RAG", "vector"), ("Text2SQL", "sql")):
        if joined[key]:
            parts.append(f"[{label}]\n" + joined[key])
    return {
        "graph": joined["graph"],
        "vector": joined["vector"],
        "sql": joined["sql"],
        "all": "\n\n".join(parts),
    }


def _is_tool_error_text(text: str) -> bool:
    """_call_knowledge_tool이 돌려주는 실패 문자열('[tool 실패: ...]') 여부."""
    t = (text or "").lstrip()
    return t.startswith("[") and "실패:" in t[:40]


# 백엔드 graph/vector 서비스가 '검색 결과 없음'을 알리는 문구들. 에러는 아니지만
# grounding에 쓸 수 없는 내용이므로 빈 결과와 동일하게 취급해야 한다(안 그러면
# graph_present=True로 오인해 7B 융합을 강제하고, 이 문장을 context로 흘려보낸다).
_NO_RESULT_MARKERS = (
    "찾지 못했습니다",
    "검색할 엔티티가 없습니다",
    "관련 정보를 찾을 수 없",
    "결과가 없습니다",
)


def _is_no_result_text(text: str) -> bool:
    t = text or ""
    return any(m in t for m in _NO_RESULT_MARKERS)


def _is_unusable_result(text: str) -> bool:
    """grounding에 쓸 수 없는 tool 출력(에러 문자열 또는 '결과 없음' 문구)."""
    t = (text or "").strip()
    return (not t) or _is_tool_error_text(t) or _is_no_result_text(t)


# 한국어 조사/어미(엔티티 뒤에 붙어 graph substring 매칭을 깨뜨리는 접미사). 긴 것부터
# 매칭해 최장 접미사를 우선 제거한다.
_KO_JOSA = (
    "으로서", "으로써", "이라고", "이란", "으로", "에서", "에게", "한테", "께서",
    "까지", "부터", "보다", "처럼", "같이", "라도", "이나", "이든", "든지", "마다",
    "조차", "밖에", "뿐", "이랑", "랑", "과", "와", "을", "를", "이", "가", "은",
    "는", "의", "에", "도", "만", "로", "나", "야",
)

# 질의에서 엔티티가 아닌 의문사·서술어·기능어. 엔티티 추출 시 제거한다.
_KO_STOPWORDS = frozenset({
    "어떻게", "어떡해", "무엇", "무엇이", "뭐", "뭐야", "뭔데", "왜", "언제", "어디",
    "어디서", "어디에", "얼마나", "몇", "해", "해줘", "해야", "하면", "되", "돼",
    "있어", "없어", "인가", "인지", "좋아", "알려줘", "대해", "관해", "경우", "및",
    "수", "때", "그", "이", "저", "것", "거", "좀", "다시", "또", "안", "못",
})


def _extract_query_terms(query: str) -> List[str]:
    """자연어 질의에서 graph 매칭용 엔티티 후보를 뽑는다.

    백엔드 graph_rag는 entities 미지정 시 query.split()(조사 포함)을 그대로 substring
    매칭에 써서 '선루프가'/'뒷좌석을'/'hud가'처럼 조사가 붙어 엔티티명과 안 맞는다.
    여기서 구두점·조사·의문사를 제거한 깨끗한 term을 만들어 entities로 넘긴다.
    """
    terms: List[str] = []
    for raw in (query or "").split():
        tok = raw.strip().strip("?？!！.,·…‘’\"'()[]{}")
        if not tok or tok in _KO_STOPWORDS:
            continue
        stem = tok
        for josa in _KO_JOSA:  # 조사가 붙어 있고 어간이 2자 이상이면 제거
            if stem.endswith(josa) and len(stem) - len(josa) >= 2:
                stem = stem[: -len(josa)]
                break
        if len(stem) < 2 or stem in _KO_STOPWORDS:
            stem = tok  # 과도한 절단 방지: 어간이 너무 짧으면 원형 유지
        if len(stem) >= 2 and stem not in terms:
            terms.append(stem)
    return terms


def _is_prompt_echo(text: str) -> bool:
    """7B(VL) 융합 모델이 답변 대신 입력 프롬프트 템플릿을 그대로 되뱉은 경우.
    _fuse_with_7b가 보내는 HumanMessage의 고정 마커('[User Query]'/'[Retrieved
    Context]')가 출력에 그대로 나타나면 echo로 본다 — 정상 답변에는 나올 수 없는
    문자열이다. 이 echo가 malformed 가드를 통과하면 사용자에게 프롬프트가 노출된다."""
    t = text or ""
    if "[Retrieved Context]" in t:
        return True
    return t.lstrip().startswith("[User Query]")


_GROUNDING_TERM_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")


def _grounding_overlap(answer: str, context: str) -> float:
    """답변의 내용어(2자+ 한글/영숫자) 중 context에 등장하는 비율(0~1).
    저비용 근거성 선판정 — 대부분의 grounded 답변은 이 값이 높아 LLM judge를 건너뛴다."""
    ans_terms = set(_GROUNDING_TERM_RE.findall((answer or "").lower()))
    if not ans_terms:
        return 1.0  # 판정할 내용어가 없음 — 통과(별도 malformed 가드가 처리)
    ctx = (context or "").lower()
    hit = sum(1 for t in ans_terms if t in ctx)
    return hit / len(ans_terms)


async def _verify_answer_grounded(answer: str, context: str) -> bool:
    """생성된 답변이 검색 context에 근거하는지(환각 아닌지) 검증한다.

    1) 어휘 중첩이 충분(>=0.5)하면 grounded로 보고 즉시 통과(LLM 호출 없음 — 지연 0).
    2) 중첩이 낮아 '의심'되는 경우에만 1.5B judge로 확정한다(judge 실패 시 보수적으로 통과).
    """
    if not answer or not answer.strip() or not context or not context.strip():
        return True  # 근거/답변 부재는 다른 가드가 처리
    overlap = _grounding_overlap(answer, context)
    if overlap >= 0.5:
        return True
    try:
        llm = ChatOpenAI(
            model=KNOWLEDGE_RETRIEVAL_MODEL, temperature=0.0, max_tokens=120,
            base_url=MODEL_SERVER_URL, model_kwargs={"response_format": {"type": "json_object"}},
        )
        resp = await llm.ainvoke([
            SystemMessage(content=GROUNDING_CHECK_PROMPT),
            HumanMessage(content=f"[RETRIEVED CONTEXT]\n{context}\n\n[ANSWER]\n{answer}"),
        ])
        raw = (resp.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return True  # 판정 파싱 실패 — 보수적으로 통과(무응답보다 낫다)
        verdict = json.loads(match.group(0))
        return bool(verdict.get("grounded", True))
    except Exception:
        logger.exception("knowledge: 근거성 judge 실패 — 보수적으로 통과 처리")
        return True


async def _deterministic_retrieve(query: str) -> Dict[str, str]:
    """1.5B ReAct가 tool을 한 번도 안 부른 경우의 안전망 — graph/vector를 직접 호출해
    grounding을 보장한다(에어백 오답처럼 검색 없이 hallucination하는 것을 차단).

    - graph는 질의에서 뽑은 깨끗한 엔티티를 넘겨 매칭률을 높인다(조사 제거).
    - graph/vector를 병렬 호출해 지연을 max(둘)로 줄인다(순차 합산 아님).
    - 에러/결과없음은 빈 결과로 정규화해 garbage가 융합 입력으로 새지 않게 한다.
    """
    entities = _extract_query_terms(query)
    graph, vector = await asyncio.gather(
        _call_knowledge_tool("graph_rag_search", {"query": query, "entities": entities}),
        _call_knowledge_tool("vector_rag_search", {"query": query}),
    )
    graph = "" if _is_unusable_result(graph) else graph
    vector = "" if _is_unusable_result(vector) else vector
    parts = []
    if graph:
        parts.append(f"[Graph RAG]\n{graph}")
    if vector:
        parts.append(f"[Vector RAG]\n{vector}")
    return {"graph": graph, "vector": vector, "sql": "", "all": "\n\n".join(parts)}


async def _fuse_with_7b(query: str, context_text: str) -> str:
    """7B(QWEN_VL)로 Graph Context Fusion + CoT 합성. tool 미사용(VL 경로는 텍스트 생성)."""
    llm = ChatOpenAI(
        model=KNOWLEDGE_FUSION_MODEL,
        # greedy(0.0). Qwen2-VL-7B은 중국어 중심 모델이라 temperature>0 + top_p 미상한
        # 조합에서 저확률 CJK/일본어 토큰으로 code-switching이 샜다(坡道/ング 혼입).
        # 융합은 근거 종합이라 창의성 불필요 — 결정적 디코딩으로 발산을 원천 차단한다.
        temperature=0.0,
        # 7B fusion 생성 시간은 출력 토큰 수에 거의 선형 — 768은 실측 ~23~60초로
        # A2A 타임아웃을 넘기는 주 병목이었다. 주행 답변은 3~5문장이면 충분하므로
        # 384로 줄여 생성 시간을 ~절반으로 낮춘다(품질 손실 없이 지연 근본 개선).
        max_tokens=384,
        base_url=MODEL_SERVER_URL,
    )
    response = await llm.ainvoke([
        SystemMessage(content=KNOWLEDGE_FUSION_PROMPT),
        HumanMessage(content=f"[User Query]\n{query}\n\n[Retrieved Context]\n{context_text}"),
    ])
    return (response.content or "").strip()


def _is_probably_korean(text: str) -> bool:
    """최종 답변이 (충분히) 한국어인지 판정한다.

    한글 음절과 라틴 문자 수를 비교해, 라틴이 지배적이면(영어 유출) False.
    프롬프트의 'Always write in Korean' 지시를 1.5B/7B가 영어 원문 컨텍스트를
    미러링하며 무시하는 경우를 결정적으로 잡기 위한 저비용 휴리스틱이다.
    """
    if not text:
        return True  # 빈 문자열은 별도 malformed 가드가 처리 — 여기선 통과
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    latin = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    if hangul + latin == 0:
        return True  # 숫자/기호만(예: 수치 답변) — 언어 판정 대상 아님
    return hangul >= latin  # 한글이 라틴 이상이면 한국어로 본다


async def _translate_to_korean(text: str) -> str:
    """영어 등으로 나온 최종 답변을 1.5B로 한국어 번역한다(fast·저비용)."""
    llm = ChatOpenAI(
        model=KNOWLEDGE_RETRIEVAL_MODEL,
        temperature=0.0,
        max_tokens=384,
        base_url=MODEL_SERVER_URL,
    )
    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are a translator. Translate the user's text into natural Korean (한국어). "
            "Output ONLY the Korean translation — no preamble, no notes, no original text. "
            "Preserve technical terms and numbers accurately."
        )),
        HumanMessage(content=text),
    ])
    return (response.content or "").strip()


async def _ensure_korean(text: str) -> str:
    """최종 답변이 한국어가 아니면 번역해 한국어를 보장한다(모든 synthesis 경로 공통 가드)."""
    if _is_probably_korean(text):
        return text
    logger.info("knowledge: 최종 답변이 비한국어로 감지됨 — 1.5B 한국어 번역 적용")
    translated = await _translate_to_korean(text)
    # 번역이 비거나 여전히 비한국어면 원문 유지(무응답 방지 — malformed 가드가 이어서 처리).
    if translated and _is_probably_korean(translated):
        return translated
    return text


async def _summarize_with_1_5b(query: str, context_text: str) -> str:
    """1.5B로 검색된 context를 요약(단순 질의 fast path). 관계형 융합/CoT 불필요."""
    llm = ChatOpenAI(
        model=KNOWLEDGE_RETRIEVAL_MODEL,
        # greedy(0.0) — CJK code-switching 발산 방지(fusion과 동일 이유).
        temperature=0.0,
        max_tokens=384,
        base_url=MODEL_SERVER_URL,
    )
    response = await llm.ainvoke([
        SystemMessage(content=KNOWLEDGE_SUMMARIZE_PROMPT),
        HumanMessage(content=f"[User Query]\n{query}\n\n[Retrieved Context]\n{context_text}"),
    ])
    return (response.content or "").strip()


async def knowledge_node(state: AgentState) -> Dict[str, Any]:
    await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": "Knowledge agent retrieving context..."}))
    
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    context_data = state.get("context_data", {})

    # CRAG 재검색 여부: transform_query_node 가 채운 refined_query 가 있으면
    # 동일 plan 스텝을 재작성된 쿼리로 재검색하는 중이다.
    refined_query = context_data.get("refined_query", "")
    is_reretrieval = bool(refined_query)

    # 1. Prepare Instructions based on Plan-and-Execute pattern
    instruction_text = ""
    if is_reretrieval:
        instruction_text = refined_query
    elif plan and len(plan) > 0:
        instruction_text = f"Execute the next step in the plan: {plan[0]}"
    else:
        instruction_text = "Answer the user's latest query based on your domain knowledge tools."

    instruction_msg = HumanMessage(content=f"[Supervisor Instruction] {instruction_text}")
    
    # 검색 대상 쿼리 — 재검색이면 refined_query, 아니면 plan 스텝/사용자 쿼리.
    user_query = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    retrieval_query = refined_query or (plan[0] if plan else user_query)

    # 2. 검색 단계 (1.5B, 단순검색) — ReAct 루프로 tool을 호출해 raw context를 모은다.
    # greedy(0.0) — CJK code-switching 발산 방지(fusion과 동일 이유).
    llm = ChatOpenAI(model=KNOWLEDGE_RETRIEVAL_MODEL, temperature=0.0, base_url=MODEL_SERVER_URL)
    tools = [vector_rag_search, graph_rag_search, text_to_sql_query]

    agent = create_react_agent(llm, tools, prompt=KNOWLEDGE_SYSTEM_PROMPT)

    messages_to_pass = messages + [instruction_msg]

    try:
        # 3. Invoke the ReAct Agent
        response = await agent.ainvoke({"messages": messages_to_pass})

        # Extract new messages generated by the Knowledge Agent
        new_messages = response["messages"][len(messages_to_pass):]

        # 검색 단계에서 1.5B ReAct가 실제로 호출한 tool 출력을 모은다.
        tool_ctx = _extract_tool_context(new_messages)
        react_grounded = bool(tool_ctx["all"])  # 1.5B가 스스로 검색했는가

        # 1.5B가 tool을 한 번도 안 부른 경우(에어백 오답처럼 검색 없이 hallucination)
        # — 결정적으로 graph/vector를 직접 호출해 grounding을 보장한다(안전망).
        if not react_grounded:
            logger.info("knowledge: ReAct가 tool 미호출 — 결정적 검색 폴백(graph+vector) 실행")
            tool_ctx = await _deterministic_retrieve(retrieval_query)

        react_answer = new_messages[-1].content if new_messages else ""
        has_context = bool(tool_ctx["all"])

        # 검색이 전부 실패/무결과라 grounding할 근거가 하나도 없으면 — 1.5B의
        # 미검증 답변(환각)을 사용자에게 내보내지 않고, garbage를 7B에 융합시키지도
        # 않는다. 명시적 실패로 raise → CRAG(transform 재검색)/observe가 처리한다.
        if not has_context:
            raise ValueError("Knowledge retrieval returned no usable context (graph/vector both empty)")

        # 4. 합성 단계 — 관계형/다중홉(graph 근거 존재)은 7B Graph Fusion/CoT로,
        #    단순 질의는 1.5B가 직접 검색·요약한 답을 그대로 쓴다.
        graph_present = bool(tool_ctx.get("graph"))
        is_complex = _needs_graph_fusion(instruction_text, user_query, graph_present)

        if is_complex:
            await _ws.websocket_manager.send_status(
                json.dumps({"type": "status", "data": "Fusing graph+vector context (7B CoT)..."})
            )
            final_answer = await _fuse_with_7b(retrieval_query or user_query, tool_ctx["all"])
            # 7B 합성이 비거나, <tool_call>이 새거나, 입력 프롬프트를 그대로 echo하면
            # 부실로 보고 1.5B ReAct 답변으로 폴백(단, grounded된 경우에만).
            if (not final_answer or not final_answer.strip()
                    or "<tool_call>" in final_answer or _is_prompt_echo(final_answer)):
                logger.warning("knowledge: 7B 융합 결과가 부실(빈값/tool_call/프롬프트 echo) — 폴백")
                final_answer = react_answer if react_grounded else ""
        elif react_grounded and react_answer and not _is_tool_error_text(react_answer):
            # 단순 검색: 1.5B가 직접 검색+요약한 답을 그대로 사용.
            final_answer = react_answer
        else:
            # 단순 질의(is_complex=False)인데 1.5B가 직접 검색은 안 함(폴백) →
            # react_answer는 미근거라 못 쓴다. 설계 스펙("요약은 1.5B")대로 검색된
            # context를 1.5B로 요약해 grounding한다 — 7B fusion(~23~60s)을 피해
            # fast path(~5s)로 끝낸다. 1.5B 요약이 부실하면 7B로만 폴백한다.
            await _ws.websocket_manager.send_status(
                json.dumps({"type": "status", "data": "Summarizing retrieved context (1.5B)..."})
            )
            final_answer = await _summarize_with_1_5b(retrieval_query or user_query, tool_ctx["all"])
            if (not final_answer or not final_answer.strip()
                    or "<tool_call>" in final_answer or _is_prompt_echo(final_answer)):
                logger.warning("knowledge: 1.5B 요약이 부실 — 7B 융합으로 폴백")
                await _ws.websocket_manager.send_status(
                    json.dumps({"type": "status", "data": "Fusing graph+vector context (7B CoT)..."})
                )
                final_answer = await _fuse_with_7b(retrieval_query or user_query, tool_ctx["all"])

        # 답변이 비었거나 <tool_call> 태그가 노출됐거나 입력 프롬프트를 그대로
        # echo한 경우 — 조용히 "성공"으로 넘기면 observe/CRAG의 실패 감지를 모두
        # 통과해 사용자에게 무응답/프롬프트 노출로 이어진다. 명시적 실패로 raise한다.
        if (not final_answer or not final_answer.strip()
                or "<tool_call>" in final_answer or _is_prompt_echo(final_answer)):
            raise ValueError(
                f"Knowledge agent produced an empty or malformed final answer: {final_answer!r}"
            )

        # 언어 보정(모든 synthesis 경로 공통): 프롬프트의 'Always write in Korean'을
        # 1.5B/7B가 영어 원문 컨텍스트를 미러링하며 무시하는 경우가 있어(예: 배터리
        # 안전 답변이 영어로 유출) — 최종 답변이 비한국어면 결정적으로 한국어 번역한다.
        final_answer = await _ensure_korean(final_answer)

        # 근거성(faithfulness) 게이트 — 형식 가드(빈값/tool_call/echo)를 통과한 '유창한
        # 환각'을 잡는다. 답변이 검색 context에 근거하지 않으면(어휘 중첩 낮음 + 1.5B judge
        # 미근거 판정) 그대로 내보내지 않는다. 아직 재검색을 안 한 첫 시도면 명시적 실패로
        # raise → 아래 except가 knowledge_failed로 처리하고 supervisor 재위임/observe
        # 보정 경로를 태운다(환각을 그대로 ship하지 않음). 이미 재검색까지 했다면
        # (is_reretrieval) 무한 재시도를 피해 best-effort로 답을 유지하되 경고만 남긴다.
        if not await _verify_answer_grounded(final_answer, tool_ctx["all"]):
            if not is_reretrieval:
                logger.warning("knowledge: 답변이 검색 context에 미근거(환각 의심) — 실패 처리로 보정 유도")
                raise ValueError("Knowledge answer not grounded in retrieved context (hallucination guard)")
            logger.warning("knowledge: 재검색 후에도 미근거 의심 — best-effort로 답변 유지(무한 재시도 방지)")

        # Context data update (accumulate findings)
        context_data["last_knowledge_result"] = final_answer
        # refined_query 는 이번 재검색으로 소비됐으므로 초기화(빈 문자열=없음).
        context_data["refined_query"] = ""
        # 이전 시도의 knowledge_failed=True 가 얕은 병합(merge_context)으로
        # 남아있지 않도록 성공 시 명시적으로 False 로 되돌린다(CRAG 노드들이
        # 이 플래그로 재작성/정제 LLM 호출을 건너뛰는 기준이므로, 낡은 True가
        # 남으면 정상 검색 결과도 실패로 오인해 정제를 건너뛰게 된다).
        context_data["knowledge_failed"] = False

        if is_reretrieval:
            # CRAG 재검색: 동일 스텝을 다시 검색한 것이므로 plan 을 재-pop 하지 않고,
            # grade/transform 의 평가 기준(crag_query)도 원본 스텝 그대로 유지한다.
            new_plan = plan
        else:
            # 신규 스텝 실행: 완료 스텝 pop + CRAG 재검색 예산 리셋.
            # grade_retrieval/transform_query 가 "실제로 실행된 쿼리"를 평가하도록
            # crag_query 를 기록한다 — pop 이후 plan[0] 이 다음 스텝으로 바뀌어
            # grader 가 엉뚱한 스텝을 평가하는 문제를 막는다.
            context_data["crag_query"] = plan[0] if plan else user_query
            new_plan = plan[1:] if plan else []
            context_data["crag_attempts"] = 0
        
        await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": "Knowledge retrieval complete. Returning to Supervisor."}))

        return {
            "messages": new_messages,
            "plan": new_plan,
            "context_data": context_data,
            # execution.py와 동일한 계약으로 tool_calls에 append한다 — observe_node는
            # tool_calls[-1]만 보고 성공/실패를 판정하는데, knowledge가 여기 아무것도
            # 안 쓰면 observe가 몇 턴 전 다른 agent의 결과를 재관측하게 된다.
            "tool_calls": [{
                "tool": "knowledge",
                "params": {"instruction": instruction_text},
                "result": final_answer,
                "status": "success",
            }],
            "next_agent": "supervisor", # Always return control to supervisor
            # error_count 반환 없음 — execution.py와 동일하게 observe_node가 단일
            # 권위자다. 여기서 같이 건드리면 observe가 tool_calls로 다시 세면서
            # 중복 카운트된다.
        }
        
    except Exception as e:
        logger.error(f"Error in Knowledge Agent Node: {e}")

        await _ws.websocket_manager.send_status(json.dumps({"type": "status", "data": f"Knowledge agent failed: {str(e)[:50]}..."}))

        return {
            "messages": [AIMessage(content=f"Knowledge Agent encountered an error: {e}")],
            "context_data": {
                # 재검색 중 실패해도 refined_query 를 반드시 초기화한다 — 남겨두면
                # 다음 knowledge 위임이 낡은 refined_query 를 재검색으로 오인해
                # 엉뚱한 instruction 을 쓰고 plan 스텝 pop 을 건너뛴다.
                "refined_query": "",
                # 실패를 last_knowledge_result 에 명시적으로 기록한다 — 비워두면
                # (a) 이번이 첫 knowledge 호출인 경우 supervisor 의 재호출 차단
                # 가드(if next_agent=="knowledge" and last_knowledge_result)가
                # 발동하지 않아 knowledge 가 계속 재위임되고, (b) 이전에 knowledge
                # 가 성공한 적이 있으면 그 낡은 결과가 그대로 남아 이번 실패를
                # 가리고 supervisor 가 무관한 답으로 턴을 종료한다.
                "last_knowledge_result": f"[knowledge 실패] {e}",
                # CRAG(grade_retrieval/route_after_grade/refine_knowledge)가 이
                # 플래그를 보고 재작성·정제 LLM 호출을 생략하고 곧장 observe로
                # 보낸다 — "검색 결과가 부실하다"는 CRAG의 전제 자체가 예외
                # 상황(agent 실행 실패)에는 맞지 않는다(재검색해도 같은 예외가
                # 다시 날 뿐이다).
                "knowledge_failed": True,
            },
            "tool_calls": [{
                "tool": "knowledge",
                "params": {"instruction": instruction_text},
                "result": str(e),
                "status": "error",
                "error_type": "parameter",
                "error_msg": str(e),
            }],
            "next_agent": "supervisor",
            # error_count 반환 없음(성공 경로와 동일 이유) — observe_node가
            # tool_calls를 보고 유일하게 카운트한다. 여기서 같이 올리면 이번 한
            # 번의 실패가 observe 카운트 + 이 카운트로 중복 집계돼, MAX_RETRY
            # 한도(parameter=2)를 첫 실패만으로 소진해버린다.
            "feedback": f"Knowledge Agent failed to execute the plan step due to: {e}"
        }
