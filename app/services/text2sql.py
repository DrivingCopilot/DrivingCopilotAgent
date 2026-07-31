import logging
import re
import sqlite3
from collections import Counter
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import (
    COLLECTION_NAME,
    DB_PATH,
    MODEL_NAME,
    MODEL_SERVER_URL,
    QDRANT_URL,
    QWEN_TEXT_MODEL_NAME,
)

logger = logging.getLogger(__name__)

def schema_Linking(query: str) -> dict[str, str]:
    """
    Schema Linking: 자연어 질문(query)과 관련된 테이블 스키마만 필터링하여 반환합니다.
    """
    # 4개의 DB 테이블 스키마 정의 (Text2SQL Agent 용)
    schemas = {
        "vehicle_telemetry": "table: vehicle_telemetry, columns: ts, speed, rpm, fuel, battery, tire (실시간 차량 센서 데이터)",
        "maintenance_log": "table: maintenance_log, columns: date, mileage, type, parts, cost (차량 정비 이력 데이터)",
        "dtc_codes": "table: dtc_codes, columns: code, description, severity, resolved (OBD-II 오류 및 진단 코드)",
        "trip_history": "table: trip_history, columns: start, end, distance, avg_speed, fuel (차량 운행 기록 데이터)"
    }
    
    # 키워드 기반 단순 룰 매칭
    keywords = {
        "vehicle_telemetry": ["현재", "실시간", "속도", "rpm", "연료", "배터리", "타이어", "센서"],
        "maintenance_log": ["정비", "오일", "교환", "교체", "비용", "마지막", "이력", "수리"],
        "dtc_codes": ["경고", "코드", "오류", "해결", "미해결", "obd", "dtc", "진단"],
        "trip_history": ["주행", "거리", "운행", "이번 달", "출발", "도착", "평균 속도"]
    }
    
    selected_schemas = {}
    query_lower = query.lower()
    
    for table, words in keywords.items():
        if any(word in query_lower for word in words):
            selected_schemas[table] = schemas[table]
            
    # 매칭되는 테이블이 하나도 없으면 전체 스키마 반환 (LLM이 자체 판단하도록)
    if not selected_schemas:
        return schemas
        
    return selected_schemas

def few_shot_sql(query: str, k: int=3) -> dict[str, str]:
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value="few_shots_examples")
            )
        ]
    )

    search_result = vectorstore.similarity_search(
        query = query,
        k=k,
        search_filter = search_filter,
    )

    prompt_text = "## 참고용 유사 SQL 작성 예시\n"

    for i, doc in enumerate(search_result,1):
        q = doc.metadata.get("question","")
        s = doc.metadata.get("sql","")
        prompt_text+= f"예시 {i}, 질문 : {q}\n      SQL: {s}\n"

    return prompt_text



def self_consistency(query: str, table_info: str, few_shot_examples: str, n: int = 5) -> str:
    """
    Self-Consistency: 주어진 query, table_info, few_shot_examples를 기반으로 N개의 SQL을 생성하고,
    다수결 투표(Majority Voting)를 통해 가장 빈도수가 높은 최종 SQL을 선택합니다.
    """
    sqls = []

    def _generate_sql_from_llm(q: str, t_info: str, few_shots: str) -> str:
        sql_prompt = PromptTemplate.from_template(
            """You are an expert SQLite Data Analyst for an On-Device Multimodal Driving Copilot system.
Your task is to convert the user's natural language question into a strictly valid and optimized SQLite query.

### Database Schema (Table Information)
{table_info}

{few_shot_examples}

### Rules and Constraints:
1. Return ONLY the raw SQL query. Do NOT include markdown formatting (e.g., ```sql), explanations, or any other text.
2. Use ONLY the tables and columns provided in the schema above. Do not invent columns.
3. If the query asks for "today", "this month", or related timeframes, use SQLite date/time functions (e.g., DATE('now'), strftime('%Y-%m', 'now')).
4. Focus on vehicle telemetry, maintenance, DTC codes, and trip history specific structures as described.
5. Provide the most efficient and robust query possible.

### User Question
{question}

### SQL Query
"""
        )
        llm = ChatOpenAI(model=QWEN_TEXT_MODEL_NAME, temperature=0.7, base_url=MODEL_SERVER_URL)
        
        
        chain = sql_prompt | llm | StrOutputParser()
        
    
        generated_sql_query = chain.invoke({
            "table_info": t_info,
            "few_shot_examples": few_shots,
            "question": q
        })
        return generated_sql_query

    for _ in range(n):
        try:
            sql = _generate_sql_from_llm(query, table_info, few_shot_examples)
            if sql:
                # 공백 및 줄바꿈 정규화
                normalized_sql = " ".join(sql.strip().split())
                sqls.append(normalized_sql)
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            
    if not sqls:
        logger.warning("No SQL queries were generated.")
        return ""
        
    # 다수결 투표를 통해 가장 빈도수가 높은 SQL 선택
    vote_counts = Counter(sqls)
    best_sql, max_votes = vote_counts.most_common(1)[0]
    
    logger.info(f"Selected SQL with {max_votes}/{len(sqls)} votes: {best_sql}")
    
    return best_sql

# ---------------------------------------------------------------------------
# SQL 검증 3단계 파이프라인 (계획서 §4.2 "구문 + EXPLAIN 분석 후 실행")
#   1) validate_syntax    — 구문/보안 정적 검증 (DB 연결 불필요)
#   2) validate_plan      — EXPLAIN QUERY PLAN 정적 분석 (실제 실행 X)
#   3) execute_validated  — 검증 통과 SQL 실제 실행
# ---------------------------------------------------------------------------

# 읽기 전용 허용 prefix / 금지 키워드 (Backend execute_db 정책과 동일선상)
_READONLY_PREFIXES = ("SELECT", "WITH", "PRAGMA")
_FORBIDDEN_KEYWORDS = (
    "DROP", "DELETE", "ALTER", "INSERT", "UPDATE",
    "CREATE", "REPLACE", "TRUNCATE", "ATTACH", "DETACH", "GRANT",
)
# 문자열 리터럴 제거용 — LIKE '%delete%' 같은 값이 키워드 오탐을 일으키지 않도록 한다.
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")


def _clean_sql(raw_sql: str) -> str:
    """LLM 출력에서 마크다운 펜스/잉여 공백/후행 세미콜론을 제거한다."""
    sql = raw_sql.strip()
    sql = re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.IGNORECASE).strip()
    return sql.rstrip(";").strip()


def validate_syntax(sql: str) -> dict[str, Any]:
    """
    [1단계] 구문/보안 검증 — DB 연결 없이 정적으로 수행.
    읽기 전용 단일 문장(SELECT/WITH/PRAGMA)만 통과시킨다.
    """
    if not sql:
        return {"success": False, "stage": "syntax", "error": "빈 SQL 입니다."}

    # 다중 문장 차단 (세미콜론을 통한 추가 구문 주입 방지)
    if ";" in sql:
        return {"success": False, "stage": "syntax",
                "error": "다중 SQL 문장은 허용되지 않습니다."}

    upper = sql.upper()

    # 화이트리스트 prefix — 읽기 전용 쿼리만 허용
    if not upper.startswith(_READONLY_PREFIXES):
        return {"success": False, "stage": "syntax",
                "error": "SELECT/WITH/PRAGMA 로 시작하는 읽기 전용 쿼리만 허용됩니다."}

    # 금지 키워드 차단 (WITH CTE 뒤에 숨은 DML/DDL 까지 검출). 문자열 리터럴은 제외.
    scan_target = _STRING_LITERAL_RE.sub("''", upper)
    if any(re.search(rf"\b{kw}\b", scan_target) for kw in _FORBIDDEN_KEYWORDS):
        return {"success": False, "stage": "syntax",
                "error": "쓰기/DDL 구문이 포함되어 있습니다. (읽기 전용만 허용)"}

    return {"success": True, "stage": "syntax", "sql": sql}


def validate_plan(sql: str, conn: sqlite3.Connection) -> dict[str, Any]:
    """
    [2단계] EXPLAIN 정적 분석 — 실제 실행 없이 테이블/컬럼/구문 유효성을 검증한다.
    잘못된 테이블·컬럼 참조나 구문 오류가 여기서 sqlite3.Error 로 잡힌다.
    """
    try:
        rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
        plan = [row[-1] for row in rows]  # 각 행의 detail(마지막) 컬럼
        return {"success": True, "stage": "explain", "plan": plan}
    except sqlite3.Error as e:
        return {"success": False, "stage": "explain", "error": f"EXPLAIN 검증 실패: {e}"}


def execute_validated(sql: str, conn: sqlite3.Connection) -> dict[str, Any]:
    """
    [3단계] 실행 — 1·2단계를 통과한 SQL을 실제 실행하고 결과를 dict 리스트로 반환한다.
    """
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return {"success": True, "stage": "execute",
                "data": [dict(zip(columns, row, strict=False)) for row in rows]}
    except sqlite3.Error as e:
        # DB Exception 캡처 (LangGraph 재생성 루프에 활용)
        return {"success": False, "stage": "execute", "error": f"DB exception: {e}"}


def validate_sql_and_execute(generated_sql: str, db_path: str = DB_PATH) -> dict[str, Any]:
    """
    SQL 검증 3단계 파이프라인 오케스트레이터.

      1. validate_syntax    → 2. validate_plan → 3. execute_validated

    한 단계라도 실패하면 즉시 중단하고 {"success": False, "stage": ..., "error": ...} 를
    반환한다. 호출부(LangGraph)는 실패 stage/error 를 읽어 SQL 재생성 루프(최대 3회)에 활용.
    성공 시 {"success": True, "stage": "execute", "data": [...], "plan": [...]} 를 반환한다.
    """
    sql = _clean_sql(generated_sql)

    # 1단계: DB 연결 전 정적 구문/보안 검증
    syntax = validate_syntax(sql)
    if not syntax["success"]:
        return syntax
    sql = syntax["sql"]

    # 읽기 전용 모드로 연결 → 드라이버 레벨에서 쓰기 자체를 차단
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # 2단계: EXPLAIN 정적 분석
        plan = validate_plan(sql, conn)
        if not plan["success"]:
            return plan

        # 3단계: 실행
        result = execute_validated(sql, conn)
        if result["success"]:
            result["plan"] = plan["plan"]
        return result
    finally:
        conn.close()
