import asyncio
import logging
from typing import Dict, Any


from mcp import ClientSession, StdioServerParameters
from langchain_community.embeddings import HuggingFaceEmbeddings
import sqlite3
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import QDRANT_URL, COLLECTION_NAME, MODEL_NAME
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

def schema_Linking(query: str) -> Dict[str, str]:
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

def few_shot_sql(query: str, k: int=3) -> Dict[str, str]:
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



from collections import Counter

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
        llm = ChatOpenAI(model="qwen2-vl-1.5b-instruct-int4", temperature=0.7)
        
        
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

def validate_sql_and_execute(generated_sql: str, db: SQLDatabase) -> Any:
    
    forbidden_statements = ["DROP", "DELETE", "ALTER", "INSERT", "UPDATE"]
    upper_sql = generated_sql.upper()
    if any (stmt in upper_sql for stmt in forbidden_statements):
        logger.error("Generated SQL contains forbidden statements.")
        return {"success": False, "error": "Generated SQL contains forbidden statements."}

    try: 
        conn = sqlite3.connect(db.db_uri)
        cursor = conn.cursor()
        explain_query = f"EXPLAIN QUERY PLAN {generated_sql}"
        cursor.execute(explain_query)

        cursor.execute(generated_sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

        return {"success": True, "data": [dict(zip(columns, row)) for row in rows]}

    except sqlite3.Error as e:
        # 에러 발생 시 DB Exception 캡처 (이후 LangGraph에서 재생성 루프에 활용) 
        return {"success": False, "error": f"DB exception: {str(e)}"}
    finally:
        conn.close()

        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    
