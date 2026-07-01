import pytest
from unittest.mock import patch, MagicMock
import sqlite3

from app.services.text2sql import (
    schema_Linking,
    few_shot_sql,
    self_consistency,
    validate_syntax,
    validate_plan,
    execute_validated,
    validate_sql_and_execute,
    _clean_sql,
)

@pytest.fixture
def dummy_schemas():
    return {
        "vehicle_telemetry": "table: vehicle_telemetry, columns: ts, speed, rpm, fuel, battery, tire (실시간 차량 센서 데이터)",
        "maintenance_log": "table: maintenance_log, columns: date, mileage, type, parts, cost (차량 정비 이력 데이터)",
        "dtc_codes": "table: dtc_codes, columns: code, description, severity, resolved (OBD-II 오류 및 진단 코드)",
        "trip_history": "table: trip_history, columns: start, end, distance, avg_speed, fuel (차량 운행 기록 데이터)"
    }

# 1. schema_Linking Tests
def test_schema_linking_match(dummy_schemas):
    # '속도'는 vehicle_telemetry 에 매칭됨
    result = schema_Linking("현재 차량의 속도를 알려줘")
    assert "vehicle_telemetry" in result
    assert result["vehicle_telemetry"] == dummy_schemas["vehicle_telemetry"]
    assert len(result) == 1

def test_schema_linking_multiple_match(dummy_schemas):
    # '속도'(vehicle_telemetry), '정비'(maintenance_log)
    result = schema_Linking("차량 속도와 정비 이력을 알려줘")
    assert "vehicle_telemetry" in result
    assert "maintenance_log" in result
    assert len(result) == 2

def test_schema_linking_no_match(dummy_schemas):
    # 매칭되는 키워드가 없으면 전체 스키마 반환
    result = schema_Linking("안녕하세요")
    assert result == dummy_schemas

# 2. few_shot_sql Tests
@patch("app.services.text2sql.QdrantVectorStore")
@patch("app.services.text2sql.QdrantClient")
@patch("app.services.text2sql.HuggingFaceEmbeddings")
def test_few_shot_sql_success(mock_embeddings, mock_qdrant_client, mock_vector_store):
    mock_vs_instance = MagicMock()
    mock_vector_store.return_value = mock_vs_instance
    
    mock_doc = MagicMock()
    mock_doc.metadata = {"question": "dummy question?", "sql": "SELECT * FROM dummy;"}
    mock_vs_instance.similarity_search.return_value = [mock_doc, mock_doc]
    
    result = few_shot_sql("dummy query", k=2)
    assert "## 참고용 유사 SQL 작성 예시" in result
    assert "예시 1, 질문 : dummy question?" in result
    assert "SQL: SELECT * FROM dummy;" in result

# 3. self_consistency Tests
@patch("app.services.text2sql.SQLDatabase")
@patch("app.services.text2sql.ChatOpenAI")
@patch("app.services.text2sql.PromptTemplate")
def test_self_consistency_success(mock_prompt, mock_chat, mock_sql_db):
    # Mock chain object inside self_consistency
    # We will patch the specific invoke method
    # Actually, we need to mock the pipeline operator or just patch _generate_sql_from_llm
    pass # Will rewrite using patch for the local function or chain

@patch("app.services.text2sql.SQLDatabase")
@patch("app.services.text2sql.ChatOpenAI")
def test_self_consistency_majority_vote(mock_chat, mock_sqldb):
    with patch("app.services.text2sql.PromptTemplate") as mock_prompt:
        # Mocking the chain behavior
        mock_chain = MagicMock()
        
        # Returns SELECT A 3 times and SELECT B 2 times
        mock_chain.invoke.side_effect = [
            "SELECT A FROM table", 
            "SELECT B FROM table", 
            "SELECT A FROM table", 
            "SELECT B FROM table", 
            "SELECT A FROM table"
        ]
        
        # To make prompt | llm | parser return mock_chain, we can mock __or__ on PromptTemplate instance
        mock_prompt_instance = MagicMock()
        mock_prompt.from_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__.return_value.__or__.return_value = mock_chain

        result = self_consistency("query", "table_info", "few_shots", n=5)
        assert result == "SELECT A FROM table"

@patch("app.services.text2sql.SQLDatabase")
@patch("app.services.text2sql.PromptTemplate")
def test_self_consistency_empty(mock_prompt, mock_sqldb):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = ""
    mock_prompt_instance = MagicMock()
    mock_prompt.from_template.return_value = mock_prompt_instance
    mock_prompt_instance.__or__.return_value.__or__.return_value = mock_chain
    
    result = self_consistency("query", "table_info", "few_shots", n=3)
    assert result == ""

# 4. SQL 검증 3단계 파이프라인 Tests

# --- 0) _clean_sql: 마크다운 펜스/세미콜론 정제 ---
def test_clean_sql_strips_markdown_fence():
    assert _clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert _clean_sql("  SELECT * FROM trip_history ;  ") == "SELECT * FROM trip_history"

# --- 1단계) validate_syntax: 구문/보안 정적 검증 ---
def test_syntax_allows_readonly_select():
    result = validate_syntax("SELECT * FROM trip_history")
    assert result["success"] is True
    assert result["stage"] == "syntax"

def test_syntax_blocks_ddl():
    result = validate_syntax("DROP TABLE users")
    assert result["success"] is False
    assert result["stage"] == "syntax"

def test_syntax_blocks_multi_statement():
    result = validate_syntax("SELECT 1; DROP TABLE users")
    assert result["success"] is False
    assert "다중" in result["error"]

def test_syntax_blocks_dml_hidden_in_cte():
    # WITH(허용 prefix) 뒤에 숨은 DELETE 까지 검출
    result = validate_syntax("WITH x AS (SELECT 1) DELETE FROM dtc_codes")
    assert result["success"] is False
    assert result["stage"] == "syntax"

def test_syntax_keyword_in_string_literal_is_ok():
    # 문자열 리터럴 안의 키워드는 오탐하지 않는다.
    result = validate_syntax("SELECT * FROM dtc_codes WHERE description LIKE '%delete%'")
    assert result["success"] is True

# --- 2단계) validate_plan: EXPLAIN 정적 분석 ---
@patch("app.services.text2sql.sqlite3.connect")
def test_validate_plan_success(mock_connect):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [(0, 0, 0, "SCAN trip_history")]
    result = validate_plan("SELECT * FROM trip_history", conn)
    assert result["success"] is True
    assert result["stage"] == "explain"
    assert result["plan"] == ["SCAN trip_history"]

def test_validate_plan_bad_reference():
    conn = MagicMock()
    conn.execute.side_effect = sqlite3.Error("no such table: ghost")
    result = validate_plan("SELECT * FROM ghost", conn)
    assert result["success"] is False
    assert result["stage"] == "explain"

# --- 3단계) execute_validated: 실제 실행 ---
def test_execute_validated_success():
    conn = MagicMock()
    cursor = conn.execute.return_value
    cursor.description = (("id", None), ("name", None))
    cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]
    result = execute_validated("SELECT * FROM users", conn)
    assert result["success"] is True
    assert result["stage"] == "execute"
    assert result["data"][0] == {"id": 1, "name": "Alice"}

def test_execute_validated_exception():
    conn = MagicMock()
    conn.execute.side_effect = sqlite3.Error("Syntax error")
    result = execute_validated("SELECT * FROM", conn)
    assert result["success"] is False
    assert "DB exception" in result["error"]

# --- 오케스트레이터) 3단계 연결 + 조기 중단 ---
def test_pipeline_stops_at_syntax_stage():
    # 1단계에서 막히면 DB 연결조차 하지 않는다.
    result = validate_sql_and_execute("DROP TABLE users", db_path="unused.db")
    assert result["success"] is False
    assert result["stage"] == "syntax"

@patch("app.services.text2sql.sqlite3.connect")
def test_pipeline_full_success(mock_connect):
    conn = MagicMock()
    mock_connect.return_value = conn

    explain_cursor = MagicMock()
    explain_cursor.fetchall.return_value = [(0, 0, 0, "SCAN trip_history")]
    exec_cursor = MagicMock()
    exec_cursor.description = (("distance", None),)
    exec_cursor.fetchall.return_value = [(120.5,)]
    # 2단계 EXPLAIN → 3단계 실행 순서로 conn.execute 가 두 번 호출된다.
    conn.execute.side_effect = [explain_cursor, exec_cursor]

    result = validate_sql_and_execute("SELECT distance FROM trip_history", db_path="dummy.db")
    assert result["success"] is True
    assert result["stage"] == "execute"
    assert result["data"] == [{"distance": 120.5}]
    assert result["plan"] == ["SCAN trip_history"]
    # 읽기 전용 모드로 연결했는지 확인
    assert "mode=ro" in mock_connect.call_args[0][0]
