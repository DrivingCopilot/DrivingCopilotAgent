import pytest
from unittest.mock import patch, MagicMock
import sqlite3

from app.services.text2sql import (
    schema_Linking,
    few_shot_sql,
    self_consistency,
    validate_sql_and_execute
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

# 4. validate_sql_and_execute Tests
def test_validate_sql_forbidden():
    mock_db = MagicMock()
    result = validate_sql_and_execute("DROP TABLE users;", mock_db)
    assert result["success"] == False
    assert "forbidden" in result["error"]

@patch("app.services.text2sql.sqlite3.connect")
def test_validate_sql_execute_success(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    
    mock_cursor.description = (("id", None), ("name", None))
    mock_cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]
    
    mock_db = MagicMock()
    mock_db.db_uri = "sqlite:///dummy.db"
    
    result = validate_sql_and_execute("SELECT * FROM users;", mock_db)
    
    assert result["success"] == True
    assert len(result["data"]) == 2
    assert result["data"][0]["id"] == 1
    assert result["data"][0]["name"] == "Alice"

@patch("app.services.text2sql.sqlite3.connect")
def test_validate_sql_execute_exception(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    
    mock_cursor.execute.side_effect = sqlite3.Error("Syntax error")
    
    mock_db = MagicMock()
    mock_db.db_uri = "sqlite:///dummy.db"
    
    result = validate_sql_and_execute("SELECT * FROM", mock_db)
    
    assert result["success"] == False
    assert "DB exception" in result["error"]
