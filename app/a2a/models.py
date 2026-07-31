# app/a2a/models.py
#
# A2A 프로토콜 공통 데이터 모델.
# Agent Card: Agent의 이름/URL/능력을 담은 명함.
# Task Request/Response: Agent 간 작업 위임 메시지 형식.

from typing import Any

from pydantic import BaseModel


class AgentCapabilities(BaseModel):
    mcp_tools: list[str] = []


class AgentCard(BaseModel):
    name: str
    description: str
    url: str                          # Agent 서버 주소 (ex: "http://localhost:8001")
    capabilities: AgentCapabilities
    version: str = "0.1.0"


class A2ATaskRequest(BaseModel):
    task_id: str
    agent_name: str
    instruction: str
    context: dict[str, Any] = {}


class A2ATaskResponse(BaseModel):
    task_id: str
    status: str                       # "success" | "error"
    result: dict[str, Any] = {}
    error: str | None = None
    error_type: str | None = None  # error 시만: "timeout" | "parameter" | "invalid_tool" | "sql"
