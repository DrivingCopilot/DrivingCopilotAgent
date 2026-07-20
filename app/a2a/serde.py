# app/a2a/serde.py
#
# AgentState.messages(LangChain 메시지 객체)를 A2ATaskRequest/Response의
# JSON 직렬화 가능한 Dict로 변환하는 얇은 래퍼.

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


def serialize_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    return messages_to_dict(messages)


def deserialize_messages(data: List[Dict[str, Any]]) -> List[BaseMessage]:
    return messages_from_dict(data)
