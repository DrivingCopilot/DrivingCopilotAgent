# scripts/test_supervisor_manual.py
#
# supervisor_node 를 서버 없이 단독으로 실행해보는 수동 테스트 스크립트.
# JSON 모드 + max_tokens 안전망이 반복 출력 버그를 막는지 확인하는 용도.
#
# 실행: python3 scripts/test_supervisor_manual.py

import asyncio
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage

from app.agents.supervisor import supervisor_node


async def main():
    state = {
        "messages": [HumanMessage(content="지금 비와?")],
        "route_type": "vision",
        "plan": [],
        "next_agent": "",
        "tool_calls": [],
        "context_data": {},
        "error_count": {},
        "feedback": "",
    }
    result = await supervisor_node(state)
    print(json.dumps(
        {k: v for k, v in result.items() if k != "messages"},
        ensure_ascii=False, indent=2,
    ))
    if "messages" in result:
        for m in result["messages"]:
            print("--- message ---")
            print(m.content[:2000])


if __name__ == "__main__":
    asyncio.run(main())
