# scripts/test_supervisor_loop_manual.py
#
# supervisor → perception → supervisor 2회 루프를 그대로 재현해,
# perception 결과를 supervisor가 실제로 보고 __end__ 로 종료하는지 확인한다.
#
# 실행: python3 scripts/test_supervisor_loop_manual.py

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
from app.agents.perception import perception_node


async def main():
    state = {
        "messages": [HumanMessage(content="밖에 비와?")],
        "route_type": "vision",
        "plan": [],
        "next_agent": "",
        "tool_calls": [],
        "context_data": {},
        "error_count": {},
        "feedback": "",
    }

    for round_num in range(1, 4):
        print(f"\n===== Round {round_num}: supervisor_node =====")
        result = await supervisor_node(state)
        print(json.dumps({k: v for k, v in result.items() if k != "messages"}, ensure_ascii=False, indent=2))
        state.update(result)

        if state["next_agent"] == "__end__":
            print(">>> __end__ 도달 — 루프 정상 종료")
            return
        if state["next_agent"] != "perception":
            print(f">>> next_agent={state['next_agent']!r} — 이 테스트는 perception 경로만 재현함")
            return

        print(f"\n===== Round {round_num}: perception_node =====")
        p_result = await perception_node(state)
        print(json.dumps(p_result, ensure_ascii=False, indent=2))
        state["context_data"] = {**state.get("context_data", {}), **p_result.get("context_data", {})}

    print(">>> 3라운드 넘어가도록 안 끝남 — 여전히 루프")


if __name__ == "__main__":
    asyncio.run(main())
