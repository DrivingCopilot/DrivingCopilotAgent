# scripts/trace_query.py
#
# 질문 하나를 그래프에 태우고 노드별 delta / LLM 호출 / 최종 state를
# traces/{MMDD_HHMMSS}.jsonl 로 남기는 디버깅용 CLI.
#
# 실행: python scripts/trace_query.py "지금 비와?" [route_type]
#   route_type: 선택. 프로덕션에서는 Backend의 classify_query()가 채워 넘기지만
#   (rag/tool/sql/vision/chat), 이 디버그 CLI는 Backend를 거치지 않으므로 기본값 ""로
#   호출된다 — 프로덕션과 같은 관측 조건을 재현하려면 명시적으로 넘긴다.

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage

from app.core.trace import TraceRecorder
from app.graph.builder import build_graph
from app.graph.state import AgentState


async def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/trace_query.py \"<질문>\" [route_type]")
    question = sys.argv[1]
    route_type = sys.argv[2] if len(sys.argv) > 2 else ""

    traces_dir = _PROJECT_ROOT / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"{datetime.now().strftime('%m%d_%H%M%S')}.jsonl"

    recorder = TraceRecorder(trace_path)
    graph = build_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "route_type": route_type,
        "plan": [],
        "next_agent": "",
        "tool_calls": [],
        "context_data": {},
        "error_count": {"timeout": 0, "parameter": 0, "invalid_tool": 0, "sql": 0},
        "feedback": "",
    }

    last_values: dict[str, Any] = dict(initial_state)
    async for mode, chunk in graph.astream(
        initial_state,
        config={"recursion_limit": 40, "callbacks": [recorder]},
        stream_mode=["updates", "values"],
    ):
        if mode == "updates":
            for node_name, delta in chunk.items():
                recorder.record_node(node_name, delta)
        elif mode == "values":
            last_values = chunk

    recorder.record_final(last_values)
    print(trace_path)


if __name__ == "__main__":
    asyncio.run(main())
