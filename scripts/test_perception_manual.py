# scripts/test_perception_manual.py
#
# perception_node 를 서버 없이 단독으로 실행해보는 수동 테스트 스크립트.
# camera_stub/sample_frame.jpg 를 실제로 읽어 Qwen2.5-VL(Ollama) 에 넘기고
# 결과를 눈으로 확인하는 용도. .env 의 PERCEPTION_VLM_BASE_URL 이 가리키는
# 서버가 떠 있어야 한다.
#
# 실행: python3 scripts/test_perception_manual.py
# (VSCode ▶ 버튼으로 실행 시 .env가 자동 로드되지 않으므로 dotenv로 직접 읽는다)

import asyncio
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from app.agents.perception import perception_node


async def main():
    state = {
        "messages": [],
        "route_type": "vision",
        "plan": [],
        "next_agent": "perception",
        "tool_calls": [],
        "context_data": {},
        "error_count": {},
        "feedback": "",
    }
    result = await perception_node(state)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
