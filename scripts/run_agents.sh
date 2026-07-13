#!/usr/bin/env bash
# Supervisor(8001) + knowledge(8002)/execution(8003)/perception(8004) A2A 서버를
# 한 번에 띄운다. Ctrl+C 시 trap으로 전부 정리된다.
set -euo pipefail
cd "$(dirname "$0")/.."

trap 'kill 0' EXIT

python main.py &
python -m app.a2a.server knowledge &
python -m app.a2a.server execution &
python -m app.a2a.server perception &

wait
