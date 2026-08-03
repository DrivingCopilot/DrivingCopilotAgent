#!/usr/bin/env bash
# 로컬 모델 서버(11500, 7B+1.5B 로드) + Supervisor(8001) +
# knowledge(8002)/execution(8003)/perception(8004) A2A 서버를 한 번에 띄운다.
# Ctrl+C 시 trap으로 전부 정리된다.
set -euo pipefail
cd "$(dirname "$0")/../.."

trap 'kill 0' EXIT

python -m app.model_server.server &

# 모델 서버가 두 모델(7B + 1.5B) 로딩을 마칠 때까지 대기 — 최초 실행 시
# HuggingFace 다운로드까지 겹치면 몇 분 걸릴 수 있다.
echo "모델 서버 로딩 대기 중..."
until curl -sf http://localhost:11500/health | grep -q '"ok"'; do
    sleep 2
done
echo "모델 서버 준비 완료."

python main.py &
python -m app.a2a.server knowledge &
python -m app.a2a.server execution &
python -m app.a2a.server perception &

wait
