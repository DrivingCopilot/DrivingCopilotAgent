# 로컬 모델 서버(11500, 7B+1.5B 로드) + Supervisor(8001) +
# knowledge(8002)/execution(8003)/perception(8004) A2A 서버를 한 번에 띄운다.
# scripts/run_agents.sh의 Windows PowerShell 포팅.
#
# 원본 sh와의 의도적 차이:
# - `trap 'kill 0' EXIT` (Ctrl+C 시 전체 일괄 종료)는 구현하지 않는다.
#   대신 서비스별로 창을 분리했으므로 종료는 각 창을 개별로 닫는 방식으로 한다.
# - 끝의 `wait`도 창 분리 방식에서는 불필요하므로 넣지 않는다.

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$activatePath = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"

function Start-AgentWindow {
    param(
        [string]$Title,
        [string]$Command
    )

    $inner = "`$host.UI.RawUI.WindowTitle = '$Title'; & '$activatePath'; $Command"
    Start-Process powershell -ArgumentList @('-NoExit', '-Command', $inner) -WorkingDirectory $repoRoot
}

Start-AgentWindow -Title "model-server" -Command "python -m app.model_server.server"

# 모델 서버가 두 모델(7B GPTQ-Int4 + 1.5B) 로딩을 마칠 때까지 대기 — 최초 실행 시
# HuggingFace 다운로드까지 겹치면 몇 분 걸릴 수 있다.
Write-Host "모델 서버 로딩 대기 중..."
while ($true) {
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:11500/health" -Method Get
        if ($resp.status -eq "ok") {
            break
        }
    } catch {
    }
    Start-Sleep -Seconds 2
}
Write-Host "모델 서버 준비 완료."

Start-AgentWindow -Title "supervisor" -Command "python main.py"
Start-AgentWindow -Title "knowledge" -Command "python -m app.a2a.server knowledge"
Start-AgentWindow -Title "execution" -Command "python -m app.a2a.server execution"
Start-AgentWindow -Title "perception" -Command "python -m app.a2a.server perception"
