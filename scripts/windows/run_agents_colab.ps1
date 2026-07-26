# Supervisor(8001) + knowledge(8002)/execution(8003)/perception(8004) A2A 서버만 띄운다.
# scripts/windows/run_agents.ps1의 변형 — 모델 서버는 로컬이 아니라 Colab GPU에서 띄운다.
#
# 선행 조건:
# 1. notebooks/colab_model_server.ipynb를 Colab에서 실행해 모델 서버 + ngrok을 띄워둔다.
# 2. 노트북이 출력한 ngrok URL을 로컬 레포의 .env에 다음과 같이 갱신한다:
#      MODEL_SERVER_URL=<ngrok-url>/v1
#
# 이 스크립트는 로컬 모델 서버(app.model_server.server)를 기동하지 않고,
# .env의 MODEL_SERVER_URL을 한 번만 확인한 뒤(하드 블록 없이) 에이전트 4개만 띄운다.

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

$activatePath = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
$envPath = Join-Path $repoRoot ".env"

function Start-AgentWindow {
    param(
        [string]$Title,
        [string]$Command
    )

    $inner = "`$host.UI.RawUI.WindowTitle = '$Title'; & '$activatePath'; $Command"
    Start-Process powershell -ArgumentList @('-NoExit', '-Command', $inner) -WorkingDirectory $repoRoot
}

# .env에서 MODEL_SERVER_URL을 읽는다 (주석/빈 줄 제외).
$modelServerUrl = $null
if (Test-Path $envPath) {
    foreach ($line in Get-Content $envPath) {
        if ($line -match '^\s*MODEL_SERVER_URL\s*=\s*(.+?)\s*$') {
            $modelServerUrl = $matches[1].Trim().Trim('"').Trim("'")
        }
    }
}

if (-not $modelServerUrl) {
    Write-Host ".env에 MODEL_SERVER_URL이 없습니다. colab_model_server.ipynb 실행 후 ngrok URL을 .env에 설정하세요." -ForegroundColor Yellow
} else {
    $healthUrl = ($modelServerUrl -replace '/v1/?$', '').TrimEnd('/') + "/health"
    Write-Host "모델 서버 상태 확인 중: $healthUrl"
    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 5
        if ($resp.status -eq "ok") {
            Write-Host "모델 서버 준비 완료 ($healthUrl)" -ForegroundColor Green
        } else {
            Write-Host "모델 서버 상태가 ok가 아닙니다 ($healthUrl): $($resp | ConvertTo-Json -Compress)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "모델 서버에 연결할 수 없습니다 ($healthUrl): $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

Start-AgentWindow -Title "supervisor" -Command "python main.py"
Start-AgentWindow -Title "knowledge" -Command "python -m app.a2a.server knowledge"
Start-AgentWindow -Title "execution" -Command "python -m app.a2a.server execution"
Start-AgentWindow -Title "perception" -Command "python -m app.a2a.server perception"
