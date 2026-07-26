# run_agents.ps1 / run_agents_colab.ps1로 띄운 창들은 Start-Process로 분리되어 있어
# 종료가 수동이다. 이 스크립트는 포트별로 LISTENING 중인 프로세스를 netstat로 찾아
# 한 번에 정리한다.

$portServiceMap = [ordered]@{
    "8001"  = "supervisor"
    "8002"  = "knowledge"
    "8003"  = "execution"
    "8004"  = "perception"
    "11500" = "model-server"
}

$netstatOutput = netstat -ano

foreach ($port in $portServiceMap.Keys) {
    $service = $portServiceMap[$port]
    $pids = @()

    foreach ($line in $netstatOutput) {
        if ($line -match '^\s*TCP') {
            $fields = ($line -split '\s+') | Where-Object { $_ -ne '' }
            if ($fields.Count -ge 5 -and $fields[3] -eq 'LISTENING') {
                $localAddr = $fields[1]
                $localPort = $localAddr.Substring($localAddr.LastIndexOf(':') + 1)
                if ($localPort -eq "$port") {
                    $pids += $fields[4]
                }
            }
        }
    }

    $pids = $pids | Select-Object -Unique

    if ($pids.Count -eq 0) {
        Write-Host "포트 $port : 실행 중 아님"
        continue
    }

    foreach ($p in $pids) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        Write-Host "포트 $port ($service) 종료: PID $p"
    }
}
