param([switch]$Install, [switch]$Seed)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
if ($Install) {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw '请安装 Python 3.11/3.12 后重试。' }
    }
    & '.venv\Scripts\python.exe' -m pip install -e '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw '依赖安装失败。' }
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw '请先运行 .\scripts\start.ps1 -Install -Seed'
}
if ($Seed) {
    & '.venv\Scripts\python.exe' -m bank_agent.cli seed-demo
    if ($LASTEXITCODE -ne 0) { throw '教学数据初始化失败。' }
}
Write-Host 'API 即将运行于 http://127.0.0.1:8000；请在另一个终端运行：'
Write-Host '.\.venv\Scripts\python.exe -m streamlit run bank_agent/ui.py'
& '.venv\Scripts\python.exe' -m uvicorn bank_agent.api:app --host 127.0.0.1 --port 8000
