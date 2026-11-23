# A-Share Alpha Studio Windows 一键启动脚本（后端 FastAPI + 前端 Vite）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "=== 启动 A-Share Alpha Studio ===" -ForegroundColor Cyan

if (-not (Test-Path "$Root\backend\.venv\Scripts\python.exe")) {
    Write-Host "[1/3] 初始化 Python 虚拟环境并安装依赖..." -ForegroundColor Yellow
    python -m venv "$Root\backend\.venv"
    & "$Root\backend\.venv\Scripts\python.exe" -m pip install -e "$Root\backend[dev]"
}

if (-not (Test-Path "$Root\frontend\node_modules")) {
    Write-Host "[2/3] 安装前端依赖..." -ForegroundColor Yellow
    Push-Location "$Root\frontend"
    npm install
    Pop-Location
}

Write-Host "[3/3] 启动后端 (http://127.0.0.1:8000) 与前端 (http://127.0.0.1:5173)..." -ForegroundColor Green

$backendJob = Start-Process -FilePath "$Root\backend\.venv\Scripts\uvicorn.exe" `
    -ArgumentList "app.main:app", "--host", "0.0.0.0", "--port", "8000" `
    -WorkingDirectory "$Root\backend" -PassThru

$frontendJob = Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory "$Root\frontend" -PassThru

Write-Host "后端 PID: $($backendJob.Id) | 前端 PID: $($frontendJob.Id)" -ForegroundColor Green
Write-Host "浏览器访问: http://localhost:5173  |  API 文档: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "按 Ctrl+C 退出（或手动关闭弹出的终端窗口）" -ForegroundColor Gray
