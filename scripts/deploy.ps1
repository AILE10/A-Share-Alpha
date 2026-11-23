<#
.SYNOPSIS
    A-Share Alpha Studio — Windows 一键部署与启动脚本（支持任意新电脑开箱即用）
.EXAMPLE
    # 首次在新电脑部署（自动装环境 + 同步 60 只代表性真实 A 股 + 跑通模型与模拟盘 + 启动前后端）
    powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1

    # 仅启动前后端（已有数据）
    powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Mode start

    # 全市场深度初始化（3000+ 活跃 A 股 + 回补至 2023-06-01，耗时较长但支持断点续传）
    powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Mode full
#>
param(
    [ValidateSet("quick", "full", "start")]
    [string]$Mode = "quick",
    [switch]$SkipTests,
    [switch]$NoOpenBrowser
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  A-Share Alpha Studio — Windows 一键部署与启动" -ForegroundColor Cyan
Write-Host "  模式: $Mode | 目录: $Root" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# 1. 环境检查
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "未检测到 Python，请先安装 Python 3.11+（https://www.python.org/downloads/）并勾选 Add to PATH。"
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "未检测到 Node.js / npm，请先安装 Node.js 20+ LTS（https://nodejs.org/）。"
}

# 2. 初始化环境配置（无个人敏感信息，默认免 Token 运行）
$EnvFile = Join-Path $Root "backend\.env"
$EnvExample = Join-Path $Root "backend\.env.example"
if (-not (Test-Path $EnvFile) -and (Test-Path $EnvExample)) {
    Copy-Item $EnvExample $EnvFile
    Write-Host "[1/5] 已自动生成 backend\.env（默认免 Token 使用 AKShare + Baostock 真实行情）" -ForegroundColor Green
} else {
    Write-Host "[1/5] 保留现有 backend\.env 配置" -ForegroundColor DarkGray
}

# 3. 后端虚拟环境与依赖安装
$VenvPy = Join-Path $Root "backend\.venv\Scripts\python.exe"
$Uvicorn = Join-Path $Root "backend\.venv\Scripts\uvicorn.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Host "[2/5] 创建 Python 虚拟环境并安装后端依赖..." -ForegroundColor Yellow
    python -m venv (Join-Path $Root "backend\.venv")
    & $VenvPy -m pip install --upgrade pip
    & $VenvPy -m pip install -e "$Root\backend[dev]"
} else {
    Write-Host "[2/5] 检测到已有 Python 虚拟环境，跳过重复安装" -ForegroundColor DarkGray
}

# 4. 前端依赖安装
$NodeModules = Join-Path $Root "frontend\node_modules"
if (-not (Test-Path $NodeModules)) {
    Write-Host "[3/5] 安装前端 React/Vite 依赖..." -ForegroundColor Yellow
    Push-Location (Join-Path $Root "frontend")
    npm install --no-audit --no-fund
    Pop-Location
} else {
    Write-Host "[3/5] 检测到已有 frontend\node_modules，跳过重复安装" -ForegroundColor DarkGray
}

# 5. 首次真实数据与模型初始化（仅当 DuckDB 尚不存在或显式选择 full 时执行）
$DuckDb = Join-Path $Root "data\market_analytics.duckdb"
if ($Mode -eq "full") {
    Write-Host "[4/5] 执行全市场真实 A 股同步 + 2023-06-01 历史回补 + 资金流 + 模型初始化..." -ForegroundColor Yellow
    & $VenvPy -u (Join-Path $Root "scripts\sync_full_market.py") --retrain
    & $VenvPy -u (Join-Path $Root "scripts\extend_history.py") --start 2023-06-01 --workers 8
    & $VenvPy -u (Join-Path $Root "scripts\sync_capital_flow.py") --workers 4
    & $VenvPy -u (Join-Path $Root "scripts\seed_backtest_run.py")
} elseif ($Mode -eq "quick" -and -not (Test-Path $DuckDb)) {
    Write-Host "[4/5] 首次部署：同步 60 只代表性真实 A 股（近 2 年）、训练初始模型并初始化模拟盘..." -ForegroundColor Yellow
    & $VenvPy -u (Join-Path $Root "scripts\m1_verify_real_data.py") --n 60 --days 730
    & $VenvPy -u (Join-Path $Root "scripts\m2_verify_backtest.py")
    & $VenvPy -u (Join-Path $Root "scripts\m3_verify_paper_loop.py")
} else {
    Write-Host "[4/5] 本地行情数据库已存在（或选择 start 模式），直接进入启动" -ForegroundColor DarkGray
}

# 可选测试检查
if (-not $SkipTests -and $Mode -ne "start" -and (Test-Path $DuckDb)) {
    Write-Host "[测试] 运行核心风控与撮合单元测试..." -ForegroundColor Cyan
    Push-Location (Join-Path $Root "backend")
    & $VenvPy -m pytest tests/test_risk_and_broker.py -q
    Pop-Location
}

# 6. 启动后端 FastAPI 与前端 Vite
Write-Host "[5/5] 启动后端 (http://127.0.0.1:8000) 与前端 (http://localhost:5173)..." -ForegroundColor Green
$backendProc = Start-Process -FilePath $Uvicorn `
    -ArgumentList "app.main:app", "--host", "0.0.0.0", "--port", "8000" `
    -WorkingDirectory (Join-Path $Root "backend") -PassThru

$frontendProc = Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory (Join-Path $Root "frontend") -PassThru

Write-Host ""
Write-Host "部署并启动成功！" -ForegroundColor Green
Write-Host "  后端 PID: $($backendProc.Id)  | Swagger API: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  前端 PID: $($frontendProc.Id) | 量化工作台:  http://localhost:5173" -ForegroundColor Cyan
Write-Host "  提示: 关闭弹出的两个终端窗口即可停止服务。" -ForegroundColor DarkGray

if (-not $NoOpenBrowser) {
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:5173"
}
