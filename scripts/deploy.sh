#!/usr/bin/env bash
# A-Share Alpha Studio — macOS / Linux 一键部署与启动脚本
# 用法:
#   ./scripts/deploy.sh          # 首次部署（默认 quick：60 只真实 A 股 + 模型 + 模拟盘 + 启动）
#   ./scripts/deploy.sh full     # 全市场深度同步（3000+ 活跃股 + 2023-06-01 历史回补）
#   ./scripts/deploy.sh start    # 仅启动前后端服务

set -euo pipefail

MODE="${1:-quick}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "====================================================="
echo "  A-Share Alpha Studio — 一键部署与启动 ($MODE)"
echo "====================================================="

command -v python3 >/dev/null 2>&1 || { echo "错误: 未检测到 python3 (需 3.11+)"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "错误: 未检测到 npm (需 Node.js 20+)"; exit 1; }

if [ ! -f "$ROOT_DIR/backend/.env" ] && [ -f "$ROOT_DIR/backend/.env.example" ]; then
  cp "$ROOT_DIR/backend/.env.example" "$ROOT_DIR/backend/.env"
  echo "[1/5] 已自动生成 backend/.env"
fi

if [ ! -x "$ROOT_DIR/backend/.venv/bin/python" ]; then
  echo "[2/5] 创建 Python 虚拟环境并安装依赖..."
  python3 -m venv "$ROOT_DIR/backend/.venv"
  "$ROOT_DIR/backend/.venv/bin/pip" install --upgrade pip
  "$ROOT_DIR/backend/.venv/bin/pip" install -e "$ROOT_DIR/backend[dev]"
fi

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  echo "[3/5] 安装前端依赖..."
  (cd "$ROOT_DIR/frontend" && npm install --no-audit --no-fund)
fi

VENV_PY="$ROOT_DIR/backend/.venv/bin/python"
DUCKDB="$ROOT_DIR/data/market_analytics.duckdb"

if [ "$MODE" = "full" ]; then
  echo "[4/5] 全市场真实 A 股同步 + 2023-06-01 历史回补..."
  "$VENV_PY" -u "$ROOT_DIR/scripts/sync_full_market.py" --retrain
  "$VENV_PY" -u "$ROOT_DIR/scripts/extend_history.py" --start 2023-06-01 --workers 8
  "$VENV_PY" -u "$ROOT_DIR/scripts/sync_capital_flow.py" --workers 4
  "$VENV_PY" -u "$ROOT_DIR/scripts/seed_backtest_run.py"
elif [ "$MODE" = "quick" ] && [ ! -f "$DUCKDB" ]; then
  echo "[4/5] 首次部署：同步 60 只代表性真实 A 股并初始化模型与模拟盘..."
  "$VENV_PY" -u "$ROOT_DIR/scripts/m1_verify_real_data.py" --n 60 --days 730
  "$VENV_PY" -u "$ROOT_DIR/scripts/m2_verify_backtest.py"
  "$VENV_PY" -u "$ROOT_DIR/scripts/m3_verify_paper_loop.py"
fi

echo "[5/5] 启动后端 (http://localhost:8000) 与前端 (http://localhost:5173)..."
(cd "$ROOT_DIR/backend" && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000) &
BACK_PID=$!
(cd "$ROOT_DIR/frontend" && npm run dev) &
FRONT_PID=$!

trap 'kill $BACK_PID $FRONT_PID 2>/dev/null || true' EXIT INT TERM
echo "量化工作台: http://localhost:5173 | API 文档: http://localhost:8000/docs"
wait
