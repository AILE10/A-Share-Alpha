#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== 启动 A-Share Alpha Studio ==="

if [ ! -d "$ROOT_DIR/backend/.venv" ]; then
  python3 -m venv "$ROOT_DIR/backend/.venv"
  "$ROOT_DIR/backend/.venv/bin/pip" install -e "$ROOT_DIR/backend[dev]"
fi

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  (cd "$ROOT_DIR/frontend" && npm install)
fi

(cd "$ROOT_DIR/backend" && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000) &
BACK_PID=$!
(cd "$ROOT_DIR/frontend" && npm run dev) &
FRONT_PID=$!

trap 'kill $BACK_PID $FRONT_PID 2>/dev/null || true' EXIT INT TERM
echo "前端工作台: http://localhost:5173 | API 文档: http://localhost:8000/docs"
wait
