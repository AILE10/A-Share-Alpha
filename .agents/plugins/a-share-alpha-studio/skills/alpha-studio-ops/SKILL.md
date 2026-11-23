---
name: alpha-studio-ops
description: >-
  Bootstrap, install dependencies, sync real A-share market/PIT financial data into DuckDB,
  start FastAPI + Vite (or Docker Compose) servers, and troubleshoot A-Share Alpha Studio
  in any fresh workspace or Antigravity environment. Use when deploying, initializing,
  syncing data, starting the workbench, or debugging runtime issues.
---

# A-Share Alpha Studio — One-Click Bootstrap & Operations Runbook

Follow this runbook whenever setting up **A-Share Alpha Studio** in a new machine or Antigravity workspace, or when syncing fresh A-share market data and starting the workbench.

## 1. Environment Preflight

Verify toolchains before installing:
- Python `>= 3.11` (`python --version`)
- Node.js `>= 18` and npm (`node --version; npm --version`)
- Workspace contains `backend/pyproject.toml`, `frontend/package.json`, `scripts/`, `docker-compose.yml`.

## 2. One-Time Dependency Installation

### Backend (Python Virtual Environment)
```powershell
# Windows PowerShell
python -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -e "backend[dev]"
```
```bash
# macOS / Linux
python3 -m venv backend/.venv
backend/.venv/bin/pip install -e "backend[dev]"
```
Optional Tushare Pro support (only if user has a `TUSHARE_TOKEN`):
```bash
backend/.venv/bin/pip install -e "backend[tushare]"
```

### Frontend (React + TypeScript + Vite)
```bash
cd frontend && npm install && npm run build
```
> Note: On Windows, package installation (`pip install` / `npm install`) and live A-share API syncing require network access outside the sandbox.

## 3. Initial Real A-Share Data Sync & Cold-Start Seeding (Plan A Full-Market Resumable)

NEVER fabricate or mock market data. All backend endpoints automatically analyze the ENTIRE synced DuckDB universe (`codes=None`), while intraday real-time quoting is strictly capped to held positions + Top-100 model candidates.

### Plan A: Token-Free Full-Market Active A-Share Resumable Sync (Recommended)
Scans all ~5,200 listed A-shares in <10s, bulk-downloads PIT quarterly financials for all A-shares, and multi-threaded resumable syncs all active liquid non-ST stocks (`>= 40M CNY` daily turnover & above bottom 8% market cap; automatically skips already up-to-date stocks on re-run):
```powershell
# Full active A-share market (~1,500-2,500 active stocks, resumable anytime)
backend\.venv\Scripts\python.exe -u scripts\sync_full_market.py --retrain

# Or fast cold-start with Top 300 most liquid active A-shares (~2 min)
backend\.venv\Scripts\python.exe -u scripts\sync_full_market.py --max-stocks 300 --retrain

# Seed initial Walk-Forward backtest report & live paper trading loop
backend\.venv\Scripts\python.exe -u scripts\seed_backtest_run.py
backend\.venv\Scripts\python.exe -u scripts\m3_verify_paper_loop.py
```

### Lightweight 60-Stock Verification Mode
```powershell
backend\.venv\Scripts\python.exe -u scripts\m1_verify_real_data.py --n 60 --days 730
```

## 4. Full Regression Verification

Run all 45 automated tests before launching servers:
```powershell
cd backend
.venv\Scripts\python.exe -m pytest
```
All tests must pass (covering A-share T+1, 100-share lots, 10%/20%/30% price limits, exact fees, PIT `ann_date` anti-lookahead, Embargo >= 5d, RiskMiddleware caps/circuit-breaker/rate-limiter, Assisted fill reconciliation, and Compliant QMT dry-run).

## 5. Starting the Full-Stack Workbench

- **Local Development (Windows)**: `powershell -ExecutionPolicy Bypass -File scripts\start_dev.ps1`
- **Local Development (macOS/Linux)**: `./scripts/start_dev.sh`
- **Manual Start** (two terminals):
  - Backend: `cd backend && .venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000`
  - Frontend: `cd frontend && npm run dev`
- **Docker Compose**: `docker compose up -d --build`

Verify in browser:
- Open `http://localhost:5173`, check top status bar shows `实时推流在线` (WebSocket connected), and inspect all 6 routes (`/`, `/factors`, `/backtest`, `/signals`, `/paper`, `/broker-risk`) for real data rendering and zero console/network errors.

## 6. Common Operational Pitfalls

1. **DuckDB single-writer lock**: Stop the running `uvicorn` process before executing offline CLI sync scripts (`m1_verify_real_data.py`), or trigger incremental sync via `POST /api/v1/data/sync` while `uvicorn` is running.
2. **Eastmoney `push2` temporary rate-limit**: `AKShareProvider` automatically cools down Eastmoney push2 and switches to Tencent CDN (`stock_zh_a_hist_tx` + `qt.gtimg.cn`), Sina indices, and Eastmoney bulk `stock_yjbb_em`; missing historical capital flow is gracefully tagged `DEGRADED_MISSING` / `PARTIAL_DEGRADED`.
3. **Baostock slow TCP**: `BaostockProvider` performs a 4-second fast-fail TCP probe so it never stalls the pipeline when Baostock's server is busy.
4. **Windows sandbox `asyncio` limitation**: Run `uvicorn` and Vite outside the sandbox (`BypassSandbox: true`) so localhost ports 8000/5173 and WebSocket connections are reachable by the browser.
