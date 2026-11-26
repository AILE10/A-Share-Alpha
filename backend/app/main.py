"""A-Share Alpha Studio FastAPI Application Entry Point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import backtest, broker, dashboard, factors, paper, signals
from app.api.ws_manager import ws_manager
from app.config import get_settings
from app.core.calendar import now_cn
from app.core.logging import get_logger, setup_logging
from app.core.task_manager import action_task_manager
from app.db.duckdb_client import get_duckdb
from app.db.sqlite_client import get_sqlite_db
from app.paper.scheduler import StudioScheduler

logger = get_logger(__name__)

_scheduler: StudioScheduler | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _scheduler
    settings = get_settings()
    setup_logging(settings.log_level)
    get_duckdb()
    get_sqlite_db()
    ws_manager.bind_loop(asyncio.get_running_loop())
    _scheduler = StudioScheduler()
    _scheduler.start()
    logger.info("A-Share Alpha Studio API started on %s:%s", settings.app_host, settings.app_port)
    try:
        yield
    finally:
        if _scheduler is not None:
            _scheduler.stop()
            _scheduler = None


app = FastAPI(
    title="A-Share Alpha Studio API",
    version="0.1.0",
    description="个人投资者 A 股全栈实时量化投研、Walk-Forward 回测优化、模拟实盘与合规辅助交易系统",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"
app.include_router(dashboard.router, prefix=API_PREFIX)
app.include_router(factors.router, prefix=API_PREFIX)
app.include_router(backtest.router, prefix=API_PREFIX)
app.include_router(signals.router, prefix=API_PREFIX)
app.include_router(paper.router, prefix=API_PREFIX)
app.include_router(broker.router, prefix=API_PREFIX)


@app.get("/api/v1/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "A-Share Alpha Studio",
        "timestamp": now_cn().isoformat(),
        "ws_clients": ws_manager.active_count,
        "scheduler_jobs": _scheduler.list_jobs() if _scheduler else [],
    }


@app.get("/api/v1/scheduler/jobs")
def list_scheduler_jobs() -> list[dict]:
    return _scheduler.list_jobs() if _scheduler else []


@app.get("/api/v1/tasks")
def list_action_tasks(
    active_only: bool = Query(default=False),
    kind: str | None = Query(default=None),
) -> list[dict]:
    return action_task_manager.list_tasks(kind=kind, active_only=active_only)


@app.get("/api/v1/tasks/{task_id}")
def get_action_task(task_id: str) -> dict:
    snap = action_task_manager.get(task_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return snap


@app.websocket("/api/v1/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws_manager.connect(ws)
    try:
        await ws.send_json({
            "channel": "system",
            "timestamp": now_cn().isoformat(),
            "payload": {"connected": True, "message": "A-Share Alpha Studio 实时流已连接"},
        })
        while True:
            msg = await ws.receive_text()
            if "ping" in msg:
                await ws.send_json({"type": "pong", "ts": now_cn().isoformat()})
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:  # noqa: BLE001
        ws_manager.disconnect(ws)
