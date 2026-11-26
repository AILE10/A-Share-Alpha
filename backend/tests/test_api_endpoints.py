"""M4/M5 FastAPI router & OpenAPI contract tests against REAL DuckDB/SQLite data (sync, sandbox-safe)."""

from __future__ import annotations

from app.api.routers import backtest, broker, dashboard, factors, paper, signals
from app.core.constants import ALL_FACTOR_NAMES
from app.main import app


def test_openapi_routes_registered():
    openapi_paths = set(app.openapi()["paths"].keys())
    ws_paths = {getattr(r, "path", "") for r in app.routes}
    paths = openapi_paths | ws_paths
    expected = {
        "/api/v1/health",
        "/api/v1/tasks",
        "/api/v1/tasks/{task_id}",
        "/api/v1/ws/stream",
        "/api/v1/dashboard/summary",
        "/api/v1/dashboard/workflow",
        "/api/v1/dashboard/regime/refresh",
        "/api/v1/dashboard/regime/refresh-async",
        "/api/v1/data/status",
        "/api/v1/factors/summary",
        "/api/v1/factors/correlation",
        "/api/v1/factors/{factor_name}/detail",
        "/api/v1/factors/compute-async",
        "/api/v1/backtest/run",
        "/api/v1/backtest/run-async",
        "/api/v1/backtest/jobs",
        "/api/v1/backtest/jobs/{job_id}",
        "/api/v1/backtest/cache-status",
        "/api/v1/backtest/cache-clear",
        "/api/v1/backtest/runs",
        "/api/v1/signals/latest",
        "/api/v1/signals/refresh",
        "/api/v1/signals/refresh-async",
        "/api/v1/signals/kline/{ts_code}",
        "/api/v1/paper/account",
        "/api/v1/paper/pre-market-unlock-async",
        "/api/v1/paper/intraday-refresh-async",
        "/api/v1/paper/rebalance-async",
        "/api/v1/paper/reset",
        "/api/v1/paper/reviews",
        "/api/v1/paper/reviews/generate-async",
        "/api/v1/models/versions",
        "/api/v1/models/retrain-async",
        "/api/v1/broker/config",
        "/api/v1/risk/config",
        "/api/v1/risk/audit-logs",
    }
    missing = expected - paths
    assert not missing, f"Missing API routes: {missing}"


def test_core_routers_on_real_data(real_db, sample_codes):
    ds = dashboard.get_data_status()
    assert ds["stocks_synced"] >= 50

    dash = dashboard.get_dashboard_summary()
    assert dash["regime"]["regime"] in ("OFFENSIVE", "NEUTRAL", "DEFENSIVE")
    assert "risk_watermarks" in dash
    assert "daily_workflow" in dash and len(dash["daily_workflow"]["steps"]) == 5

    wf = dashboard.get_daily_workflow()
    assert len(wf["steps"]) == 5

    fac = factors.get_factors_summary(horizon=5, decay_window=20)
    assert len(fac["factors"]) == len(ALL_FACTOR_NAMES) == 20

    corr = factors.get_factor_correlation()
    assert len(corr["matrix"]) == len(ALL_FACTOR_NAMES) == 20

    detail = factors.get_factor_detail("amihud_20d", horizon=5)
    assert len(detail["quintile_nav"]) > 0

    kline = signals.get_stock_kline(sample_codes[0], limit=120)
    assert len(kline["candles"]) >= 30
    assert kline["ts_code"] == sample_codes[0]

    risk = broker.get_risk_config()
    assert risk["max_single_weight"] == 0.12
    assert risk["max_industry_weight"] == 0.25

    brk = broker.get_broker_config()
    assert brk["active_mode"] in ("PAPER", "ASSISTED", "QMT")
    assert set(brk["adapters_health"].keys()) == {"PAPER", "ASSISTED", "QMT"}

    acct = paper.get_paper_account()
    assert "account" in acct and "positions" in acct
    assert isinstance(backtest.list_backtest_runs(), list)


def test_background_task_manager_lifecycle_and_dedup():
    import threading
    import time

    from app.core.task_manager import BackgroundTaskManager

    mgr = BackgroundTaskManager()
    gate = threading.Event()
    stages_seen: list[tuple[str, float]] = []

    def _work(emit):
        emit("step 1", 0.35)
        stages_seen.append(("step 1", 0.35))
        gate.wait(timeout=3.0)
        emit("step 2", 0.85)
        stages_seen.append(("step 2", 0.85))
        return {"ok": True, "count": 42}

    t1 = mgr.submit(kind="unit_test_task", title="单元测试任务", fn=_work)
    assert t1["status"] == "RUNNING"
    # Concurrent submit of same kind must deduplicate & return existing running task
    t2 = mgr.submit(kind="unit_test_task", title="单元测试任务", fn=_work)
    assert t2["task_id"] == t1["task_id"]
    assert len(mgr.list_tasks(active_only=True, kind="unit_test_task")) == 1

    gate.set()
    deadline = time.monotonic() + 3.0
    final = None
    while time.monotonic() < deadline:
        snap = mgr.get(t1["task_id"])
        if snap and snap["status"] == "DONE":
            final = snap
            break
        time.sleep(0.02)

    assert final is not None
    assert final["pct"] == 1.0
    assert final["result"] == {"ok": True, "count": 42}
    assert len(stages_seen) == 2

    # Failing task transitions cleanly to FAILED with error string
    def _fail(emit):
        emit("about to fail", 0.2)
        raise RuntimeError("boom-expected")

    tf = mgr.submit(kind="unit_fail_task", title="失败测试", fn=_fail)
    deadline = time.monotonic() + 3.0
    failed = None
    while time.monotonic() < deadline:
        snap = mgr.get(tf["task_id"])
        if snap and snap["status"] == "FAILED":
            failed = snap
            break
        time.sleep(0.02)
    assert failed is not None
    assert "boom-expected" in failed["error"]
