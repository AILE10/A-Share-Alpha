"""Backtest & Walk-Forward Out-of-Sample Optimization Center endpoints.

Supports both synchronous `/backtest/run` (CLI/scripts) and zero-timeout asynchronous
`/backtest/run-async` with live WebSocket progress streaming, polling recovery,
two-level DuckDB-watermark research caching, and multi-experiment comparison.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.ws_manager import ws_manager
from app.backtest.engine import BacktestConfig
from app.db.duckdb_client import get_duckdb
from app.db.models import BacktestRun
from app.db.sqlite_client import get_sqlite_db
from app.factors.service import to_jsonable
from app.ml.lgbm_scorer import LGBMConfig
from app.strategy.alpha_strategy import (
    AlphaStrategyService,
    clear_research_cache,
    get_research_cache_status,
)

router = APIRouter(prefix="/backtest", tags=["Backtest & Optimization"])

_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}


class BacktestRunRequest(BaseModel):
    name: str = "A-Share Alpha 全市场回测"
    start_date: date | None = None
    end_date: date | None = None
    model_mode: str = Field(default="regression", pattern="^(lambdarank|regression)$")
    label_horizon: int = Field(default=10, ge=5, le=20)
    ensemble_seeds_count: int = Field(default=6, ge=1, le=12)
    weight_method: str = Field(default="risk_parity", pattern="^(risk_parity|equal_weight)$")
    holding_mode: str = Field(default="dynamic", pattern="^(dynamic|fixed)$")
    rebalance_days: int = Field(default=10, ge=1, le=20)
    min_hold_days: int = Field(default=10, ge=1, le=30)
    deep_exit_rank: int = Field(default=150, ge=40, le=500)
    exit_confirm_days: int = Field(default=3, ge=1, le=10)
    swap_score_gap: float = Field(default=0.70, ge=0.0, le=2.0)
    profit_ride_pct: float = Field(default=0.10, ge=0.03, le=0.50)
    profit_protect_pct: float = Field(default=0.15, ge=0.05, le=0.50)
    stop_cooldown_days: int = Field(default=5, ge=0, le=15)
    target_holdings: int = Field(default=10, ge=5, le=20)
    buffer_rank: int = Field(default=35, ge=10, le=80)
    max_single_weight: float = Field(default=0.12, ge=0.05, le=0.25)
    max_industry_weight: float = Field(default=0.25, ge=0.10, le=0.50)
    hard_stop_pct: float = Field(default=0.08, ge=0.03, le=0.20)
    atr_stop_mult: float = Field(default=2.5, ge=1.0, le=5.0)
    slippage_bps: float = Field(default=10.0, ge=0.0, le=50.0)
    enable_regime_timing: bool = True
    enable_technical_filter: bool = True
    bull_mom_h120_weight: float = Field(default=0.10, ge=0.0, le=0.50)
    bull_mom_ret20_weight: float = Field(default=0.08, ge=0.0, le=0.50)
    cash_yield_rate: float = Field(default=0.0, ge=0.0, le=0.05)
    include_heatmap: bool = True
    use_cache: bool = True


class WalkForwardOptRequest(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    model_mode: str = "regression"
    weight_method: str = "risk_parity"
    x_param: str = "rebalance_days"
    x_values: list[int | float] = [3, 5, 10]
    y_param: str = "target_holdings"
    y_values: list[int | float] = [8, 10, 12]
    metric: str = "sharpe_ratio"


def _sample_codes() -> list[str] | None:
    """Use ALL stocks synced in DuckDB for full-universe Walk-Forward backtesting."""
    return None


def _build_configs(req: BacktestRunRequest) -> tuple[LGBMConfig, BacktestConfig]:
    seed_pool = (42, 1024, 2026, 7, 99, 777, 11, 22, 33, 44, 55, 66)
    seeds = seed_pool[: max(1, int(req.ensemble_seeds_count))]
    embargo = max(5, int(req.label_horizon))
    lgbm_cfg = LGBMConfig(
        mode=req.model_mode,  # type: ignore[arg-type]
        label_horizon=req.label_horizon,
        embargo_days=embargo,
        ensemble_seeds=seeds,
    )
    bt_cfg = BacktestConfig(
        holding_mode=req.holding_mode,  # type: ignore[arg-type]
        rebalance_days=req.rebalance_days,
        min_hold_days=req.min_hold_days,
        deep_exit_rank=req.deep_exit_rank,
        exit_confirm_days=req.exit_confirm_days,
        swap_score_gap=req.swap_score_gap,
        profit_ride_pct=req.profit_ride_pct,
        profit_protect_pct=req.profit_protect_pct,
        stop_cooldown_days=req.stop_cooldown_days,
        target_holdings=req.target_holdings,
        buffer_rank=req.buffer_rank,
        max_single_weight=req.max_single_weight,
        max_industry_weight=req.max_industry_weight,
        weight_method=req.weight_method,  # type: ignore[arg-type]
        slippage_bps=req.slippage_bps,
        hard_stop_pct=req.hard_stop_pct,
        atr_stop_mult=req.atr_stop_mult,
        enable_regime_timing=req.enable_regime_timing,
        enable_technical_filter=req.enable_technical_filter,
        bull_mom_h120_weight=req.bull_mom_h120_weight,
        bull_mom_ret20_weight=req.bull_mom_ret20_weight,
        cash_yield_rate=req.cash_yield_rate,
    )
    return lgbm_cfg, bt_cfg


def _execute_and_persist(
    run_id: str,
    req: BacktestRunRequest,
    progress_cb=None,
    row_exists: bool = False,
) -> dict:
    t0 = time.monotonic()
    db = get_duckdb()
    sqlite = get_sqlite_db()
    svc = AlphaStrategyService(db)
    lgbm_cfg, bt_cfg = _build_configs(req)

    bundle = svc.run_full_pipeline(
        codes=_sample_codes(),
        start=req.start_date,
        end=req.end_date,
        lgbm_config=lgbm_cfg,
        bt_config=bt_cfg,
        include_ew_compare=True,
        include_heatmap=req.include_heatmap,
        use_cache=req.use_cache,
        progress_cb=progress_cb,
    )

    elapsed = round(time.monotonic() - t0, 2)
    m = bundle.backtest.metrics

    # Compute window-matched OOS IC / RankIC / ICIR so sliced windows (e.g. 2025-01~present,
    # 2025 only, 2026 only) report the exact out-of-sample predictive power of that period.
    ic_s = bundle.ml_result.oos_ic_series
    ric_s = bundle.ml_result.oos_rank_ic_series
    sim_s_str = str(m.get("start_date", "") or "")
    sim_e_str = str(m.get("end_date", "") or "")
    if len(ric_s) and sim_s_str and sim_e_str:
        ic_win = ic_s.loc[(ic_s.index >= sim_s_str) & (ic_s.index <= sim_e_str)]
        ric_win = ric_s.loc[(ric_s.index >= sim_s_str) & (ric_s.index <= sim_e_str)]
    else:
        ic_win, ric_win = ic_s, ric_s
    step = max(1, int(lgbm_cfg.label_horizon))
    if len(ric_win):
        nov_ric = ric_win.iloc[::step]
        oos_ic_val = float(ic_win.iloc[::step].mean()) if len(ic_win) else float(bundle.ml_result.oos_ic_mean)
        oos_ric_val = float(nov_ric.mean())
        ric_std = float(nov_ric.std(ddof=1)) if len(nov_ric) > 1 else 0.0
        oos_icir_val = float(oos_ric_val / ric_std) if ric_std > 1e-12 else 0.0
    else:
        oos_ic_val = float(bundle.ml_result.oos_ic_mean)
        oos_ric_val = float(bundle.ml_result.oos_rank_ic_mean)
        oos_icir_val = float(bundle.ml_result.oos_icir)

    payload_metrics = {
        **m,
        "oos_ic": oos_ic_val,
        "oos_rank_ic": oos_ric_val,
        "oos_icir": oos_icir_val,
        "full_oos_rank_ic": float(bundle.ml_result.oos_rank_ic_mean),
        "folds": bundle.ml_result.folds,
        "feature_importance": bundle.ml_result.feature_importance,
        "comparison": bundle.equal_weight_comparison,
        "rejected_orders_count": len(bundle.backtest.rejected_orders),
        "cache_info": bundle.cache_info,
        "elapsed_sec": elapsed,
    }

    with sqlite.session_scope() as session:
        row = session.get(BacktestRun, run_id) if row_exists else None
        if row is None:
            row = BacktestRun(id=run_id, name=req.name)
            session.add(row)
        row.name = req.name
        row.status = "DONE"
        row.start_date = str(m.get("start_date", ""))
        row.end_date = str(m.get("end_date", ""))
        row.config_json = json.dumps(to_jsonable(req.model_dump()), ensure_ascii=False)
        row.metrics_json = json.dumps(to_jsonable(payload_metrics), ensure_ascii=False)
        row.equity_curve_json = json.dumps(to_jsonable(bundle.backtest.equity_curve), ensure_ascii=False)
        row.drawdown_curve_json = json.dumps(to_jsonable(bundle.backtest.drawdown_curve), ensure_ascii=False)
        row.trades_json = json.dumps(to_jsonable(bundle.backtest.trades), ensure_ascii=False)
        row.wf_heatmap_json = json.dumps(to_jsonable(bundle.heatmap or {}), ensure_ascii=False)
        row.error_message = ""

    return to_jsonable({
        "id": run_id,
        "name": req.name,
        "status": "DONE",
        "config": req.model_dump(),
        "metrics": payload_metrics,
        "equity_curve": bundle.backtest.equity_curve,
        "drawdown_curve": bundle.backtest.drawdown_curve,
        "trades": bundle.backtest.trades,
        "wf_heatmap": bundle.heatmap or {},
    })


@router.post("/run")
def run_backtest(req: BacktestRunRequest) -> dict:
    run_id = f"bt_{uuid.uuid4().hex[:10]}"
    try:
        return _execute_and_persist(run_id, req, row_exists=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"回测执行失败: {exc}") from exc


@router.post("/run-async")
def run_backtest_async(req: BacktestRunRequest) -> dict:
    """Launch a zero-timeout background backtest with real-time WebSocket & polling progress."""
    with _JOBS_LOCK:
        running = [j for j in _JOBS.values() if j["status"] == "RUNNING"]
        if running:
            raise HTTPException(
                status_code=409,
                detail=f"已有回测任务正在运行 ({running[0]['run_id']} · {running[0]['stage']})，请等待其完成",
            )

    run_id = f"bt_{uuid.uuid4().hex[:10]}"
    started_mono = time.monotonic()

    sqlite = get_sqlite_db()
    with sqlite.session_scope() as session:
        session.add(
            BacktestRun(
                id=run_id,
                name=req.name,
                status="RUNNING",
                start_date=str(req.start_date or ""),
                end_date=str(req.end_date or ""),
                config_json=json.dumps(to_jsonable(req.model_dump()), ensure_ascii=False),
            )
        )

    job_init = {
        "job_id": run_id,
        "run_id": run_id,
        "name": req.name,
        "status": "RUNNING",
        "stage": "正在启动全市场 Purged Walk-Forward 回测任务...",
        "pct": 0.02,
        "elapsed_sec": 0.0,
        "cache_info": {},
        "error": "",
    }
    with _JOBS_LOCK:
        _JOBS[run_id] = job_init

    def _emit(stage: str, pct: float) -> None:
        elapsed = round(time.monotonic() - started_mono, 1)
        with _JOBS_LOCK:
            if run_id in _JOBS:
                _JOBS[run_id]["stage"] = stage
                _JOBS[run_id]["pct"] = round(float(pct), 4)
                _JOBS[run_id]["elapsed_sec"] = elapsed
        ws_manager.broadcast_threadsafe(
            "task_progress",
            {
                "mode": "backtest",
                "task_id": run_id,
                "run_id": run_id,
                "name": req.name,
                "stage": stage,
                "pct": round(float(pct), 4),
                "elapsed_sec": elapsed,
            },
        )

    def _worker() -> None:
        try:
            _emit("正在初始化回测上下文并检查两级全市场缓存...", 0.03)
            result = _execute_and_persist(run_id, req, progress_cb=_emit, row_exists=True)
            elapsed = round(time.monotonic() - started_mono, 1)
            cinfo = result.get("metrics", {}).get("cache_info", {})
            with _JOBS_LOCK:
                if run_id in _JOBS:
                    _JOBS[run_id].update({
                        "status": "DONE",
                        "stage": "done",
                        "pct": 1.0,
                        "elapsed_sec": elapsed,
                        "cache_info": cinfo,
                    })
            ws_manager.broadcast_threadsafe(
                "task_progress",
                {
                    "mode": "backtest",
                    "task_id": run_id,
                    "run_id": run_id,
                    "name": req.name,
                    "stage": "done",
                    "pct": 1.0,
                    "elapsed_sec": elapsed,
                    "cache_info": cinfo,
                },
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.monotonic() - started_mono, 1)
            err_msg = str(exc)
            with sqlite.session_scope() as session:
                row = session.get(BacktestRun, run_id)
                if row is not None:
                    row.status = "FAILED"
                    row.error_message = err_msg
            with _JOBS_LOCK:
                if run_id in _JOBS:
                    _JOBS[run_id].update({
                        "status": "FAILED",
                        "stage": "error",
                        "elapsed_sec": elapsed,
                        "error": err_msg,
                    })
            ws_manager.broadcast_threadsafe(
                "task_progress",
                {
                    "mode": "backtest",
                    "task_id": run_id,
                    "run_id": run_id,
                    "name": req.name,
                    "stage": "error",
                    "error": err_msg,
                    "elapsed_sec": elapsed,
                },
            )

    threading.Thread(target=_worker, daemon=True).start()
    return to_jsonable(job_init)


@router.get("/jobs")
def list_backtest_jobs() -> list[dict]:
    with _JOBS_LOCK:
        return to_jsonable(list(_JOBS.values())[-10:][::-1])


@router.get("/jobs/{job_id}")
def get_backtest_job(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            return to_jsonable(dict(job))
    # Fallback to SQLite in case backend restarted
    sqlite = get_sqlite_db()
    with sqlite.session_scope() as session:
        row = session.get(BacktestRun, job_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Backtest job not found: {job_id}")
        return to_jsonable({
            "job_id": row.id,
            "run_id": row.id,
            "name": row.name,
            "status": row.status,
            "stage": "done" if row.status == "DONE" else ("error" if row.status == "FAILED" else "running"),
            "pct": 1.0 if row.status == "DONE" else 0.0,
            "error": row.error_message or "",
        })


@router.get("/cache-status")
def backtest_cache_status() -> dict:
    return to_jsonable(get_research_cache_status(get_duckdb()))


@router.post("/cache-clear")
def backtest_cache_clear() -> dict:
    res = clear_research_cache()
    return to_jsonable({"ok": True, **res})


@router.get("/runs")
def list_backtest_runs(limit: int = 30) -> list[dict]:
    sqlite = get_sqlite_db()
    with sqlite.session_scope() as session:
        rows = session.query(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(limit).all()
        out = []
        for r in rows:
            m = json.loads(r.metrics_json or "{}")
            cfg = json.loads(r.config_json or "{}")
            cinfo = m.get("cache_info") or {}
            out.append({
                "id": r.id,
                "name": r.name,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "status": r.status,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "model_mode": cfg.get("model_mode", "lambdarank"),
                "weight_method": cfg.get("weight_method", "risk_parity"),
                "holding_mode": cfg.get("holding_mode", m.get("holding_mode", "fixed")),
                "rebalance_days": cfg.get("rebalance_days", 5),
                "min_hold_days": cfg.get("min_hold_days", 10),
                "stop_cooldown_days": cfg.get("stop_cooldown_days", 5),
                "avg_hold_days": m.get("avg_hold_days"),
                "target_holdings": cfg.get("target_holdings", 10),
                "buffer_rank": cfg.get("buffer_rank", 25),
                "slippage_bps": cfg.get("slippage_bps", 10),
                "hard_stop_pct": cfg.get("hard_stop_pct", 0.08),
                "atr_stop_mult": cfg.get("atr_stop_mult", 2.5),
                "enable_regime_timing": cfg.get("enable_regime_timing", True),
                "enable_technical_filter": cfg.get("enable_technical_filter", True),
                "annualized_return": m.get("annualized_return"),
                "excess_return_csi1000": m.get("excess_return_csi1000"),
                "excess_annualized_csi1000": m.get("excess_annualized_csi1000"),
                "sharpe_ratio": m.get("sharpe_ratio"),
                "calmar_ratio": m.get("calmar_ratio"),
                "max_drawdown": m.get("max_drawdown"),
                "win_rate": m.get("win_rate"),
                "profit_loss_ratio": m.get("profit_loss_ratio"),
                "annualized_turnover": m.get("annualized_turnover"),
                "oos_rank_ic": m.get("oos_rank_ic"),
                "total_cost": m.get("total_cost"),
                "stocks_count": cinfo.get("stocks_count"),
                "elapsed_sec": m.get("elapsed_sec"),
                "error_message": r.error_message or "",
            })
        return to_jsonable(out)


@router.get("/runs/{run_id}")
def get_backtest_run(run_id: str) -> dict:
    sqlite = get_sqlite_db()
    with sqlite.session_scope() as session:
        r = session.get(BacktestRun, run_id)
        if r is None:
            raise HTTPException(status_code=404, detail=f"Backtest run not found: {run_id}")
        return to_jsonable({
            "id": r.id,
            "name": r.name,
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "status": r.status,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "config": json.loads(r.config_json or "{}"),
            "metrics": json.loads(r.metrics_json or "{}"),
            "equity_curve": json.loads(r.equity_curve_json or "[]"),
            "drawdown_curve": json.loads(r.drawdown_curve_json or "[]"),
            "trades": json.loads(r.trades_json or "[]"),
            "wf_heatmap": json.loads(r.wf_heatmap_json or "{}"),
            "error_message": r.error_message or "",
        })


@router.delete("/runs/{run_id}")
def delete_backtest_run(run_id: str) -> dict:
    sqlite = get_sqlite_db()
    with sqlite.session_scope() as session:
        r = session.get(BacktestRun, run_id)
        if r is None:
            raise HTTPException(status_code=404, detail=f"Backtest run not found: {run_id}")
        session.delete(r)
    return {"ok": True, "deleted": run_id}


@router.post("/walk-forward-opt")
def walk_forward_opt(req: WalkForwardOptRequest) -> dict:
    from app.backtest.walk_forward_opt import run_walk_forward_heatmap

    db = get_duckdb()
    svc = AlphaStrategyService(db)
    bundle = svc.run_full_pipeline(
        codes=_sample_codes(),
        start=req.start_date,
        end=req.end_date,
        lgbm_config=LGBMConfig(mode=req.model_mode),  # type: ignore[arg-type]
        bt_config=BacktestConfig(weight_method=req.weight_method),  # type: ignore[arg-type]
        include_ew_compare=False,
        include_heatmap=False,
        use_cache=True,
    )
    hm = run_walk_forward_heatmap(
        panel=bundle.factor_run.panel,
        oos_scores=bundle.ml_result.oos_scores,
        eligible=bundle.factor_run.universe.eligible,
        technical=bundle.technical,
        regime_df=bundle.regime_df,
        base_config=BacktestConfig(weight_method=req.weight_method),  # type: ignore[arg-type]
        x_param=req.x_param,
        x_values=req.x_values,
        y_param=req.y_param,
        y_values=req.y_values,
        metric=req.metric,
        start_date=req.start_date,
        end_date=req.end_date,
    )
    return to_jsonable(hm)
