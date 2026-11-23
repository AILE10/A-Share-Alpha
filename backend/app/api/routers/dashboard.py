"""Dashboard summary, market regime overview, risk gauges & data sync status endpoints."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import timedelta

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from app.api.ws_manager import ws_manager
from app.broker.gateway import BrokerGateway
from app.config import get_settings
from app.core.calendar import today_cn
from app.core.constants import BENCHMARK_CSI1000, BENCHMARK_CSI300
from app.data.sync_service import DataSyncService
from app.db.duckdb_client import get_duckdb
from app.db.models import BacktestRun
from app.db.sqlite_client import get_sqlite_db
from app.factors.service import FactorEngine, to_jsonable
from app.paper.service import PaperTradingService
from app.strategy.regime import compute_market_regime, persist_market_regime

router = APIRouter(tags=["Dashboard & Data"])


class SyncRequest(BaseModel):
    mode: str = "incremental"  # incremental | active_market | sample_60 | top100_snapshot
    n_stocks: int = 60
    max_stocks: int | None = None
    min_amount: float = 40_000_000.0
    days: int = 730
    workers: int = 4


@router.get("/dashboard/summary")
def get_dashboard_summary() -> dict:
    db = get_duckdb()
    sqlite = get_sqlite_db()
    gw = BrokerGateway(sqlite)
    paper = PaperTradingService(db, sqlite, gw)

    # 1) Market Regime (from DuckDB market_regime_daily or compute on the fly)
    reg_df = db.query_df("SELECT * FROM market_regime_daily ORDER BY trade_date DESC LIMIT 60")
    if reg_df.empty:
        try:
            f_run = FactorEngine(db).build()
            r_all = compute_market_regime(f_run.panel)
            persist_market_regime(r_all, db)
            reg_df = db.query_df("SELECT * FROM market_regime_daily ORDER BY trade_date DESC LIMIT 60")
        except Exception:  # noqa: BLE001
            reg_df = pd.DataFrame()

    if not reg_df.empty:
        latest = reg_df.iloc[0]
        regime_card = {
            "trade_date": str(pd.to_datetime(latest["trade_date"]).date()),
            "regime": str(latest["regime"]),
            "target_exposure": float(latest["target_exposure"]),
            "csi1000_close": round(float(latest["csi1000_close"]), 2),
            "csi1000_ma20": round(float(latest["csi1000_ma20"]), 2) if pd.notna(latest["csi1000_ma20"]) else None,
            "csi1000_ma60": round(float(latest["csi1000_ma60"]), 2) if pd.notna(latest["csi1000_ma60"]) else None,
            "breadth_above_ma20": round(float(latest["breadth_above_ma20"]), 4),
            "history": [
                {
                    "date": str(pd.to_datetime(r["trade_date"]).date()),
                    "csi1000_close": round(float(r["csi1000_close"]), 2),
                    "breadth": round(float(r["breadth_above_ma20"]), 4),
                    "regime": str(r["regime"]),
                    "target_exposure": float(r["target_exposure"]),
                }
                for _, r in reg_df.iloc[::-1].iterrows()
            ],
        }
    else:
        regime_card = {
            "trade_date": "",
            "regime": "UNKNOWN",
            "target_exposure": 0.0,
            "csi1000_close": None,
            "csi1000_ma20": None,
            "csi1000_ma60": None,
            "breadth_above_ma20": None,
            "history": [],
        }

    # 2) Paper account, holdings & NAV curve
    acct_state = gw.paper.get_account_state()
    nav_history = paper.get_nav_history()
    risk_cfg = gw.risk.get_config()
    broker_cfg = gw.get_config()

    # Risk watermark computation
    positions = acct_state.get("positions", [])
    max_single_w = max((float(p.get("weight", 0.0)) for p in positions), default=0.0)
    ind_weights: dict[str, float] = {}
    for p in positions:
        ind = p.get("sw_l1_name") or "未分类"
        ind_weights[ind] = ind_weights.get(ind, 0.0) + float(p.get("weight", 0.0))
    max_ind_name, max_ind_w = ("无", 0.0)
    if ind_weights:
        max_ind_name, max_ind_w = max(ind_weights.items(), key=lambda kv: kv[1])

    # Fallback curve from latest completed backtest if paper has < 2 days so dashboard chart is always informative
    latest_bt_curve = []
    with sqlite.session_scope() as session:
        bt = session.query(BacktestRun).filter(BacktestRun.status == "DONE").order_by(BacktestRun.created_at.desc()).first()
        if bt is not None:
            latest_bt_curve = json.loads(bt.equity_curve_json or "[]")[-120:]

    # Benchmark index quick snapshot
    idx_cards = {}
    for code, label in ((BENCHMARK_CSI1000, "中证1000"), (BENCHMARK_CSI300, "沪深300")):
        idf = db.query_df(
            "SELECT trade_date, close, pct_chg FROM index_daily WHERE index_code = ? ORDER BY trade_date DESC LIMIT 2",
            [code],
        )
        if not idf.empty:
            r = idf.iloc[0]
            idx_cards[code] = {
                "name": label,
                "trade_date": str(pd.to_datetime(r["trade_date"]).date()),
                "close": round(float(r["close"]), 2),
                "pct_chg": round(float(r["pct_chg"]), 2) if pd.notna(r["pct_chg"]) else 0.0,
            }

    return to_jsonable({
        "regime": regime_card,
        "benchmarks": idx_cards,
        "account": acct_state.get("account", {}),
        "positions": positions,
        "paper_nav_curve": nav_history,
        "backtest_reference_curve": latest_bt_curve,
        "broker_mode": broker_cfg["active_mode"],
        "risk_watermarks": {
            "max_single_weight": round(max_single_w, 4),
            "single_limit": risk_cfg["max_single_weight"],
            "max_industry_name": max_ind_name,
            "max_industry_weight": round(max_ind_w, 4),
            "industry_limit": risk_cfg["max_industry_weight"],
            "current_drawdown": acct_state.get("account", {}).get("current_drawdown", 0.0),
            "drawdown_warn": -risk_cfg["drawdown_warn_pct"],
            "drawdown_halt": -risk_cfg["drawdown_halt_pct"],
            "fuse_status": acct_state.get("account", {}).get("fuse_status", "NORMAL"),
        },
        "ws_clients": ws_manager.active_count,
    })


@router.get("/data/status")
def get_data_status() -> dict:
    db = get_duckdb()
    total_listed = int(db.query_scalar("SELECT COUNT(*) FROM stock_basic") or 0)
    stocks = int(db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM daily_bars") or 0)
    bar_rows = int(db.query_scalar("SELECT COUNT(*) FROM daily_bars") or 0)
    dr = db.query_df("SELECT MIN(trade_date) min_d, MAX(trade_date) max_d FROM daily_bars").iloc[0]
    fin_rows = int(db.query_scalar("SELECT COUNT(*) FROM financial_pit") or 0)
    fin_stocks = int(db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM financial_pit") or 0)
    cf_ok = int(db.query_scalar("SELECT COUNT(*) FROM capital_flow_daily WHERE status='OK'") or 0)
    cf_stocks = int(db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM capital_flow_daily WHERE status='OK'") or 0)
    cf_dr = db.query_df("SELECT MIN(trade_date) min_d, MAX(trade_date) max_d FROM capital_flow_daily WHERE status='OK'").iloc[0]
    fac_rows = int(db.query_scalar("SELECT COUNT(*) FROM factor_values") or 0)
    ic_rows = int(db.query_scalar("SELECT COUNT(*) FROM factor_ic_daily") or 0)
    sources = db.query_df("SELECT source, COUNT(DISTINCT ts_code) stocks FROM daily_bars GROUP BY 1").to_dict("records")
    boards = db.query_df(
        "SELECT s.board, COUNT(DISTINCT b.ts_code) stocks FROM daily_bars b JOIN stock_basic s USING (ts_code) GROUP BY 1"
    ).to_dict("records")
    ckpt_path = get_settings().data_dir / "sync_checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else None
    cf_ratio = cf_stocks / max(stocks, 1)
    cf_status = "OK" if cf_ratio >= 0.70 else ("PARTIAL_DEGRADED" if cf_ok > 0 else "DEGRADED_MISSING")
    return to_jsonable({
        "total_listed_stocks": total_listed,
        "stocks_synced": stocks,
        "daily_bar_rows": bar_rows,
        "date_range": [
            str(pd.to_datetime(dr["min_d"]).date()) if pd.notna(dr["min_d"]) else None,
            str(pd.to_datetime(dr["max_d"]).date()) if pd.notna(dr["max_d"]) else None,
        ],
        "financial_pit_rows": fin_rows,
        "financial_pit_stocks": fin_stocks,
        "capital_flow_ok_rows": cf_ok,
        "capital_flow_stocks": cf_stocks,
        "capital_flow_date_range": [
            str(pd.to_datetime(cf_dr["min_d"]).date()) if pd.notna(cf_dr["min_d"]) else None,
            str(pd.to_datetime(cf_dr["max_d"]).date()) if pd.notna(cf_dr["max_d"]) else None,
        ],
        "capital_flow_status": cf_status,
        "factor_values_rows": fac_rows,
        "factor_ic_rows": ic_rows,
        "sources": sources,
        "boards": boards,
        "sync_checkpoint": ckpt,
    })


@router.post("/data/sync")
def trigger_data_sync(req: SyncRequest) -> dict:
    task_id = f"sync_{uuid.uuid4().hex[:10]}"

    def _worker() -> None:
        svc = DataSyncService()
        try:
            def _emit(stage: str, pct: float) -> None:
                ws_manager.broadcast_threadsafe(
                    "task_progress",
                    {"task_id": task_id, "mode": req.mode, "stage": stage, "pct": round(float(pct), 4)},
                )

            _emit("starting", 0.02)
            if req.mode == "top100_snapshot":
                codes = svc.db.query_df(
                    "SELECT ts_code FROM daily_bars GROUP BY ts_code ORDER BY MAX(amount) DESC LIMIT 100"
                )["ts_code"].tolist()
                df = svc.snapshot_top(codes, max_n=100)
                ws_manager.broadcast_threadsafe(
                    "task_progress",
                    {"task_id": task_id, "mode": req.mode, "stage": "done", "pct": 1.0, "quoted": len(df)},
                )
            elif req.mode == "active_market":
                end = today_cn()
                start = end - timedelta(days=req.days)
                _emit("syncing calendar & full A-share listing", 0.05)
                svc.sync_calendar(start - timedelta(days=365), end)
                if (svc.db.query_scalar("SELECT COUNT(*) FROM stock_basic") or 0) < 3000:
                    svc.sync_stock_basic(with_industry=True)
                _emit("scanning full-market liquidity", 0.10)
                codes, scan_stats = svc.select_active_market_universe(
                    min_amount=req.min_amount,
                    max_stocks=req.max_stocks,
                )
                _emit(f"syncing bulk PIT financials ({len(codes)} active stocks)", 0.15)
                if (svc.db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM financial_pit") or 0) < max(500, int(len(codes) * 0.7)):
                    svc.sync_financials(None, start.year - 1, end.year)
                _emit(f"resumable parallel bars sync ({len(codes)} stocks)", 0.25)
                rep = svc.sync_daily_bars_parallel(
                    codes,
                    start,
                    end,
                    workers=max(1, min(req.workers, 6)),
                    resume=True,
                    progress=lambda msg, p: _emit(msg, 0.25 + 0.60 * p),
                )
                svc.sync_index(start - timedelta(days=120), end)
                _emit("syncing real capital flow (EM push2his + bulk snapshot)", 0.86)
                svc.sync_capital_flow(codes, rep, workers=min(6, max(2, req.workers)), resume=True, include_today_bulk=True)
                _emit("recomputing 16 factors & market regime", 0.92)
                eng = FactorEngine(svc.db)
                f_run = eng.build(codes=None)
                eng.persist(f_run)
                persist_market_regime(compute_market_regime(f_run.panel), svc.db)
                ws_manager.broadcast_threadsafe(
                    "task_progress",
                    {
                        "task_id": task_id,
                        "mode": req.mode,
                        "stage": "done",
                        "pct": 1.0,
                        "scan_stats": scan_stats,
                        "report": rep.to_dict(),
                    },
                )
            elif req.mode == "sample_60":
                end = today_cn()
                start = end - timedelta(days=req.days)
                svc.sync_calendar(start - timedelta(days=365), end)
                svc.sync_stock_basic(with_industry=True)
                codes = svc.select_sample_universe(n=req.n_stocks)
                svc.sync_financials(codes, start.year - 1, end.year)
                rep = svc.sync_daily_bars_parallel(codes, start, end, workers=req.workers, resume=True)
                svc.sync_index(start - timedelta(days=120), end)
                svc.sync_capital_flow(codes, rep)
                ws_manager.broadcast_threadsafe(
                    "task_progress",
                    {"task_id": task_id, "mode": req.mode, "stage": "done", "pct": 1.0, "report": rep.to_dict()},
                )
            else:
                rep = svc.incremental_sync(workers=req.workers)
                ws_manager.broadcast_threadsafe(
                    "task_progress",
                    {"task_id": task_id, "mode": req.mode, "stage": "done", "pct": 1.0, "report": rep.to_dict()},
                )
        except Exception as exc:  # noqa: BLE001
            ws_manager.broadcast_threadsafe(
                "task_progress", {"task_id": task_id, "mode": req.mode, "stage": "error", "error": str(exc)}
            )
        finally:
            svc.close()

    threading.Thread(target=_worker, daemon=True).start()
    return {"task_id": task_id, "mode": req.mode, "status": "RUNNING"}
