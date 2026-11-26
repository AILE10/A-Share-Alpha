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
from app.core.task_manager import action_task_manager
from app.data.sync_service import DataSyncService
from app.db.duckdb_client import get_duckdb
from app.db.models import BacktestRun
from app.db.sqlite_client import get_sqlite_db
from app.factors.service import FactorEngine, to_jsonable
from app.paper.service import PaperTradingService
from app.strategy.realtime_regime import get_realtime_regime_service
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
            "style_mode": str(latest["style_mode"]) if ("style_mode" in latest.index and pd.notna(latest["style_mode"])) else "ALL_CAP",
            "min_mcap_quantile": float(latest["min_mcap_quantile"]) if ("min_mcap_quantile" in latest.index and pd.notna(latest["min_mcap_quantile"])) else 0.0,
            "csi1000_close": round(float(latest["csi1000_close"]), 2),
            "csi1000_ma20": round(float(latest["csi1000_ma20"]), 2) if pd.notna(latest["csi1000_ma20"]) else None,
            "csi1000_ma60": round(float(latest["csi1000_ma60"]), 2) if pd.notna(latest["csi1000_ma60"]) else None,
            "csi300_close": round(float(latest["csi300_close"]), 2) if ("csi300_close" in latest.index and pd.notna(latest["csi300_close"])) else None,
            "csi300_ma20": round(float(latest["csi300_ma20"]), 2) if ("csi300_ma20" in latest.index and pd.notna(latest["csi300_ma20"])) else None,
            "csi300_ma60": round(float(latest["csi300_ma60"]), 2) if ("csi300_ma60" in latest.index and pd.notna(latest["csi300_ma60"])) else None,
            "breadth_above_ma20": round(float(latest["breadth_above_ma20"]), 4),
            "is_realtime": False,
            "updated_at": str(pd.to_datetime(latest["trade_date"]).date()),
            "history": [
                {
                    "date": str(pd.to_datetime(r["trade_date"]).date()),
                    "csi1000_close": round(float(r["csi1000_close"]), 2),
                    "breadth": round(float(r["breadth_above_ma20"]), 4),
                    "regime": str(r["regime"]),
                    "target_exposure": float(r["target_exposure"]),
                    "style_mode": str(r["style_mode"]) if ("style_mode" in r.index and pd.notna(r["style_mode"])) else "ALL_CAP",
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
            "is_realtime": False,
            "updated_at": "",
            "history": [],
        }

    # Overlay fresh in-memory real-time intraday regime snapshot (zero DuckDB pollution)
    rt_snap = get_realtime_regime_service().get_cached(max_age_seconds=900.0)
    if rt_snap and rt_snap.get("trade_date") and rt_snap["trade_date"] >= (regime_card.get("trade_date") or ""):
        hist = list(regime_card.get("history") or [])
        rt_point = {
            "date": rt_snap["trade_date"],
            "csi1000_close": rt_snap["csi1000_close"],
            "breadth": rt_snap["breadth_above_ma20"],
            "regime": rt_snap["regime"],
            "target_exposure": rt_snap["target_exposure"],
        }
        if hist and hist[-1]["date"] == rt_point["date"]:
            hist[-1] = rt_point
        else:
            hist.append(rt_point)
        regime_card = {
            **regime_card,
            "trade_date": rt_snap["trade_date"],
            "quote_time": rt_snap.get("quote_time", ""),
            "updated_at": rt_snap.get("updated_at", rt_snap["trade_date"]),
            "is_realtime": bool(rt_snap.get("is_realtime", True)),
            "regime": rt_snap["regime"],
            "target_exposure": float(rt_snap["target_exposure"]),
            "style_mode": str(rt_snap.get("style_mode", "ALL_CAP")),
            "min_mcap_quantile": float(rt_snap.get("min_mcap_quantile", 0.0)),
            "csi1000_close": rt_snap["csi1000_close"],
            "csi1000_pct_chg": rt_snap.get("csi1000_pct_chg"),
            "csi1000_ma20": rt_snap["csi1000_ma20"],
            "csi1000_ma60": rt_snap["csi1000_ma60"],
            "csi300_close": rt_snap.get("csi300_close"),
            "csi300_pct_chg": rt_snap.get("csi300_pct_chg"),
            "csi300_ma20": rt_snap.get("csi300_ma20"),
            "csi300_ma60": rt_snap.get("csi300_ma60"),
            "ret_5d": rt_snap.get("ret_5d"),
            "breadth_above_ma20": rt_snap["breadth_above_ma20"],
            "quoted_count": rt_snap.get("quoted_count"),
            "universe_count": rt_snap.get("universe_count"),
            "advancers": rt_snap.get("advancers"),
            "decliners": rt_snap.get("decliners"),
            "flat": rt_snap.get("flat"),
            "history": hist[-60:],
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

    # Benchmark index quick snapshot (overlay realtime index quotes when cached)
    idx_cards = {}
    rt_indices = (rt_snap or {}).get("indices") or {}
    for code, label in ((BENCHMARK_CSI1000, "中证1000"), (BENCHMARK_CSI300, "沪深300")):
        if code in rt_indices:
            rq = rt_indices[code]
            idx_cards[code] = {
                "name": label,
                "trade_date": rq.get("quote_date", ""),
                "quote_time": rq.get("quote_time", ""),
                "close": rq["close"],
                "pct_chg": rq["pct_chg"],
                "is_realtime": bool((rt_snap or {}).get("is_realtime", True)),
            }
            continue
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
                "is_realtime": False,
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


@router.post("/dashboard/regime/refresh")
def refresh_realtime_regime() -> dict:
    """Force-refresh exact full-market real-time Market Regime (~2-3s) and broadcast over WebSocket."""
    from fastapi import HTTPException

    try:
        snap = get_realtime_regime_service().refresh(db=get_duckdb(), broadcast=True)
        return to_jsonable(snap)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"实时大盘状态刷新失败: {exc}") from exc


@router.post("/dashboard/regime/refresh-async")
def refresh_realtime_regime_async() -> dict:
    """Non-blocking real-time full-market Regime refresh with live WebSocket & polling progress."""
    def _run(emit):
        return get_realtime_regime_service().refresh(db=get_duckdb(), broadcast=True, progress_cb=emit)

    return action_task_manager.submit(
        kind="regime_refresh",
        title="全市场实时大盘状态刷新",
        fn=_run,
    )


@router.get("/data/status")
def get_data_status() -> dict:
    db = get_duckdb()
    total_listed = int(db.query_scalar("SELECT COUNT(*) FROM stock_basic") or 0)
    stocks = int(db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM daily_bars") or 0)
    bar_rows = int(db.query_scalar("SELECT COUNT(*) FROM daily_bars") or 0)
    dr = db.query_df("SELECT MIN(trade_date) min_d, MAX(trade_date) max_d FROM daily_bars").iloc[0]
    complete_max_d = db.query_scalar(
        """
        WITH dc AS (SELECT trade_date, COUNT(DISTINCT ts_code) AS cnt FROM daily_bars GROUP BY 1),
             mx AS (SELECT MAX(cnt) AS peak FROM dc)
        SELECT MAX(dc.trade_date) FROM dc, mx WHERE dc.cnt >= GREATEST(1, mx.peak * 0.30)
        """
    )
    raw_max_d = dr["max_d"]
    eff_max_d = complete_max_d if complete_max_d is not None else raw_max_d
    partial_date = None
    partial_stocks = 0
    if pd.notna(raw_max_d) and complete_max_d is not None and pd.to_datetime(raw_max_d).date() > pd.to_datetime(complete_max_d).date():
        partial_date = str(pd.to_datetime(raw_max_d).date())
        partial_stocks = int(
            db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM daily_bars WHERE trade_date = ?", [raw_max_d]) or 0
        )
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
            str(pd.to_datetime(eff_max_d).date()) if pd.notna(eff_max_d) else None,
        ],
        "partial_trailing_date": partial_date,
        "partial_trailing_stocks": partial_stocks,
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
    title_map = {
        "active_market": "全市场活跃 A 股断点续传同步",
        "incremental": "收盘日线与大盘状态增量同步",
        "sample_60": "60 只样本股初始化同步",
        "top100_snapshot": "Top-100 实时快照拉取",
    }
    kind = f"data_sync_{req.mode}"

    def _run(emit) -> dict:
        svc = DataSyncService()
        try:
            def _both(stage: str, pct: float) -> None:
                emit(stage, pct)
                ws_manager.broadcast_threadsafe(
                    "task_progress",
                    {"mode": req.mode, "stage": stage, "pct": round(float(pct), 4)},
                )

            _both("正在初始化数据同步服务...", 0.02)
            if req.mode == "top100_snapshot":
                codes = svc.db.query_df(
                    "SELECT ts_code FROM daily_bars GROUP BY ts_code ORDER BY MAX(amount) DESC LIMIT 100"
                )["ts_code"].tolist()
                _both(f"正在拉取 Top-{len(codes)} 实时 L1 行情快照...", 0.45)
                df = svc.snapshot_top(codes, max_n=100)
                out = {"quoted": len(df)}
            elif req.mode == "active_market":
                end = today_cn()
                start = svc.resolve_aligned_start_date(end, req.days)
                _both(f"正在同步交易日历与全市场股票列表 ({start} ~ {end})...", 0.05)
                svc.sync_calendar(start - timedelta(days=365), end + timedelta(days=30))
                if (svc.db.query_scalar("SELECT COUNT(*) FROM stock_basic") or 0) < 3000:
                    svc.sync_stock_basic(with_industry=True)
                _both("正在扫描全市场 4,600+ 只股票流动性与市值门槛...", 0.10)
                codes, scan_stats = svc.select_active_market_universe(
                    min_amount=req.min_amount,
                    max_stocks=req.max_stocks,
                )
                _both(f"正在核验全市场 PIT 财报公告库 ({len(codes)} 只活跃股)...", 0.15)
                if (svc.db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM financial_pit") or 0) < max(500, int(len(codes) * 0.7)):
                    svc.sync_financials(None, start.year - 1, end.year)
                _both(f"正在断点续传并发拉取日线 ({len(codes)} 只股票)...", 0.25)
                rep = svc.sync_daily_bars_parallel(
                    codes,
                    start,
                    end,
                    workers=max(1, min(req.workers, 6)),
                    resume=True,
                    progress=lambda msg, p: _both(msg, 0.25 + 0.58 * p),
                )
                svc.sync_index(start - timedelta(days=120), end)
                _both("正在同步真实主力资金流 (东财/同花顺)...", 0.85)
                svc.sync_capital_flow(codes, rep, workers=min(6, max(2, req.workers)), resume=True, include_today_bulk=True)
                _both("正在重算四维 20 因子与最新大盘择时状态...", 0.92)
                eng = FactorEngine(svc.db)
                f_run = eng.build(codes=None)
                eng.persist(f_run)
                persist_market_regime(compute_market_regime(f_run.panel), svc.db)
                get_realtime_regime_service().clear_cache()
                out = {"scan_stats": scan_stats, "report": rep.to_dict()}
            elif req.mode == "sample_60":
                end = today_cn()
                start = end - timedelta(days=req.days)
                _both("正在同步样本股日历、基础信息与 PIT 财报...", 0.15)
                svc.sync_calendar(start - timedelta(days=365), end + timedelta(days=30))
                svc.sync_stock_basic(with_industry=True)
                codes = svc.select_sample_universe(n=req.n_stocks)
                svc.sync_financials(codes, start.year - 1, end.year)
                rep = svc.sync_daily_bars_parallel(
                    codes, start, end, workers=req.workers, resume=True, progress=lambda msg, p: _both(msg, 0.30 + 0.55 * p)
                )
                svc.sync_index(start - timedelta(days=120), end)
                svc.sync_capital_flow(codes, rep)
                out = {"report": rep.to_dict()}
            else:
                rep = svc.incremental_sync(
                    workers=max(1, min(req.workers, 12)),
                    progress=lambda msg, p: _both(msg, 0.04 + 0.80 * p),
                )
                _both("正在重算最新收盘大盘择时状态与四维因子...", 0.88)
                eng = FactorEngine(svc.db)
                f_run = eng.build(codes=None, compute_analysis=True)
                eng.persist(f_run)
                persist_market_regime(compute_market_regime(f_run.panel), svc.db)
                get_realtime_regime_service().clear_cache()
                try:
                    get_realtime_regime_service().refresh(db=svc.db, broadcast=True)
                except Exception:  # noqa: BLE001
                    pass
                out = {"report": rep.to_dict()}

            ws_manager.broadcast_threadsafe(
                "task_progress",
                {"mode": req.mode, "stage": "done", "pct": 1.0, **out},
            )
            return out
        except Exception as exc:  # noqa: BLE001
            ws_manager.broadcast_threadsafe(
                "task_progress", {"mode": req.mode, "stage": "error", "error": str(exc)}
            )
            raise
        finally:
            svc.close()

    snap = action_task_manager.submit(
        kind=kind,
        title=title_map.get(req.mode, f"数据同步 ({req.mode})"),
        fn=_run,
    )
    return {**snap, "mode": req.mode}
