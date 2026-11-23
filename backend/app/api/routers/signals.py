"""Real-time explainable stock recommendations & individual stock K-line + technical/signal annotations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.core.constants import normalize_ts_code
from app.data.sync_service import DataSyncService
from app.db.duckdb_client import get_duckdb
from app.db.models import PaperOrder, SignalRecommendation
from app.db.sqlite_client import get_sqlite_db
from app.factors.service import to_jsonable
from app.paper.recommender import RecommendationEngine

router = APIRouter(prefix="/signals", tags=["Signals & K-Line"])


def _sample_codes() -> list[str] | None:
    """Score ALL stocks synced in DuckDB (never restricted to sample_universe.json)."""
    return None


@router.get("/latest")
def get_latest_signals() -> dict:
    db = get_duckdb()
    sqlite = get_sqlite_db()
    rec_eng = RecommendationEngine(db, sqlite)
    items = rec_eng.get_latest_persisted()
    if not items:
        bundle = rec_eng.generate(codes=_sample_codes(), persist=True, broadcast=False)
        items = bundle.recommendations
        regime = bundle.regime
        model_vid = bundle.model_version_id
    else:
        reg_df = db.query_df("SELECT * FROM market_regime_daily ORDER BY trade_date DESC LIMIT 1")
        regime = (
            {
                "trade_date": str(pd.to_datetime(reg_df.iloc[0]["trade_date"]).date()),
                "regime": str(reg_df.iloc[0]["regime"]),
                "target_exposure": float(reg_df.iloc[0]["target_exposure"]),
                "breadth_above_ma20": float(reg_df.iloc[0]["breadth_above_ma20"]),
            }
            if not reg_df.empty
            else {}
        )
        active = rec_eng.registry.get_active()
        model_vid = active["version_id"] if active else ""
    return to_jsonable({"regime": regime, "model_version_id": model_vid, "items": items})


@router.post("/refresh")
def refresh_signals(fetch_live_quotes: bool = True) -> dict:
    db = get_duckdb()
    sqlite = get_sqlite_db()
    rec_eng = RecommendationEngine(db, sqlite)
    # 1) Score ALL stocks in DuckDB to determine real Top-100 candidates
    bundle = rec_eng.generate(codes=None, live_quotes=None, persist=not fetch_live_quotes, broadcast=not fetch_live_quotes)
    if fetch_live_quotes and bundle.top_candidates_codes:
        svc = DataSyncService(db)
        try:
            # 2) Fetch intraday L1 quotes ONLY for Top-100 candidates to avoid rate limits
            live_quotes = svc.snapshot_top(bundle.top_candidates_codes, max_n=100)
        except Exception:  # noqa: BLE001
            live_quotes = None
        finally:
            svc.close()
        bundle = rec_eng.generate(codes=None, live_quotes=live_quotes, persist=True, broadcast=True)
    return to_jsonable({
        "trade_date": bundle.trade_date,
        "regime": bundle.regime,
        "model_version_id": bundle.model_version_id,
        "items": bundle.recommendations,
    })


@router.get("/kline/{ts_code}")
def get_stock_kline(ts_code: str, limit: int = Query(default=160, ge=30, le=500)) -> dict:
    """Return K-line OHLCV (unadjusted for exact price levels & stop lines), MA5/10/20/60, MACD,
    historical paper trade markers, and current recommendation stop-loss / take-profit lines."""
    code = normalize_ts_code(ts_code)
    db = get_duckdb()
    sqlite = get_sqlite_db()

    bars = db.query_df(
        "SELECT trade_date, open, high, low, close, volume, amount, turnover_rate, adj_factor, limit_up, limit_down "
        "FROM daily_bars WHERE ts_code = ? ORDER BY trade_date ASC",
        [code],
    )
    if bars.empty:
        raise HTTPException(status_code=404, detail=f"No historical bars found for {code}")

    basic = db.query_df("SELECT name, board, sw_l1_name FROM stock_basic WHERE ts_code = ?", [code])
    stock_name = str(basic.iloc[0]["name"]) if not basic.empty else code
    board = str(basic.iloc[0]["board"]) if not basic.empty else ""
    industry = str(basic.iloc[0]["sw_l1_name"]) if not basic.empty else "未分类"

    bars["trade_date"] = pd.to_datetime(bars["trade_date"])
    c = bars["close"].astype(float)
    bars["ma5"] = c.rolling(5, min_periods=5).mean()
    bars["ma10"] = c.rolling(10, min_periods=10).mean()
    bars["ma20"] = c.rolling(20, min_periods=15).mean()
    bars["ma60"] = c.rolling(60, min_periods=40).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    bars["macd_dif"] = ema12 - ema26
    bars["macd_dea"] = bars["macd_dif"].ewm(span=9, adjust=False).mean()
    bars["macd_hist"] = (bars["macd_dif"] - bars["macd_dea"]) * 2.0

    # ATR(14) on raw prices for visual stop-loss band
    prev_c = c.shift(1)
    tr = np.maximum(
        bars["high"] - bars["low"],
        np.maximum((bars["high"] - prev_c).abs(), (bars["low"] - prev_c).abs()),
    )
    bars["atr14"] = tr.rolling(14, min_periods=10).mean()
    tail = bars.tail(limit).reset_index(drop=True)

    candles = [
        {
            "time": str(r["trade_date"].date()),
            "open": round(float(r["open"]), 2),
            "high": round(float(r["high"]), 2),
            "low": round(float(r["low"]), 2),
            "close": round(float(r["close"]), 2),
            "volume": round(float(r["volume"]), 0),
            "amount": round(float(r["amount"]), 2) if pd.notna(r["amount"]) else 0.0,
            "ma5": round(float(r["ma5"]), 2) if pd.notna(r["ma5"]) else None,
            "ma10": round(float(r["ma10"]), 2) if pd.notna(r["ma10"]) else None,
            "ma20": round(float(r["ma20"]), 2) if pd.notna(r["ma20"]) else None,
            "ma60": round(float(r["ma60"]), 2) if pd.notna(r["ma60"]) else None,
            "macd_dif": round(float(r["macd_dif"]), 4) if pd.notna(r["macd_dif"]) else None,
            "macd_dea": round(float(r["macd_dea"]), 4) if pd.notna(r["macd_dea"]) else None,
            "macd_hist": round(float(r["macd_hist"]), 4) if pd.notna(r["macd_hist"]) else None,
            "atr14": round(float(r["atr14"]), 3) if pd.notna(r["atr14"]) else None,
        }
        for _, r in tail.iterrows()
    ]

    # Collect active recommendation + real paper orders as chart markers & price lines
    markers = []
    rec_card = None
    with sqlite.session_scope() as session:
        rec = (
            session.query(SignalRecommendation)
            .filter(SignalRecommendation.ts_code == code, SignalRecommendation.status == "ACTIVE")
            .order_by(SignalRecommendation.generated_at.desc())
            .first()
        )
        if rec is not None:
            rec_card = {
                "signal_id": rec.signal_id,
                "trade_date": rec.trade_date,
                "action": rec.action,
                "composite_rank": rec.composite_rank,
                "model_score": rec.model_score,
                "suggested_weight": rec.suggested_weight,
                "ref_price": rec.ref_price,
                "stop_loss_price": rec.stop_loss_price,
                "take_profit_price": rec.take_profit_price,
                "top_factors": json.loads(rec.top_factors_json or "[]"),
                "tech_patterns": json.loads(rec.tech_patterns_json or "[]"),
                "rationale": rec.rationale,
            }
            markers.append({
                "time": rec.trade_date,
                "position": "belowBar" if rec.action == "BUY" else ("aboveBar" if rec.action == "SELL" else "belowBar"),
                "color": "#F43F5E" if rec.action == "BUY" else ("#10B981" if rec.action == "SELL" else "#38BDF8"),
                "shape": "arrowUp" if rec.action == "BUY" else ("arrowDown" if rec.action == "SELL" else "circle"),
                "text": f"{rec.action} #{rec.composite_rank}",
            })

        orders = (
            session.query(PaperOrder)
            .filter(PaperOrder.ts_code == code, PaperOrder.status == "FILLED")
            .order_by(PaperOrder.created_at.asc())
            .all()
        )
        for o in orders:
            t_str = o.created_at.strftime("%Y-%m-%d") if o.created_at else candles[-1]["time"]
            # Align marker date to an existing candle date if needed
            candle_dates = {cd["time"] for cd in candles}
            if t_str not in candle_dates and candles:
                t_str = candles[-1]["time"]
            markers.append({
                "time": t_str,
                "position": "belowBar" if o.side == "BUY" else "aboveBar",
                "color": "#FB7185" if o.side == "BUY" else "#34D399",
                "shape": "arrowUp" if o.side == "BUY" else "arrowDown",
                "text": f"成交{o.side} ¥{o.avg_fill_price:.2f}",
            })

    # Deduplicate & sort markers chronologically (required by Lightweight Charts!)
    markers.sort(key=lambda m: m["time"])

    return to_jsonable({
        "ts_code": code,
        "name": stock_name,
        "board": board,
        "sw_l1_name": industry,
        "candles": candles,
        "markers": markers,
        "recommendation": rec_card,
    })
