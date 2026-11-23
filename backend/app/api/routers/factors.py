"""Factor Lab endpoints: single-factor IC/ICIR ranking, 5-quantile NAV curves, correlation heatmap & recompute."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.config import get_settings
from app.core.constants import ALL_FACTOR_NAMES, FACTOR_META_MAP
from app.db.duckdb_client import get_duckdb
from app.factors.service import FactorEngine, to_jsonable

router = APIRouter(prefix="/factors", tags=["Factor Lab"])


class ComputeFactorsRequest(BaseModel):
    horizon: int = 5
    start_date: date | None = None
    end_date: date | None = None


def _sample_codes_or_all() -> list[str] | None:
    """Analyze ALL stocks synced in DuckDB (full-market Plan A default)."""
    return None


@router.get("/summary")
def get_factors_summary(
    horizon: int = Query(default=5, ge=1, le=20),
    decay_window: int = Query(default=20, ge=5, le=60),
) -> dict:
    db = get_duckdb()
    ic_df = db.query_df(
        "SELECT * FROM factor_ic_daily WHERE horizon = ? ORDER BY factor_name, trade_date", [horizon]
    )
    if ic_df.empty:
        # Build & persist once on demand from real DuckDB bars
        eng = FactorEngine(db)
        res = eng.build(codes=_sample_codes_or_all(), horizon=horizon)
        eng.persist(res)
        ic_df = db.query_df(
            "SELECT * FROM factor_ic_daily WHERE horizon = ? ORDER BY factor_name, trade_date", [horizon]
        )

    # Check degradation flags over trailing 60 trading days in factor_values
    # (matches free ~120d real capital-flow window without penalizing older history)
    deg_df = db.query_df(
        "WITH recent_dates AS ("
        "  SELECT DISTINCT trade_date FROM factor_values ORDER BY trade_date DESC LIMIT 60"
        ") "
        "SELECT fv.factor_name, AVG(CAST(fv.is_degraded AS DOUBLE)) AS deg_ratio, COUNT(*) AS rows "
        "FROM factor_values fv JOIN recent_dates rd ON fv.trade_date = rd.trade_date "
        "GROUP BY 1"
    )
    deg_map = dict(zip(deg_df["factor_name"], deg_df["deg_ratio"])) if not deg_df.empty else {}

    periods_per_year = 252.0 / horizon
    items = []
    for fname in ALL_FACTOR_NAMES:
        grp = ic_df[ic_df["factor_name"] == fname].sort_values("trade_date")
        meta = FACTOR_META_MAP.get(fname, {})
        deg_ratio = float(deg_map.get(fname, 0.0))
        data_status = "DEGRADED_MISSING" if deg_ratio >= 0.99 else ("PARTIAL_DEGRADED" if deg_ratio > 0.30 else "OK")
        cov_60d = round(max(0.0, 1.0 - deg_ratio), 4)
        if grp.empty:
            items.append({
                "factor_name": fname,
                "label": meta.get("label", fname),
                "dimension": meta.get("dimension", ""),
                "ic_mean": None, "rank_ic_mean": None, "icir": None, "rank_icir": None,
                "t_stat": None, "ic_win_rate": None, "recent_rank_ic_20d": None,
                "ls_annual_return": None, "decay_status": "INSUFFICIENT",
                "n_periods": 0, "data_status": data_status, "coverage_recent_60d": cov_60d,
            })
            continue

        nov = grp.iloc[::horizon]
        ic_nov = nov["ic"].dropna()
        ric_nov = nov["rank_ic"].dropna()
        ic_m = float(ic_nov.mean()) if len(ic_nov) else np.nan
        ic_s = float(ic_nov.std(ddof=1)) if len(ic_nov) > 1 else np.nan
        ric_m = float(ric_nov.mean()) if len(ric_nov) else np.nan
        ric_s = float(ric_nov.std(ddof=1)) if len(ric_nov) > 1 else np.nan
        icir = ic_m / ic_s if ic_s and ic_s > 1e-12 else np.nan
        ricir = ric_m / ric_s if ric_s and ric_s > 1e-12 else np.nan
        t_stat = ricir * np.sqrt(len(ric_nov)) if pd.notna(ricir) else np.nan
        win = float((ric_nov > 0).mean()) if len(ric_nov) else np.nan
        recent = grp["rank_ic"].dropna().tail(decay_window)
        rec_m = float(recent.mean()) if len(recent) else np.nan
        if pd.isna(ric_m) or pd.isna(rec_m):
            decay = "INSUFFICIENT"
        elif np.sign(rec_m) != np.sign(ric_m) and abs(ric_m) > 0.01:
            decay = "REVERSED"
        elif abs(rec_m) < 0.5 * abs(ric_m):
            decay = "DECAYING"
        else:
            decay = "STABLE"
        ls_s = grp["ls_ret"].dropna()
        ls_ann = float((1.0 + ls_s).prod() ** (periods_per_year / max(1, len(ls_s))) - 1.0) if len(ls_s) else np.nan

        items.append({
            "factor_name": fname,
            "label": meta.get("label", fname),
            "dimension": meta.get("dimension", ""),
            "ic_mean": ic_m,
            "rank_ic_mean": ric_m,
            "icir": icir,
            "rank_icir": ricir,
            "t_stat": t_stat,
            "ic_win_rate": win,
            "recent_rank_ic_20d": rec_m,
            "ls_annual_return": ls_ann,
            "decay_status": decay,
            "n_periods": int(len(ric_nov)),
            "data_status": data_status,
            "coverage_recent_60d": cov_60d,
        })

    items.sort(key=lambda x: abs(x["rank_icir"]) if x["rank_icir"] is not None and not np.isnan(x["rank_icir"]) else -1.0, reverse=True)
    return to_jsonable({"horizon": horizon, "factors": items})


@router.get("/correlation")
def get_factor_correlation() -> dict:
    db = get_duckdb()
    # Compute cross-sectional Spearman correlation on sampled recent dates from persisted std_value
    df = db.query_df(
        "SELECT trade_date, ts_code, factor_name, std_value FROM factor_values "
        "WHERE std_value IS NOT NULL"
    )
    if df.empty:
        return {"factors": ALL_FACTOR_NAMES, "labels": [FACTOR_META_MAP[n]["label"] for n in ALL_FACTOR_NAMES], "matrix": []}

    dates = sorted(df["trade_date"].unique())[::5]
    sub = df[df["trade_date"].isin(dates)]
    k = len(ALL_FACTOR_NAMES)
    acc = np.zeros((k, k), dtype=float)
    pcnt = np.zeros((k, k), dtype=float)
    for _, grp in sub.groupby("trade_date"):
        piv = grp.pivot(index="ts_code", columns="factor_name", values="std_value").reindex(columns=ALL_FACTOR_NAMES)
        if len(piv) < 10:
            continue
        for col in ALL_FACTOR_NAMES:
            if piv[col].std() < 1e-12:
                piv[col] = np.nan
        c = piv.rank().corr().to_numpy(dtype=float)
        valid = ~np.isnan(c)
        acc[valid] += c[valid]
        pcnt[valid] += 1.0
    mat = np.where(pcnt > 0, acc / np.maximum(pcnt, 1.0), 0.0).round(4)
    return to_jsonable({
        "factors": ALL_FACTOR_NAMES,
        "labels": [FACTOR_META_MAP[n]["label"] for n in ALL_FACTOR_NAMES],
        "matrix": mat.tolist(),
    })


@router.get("/{factor_name}/detail")
def get_factor_detail(factor_name: str, horizon: int = Query(default=5, ge=1, le=20)) -> dict:
    if factor_name not in FACTOR_META_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown factor: {factor_name}")
    db = get_duckdb()
    df = db.query_df(
        "SELECT * FROM factor_ic_daily WHERE factor_name = ? AND horizon = ? ORDER BY trade_date",
        [factor_name, horizon],
    )
    if df.empty:
        return {"factor_name": factor_name, "meta": FACTOR_META_MAP[factor_name], "ic_series": [], "quintile_nav": []}

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["cum_rank_ic"] = df["rank_ic"].fillna(0.0).cumsum()
    df["rolling_20d_ic"] = df["rank_ic"].rolling(20, min_periods=5).mean()

    ic_series = [
        {
            "date": str(r["trade_date"].date()),
            "ic": r["ic"],
            "rank_ic": r["rank_ic"],
            "cum_rank_ic": r["cum_rank_ic"],
            "rolling_20d_ic": r["rolling_20d_ic"],
        }
        for _, r in df.iterrows()
    ]

    q_df = df.dropna(subset=["q1_ret", "q5_ret"]).copy()
    quintile_nav = []
    if not q_df.empty:
        nav_cols = {}
        for q in ("q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret", "ls_ret"):
            nav_cols[q] = (1.0 + q_df[q].fillna(0.0)).cumprod().to_numpy()
        for i, dt in enumerate(q_df["trade_date"]):
            quintile_nav.append({
                "date": str(dt.date()),
                "Q1": nav_cols["q1_ret"][i],
                "Q2": nav_cols["q2_ret"][i],
                "Q3": nav_cols["q3_ret"][i],
                "Q4": nav_cols["q4_ret"][i],
                "Q5": nav_cols["q5_ret"][i],
                "LS": nav_cols["ls_ret"][i],
            })

    return to_jsonable({
        "factor_name": factor_name,
        "meta": FACTOR_META_MAP[factor_name],
        "ic_series": ic_series,
        "quintile_nav": quintile_nav,
    })


@router.post("/compute")
def recompute_factors(req: ComputeFactorsRequest) -> dict:
    db = get_duckdb()
    eng = FactorEngine(db)
    res = eng.build(codes=_sample_codes_or_all(), start=req.start_date, end=req.end_date, horizon=req.horizon)
    persisted = eng.persist(res)
    return to_jsonable({
        "ok": True,
        "universe": res.universe.summary(),
        "persisted": persisted,
        "factor_status": res.factors.status,
    })
