"""Factor research engine: universe -> raw factors -> standardization -> analytics -> DuckDB persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.db.duckdb_client import DuckDBClient, get_duckdb
from app.factors.analyzer import FactorAnalysis, analyze_factors
from app.factors.definitions import FactorSet, compute_factors, forward_returns
from app.factors.panel import MarketPanel, load_market_panel
from app.factors.pipeline import standardize_panel
from app.factors.universe import UniverseConfig, UniverseResult, apply_universe_filter

logger = get_logger(__name__)


@dataclass
class FactorRunResult:
    panel: MarketPanel
    universe: UniverseResult
    factors: FactorSet
    std: dict[str, pd.DataFrame]
    analysis: FactorAnalysis | None


class FactorEngine:
    def __init__(self, db: DuckDBClient | None = None) -> None:
        self.db = db or get_duckdb()

    def all_trade_dates(self) -> pd.DatetimeIndex:
        cal = self.db.query_df("SELECT trade_date FROM trade_calendar WHERE is_open ORDER BY trade_date")
        return pd.DatetimeIndex(pd.to_datetime(cal["trade_date"])) if not cal.empty else None

    def build(
        self,
        codes: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        horizon: int = 5,
        universe_config: UniverseConfig | None = None,
        panel: MarketPanel | None = None,
        compute_analysis: bool = True,
    ) -> FactorRunResult:
        panel = panel or load_market_panel(codes, start, end, db=self.db)
        universe = apply_universe_filter(panel, universe_config, self.all_trade_dates())
        factors = compute_factors(panel)
        std = standardize_panel(factors.raw, universe.eligible, panel.industry_of(), panel.total_mv)
        analysis: FactorAnalysis | None = None
        if compute_analysis:
            fwd = forward_returns(panel, horizon)
            analysis = analyze_factors(std, fwd, universe.eligible, horizon=horizon, status=factors.status)
        return FactorRunResult(panel=panel, universe=universe, factors=factors, std=std, analysis=analysis)

    def persist(self, res: FactorRunResult, factor_values_tail_days: int | None = None) -> dict[str, int]:
        uni = res.universe.to_long()
        n_uni = self.db.upsert_df("universe_daily", uni)

        # For full-market runs (>250 stocks), persist the trailing 60 trading days of raw/std factor_values
        # (sufficient for 16x16 Spearman correlation & degradation tracking while keeping DuckDB compact),
        # while always persisting the ENTIRE history of universe_daily & factor_ic_daily.
        tail_n = factor_values_tail_days
        if tail_n is None and len(res.panel.codes) > 250:
            tail_n = 60

        raw_long = res.factors.to_long(tail_n=tail_n)

        std_parts = []
        for name, df in res.std.items():
            sub_df = df.iloc[-tail_n:] if (tail_n and len(df) > tail_n) else df
            s = sub_df.stack(future_stack=True)
            part = s.reset_index()
            part.columns = ["trade_date", "ts_code", "std_value"]
            part["factor_name"] = name
            std_parts.append(part)
        std_long = pd.concat(std_parts, ignore_index=True)
        merged = raw_long.merge(std_long, on=["trade_date", "ts_code", "factor_name"], how="left")
        merged = merged[merged["raw_value"].notna() | merged["std_value"].notna() | merged["is_degraded"]]
        merged["trade_date"] = pd.to_datetime(merged["trade_date"]).dt.date
        n_fac = self.db.upsert_df("factor_values", merged)

        n_ic = 0
        if res.analysis is not None:
            ic = res.analysis.ic_daily.copy()
            ic = ic[ic["ic"].notna() | ic["rank_ic"].notna()]
            n_ic = self.db.upsert_df("factor_ic_daily", ic)
        logger.info("Persisted universe=%d factor_values=%d factor_ic=%d", n_uni, n_fac, n_ic)
        return {"universe_daily": n_uni, "factor_values": n_fac, "factor_ic_daily": n_ic}


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else round(float(obj), 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, date)):
        return obj.isoformat()[:10]
    return obj
