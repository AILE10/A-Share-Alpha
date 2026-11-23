"""Market Regime timing engine (CSI 1000 MA20/MA60 + market breadth above MA20)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.constants import BENCHMARK_CSI1000
from app.db.duckdb_client import DuckDBClient
from app.factors.panel import MarketPanel


@dataclass
class RegimeConfig:
    offensive_exposure: float = 0.90     # 强势进攻 80%~95%
    neutral_exposure: float = 0.55       # 震荡中性 50%~60%
    defensive_exposure: float = 0.15     # 弱势防守 0%~20%
    extreme_exposure: float = 0.10       # 极端风险轻仓试探/底仓（0%~20%）
    breadth_bull: float = 0.55
    breadth_bear: float = 0.35
    extreme_breadth: float = 0.18
    extreme_drop_5d: float = -0.08


def compute_market_regime(panel: MarketPanel, config: RegimeConfig | None = None) -> pd.DataFrame:
    """Compute daily market regime & target exposure using ONLY information available at day t close."""
    cfg = config or RegimeConfig()
    dates = panel.dates
    hc = panel.hfq_close
    ma20_stocks = hc.rolling(20, min_periods=15).mean()
    active = (~panel.is_suspended) & ma20_stocks.notna()
    breadth = ((hc > ma20_stocks) & active).sum(axis=1) / active.sum(axis=1).replace(0, np.nan)
    breadth = breadth.fillna(0.5)

    idx = panel.index_close.get(BENCHMARK_CSI1000)
    if idx is None or idx.dropna().empty:
        # Fallback to equal-weighted market proxy if index missing
        idx = hc.pct_change(fill_method=None).mean(axis=1).fillna(0.0).add(1.0).cumprod() * 1000.0
    else:
        idx = idx.reindex(dates).ffill()

    ma20 = idx.rolling(20, min_periods=15).mean()
    ma60 = idx.rolling(60, min_periods=30).mean()
    ret_5d = idx / idx.shift(5) - 1.0

    regimes = []
    exposures = []
    for dt in dates:
        c = idx.loc[dt]
        m20 = ma20.loc[dt]
        m60 = ma60.loc[dt]
        b = float(breadth.loc[dt])
        r5 = float(ret_5d.loc[dt]) if pd.notna(ret_5d.loc[dt]) else 0.0

        if pd.isna(m20) or pd.isna(m60):
            regimes.append("NEUTRAL")
            exposures.append(cfg.neutral_exposure)
            continue

        extreme_risk = (r5 <= cfg.extreme_drop_5d) or (b < cfg.extreme_breadth)
        bearish = (c < m20) and (c < m60) and (b < cfg.breadth_bear)
        bullish = (c > m20) and (c > m60) and (b >= cfg.breadth_bull)

        if extreme_risk:
            regimes.append("DEFENSIVE")
            exposures.append(cfg.extreme_exposure if b >= 0.12 else 0.0)
        elif bearish:
            regimes.append("DEFENSIVE")
            exposures.append(cfg.defensive_exposure)
        elif bullish:
            regimes.append("OFFENSIVE")
            exposures.append(cfg.offensive_exposure)
        else:
            regimes.append("NEUTRAL")
            exposures.append(cfg.neutral_exposure)

    return pd.DataFrame(
        {
            "trade_date": dates.date,
            "csi1000_close": idx.to_numpy(dtype=float),
            "csi1000_ma20": ma20.to_numpy(dtype=float),
            "csi1000_ma60": ma60.to_numpy(dtype=float),
            "breadth_above_ma20": breadth.to_numpy(dtype=float),
            "regime": regimes,
            "target_exposure": exposures,
        },
        index=dates,
    )


def persist_market_regime(regime_df: pd.DataFrame, db: DuckDBClient) -> int:
    return db.upsert_df("market_regime_daily", regime_df.reset_index(drop=True))
