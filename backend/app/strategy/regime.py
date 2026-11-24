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
    offensive_exposure: float = 0.98          # 强牛主升满仓进攻（消除10%永久现金拖累）
    bull_hold_exposure: float = 0.75          # 金叉结构牛/顶部安全区仓位（MA20>MA60 且收盘>=MA20*0.985 且宽度>=40%）
    bull_pullback_exposure: float = 0.45      # 金叉回踩破位预警仓位（跌破MA20*0.985 及时防守）
    neutral_exposure: float = 0.30            # 死叉震荡/熊市反弹中性仓位（压缩死猫跳敞口至30%）
    defensive_exposure: float = 0.10          # 确认弱势防守轻仓
    extreme_exposure: float = 0.0             # 极端风险/深度熊市空仓保护（0%）
    breadth_bull: float = 0.55                # 非金叉期强牛宽度门槛
    breadth_golden_bull: float = 0.50         # 金叉多头期强牛宽度门槛
    breadth_bull_hold: float = 0.40           # 金叉结构牛续持宽度门槛
    breadth_bear: float = 0.35                # 弱势防守宽度门槛
    breadth_deep_bear: float = 0.25           # 深度熊市直接空仓宽度门槛
    extreme_breadth: float = 0.18             # 极端冰点空仓宽度门槛
    extreme_drop_5d: float = -0.07            # 5日急跌>7% 立即空仓保护
    top_safe_ma20_ratio: float = 0.985        # 金叉顶部安全线（跌破MA20*0.985 提前降仓）


def classify_regime_point(
    close: float,
    ma20: float,
    ma60: float,
    breadth: float,
    ret_5d: float,
    config: RegimeConfig | None = None,
) -> tuple[str, float]:
    """Classify a single point (EOD close or live intraday snapshot) into (regime, target_exposure).

    Asymmetric Golden-Cross TopSafe Timing (shared identically by backtest & live realtime regime):
      - Bull structure (MA20 > MA60): captures 98% in strong bull & 75% in healthy bull hold,
        dropping promptly to 45% if price breaks below MA20 * 0.985 (top-crash protection).
      - Bear structure (MA20 <= MA60): caps dead-cat-bounce neutral exposure at 30%,
        10% in confirmed bear, and 0% full cash on deep bear (<25% breadth) or crash (<-7% 5d / <18% breadth).
    """
    cfg = config or RegimeConfig()
    if pd.isna(ma20) or pd.isna(ma60) or pd.isna(close):
        return ("NEUTRAL", 0.55)

    b = float(breadth) if pd.notna(breadth) else 0.5
    r5 = float(ret_5d) if pd.notna(ret_5d) else 0.0
    golden_cross = ma20 > ma60
    bearish = (close < ma20) and (close < ma60) and (b < cfg.breadth_bear)
    extreme_or_deep_bear = (
        (r5 <= cfg.extreme_drop_5d)
        or (b < cfg.extreme_breadth)
        or (bearish and b < cfg.breadth_deep_bear)
    )

    if extreme_or_deep_bear:
        return ("DEFENSIVE", cfg.extreme_exposure)
    if bearish:
        return ("DEFENSIVE", cfg.defensive_exposure)

    bull_thresh = cfg.breadth_golden_bull if golden_cross else cfg.breadth_bull
    if (close > ma20) and (close > ma60) and (b >= bull_thresh):
        return ("OFFENSIVE", cfg.offensive_exposure)
    if golden_cross and (close > ma60) and (close >= ma20 * cfg.top_safe_ma20_ratio) and (b >= cfg.breadth_bull_hold):
        return ("NEUTRAL", cfg.bull_hold_exposure)
    if golden_cross and (close > ma60):
        return ("NEUTRAL", cfg.bull_pullback_exposure)
    return ("NEUTRAL", cfg.neutral_exposure)


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
        reg, exp = classify_regime_point(
            float(idx.loc[dt]) if pd.notna(idx.loc[dt]) else np.nan,
            float(ma20.loc[dt]) if pd.notna(ma20.loc[dt]) else np.nan,
            float(ma60.loc[dt]) if pd.notna(ma60.loc[dt]) else np.nan,
            float(breadth.loc[dt]),
            float(ret_5d.loc[dt]) if pd.notna(ret_5d.loc[dt]) else 0.0,
            cfg,
        )
        regimes.append(reg)
        exposures.append(exp)

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
