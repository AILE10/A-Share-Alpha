"""Technical resonance filters (MA20 / bullish alignment / shrinking-volume pullback,
anti-overheating 3d gain <= 20%, anti-MACD bearish divergence) + ATR(14) calculation.

All indicators are computed strictly on backward-adjusted (hfq) prices up to date t close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.factors.panel import MarketPanel


@dataclass
class TechnicalResult:
    passed: pd.DataFrame            # bool: date x ts_code
    overheated: pd.DataFrame        # bool: 3d return > 20%
    macd_top_div: pd.DataFrame      # bool: MACD bearish divergence
    above_ma20: pd.DataFrame        # bool
    bull_ma: pd.DataFrame           # bool: MA5 > MA10 > MA20
    pullback_vol: pd.DataFrame      # bool: pullback near MA20 with shrinking volume & close >= open
    macd_golden: pd.DataFrame       # bool: DIF crosses above DEA or expanding red histogram
    atr_14_raw: pd.DataFrame        # raw-price-scale ATR(14) for stop-loss calculation
    ma5: pd.DataFrame
    ma10: pd.DataFrame
    ma20: pd.DataFrame
    ma60: pd.DataFrame
    macd_dif: pd.DataFrame
    macd_dea: pd.DataFrame
    macd_hist: pd.DataFrame

    def describe_stock(self, dt: pd.Timestamp, ts_code: str) -> list[str]:
        """Return human-readable Chinese technical pattern tags for UI explainability cards."""
        if dt not in self.passed.index or ts_code not in self.passed.columns:
            return []
        tags: list[str] = []
        if bool(self.bull_ma.loc[dt, ts_code]):
            tags.append("均线多头排列(MA5>10>20)")
        elif bool(self.above_ma20.loc[dt, ts_code]):
            tags.append("站上20日均线")
        if bool(self.pullback_vol.loc[dt, ts_code]):
            tags.append("回踩MA20缩量企稳")
        if bool(self.macd_golden.loc[dt, ts_code]):
            tags.append("MACD多头共振")
        if bool(self.overheated.loc[dt, ts_code]):
            tags.append("风险:近3日涨幅>20%过热")
        if bool(self.macd_top_div.loc[dt, ts_code]):
            tags.append("风险:MACD顶背离")
        return tags


def compute_technical_signals(
    panel: MarketPanel,
    max_3d_return: float = 0.20,
    div_lookback: int = 25,
    require_trend: bool = False,
) -> TechnicalResult:
    hc = panel.hfq_close
    ho = panel.hfq_open
    hh = panel.hfq_high
    hl = panel.hfq_low
    vol = panel.volume

    ma5 = hc.rolling(5, min_periods=5).mean()
    ma10 = hc.rolling(10, min_periods=10).mean()
    ma20 = hc.rolling(20, min_periods=15).mean()
    ma60 = hc.rolling(60, min_periods=40).mean()
    vol_ma20 = vol.rolling(20, min_periods=15).mean()

    above_ma20 = (hc > ma20).fillna(False)
    bull_ma = ((ma5 > ma10) & (ma10 > ma20)).fillna(False)
    pullback_vol = (
        (hc >= 0.98 * ma20)
        & (hc <= 1.03 * ma20)
        & (vol < 0.85 * vol_ma20)
        & (hc >= ho)
    ).fillna(False)

    # Anti-overheating: 3-day cumulative return <= 20%
    ret_3d = hc / hc.shift(3) - 1.0
    overheated = (ret_3d > max_3d_return).fillna(False)

    # MACD (12, 26, 9)
    ema12 = hc.ewm(span=12, adjust=False).mean()
    ema26 = hc.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2.0

    macd_golden = (((dif > dea) & (hist > hist.shift(1))) | ((dif > dea) & (dif.shift(1) <= dea.shift(1)))).fillna(False)

    # MACD Bearish Divergence (顶背离):
    # Price makes a higher peak in recent 7 days vs prior [8..div_lookback] days,
    # while MACD DIF fails to confirm (lower peak) AND histogram is turning down.
    recent_p_max = hh.rolling(7, min_periods=4).max()
    prior_p_max = hh.shift(7).rolling(max(10, div_lookback - 7), min_periods=8).max()
    recent_dif_max = dif.rolling(7, min_periods=4).max()
    prior_dif_max = dif.shift(7).rolling(max(10, div_lookback - 7), min_periods=8).max()

    macd_top_div = (
        (recent_p_max > prior_p_max * 1.01)
        & (recent_dif_max < prior_dif_max * 0.95)
        & (prior_dif_max > 0)
        & (hist < hist.shift(1))
    ).fillna(False)

    # ATR(14) computed on hfq prices then converted back to raw-price scale for exact stop-loss lines
    prev_hc = hc.shift(1)
    tr = pd.concat(
        [(hh - hl), (hh - prev_hc).abs(), (hl - prev_hc).abs()],
        axis=0,
    )
    # Vectorized max across the 3 true-range components
    tr_arr = np.maximum(
        (hh - hl).to_numpy(dtype=float),
        np.maximum((hh - prev_hc).abs().to_numpy(dtype=float), (hl - prev_hc).abs().to_numpy(dtype=float)),
    )
    tr_df = pd.DataFrame(tr_arr, index=hc.index, columns=hc.columns)
    atr_hfq = tr_df.rolling(14, min_periods=10).mean()
    atr_raw = atr_hfq / panel.adj.replace(0, np.nan)

    # Anti-Chase technical guard (default require_trend=False):
    # Blocks 3-day overheated chasing (>20%) and MACD bearish top divergence without
    # conflicting with short-term reversal pullbacks (rev_5d / price_pct_60d) below MA20.
    # All trend indicators (above_ma20 / bull_ma / pullback_vol / macd_golden) remain
    # computed for UI explainability tags and Dynamic Holding profit-ride checks.
    trend_ok = above_ma20 | bull_ma | pullback_vol
    if require_trend:
        passed = trend_ok & (~overheated) & (~macd_top_div)
    else:
        passed = (~overheated) & (~macd_top_div)

    return TechnicalResult(
        passed=passed,
        overheated=overheated,
        macd_top_div=macd_top_div,
        above_ma20=above_ma20,
        bull_ma=bull_ma,
        pullback_vol=pullback_vol,
        macd_golden=macd_golden,
        atr_14_raw=atr_raw,
        ma5=ma5,
        ma10=ma10,
        ma20=ma20,
        ma60=ma60,
        macd_dif=dif,
        macd_dea=dea,
        macd_hist=hist,
    )
