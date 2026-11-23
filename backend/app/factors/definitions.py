"""Four-dimension orthogonal factor library (16 factors), fully vectorized on the wide panel.

Every factor value at date t uses information up to and including the close of day t only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.constants import ALL_FACTOR_NAMES, BENCHMARK_CSI1000
from app.factors.panel import MarketPanel, np_nan_like

ANNUALIZE = np.sqrt(252.0)


@dataclass
class FactorSet:
    raw: dict[str, pd.DataFrame]          # factor_name -> date x code raw values
    degraded: dict[str, pd.DataFrame]     # factor_name -> date x code bool (source missing)
    status: dict[str, str]                # factor_name -> OK / PARTIAL_DEGRADED / DEGRADED_MISSING

    def to_long(self, tail_n: int | None = None) -> pd.DataFrame:
        parts = []
        for name, df in self.raw.items():
            sub_df = df.iloc[-tail_n:] if (tail_n and len(df) > tail_n) else df
            s = sub_df.stack(future_stack=True)
            deg = self.degraded[name].reindex_like(sub_df).stack(future_stack=True)
            part = pd.DataFrame({"raw_value": s, "is_degraded": deg.astype(bool)})
            part["factor_name"] = name
            parts.append(part)
        out = pd.concat(parts).reset_index()
        out = out.rename(columns={"level_0": "trade_date", "level_1": "ts_code"})
        out.columns = ["trade_date", "ts_code", "raw_value", "is_degraded", "factor_name"]
        return out


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    res = a / b.where(b.abs() > 1e-12)
    return res.replace([np.inf, -np.inf], np.nan)


def rolling_corr_frame(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    mx = x.rolling(window, min_periods=window).mean()
    my = y.rolling(window, min_periods=window).mean()
    cov = (x * y).rolling(window, min_periods=window).mean() - mx * my
    vx = (x * x).rolling(window, min_periods=window).mean() - mx * mx
    vy = (y * y).rolling(window, min_periods=window).mean() - my * my
    corr = cov / np.sqrt((vx * vy).where((vx > 1e-18) & (vy > 1e-18)))
    return corr.clip(-1, 1)


def compute_factors(panel: MarketPanel) -> FactorSet:
    hc = panel.hfq_close
    hh, hl = panel.hfq_high, panel.hfq_low
    ret = panel.returns
    active = ~panel.is_suspended
    vol = panel.volume.where(active)
    amt = panel.amount.where(active)
    turn = panel.turnover.where(active)

    raw: dict[str, pd.DataFrame] = {}
    degraded: dict[str, pd.DataFrame] = {}
    status: dict[str, str] = {}

    # 1) 量价与动量反转 --------------------------------------------------------
    raw["ret_20d"] = hc / hc.shift(20) - 1.0
    raw["ret_60d"] = hc / hc.shift(60) - 1.0
    raw["rev_5d"] = -(hc / hc.shift(5) - 1.0)
    raw["pv_corr_20d"] = rolling_corr_frame(hc, vol.fillna(0.0), 20)
    rng = (hh - hl)
    raw["intraday_pos"] = _safe_div(hc - hl, rng).where(rng > 1e-9, 0.5).where(active)

    # 2) 流动性与资金流 --------------------------------------------------------
    raw["turnover_ratio_20_60"] = _safe_div(turn.rolling(20, min_periods=15).mean(), turn.rolling(60, min_periods=40).mean())
    raw["turn_cv_20d"] = _safe_div(turn.rolling(20, min_periods=15).std(), turn.rolling(20, min_periods=15).mean())
    illiq = _safe_div(ret.abs(), amt) * 1e8  # |r| per 100M CNY traded
    raw["amihud_20d"] = illiq.rolling(20, min_periods=15).mean()
    raw["vol_ratio_5d"] = _safe_div(vol, vol.shift(1).rolling(5, min_periods=4).mean())

    if panel.capital_flow_ratio is not None and panel.capital_flow_ratio.notna().any().any():
        cfr = panel.capital_flow_ratio.reindex_like(hc)
        # Strict no-smear 5-day rolling mean of real main-force net inflow ratio
        raw["main_net_inflow_ratio"] = cfr.rolling(5, min_periods=1).mean().where(cfr.notna())
        degraded["main_net_inflow_ratio"] = raw["main_net_inflow_ratio"].isna()
        # Evaluate live health over trailing 60 trading days (free Eastmoney push2his provides ~120d real history)
        recent_win = raw["main_net_inflow_ratio"].tail(min(60, len(hc)))
        recent_cov = float(recent_win.notna().mean().mean()) if len(recent_win) else 0.0
        status["main_net_inflow_ratio"] = "OK" if recent_cov >= 0.70 else "PARTIAL_DEGRADED"
    else:
        # Interface unavailable: NaN + degraded flag -> pipeline fills cross-sectional neutral value.
        raw["main_net_inflow_ratio"] = np_nan_like(hc)
        degraded["main_net_inflow_ratio"] = pd.DataFrame(True, index=hc.index, columns=hc.columns)
        status["main_net_inflow_ratio"] = "DEGRADED_MISSING"

    # 3) 波动率与筹码结构 ------------------------------------------------------
    bench = panel.index_close.get(BENCHMARK_CSI1000)
    if bench is not None:
        m = bench.reindex(hc.index).pct_change(fill_method=None)
        mkt = pd.DataFrame(np.repeat(m.to_numpy()[:, None], hc.shape[1], axis=1), index=hc.index, columns=hc.columns)
        corr = rolling_corr_frame(ret, mkt, 20)
        var_r = ret.rolling(20, min_periods=15).var()
        raw["idio_vol_20d"] = np.sqrt((var_r * (1.0 - corr ** 2)).clip(lower=0)) * ANNUALIZE
        status["idio_vol_20d"] = "OK"
    else:
        raw["idio_vol_20d"] = ret.rolling(20, min_periods=15).std() * ANNUALIZE
        status["idio_vol_20d"] = "PARTIAL_DEGRADED"
    down_sq = (ret.where(ret < 0, 0.0) ** 2).rolling(20, min_periods=15).sum()
    all_sq = (ret ** 2).rolling(20, min_periods=15).sum()
    raw["downside_vol_ratio_20d"] = _safe_div(down_sq, all_sq)
    raw["price_pct_60d"] = hc.rolling(60, min_periods=40).rank(pct=True)

    # 4) 基本面质量与排雷 (PIT aligned by ann_date) ----------------------------
    pe = panel.pe_ttm.where(panel.pe_ttm.abs() > 1e-6)
    pb = panel.pb.where(panel.pb.abs() > 1e-6)
    raw["ep_ttm"] = (1.0 / pe).replace([np.inf, -np.inf], np.nan)
    raw["bp"] = (1.0 / pb).replace([np.inf, -np.inf], np.nan)
    for col in ("roe_ttm", "np_yoy_q"):
        f = panel.fundamentals.get(col)
        if f is not None:
            raw[col] = f.reindex_like(hc).astype(float).clip(-5, 5)
        else:
            raw[col] = np_nan_like(hc)
            status[col] = "DEGRADED_MISSING"

    for name in ALL_FACTOR_NAMES:
        raw[name] = raw[name].where(active).astype(float)
        if name not in degraded:
            degraded[name] = pd.DataFrame(False, index=hc.index, columns=hc.columns)
        status.setdefault(name, "OK")
    return FactorSet(raw={k: raw[k] for k in ALL_FACTOR_NAMES}, degraded=degraded, status=status)


def forward_returns(panel: MarketPanel, horizon: int = 5) -> pd.DataFrame:
    """Forward close-to-close hfq return from t to t+h (LABEL ONLY - never a feature)."""
    hc = panel.hfq_close
    return hc.shift(-horizon) / hc - 1.0


def forward_excess_returns(panel: MarketPanel, horizon: int = 5) -> pd.DataFrame:
    fwd = forward_returns(panel, horizon)
    bench = panel.index_close.get(BENCHMARK_CSI1000)
    if bench is None:
        return fwd.sub(fwd.mean(axis=1), axis=0)
    b = bench.reindex(fwd.index)
    bret = b.shift(-horizon) / b - 1.0
    return fwd.sub(bret, axis=0)
