"""Dynamic universe filter applied on every trading / rebalance day."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.constants import DEFAULT_MICRO_CAP_QUANTILE, DEFAULT_MIN_AVG_AMOUNT_20D, DEFAULT_MIN_LIST_DAYS
from app.factors.panel import MarketPanel, trading_days_since_listing


@dataclass
class UniverseConfig:
    min_list_days: int = DEFAULT_MIN_LIST_DAYS
    min_avg_amount_20d: float = DEFAULT_MIN_AVG_AMOUNT_20D
    micro_cap_quantile: float = DEFAULT_MICRO_CAP_QUANTILE


@dataclass
class UniverseResult:
    eligible: pd.DataFrame        # bool, date x code
    reason: pd.DataFrame          # str, date x code ('' if eligible)
    avg_amount_20d: pd.DataFrame
    mcap_pct_rank: pd.DataFrame
    list_days: pd.DataFrame

    def to_long(self) -> pd.DataFrame:
        frames = {
            "is_eligible": self.eligible,
            "exclude_reason": self.reason,
            "avg_amount_20d": self.avg_amount_20d,
            "mcap_pct_rank": self.mcap_pct_rank,
            "list_days": self.list_days,
        }
        parts = []
        for name, df in frames.items():
            s = df.stack(future_stack=True)
            s.name = name
            parts.append(s)
        out = pd.concat(parts, axis=1).reset_index()
        out.columns = ["trade_date", "ts_code", *frames.keys()]
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
        out["list_days"] = out["list_days"].fillna(0).astype(int)
        out["is_eligible"] = out["is_eligible"].astype(bool)
        return out

    def summary(self) -> dict:
        last = self.eligible.index[-1]
        reasons = self.reason.loc[last]
        counts = reasons[reasons != ""].str.split("|").explode().value_counts().to_dict()
        return {
            "as_of": str(last.date()),
            "total": int(self.eligible.shape[1]),
            "eligible": int(self.eligible.loc[last].sum()),
            "excluded_by_reason": {k: int(v) for k, v in counts.items()},
            "avg_eligible_per_day": float(self.eligible.sum(axis=1).mean()),
        }


def apply_universe_filter(
    panel: MarketPanel,
    config: UniverseConfig | None = None,
    all_trade_dates: pd.DatetimeIndex | None = None,
) -> UniverseResult:
    """Exclude ST/*ST, suspended, IPO < 120 trading days, 20d avg amount < 50M CNY, bottom-10% market cap.

    All checks use information available at the close of day t only.
    """
    cfg = config or UniverseConfig()
    st = panel.is_st
    susp = panel.is_suspended
    list_days = trading_days_since_listing(panel, all_trade_dates)
    avg_amt = panel.amount.rolling(20, min_periods=20).mean()
    mcap = panel.total_mv.where(~susp)
    mcap_rank = mcap.rank(axis=1, pct=True)

    reason = pd.DataFrame("", index=panel.dates, columns=panel.codes, dtype=object)

    def tag(mask: pd.DataFrame, label: str) -> None:
        m = mask.fillna(False).to_numpy(dtype=bool)
        cur = reason.to_numpy()
        cur = np.where(m, np.where(cur == "", label, cur + "|" + label), cur)
        reason.loc[:, :] = cur

    tag(st, "ST")
    tag(susp, "SUSPENDED")
    tag(list_days < cfg.min_list_days, f"SUB_NEW_{cfg.min_list_days}D")
    tag(avg_amt.isna() | (avg_amt < cfg.min_avg_amount_20d), "LOW_LIQ_50M")
    tag(mcap_rank.isna() | (mcap_rank < cfg.micro_cap_quantile), "MICRO_CAP_10P")

    eligible = reason == ""
    return UniverseResult(
        eligible=eligible,
        reason=reason,
        avg_amount_20d=avg_amt,
        mcap_pct_rank=mcap_rank,
        list_days=list_days,
    )
