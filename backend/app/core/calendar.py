"""A-Share trading calendar utilities backed by DuckDB trade_calendar table."""

from __future__ import annotations

from datetime import date, datetime, time, timezone, timedelta
from typing import Sequence

import pandas as pd

CN_TZ = timezone(timedelta(hours=8))


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def today_cn() -> date:
    return now_cn().date()


def is_trading_session(dt: datetime | None = None) -> bool:
    """Check if current Beijing time is inside A-share continuous trading hours."""
    cur = dt or now_cn()
    t = cur.time()
    morning = time(9, 25) <= t <= time(11, 31)
    afternoon = time(13, 0) <= t <= time(15, 1)
    return morning or afternoon


def get_trading_dates(
    dates_pool: Sequence[date],
    start: date | None = None,
    end: date | None = None,
) -> list[date]:
    sorted_dates = sorted(set(dates_pool))
    if start is not None:
        sorted_dates = [d for d in sorted_dates if d >= start]
    if end is not None:
        sorted_dates = [d for d in sorted_dates if d <= end]
    return sorted_dates


def offset_trading_date(sorted_dates: Sequence[date], base_date: date, offset: int) -> date | None:
    """Return trading date shifted by `offset` trading days from `base_date`."""
    if not sorted_dates:
        return None
    idx = pd.Index(sorted_dates).searchsorted(base_date, side="left")
    if idx >= len(sorted_dates) or sorted_dates[idx] != base_date:
        if offset <= 0:
            idx = max(0, idx - 1)
        else:
            idx = min(len(sorted_dates) - 1, idx)
    target = idx + offset
    if 0 <= target < len(sorted_dates):
        return sorted_dates[target]
    return None
