"""Abstract DataProvider contract and standard real-market DataFrame schemas.

All providers MUST return real exchange data fetched from public A-share APIs.
Returning random / synthetic prices is strictly forbidden (see quant-guardrails rule #1).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import date
from functools import wraps
from typing import Callable, TypeVar

import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Standard column contracts --------------------------------------------------
STOCK_LIST_COLUMNS = ["ts_code", "symbol", "name", "exchange", "board", "list_date", "delist_date"]
DAILY_BAR_COLUMNS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
    "volume", "amount", "turnover_rate", "amplitude",
]
VALUATION_COLUMNS = [
    "ts_code", "trade_date", "pre_close", "pe_ttm", "pb", "is_st", "is_suspended",
    "turnover_rate", "close", "volume", "amount", "open", "high", "low",
]
ADJ_FACTOR_COLUMNS = ["ts_code", "trade_date", "adj_factor"]
INDEX_BAR_COLUMNS = ["index_code", "trade_date", "open", "high", "low", "close", "pre_close", "volume", "amount", "pct_chg"]
CAPITAL_FLOW_COLUMNS = [
    "ts_code", "trade_date", "main_net_inflow", "main_net_inflow_ratio", "super_large_net", "large_net",
]
FINANCIAL_COLUMNS = ["ts_code", "ann_date", "end_date", "net_profit_cum", "roe_avg_cum", "total_share", "revenue_cum"]
INDUSTRY_COLUMNS = ["ts_code", "sw_l1_code", "sw_l1_name"]
SPOT_COLUMNS = [
    "ts_code", "name", "last_price", "pre_close", "open", "high", "low", "volume", "amount",
    "bid1", "ask1", "pct_chg", "turnover_rate", "pe_ttm", "pb", "total_mv", "circ_mv",
]


def retry_call(retries: int = 3, backoff: float = 1.5, exceptions: tuple[type[BaseException], ...] = (Exception,)):
    """Retry decorator with exponential backoff for flaky public data endpoints."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203
                    last_exc = exc
                    wait = backoff ** attempt
                    logger.warning("%s attempt %d/%d failed: %s (retry in %.1fs)", func.__name__, attempt, retries, exc, wait)
                    if attempt < retries:
                        time.sleep(wait)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


class DataProvider(ABC):
    """Abstract real A-share market data provider."""

    name: str = "abstract"

    def login(self) -> None:  # pragma: no cover - default noop
        return None

    def logout(self) -> None:  # pragma: no cover - default noop
        return None

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """Return full A-share listing with STOCK_LIST_COLUMNS."""

    @abstractmethod
    def get_trade_calendar(self, start: date, end: date) -> list[date]:
        """Return sorted list of real exchange open dates in [start, end]."""

    @abstractmethod
    def get_daily_bars(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        """Return UNADJUSTED daily OHLCV bars with DAILY_BAR_COLUMNS (volume in shares, amount in CNY)."""

    @abstractmethod
    def get_index_daily(self, index_code: str, start: date, end: date) -> pd.DataFrame:
        """Return benchmark index daily bars with INDEX_BAR_COLUMNS."""

    def get_daily_valuation(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        """Return daily PE_TTM / PB / ST / suspension status with VALUATION_COLUMNS."""
        raise NotImplementedError(f"{self.name} does not support daily valuation")

    def get_adj_factor(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        """Return backward adjustment factor (hfq_price = raw_price * adj_factor)."""
        raise NotImplementedError(f"{self.name} does not support adj factor")

    def get_capital_flow(self, ts_code: str) -> pd.DataFrame:
        """Return recent daily main-force capital flow with CAPITAL_FLOW_COLUMNS."""
        raise NotImplementedError(f"{self.name} does not support capital flow")

    def get_financials(self, ts_code: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Return quarterly cumulative financial data with REAL announcement date (ann_date)."""
        raise NotImplementedError(f"{self.name} does not support financials")

    def get_industry_map(self) -> pd.DataFrame:
        """Return Shenwan level-1 industry mapping with INDUSTRY_COLUMNS."""
        raise NotImplementedError(f"{self.name} does not support industry map")

    def get_realtime_spot(self, ts_codes: list[str] | None = None) -> pd.DataFrame:
        """Return realtime snapshot with SPOT_COLUMNS (optionally filtered)."""
        raise NotImplementedError(f"{self.name} does not support realtime spot")

    def health_check(self) -> dict:
        return {"provider": self.name, "ok": True}
