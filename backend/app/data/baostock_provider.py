"""Baostock backup real-market data provider.

Baostock uniquely provides, in one call, UNADJUSTED daily bars with exchange pre-close,
point-in-time PE_TTM / PB_MRQ, ST flag and suspension status, plus quarterly financial
statements with their REAL first announcement date (pubDate) -> used for PIT alignment.
"""

from __future__ import annotations

import threading
from datetime import date

import numpy as np
import pandas as pd

from app.core.constants import BENCHMARK_INDEX_META, detect_board, normalize_ts_code, ts_code_to_baostock
from app.core.logging import get_logger
from app.data.base import (
    DAILY_BAR_COLUMNS,
    FINANCIAL_COLUMNS,
    INDEX_BAR_COLUMNS,
    INDUSTRY_COLUMNS,
    STOCK_LIST_COLUMNS,
    VALUATION_COLUMNS,
    ADJ_FACTOR_COLUMNS,
    DataProvider,
    empty_frame,
    retry_call,
)
from app.core.exceptions import DataProviderError

logger = get_logger(__name__)

_BS_LOCK = threading.RLock()


def _bs():
    import baostock as bs

    return bs


def _rs_to_df(rs) -> pd.DataFrame:
    if rs is None:
        return pd.DataFrame()
    if rs.error_code != "0":
        raise DataProviderError(f"baostock error {rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


class BaostockProvider(DataProvider):
    name = "baostock"

    def __init__(self, connect_timeout: float = 4.0) -> None:
        self._logged_in = False
        self._connect_timeout = connect_timeout
        self._disabled_until: float = 0.0

    def _probe_server(self) -> bool:
        import socket
        import time

        if time.monotonic() < self._disabled_until:
            return False
        try:
            with socket.create_connection(("www.baostock.com", 10030), timeout=self._connect_timeout):
                return True
        except OSError:
            self._disabled_until = time.monotonic() + 180.0
            logger.warning("Baostock TCP server unreachable/slow; backing off 180s (AKShare primary active)")
            return False

    def login(self) -> None:
        import concurrent.futures
        import time

        with _BS_LOCK:
            if self._logged_in:
                return
            if not self._probe_server():
                raise DataProviderError("baostock server unreachable (fast fail)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_bs().login)
                try:
                    lg = fut.result(timeout=self._connect_timeout + 4.0)
                except concurrent.futures.TimeoutError as exc:
                    self._disabled_until = time.monotonic() + 180.0
                    raise DataProviderError("baostock login timed out (fast fail)") from exc
            if lg.error_code != "0":
                raise DataProviderError(f"baostock login failed: {lg.error_msg}")
            self._logged_in = True

    def logout(self) -> None:
        with _BS_LOCK:
            if self._logged_in:
                try:
                    _bs().logout()
                finally:
                    self._logged_in = False

    def _query(self, fn_name: str, *args, **kwargs) -> pd.DataFrame:
        with _BS_LOCK:
            self.login()
            fn = getattr(_bs(), fn_name)
            rs = fn(*args, **kwargs)
            try:
                return _rs_to_df(rs)
            except DataProviderError as exc:
                # session may have expired -> relogin once
                if "login" in str(exc).lower() or "10001001" in str(exc):
                    self._logged_in = False
                    self.login()
                    return _rs_to_df(fn(*args, **kwargs))
                raise

    # ------------------------------------------------------------------ listing
    @retry_call(retries=3)
    def get_stock_list(self) -> pd.DataFrame:
        raw = self._query("query_stock_basic")
        if raw.empty:
            return empty_frame(STOCK_LIST_COLUMNS)
        raw = raw[raw["type"] == "1"]  # 1 = stock
        df = pd.DataFrame()
        df["ts_code"] = raw["code"].map(normalize_ts_code)
        df["symbol"] = df["ts_code"].str.split(".").str[0]
        df["name"] = raw["code_name"].astype(str).str.replace(" ", "", regex=False)
        df["exchange"] = df["ts_code"].str.split(".").str[1]
        df["board"] = df["ts_code"].map(detect_board)
        df["list_date"] = pd.to_datetime(raw["ipoDate"].replace("", None), errors="coerce").dt.date
        df["delist_date"] = pd.to_datetime(raw["outDate"].replace("", None), errors="coerce").dt.date
        return df[STOCK_LIST_COLUMNS].reset_index(drop=True)

    @retry_call(retries=3)
    def get_trade_calendar(self, start: date, end: date) -> list[date]:
        raw = self._query("query_trade_dates", start_date=start.isoformat(), end_date=end.isoformat())
        raw = raw[raw["is_trading_day"] == "1"]
        return sorted(pd.to_datetime(raw["calendar_date"]).dt.date.tolist())

    # ------------------------------------------------------------------ bars
    _K_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,isST"

    @retry_call(retries=3)
    def _raw_k(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        return self._query(
            "query_history_k_data_plus",
            ts_code_to_baostock(ts_code),
            self._K_FIELDS,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            frequency="d",
            adjustflag="3",  # 3 = unadjusted raw exchange price
        )

    def get_daily_valuation(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        raw = self._raw_k(ts_code, start, end)
        if raw.empty:
            return empty_frame(VALUATION_COLUMNS)
        df = pd.DataFrame()
        df["trade_date"] = pd.to_datetime(raw["date"]).dt.date
        df["ts_code"] = normalize_ts_code(ts_code)
        df["open"] = _num(raw["open"])
        df["high"] = _num(raw["high"])
        df["low"] = _num(raw["low"])
        df["close"] = _num(raw["close"])
        df["pre_close"] = _num(raw["preclose"])
        df["volume"] = _num(raw["volume"])
        df["amount"] = _num(raw["amount"])
        df["turnover_rate"] = _num(raw["turn"])
        df["pe_ttm"] = _num(raw["peTTM"])
        df["pb"] = _num(raw["pbMRQ"])
        df["is_st"] = raw["isST"].astype(str) == "1"
        df["is_suspended"] = raw["tradestatus"].astype(str) != "1"
        return df[VALUATION_COLUMNS]

    def get_daily_bars(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        val = self.get_daily_valuation(ts_code, start, end)
        if val.empty:
            return empty_frame(DAILY_BAR_COLUMNS)
        df = val.copy()
        df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"] * 100.0
        return df[DAILY_BAR_COLUMNS]

    @retry_call(retries=3)
    def get_adj_factor(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        """Backward adj factor event table (hfq = raw * backAdjustFactor) forward-filled later."""
        raw = self._query(
            "query_adjust_factor",
            code=ts_code_to_baostock(ts_code),
            start_date="1990-01-01",
            end_date=end.isoformat(),
        )
        if raw.empty:
            return empty_frame(ADJ_FACTOR_COLUMNS)
        df = pd.DataFrame()
        df["trade_date"] = pd.to_datetime(raw["dividOperateDate"]).dt.date
        df["ts_code"] = normalize_ts_code(ts_code)
        df["adj_factor"] = _num(raw["backAdjustFactor"])
        return df[ADJ_FACTOR_COLUMNS].sort_values("trade_date").reset_index(drop=True)

    @retry_call(retries=3)
    def get_index_daily(self, index_code: str, start: date, end: date) -> pd.DataFrame:
        meta = BENCHMARK_INDEX_META.get(index_code)
        bs_code = meta["bs_code"] if meta else ts_code_to_baostock(index_code)
        raw = self._query(
            "query_history_k_data_plus",
            bs_code,
            "date,open,high,low,close,preclose,volume,amount,pctChg",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            frequency="d",
        )
        if raw.empty:
            return empty_frame(INDEX_BAR_COLUMNS)
        df = pd.DataFrame()
        df["trade_date"] = pd.to_datetime(raw["date"]).dt.date
        df["index_code"] = index_code
        df["open"] = _num(raw["open"])
        df["high"] = _num(raw["high"])
        df["low"] = _num(raw["low"])
        df["close"] = _num(raw["close"])
        df["pre_close"] = _num(raw["preclose"])
        df["volume"] = _num(raw["volume"])
        df["amount"] = _num(raw["amount"])
        df["pct_chg"] = _num(raw["pctChg"])
        return df[INDEX_BAR_COLUMNS]

    # ------------------------------------------------------------------ financials (PIT)
    def get_financials(self, ts_code: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Quarterly cumulative profit statements with real announcement date `pubDate`."""
        code = ts_code_to_baostock(ts_code)
        frames = []
        for year in range(start_year, end_year + 1):
            for quarter in (1, 2, 3, 4):
                try:
                    part = retry_call(retries=2)(self._query)("query_profit_data", code=code, year=year, quarter=quarter)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("baostock profit %s %sQ%s failed: %s", ts_code, year, quarter, exc)
                    continue
                if not part.empty:
                    frames.append(part)
        if not frames:
            return empty_frame(FINANCIAL_COLUMNS)
        raw = pd.concat(frames, ignore_index=True)
        df = pd.DataFrame()
        df["ts_code"] = [normalize_ts_code(ts_code)] * len(raw)
        df["ann_date"] = pd.to_datetime(raw["pubDate"], errors="coerce").dt.date
        df["end_date"] = pd.to_datetime(raw["statDate"], errors="coerce").dt.date
        df["net_profit_cum"] = _num(raw["netProfit"])
        df["roe_avg_cum"] = _num(raw["roeAvg"])
        df["total_share"] = _num(raw["totalShare"])
        df["revenue_cum"] = _num(raw["MBRevenue"]) if "MBRevenue" in raw.columns else np.nan
        df = df.dropna(subset=["ann_date", "end_date"])
        return df[FINANCIAL_COLUMNS].sort_values("end_date").reset_index(drop=True)

    @retry_call(retries=3)
    def get_industry_map(self) -> pd.DataFrame:
        """Fallback industry map (CSRC classification via Baostock) when SW L1 is unavailable."""
        raw = self._query("query_stock_industry")
        if raw.empty:
            return empty_frame(INDUSTRY_COLUMNS)
        df = pd.DataFrame()
        df["ts_code"] = raw["code"].map(normalize_ts_code)
        ind = raw["industry"].replace("", "未分类").fillna("未分类")
        df["sw_l1_code"] = ind.str[:3]
        df["sw_l1_name"] = ind
        return df[INDUSTRY_COLUMNS]

    def health_check(self) -> dict:
        try:
            cal = self.get_trade_calendar(date(2024, 1, 1), date(2024, 1, 31))
            return {"provider": self.name, "ok": len(cal) > 0}
        except Exception as exc:  # noqa: BLE001
            return {"provider": self.name, "ok": False, "error": str(exc)}
