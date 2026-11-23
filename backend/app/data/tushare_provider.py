"""Tushare Pro adapter (reserved). Activated only when TUSHARE_TOKEN is configured."""

from __future__ import annotations

from datetime import date

import pandas as pd

from app.config import get_settings
from app.core.constants import detect_board, normalize_ts_code
from app.core.exceptions import DataProviderError
from app.data.base import DAILY_BAR_COLUMNS, INDEX_BAR_COLUMNS, STOCK_LIST_COLUMNS, DataProvider, empty_frame


class TushareProvider(DataProvider):
    name = "tushare"

    def __init__(self, token: str | None = None) -> None:
        self.token = token if token is not None else get_settings().tushare_token
        self._pro = None

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def _client(self):
        if not self.enabled:
            raise DataProviderError("Tushare token not configured (set TUSHARE_TOKEN in backend/.env)")
        if self._pro is None:
            try:
                import tushare as ts  # optional dependency: pip install -e "backend[tushare]"
            except ImportError as exc:
                raise DataProviderError("tushare package not installed: pip install -e 'backend[tushare]'") from exc
            self._pro = ts.pro_api(self.token)
        return self._pro

    def get_stock_list(self) -> pd.DataFrame:
        raw = self._client().stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,list_date,delist_date")
        if raw is None or raw.empty:
            return empty_frame(STOCK_LIST_COLUMNS)
        df = raw.copy()
        df["ts_code"] = df["ts_code"].map(normalize_ts_code)
        df["exchange"] = df["ts_code"].str.split(".").str[1]
        df["board"] = df["ts_code"].map(detect_board)
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce").dt.date
        df["delist_date"] = pd.to_datetime(df["delist_date"], errors="coerce").dt.date
        return df[STOCK_LIST_COLUMNS]

    def get_trade_calendar(self, start: date, end: date) -> list[date]:
        raw = self._client().trade_cal(exchange="SSE", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), is_open="1")
        return sorted(pd.to_datetime(raw["cal_date"]).dt.date.tolist())

    def get_daily_bars(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        raw = self._client().daily(ts_code=normalize_ts_code(ts_code), start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        if raw is None or raw.empty:
            return empty_frame(DAILY_BAR_COLUMNS)
        df = pd.DataFrame()
        df["ts_code"] = raw["ts_code"].map(normalize_ts_code)
        df["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.date
        for col in ("open", "high", "low", "close", "pre_close"):
            df[col] = raw[col]
        df["volume"] = raw["vol"] * 100.0
        df["amount"] = raw["amount"] * 1000.0
        df["turnover_rate"] = pd.NA
        df["amplitude"] = (raw["high"] - raw["low"]) / raw["pre_close"] * 100.0
        return df[DAILY_BAR_COLUMNS].sort_values("trade_date").reset_index(drop=True)

    def get_index_daily(self, index_code: str, start: date, end: date) -> pd.DataFrame:
        raw = self._client().index_daily(ts_code=index_code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        if raw is None or raw.empty:
            return empty_frame(INDEX_BAR_COLUMNS)
        df = pd.DataFrame()
        df["index_code"] = index_code
        df["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.date
        for col in ("open", "high", "low", "close", "pre_close", "pct_chg"):
            df[col] = raw[col]
        df["volume"] = raw["vol"] * 100.0
        df["amount"] = raw["amount"] * 1000.0
        return df[INDEX_BAR_COLUMNS].sort_values("trade_date").reset_index(drop=True)

    def health_check(self) -> dict:
        if not self.enabled:
            return {"provider": self.name, "ok": False, "error": "TUSHARE_TOKEN not configured (reserved adapter)"}
        try:
            cal = self.get_trade_calendar(date(2024, 1, 1), date(2024, 1, 31))
            return {"provider": self.name, "ok": len(cal) > 0}
        except Exception as exc:  # noqa: BLE001
            return {"provider": self.name, "ok": False, "error": str(exc)}
