"""AKShare primary real-market data provider (AKShare 1.20+ Eastmoney / Tencent / Sina / SW endpoints).

Automatically fails fast on Eastmoney push2 rate-limit errors (to avoid extending cooldown)
and seamlessly falls back to Tencent / Sina endpoints inside AKShare, with Baostock as
full backup in DataSyncService.
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd

from app.core.constants import (
    BENCHMARK_INDEX_META,
    detect_board,
    normalize_ts_code,
    round_price,
    ts_code_to_ak_prefix,
    ts_code_to_symbol,
)
from app.core.logging import get_logger
from app.data.base import (
    CAPITAL_FLOW_COLUMNS,
    DAILY_BAR_COLUMNS,
    INDEX_BAR_COLUMNS,
    INDUSTRY_COLUMNS,
    SPOT_COLUMNS,
    STOCK_LIST_COLUMNS,
    DataProvider,
    empty_frame,
    retry_call,
)

logger = get_logger(__name__)


def _ak():
    import akshare as ak  # lazy import: heavy module

    return ak


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_cn_amount(val: object) -> float:
    """Parse Chinese financial unit strings ('1.46亿', '-9068.88万', '12.5%') into raw float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip().replace(",", "")
    if not s or s in ("-", "--", "nan", "None"):
        return float("nan")
    mult = 1.0
    if s.endswith("亿"):
        mult = 1e8
        s = s[:-1]
    elif s.endswith("万"):
        mult = 1e4
        s = s[:-1]
    elif s.endswith("元"):
        s = s[:-1]
    elif s.endswith("%"):
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return float("nan")


def _is_em_rate_limit(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "ratelimit" in name or "push2" in msg or "限流" in str(exc)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)


def parse_tencent_fh_dict(fh_dict: object) -> dict | None:
    """Parse Tencent kline r[6] corporate action payload into per-share ratios.

    Handles annual/interim/special cash dividends ('10派X元'), bonus shares ('10送X股'),
    capital reserve transfers ('10转X股'), and marks rights-issue ex-dates where
    Tencent emits a non-empty dict with empty FHcontent.
    """
    import re

    if not isinstance(fh_dict, dict) or not fh_dict:
        return None
    txt = str(fh_dict.get("FHcontent") or "").strip()

    def _extract(pattern: str) -> float:
        m = re.search(pattern, txt)
        if not m:
            return 0.0
        base = float(m.group(1))
        val = float(m.group(2))
        return val / base if base > 0 else 0.0

    bonus = _extract(r"(\d+(?:\.\d+)?)\s*股?送\s*(\d+(?:\.\d+)?)")
    transfer = _extract(r"(\d+(?:\.\d+)?)\s*股?转(?:增)?\s*(\d+(?:\.\d+)?)")
    cash = _extract(r"(\d+(?:\.\d+)?)\s*股?派(?:现|息)?\s*(\d+(?:\.\d+)?)")
    if cash == 0.0 and fh_dict.get("fh_sh") not in (None, "", "0", 0):
        try:
            fh_val = float(fh_dict["fh_sh"])
            if fh_val > 0:
                cash = fh_val / 10.0
        except (TypeError, ValueError):
            pass
    rights = _extract(r"(\d+(?:\.\d+)?)\s*股?配\s*(\d+(?:\.\d+)?)")
    rights_price = 0.0
    if rights > 0:
        m_p = re.search(r"配股价\s*(\d+(?:\.\d+)?)", txt)
        if m_p:
            rights_price = float(m_p.group(1))
    split_mult = 1.0 + bonus + transfer
    return {
        "bonus_ratio": round(bonus, 6),
        "transfer_ratio": round(transfer, 6),
        "cash_div": round(cash, 6),
        "rights_ratio": round(rights, 6),
        "rights_price": round(rights_price, 4),
        "split_mult": round(split_mult, 6),
        "raw_content": txt,
        "is_unparsed_event": (split_mult <= 1.0 + 1e-9 and cash <= 1e-9 and rights <= 1e-9),
    }


class AKShareProvider(DataProvider):
    name = "akshare"

    def __init__(self, request_interval: float = 0.25) -> None:
        import threading

        self.request_interval = request_interval
        self._last_call = 0.0
        self._em_cooldown_until: float = 0.0
        self._em_his_cooldown_until: float = 0.0
        self._tx_client = None
        self._tx_lock = threading.Lock()

    def _get_tx_client(self):
        if self._tx_client is None:
            with self._tx_lock:
                if self._tx_client is None:
                    import httpx

                    self._tx_client = httpx.Client(
                        timeout=12.0,
                        limits=httpx.Limits(max_connections=24, max_keepalive_connections=20),
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
        return self._tx_client

    def close(self) -> None:
        if self._tx_client is not None:
            try:
                self._tx_client.close()
            except Exception:  # noqa: BLE001
                pass
            self._tx_client = None

    def _fetch_tx_kline_fast(self, ts_code: str, start: date, end: date, adjust: str = "") -> pd.DataFrame:
        """Direct keep-alive Tencent newfqkline fetcher (exact same endpoint & unit scaling as ak.stock_zh_a_hist_tx,
        skipping redundant per-stock get_tx_start_year probes and tqdm overhead -> ~10x faster)."""
        import json

        symbol = ts_code_to_ak_prefix(ts_code)
        vol_mult = 1.0 if symbol.startswith(("sh688", "sh689")) else 100.0
        url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
        client = self._get_tx_client()

        # Chunk by <= 550 calendar days (~380 trading days < 640-bar cap)
        windows: list[tuple[date, date]] = []
        cur = start
        while cur <= end:
            nxt = date(min(end.year, cur.year + 1), 12, 31)
            if nxt > end:
                nxt = end
            windows.append((cur, nxt))
            cur = date(nxt.year + 1, 1, 1)

        rows: list[list] = []
        key_order = ("hfqday", "day", "qfqday") if adjust == "hfq" else ("day", "hfqday", "qfqday")
        for w_start, w_end in windows:
            params = {
                "_var": f"kline_day{adjust}{w_start.year}",
                "param": f"{symbol},day,{w_start.isoformat()},{w_end.isoformat()},640,{adjust}",
            }
            resp = client.get(url, params=params)
            resp.raise_for_status()
            txt = resp.text
            idx = txt.find("={")
            if idx < 0:
                continue
            payload = json.loads(txt[idx + 1 :])
            node = (payload.get("data") or {}).get(symbol)
            if not isinstance(node, dict):
                continue
            day_rows = None
            for k in key_order:
                if k in node and node[k]:
                    day_rows = node[k]
                    break
            if day_rows:
                rows.extend(day_rows)

        if not rows:
            return empty_frame(DAILY_BAR_COLUMNS)

        parsed = []
        raw_actions: dict[date, dict] = {}
        for r in rows:
            if not isinstance(r, list) or len(r) < 6:
                continue
            d_str = str(r[0])[:10]
            try:
                d_val = date.fromisoformat(d_str)
            except ValueError:
                continue
            if d_val < start or d_val > end:
                continue
            try:
                op = float(r[1])
                cl = float(r[2])
                hi = float(r[3])
                lo = float(r[4])
                vol = float(r[5]) * vol_mult
            except (TypeError, ValueError):
                continue
            if adjust == "" and len(r) > 6 and isinstance(r[6], dict) and r[6]:
                parsed_act = parse_tencent_fh_dict(r[6])
                if parsed_act is not None:
                    raw_actions[d_val] = parsed_act
            turn_pct = np.nan
            amt = np.nan
            if len(r) > 7 and r[7] not in ("", None):
                try:
                    turn_pct = float(r[7])
                except (TypeError, ValueError):
                    pass
            if len(r) > 8 and r[8] not in ("", None):
                try:
                    amt = float(r[8]) * 10000.0
                except (TypeError, ValueError):
                    pass
            parsed.append((d_val, op, hi, lo, cl, vol, amt, turn_pct))

        if not parsed:
            return empty_frame(DAILY_BAR_COLUMNS)

        norm_code = normalize_ts_code(ts_code)
        df = pd.DataFrame(parsed, columns=["trade_date", "open", "high", "low", "close", "volume", "amount", "turnover_rate"])
        df = df.drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
        df["ts_code"] = norm_code

        # Compute exact exchange ex-right reference `pre_close` and true multiplicative `adj_factor`
        close_arr = df["close"].to_numpy(dtype=float)
        open_arr = df["open"].to_numpy(dtype=float)
        dates_arr = df["trade_date"].tolist()
        n = len(df)
        pre_close_arr = np.full(n, np.nan, dtype=float)
        adj_arr = np.ones(n, dtype=float)

        # Rare rights-issue days (empty FHcontent dict) -> resolve exact ex_ref via single-stock hfq fallback
        unparsed_dates = {d for d, a in raw_actions.items() if a.get("is_unparsed_event")}
        hfq_close_map: dict[date, float] = {}
        if adjust == "" and unparsed_dates:
            try:
                hfq_df = self._fetch_tx_kline_fast(ts_code, start, end, adjust="hfq")
                if not hfq_df.empty:
                    hfq_close_map = dict(zip(hfq_df["trade_date"], hfq_df["close"].astype(float)))
            except Exception:  # noqa: BLE001
                hfq_close_map = {}

        ca_records: list[dict] = []
        for i in range(n):
            d_i = dates_arr[i]
            act = raw_actions.get(d_i) if adjust == "" else None
            if i == 0:
                if act is not None and not act.get("is_unparsed_event"):
                    ca_records.append({
                        "ts_code": norm_code,
                        "ex_date": d_i,
                        "bonus_ratio": act["bonus_ratio"],
                        "transfer_ratio": act["transfer_ratio"],
                        "cash_div": act["cash_div"],
                        "rights_ratio": act["rights_ratio"],
                        "rights_price": act["rights_price"],
                        "split_mult": act["split_mult"],
                        "raw_content": act["raw_content"],
                        "source": "tencent",
                    })
                continue

            prev_c = close_arr[i - 1]
            ex_ref = prev_c
            if act is not None and prev_c > 0:
                if not act.get("is_unparsed_event"):
                    k_tot = act["split_mult"] + act["rights_ratio"]
                    num = prev_c - act["cash_div"] + act["rights_price"] * act["rights_ratio"]
                    cand = round_price(num / k_tot) if k_tot > 0 else prev_c
                    if cand > 0.01:
                        ex_ref = cand
                    ca_records.append({
                        "ts_code": norm_code,
                        "ex_date": d_i,
                        "bonus_ratio": act["bonus_ratio"],
                        "transfer_ratio": act["transfer_ratio"],
                        "cash_div": act["cash_div"],
                        "rights_ratio": act["rights_ratio"],
                        "rights_price": act["rights_price"],
                        "split_mult": act["split_mult"],
                        "raw_content": act["raw_content"],
                        "source": "tencent",
                    })
                else:
                    # Rights-issue ex-date with empty FHcontent: infer exchange ex_ref from hfq continuity
                    h_prev = hfq_close_map.get(dates_arr[i - 1], np.nan)
                    h_cur = hfq_close_map.get(d_i, np.nan)
                    if np.isfinite(h_prev) and np.isfinite(h_cur) and h_prev > 0 and h_cur > 0 and close_arr[i] > 0:
                        implied = round_price(close_arr[i] * (h_prev / h_cur))
                        if 0.1 * prev_c <= implied <= 1.2 * prev_c:
                            ex_ref = implied
                            implied_k = prev_c / ex_ref
                            ca_records.append({
                                "ts_code": norm_code,
                                "ex_date": d_i,
                                "bonus_ratio": 0.0,
                                "transfer_ratio": 0.0,
                                "cash_div": round(max(0.0, prev_c - ex_ref), 6) if implied_k < 1.15 else 0.0,
                                "rights_ratio": round(max(0.0, implied_k - 1.0), 6) if implied_k >= 1.15 else 0.0,
                                "rights_price": 0.0,
                                "split_mult": round(implied_k, 6) if implied_k >= 1.15 else 1.0,
                                "raw_content": "rights_or_unparsed_cqr",
                                "source": "tencent_hfq_inferred",
                            })
                    elif open_arr[i] > 0 and abs(open_arr[i] - prev_c) / prev_c > 0.12:
                        ex_ref = round_price(open_arr[i])

            pre_close_arr[i] = ex_ref
            if ex_ref > 0 and prev_c > 0 and abs(ex_ref - prev_c) > 1e-6:
                adj_arr[i] = adj_arr[i - 1] * (prev_c / ex_ref)
            else:
                adj_arr[i] = adj_arr[i - 1]

        df["pre_close"] = pre_close_arr
        df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"] * 100.0
        out = df[DAILY_BAR_COLUMNS].copy()
        if adjust == "":
            out["adj_factor"] = adj_arr
            out.attrs["corporate_actions"] = pd.DataFrame(ca_records) if ca_records else pd.DataFrame()
        return out

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_call = time.monotonic()

    def _em_available(self) -> bool:
        return time.monotonic() >= self._em_cooldown_until

    def _em_his_available(self) -> bool:
        return time.monotonic() >= self._em_his_cooldown_until

    def _trip_em_cooldown(self, seconds: float = 300.0) -> None:
        self._em_cooldown_until = time.monotonic() + seconds
        logger.warning("Eastmoney push2 rate-limited; cooling down %.0fs and using AKShare Tencent/Sina/THS endpoints", seconds)

    def _trip_em_his_cooldown(self, seconds: float = 300.0) -> None:
        self._em_his_cooldown_until = time.monotonic() + seconds
        logger.warning("Eastmoney push2his rate-limited; cooling down %.0fs and falling back to THS bulk capital flow", seconds)

    # ------------------------------------------------------------------ listing
    def get_stock_list(self) -> pd.DataFrame:
        """Fetch official SSE & SZSE exchange stock listings with real IPO list_date."""
        ak = _ak()
        frames: list[pd.DataFrame] = []

        # 1) Official SSE Main Board & STAR Board
        for kind in ("主板A股", "科创板"):
            try:
                self._throttle()
                raw = retry_call(retries=2)(ak.stock_info_sh_name_code)(symbol=kind)
                if raw is not None and not raw.empty:
                    p = pd.DataFrame()
                    p["symbol"] = raw["证券代码"].astype(str).str.zfill(6)
                    p["ts_code"] = p["symbol"].map(normalize_ts_code)
                    p["name"] = raw["证券简称"].astype(str).str.replace(" ", "", regex=False)
                    p["exchange"] = "SH"
                    p["board"] = p["ts_code"].map(detect_board)
                    p["list_date"] = pd.to_datetime(raw["上市日期"], errors="coerce").dt.date
                    p["delist_date"] = pd.NaT
                    frames.append(p)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SSE %s list failed: %s", kind, exc)

        # 2) Official SZSE Main & ChiNext Board
        try:
            self._throttle()
            raw = retry_call(retries=2)(ak.stock_info_sz_name_code)(symbol="A股列表")
            if raw is not None and not raw.empty:
                p = pd.DataFrame()
                p["symbol"] = raw["A股代码"].astype(str).str.zfill(6)
                p["ts_code"] = p["symbol"].map(normalize_ts_code)
                p["name"] = raw["A股简称"].astype(str).str.replace(" ", "", regex=False)
                p["exchange"] = "SZ"
                p["board"] = p["ts_code"].map(detect_board)
                p["list_date"] = pd.to_datetime(raw["A股上市日期"], errors="coerce").dt.date
                p["delist_date"] = pd.NaT
                frames.append(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SZSE A-share list failed: %s", exc)

        if frames:
            out = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
            return out[STOCK_LIST_COLUMNS].reset_index(drop=True)

        # Fallback: code/name without IPO date
        self._throttle()
        raw = retry_call(retries=3)(ak.stock_info_a_code_name)()
        if raw is None or raw.empty:
            return empty_frame(STOCK_LIST_COLUMNS)
        df = pd.DataFrame()
        df["ts_code"] = raw["code"].astype(str).map(normalize_ts_code)
        df["symbol"] = raw["code"].astype(str).str.zfill(6)
        df["name"] = raw["name"].astype(str).str.replace(" ", "", regex=False)
        df["exchange"] = df["ts_code"].str.split(".").str[1]
        df["board"] = df["ts_code"].map(detect_board)
        df["list_date"] = pd.NaT
        df["delist_date"] = pd.NaT
        return df[STOCK_LIST_COLUMNS]

    @retry_call(retries=3)
    def get_trade_calendar(self, start: date, end: date) -> list[date]:
        self._throttle()
        ak = _ak()
        fn = getattr(ak, "calendar_trade_date_hist_sina", None) or getattr(ak, "tool_trade_date_hist_sina")
        raw = fn()
        dates = pd.to_datetime(raw["trade_date"]).dt.date
        return sorted(d for d in dates if start <= d <= end)

    # ------------------------------------------------------------------ bars
    def _bars_em(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        self._throttle()
        ak = _ak()
        fn = getattr(ak, "stock_zh_a_hist_em", None) or getattr(ak, "stock_zh_a_hist")
        raw = fn(
            symbol=ts_code_to_symbol(ts_code),
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="",
        )
        if raw is None or raw.empty:
            return empty_frame(DAILY_BAR_COLUMNS)
        df = pd.DataFrame()
        df["trade_date"] = pd.to_datetime(raw["日期"]).dt.date
        df["ts_code"] = normalize_ts_code(ts_code)
        df["open"] = _to_float(raw["开盘"])
        df["high"] = _to_float(raw["最高"])
        df["low"] = _to_float(raw["最低"])
        df["close"] = _to_float(raw["收盘"])
        chg = _to_float(raw["涨跌额"]) if "涨跌额" in raw.columns else np.nan
        df["pre_close"] = (df["close"] - chg).round(2)
        df["volume"] = _to_float(raw["成交量"]) * 100.0  # 手 -> 股
        df["amount"] = _to_float(raw["成交额"])
        df["turnover_rate"] = _to_float(raw["换手率"]) if "换手率" in raw.columns else np.nan
        df["amplitude"] = _to_float(raw["振幅"]) if "振幅" in raw.columns else np.nan
        return df[DAILY_BAR_COLUMNS]

    def _bars_tx(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        self._throttle()
        raw = _ak().stock_zh_a_hist_tx(
            symbol=ts_code_to_ak_prefix(ts_code),
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="",
        )
        if raw is None or raw.empty:
            return empty_frame(DAILY_BAR_COLUMNS)
        df = pd.DataFrame()
        df["trade_date"] = pd.to_datetime(raw["date"]).dt.date
        df["ts_code"] = normalize_ts_code(ts_code)
        df["open"] = _to_float(raw["open"])
        df["high"] = _to_float(raw["high"])
        df["low"] = _to_float(raw["low"])
        df["close"] = _to_float(raw["close"])
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["pre_close"] = df["close"].shift(1)
        df["volume"] = _to_float(raw["volume"])  # already in shares per akshare docstring
        df["amount"] = _to_float(raw["amount"])
        # tx turnover is fraction (e.g. 0.0040 = 0.40%) -> convert to percentage points
        turn = _to_float(raw["turnover"]) if "turnover" in raw.columns else pd.Series(np.nan, index=raw.index)
        df["turnover_rate"] = turn * 100.0
        df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"] * 100.0
        return df[DAILY_BAR_COLUMNS]

    def get_daily_bars(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        if self._em_available():
            try:
                df = self._bars_em(ts_code, start, end)
                if not df.empty:
                    return df
            except Exception as exc:  # noqa: BLE001
                if _is_em_rate_limit(exc):
                    self._trip_em_cooldown()
                else:
                    logger.warning("AKShare EM bars %s failed: %s", ts_code, exc)
        return retry_call(retries=2)(self._bars_tx)(ts_code, start, end)

    @retry_call(retries=3)
    def get_index_daily(self, index_code: str, start: date, end: date) -> pd.DataFrame:
        self._throttle()
        meta = BENCHMARK_INDEX_META.get(index_code)
        ak_symbol = meta["ak_symbol"] if meta else ts_code_to_ak_prefix(index_code)
        ak = _ak()
        fn = getattr(ak, "index_zh_daily_sina", None) or getattr(ak, "stock_zh_index_daily")
        raw = fn(symbol=ak_symbol)
        if raw is None or raw.empty:
            return empty_frame(INDEX_BAR_COLUMNS)
        df = pd.DataFrame()
        df["trade_date"] = pd.to_datetime(raw["date"]).dt.date
        df["index_code"] = index_code
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = _to_float(raw[col])
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["pre_close"] = df["close"].shift(1)
        df["amount"] = np.nan
        df["pct_chg"] = (df["close"] / df["pre_close"] - 1.0) * 100.0
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        return df[INDEX_BAR_COLUMNS].reset_index(drop=True)

    # ------------------------------------------------------------------ capital flow
    def get_capital_flow(self, ts_code: str) -> pd.DataFrame:
        """Fetch ~120 recent real trading days of individual stock capital flow from Eastmoney push2his."""
        if not self._em_his_available():
            return empty_frame(CAPITAL_FLOW_COLUMNS)
        self._throttle()
        norm = normalize_ts_code(ts_code)
        symbol, exch = norm.split(".")
        market = {"SH": "sh", "SZ": "sz", "BJ": "bj"}[exch]
        ak = _ak()
        fn = getattr(ak, "stock_fund_flow_individual_hist_em", None) or getattr(ak, "stock_individual_fund_flow")
        try:
            raw = fn(symbol=symbol, market=market) if "symbol" in fn.__code__.co_varnames else fn(stock=symbol, market=market)
        except Exception as exc:  # noqa: BLE001
            if _is_em_rate_limit(exc):
                self._trip_em_his_cooldown()
                return empty_frame(CAPITAL_FLOW_COLUMNS)
            raise
        if raw is None or raw.empty:
            return empty_frame(CAPITAL_FLOW_COLUMNS)
        df = pd.DataFrame()
        df["trade_date"] = pd.to_datetime(raw["日期"]).dt.date
        df["ts_code"] = norm
        df["main_net_inflow"] = _to_float(raw["主力净流入-净额"])
        df["main_net_inflow_ratio"] = _to_float(raw["主力净流入-净占比"])
        df["super_large_net"] = _to_float(raw.get("超大单净流入-净额", pd.Series(np.nan, index=raw.index)))
        df["large_net"] = _to_float(raw.get("大单净流入-净额", pd.Series(np.nan, index=raw.index)))
        return df[CAPITAL_FLOW_COLUMNS]

    def get_bulk_capital_flow_rank_em(
        self,
        trade_date: date,
        indicator: str = "今日",
        ts_codes: set[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch ALL ~5,300 A-shares' real main capital flow in a single bulk Eastmoney ranking call."""
        if not self._em_available():
            return empty_frame(CAPITAL_FLOW_COLUMNS)
        ak = _ak()
        fn = getattr(ak, "stock_fund_flow_individual_rank_em", None)
        if fn is None:
            return empty_frame(CAPITAL_FLOW_COLUMNS)
        self._throttle()
        try:
            raw = fn(indicator=indicator)
        except Exception as exc:  # noqa: BLE001
            if _is_em_rate_limit(exc):
                self._trip_em_cooldown()
                return empty_frame(CAPITAL_FLOW_COLUMNS)
            raise
        if raw is None or raw.empty:
            return empty_frame(CAPITAL_FLOW_COLUMNS)

        prefix = indicator
        df = pd.DataFrame()
        df["ts_code"] = raw["代码"].astype(str).str.zfill(6).map(normalize_ts_code)
        df["trade_date"] = trade_date
        df["main_net_inflow"] = _to_float(_pick_col(raw, [f"{prefix}主力净流入-净额", "主力净流入-净额"]))
        df["main_net_inflow_ratio"] = _to_float(_pick_col(raw, [f"{prefix}主力净流入-净占比", "主力净流入-净占比"]))
        df["super_large_net"] = _to_float(_pick_col(raw, [f"{prefix}超大单净流入-净额", "超大单净流入-净额"]))
        df["large_net"] = _to_float(_pick_col(raw, [f"{prefix}大单净流入-净额", "大单净流入-净额"]))
        if ts_codes is not None:
            df = df[df["ts_code"].isin(ts_codes)]
        df = df.dropna(subset=["main_net_inflow_ratio"]).drop_duplicates(["ts_code", "trade_date"], keep="first")
        return df[CAPITAL_FLOW_COLUMNS].reset_index(drop=True)

    def get_bulk_capital_flow_ths(
        self,
        trade_date: date,
        ts_codes: set[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch ALL ~5,200 A-shares' real instant capital flow from Tonghuashun (independent of Eastmoney push2)."""
        ak = _ak()
        fn = getattr(ak, "stock_fund_flow_individual_ths", None)
        if fn is None:
            return empty_frame(CAPITAL_FLOW_COLUMNS)
        self._throttle()
        raw = retry_call(retries=2)(fn)(symbol="即时")
        if raw is None or raw.empty:
            return empty_frame(CAPITAL_FLOW_COLUMNS)

        df = pd.DataFrame()
        df["ts_code"] = raw["股票代码"].astype(str).str.zfill(6).map(normalize_ts_code)
        df["trade_date"] = trade_date
        net_in = raw["净额"].map(parse_cn_amount).astype(float)
        amt = raw["成交额"].map(parse_cn_amount).astype(float)
        ratio = np.where((amt > 1e-6) & (~np.isnan(net_in)), (net_in / amt) * 100.0, np.nan)
        ratio = np.clip(ratio, -100.0, 100.0)
        df["main_net_inflow"] = net_in
        df["main_net_inflow_ratio"] = ratio
        df["super_large_net"] = np.nan
        df["large_net"] = np.nan
        if ts_codes is not None:
            df = df[df["ts_code"].isin(ts_codes)]
        df = df.dropna(subset=["main_net_inflow_ratio"]).drop_duplicates(["ts_code", "trade_date"], keep="first")
        return df[CAPITAL_FLOW_COLUMNS].reset_index(drop=True)

    def get_bulk_capital_flow_today(
        self,
        trade_date: date,
        ts_codes: set[str] | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Real full-market daily capital flow: Eastmoney bulk rank (primary) -> Tonghuashun bulk instant (backup)."""
        try:
            df = self.get_bulk_capital_flow_rank_em(trade_date=trade_date, indicator="今日", ts_codes=ts_codes)
            if not df.empty:
                return df, "akshare_em_rank"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Eastmoney bulk capital flow rank failed, falling back to THS: %s", exc)
        try:
            df = self.get_bulk_capital_flow_ths(trade_date=trade_date, ts_codes=ts_codes)
            if not df.empty:
                return df, "akshare_ths"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tonghuashun bulk capital flow failed: %s", exc)
        return empty_frame(CAPITAL_FLOW_COLUMNS), ""

    def get_daily_bars_with_hfq(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        """Fetch UNADJUSTED bars + exact exchange ex-right pre_close + true multiplicative total-return
        adj_factor in a SINGLE keep-alive Tencent CDN request (with corporate_actions attached in attrs)."""
        try:
            raw = self._fetch_tx_kline_fast(ts_code, start, end, adjust="")
            if not raw.empty:
                if "adj_factor" not in raw.columns:
                    raw["adj_factor"] = 1.0
                return raw
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fast TX kline %s failed, falling back to ak.stock_zh_a_hist_tx: %s", ts_code, exc)

        raw = self._bars_tx(ts_code, start, end)
        if raw.empty:
            return raw
        self._throttle()
        try:
            hfq = retry_call(retries=2)(_ak().stock_zh_a_hist_tx)(
                symbol=ts_code_to_ak_prefix(ts_code),
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="hfq",
            )
            if hfq is not None and not hfq.empty:
                hfq_map = dict(zip(pd.to_datetime(hfq["date"]).dt.date, _to_float(hfq["close"])))
                hfq_c = raw["trade_date"].map(hfq_map).astype(float)
                adj = (hfq_c / raw["close"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                raw["adj_factor"] = adj.ffill().bfill().fillna(1.0)
            else:
                raw["adj_factor"] = 1.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("AKShare TX hfq %s failed (using 1.0): %s", ts_code, exc)
            raw["adj_factor"] = 1.0
        return raw

    def get_bulk_financials_yjbb(self, report_dates: list[str], ts_codes: set[str] | None = None) -> pd.DataFrame:
        """Fetch quarterly statements across ALL A-shares in bulk via Eastmoney 业绩报表 (includes REAL 最新公告日期)."""
        ak = _ak()
        frames: list[pd.DataFrame] = []
        for rdate in report_dates:
            self._throttle()
            try:
                raw = retry_call(retries=2)(ak.stock_yjbb_em)(date=rdate)
            except Exception as exc:  # noqa: BLE001
                logger.warning("stock_yjbb_em %s failed: %s", rdate, exc)
                continue
            if raw is None or raw.empty:
                continue
            df = pd.DataFrame()
            df["ts_code"] = raw["股票代码"].astype(str).str.zfill(6).map(normalize_ts_code)
            if ts_codes is not None:
                df = df[df["ts_code"].isin(ts_codes)].copy()
                raw = raw.loc[df.index]
            if df.empty:
                continue
            df["ann_date"] = pd.to_datetime(raw["最新公告日期"], errors="coerce").dt.date
            df["end_date"] = pd.to_datetime(rdate).date()
            df["net_profit_cum"] = _to_float(raw["净利润-净利润"])
            df["eps_cum"] = _to_float(raw["每股收益"])
            df["bps"] = _to_float(raw["每股净资产"])
            df["roe_avg_cum"] = _to_float(raw["净资产收益率"]) / 100.0
            df["revenue_cum"] = _to_float(raw["营业总收入-营业总收入"])
            df["np_yoy_cum"] = _to_float(raw["净利润-同比增长"]) / 100.0
            df["revenue_yoy_cum"] = _to_float(raw["营业总收入-同比增长"]) / 100.0
            df["industry_em"] = raw["所处行业"].astype(str).replace("nan", "")
            # Implied total shares from cumulative net profit / EPS when valid
            eps_safe = df["eps_cum"].where(df["eps_cum"].abs() > 1e-4)
            ts_share = (df["net_profit_cum"] / eps_safe).where(lambda s: s > 1e6)
            df["total_share"] = ts_share
            df = df.dropna(subset=["ann_date", "end_date"])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ industry
    def get_industry_map(self) -> pd.DataFrame:
        """Shenwan level-1 constituents via index_first_info_sw + index_component_sw (timeout-protected)."""
        import concurrent.futures

        ak = _ak()
        self._throttle()
        first_fn = getattr(ak, "index_first_info_sw", None) or getattr(ak, "sw_index_first_info")
        first = retry_call(retries=2)(first_fn)()
        if first is None or first.empty:
            return empty_frame(INDUSTRY_COLUMNS)

        def _fetch_one(code: str, name: str) -> pd.DataFrame | None:
            comp = ak.index_component_sw(symbol=code)
            if comp is None or comp.empty:
                return None
            part = pd.DataFrame()
            part["ts_code"] = comp["证券代码"].astype(str).map(normalize_ts_code)
            part["sw_l1_code"] = code
            part["sw_l1_name"] = name
            return part

        rows: list[pd.DataFrame] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futs = {
                pool.submit(_fetch_one, str(row["行业代码"]).split(".")[0], str(row["行业名称"])): str(row["行业名称"])
                for _, row in first.iterrows()
            }
            for fut, name in futs.items():
                try:
                    part = fut.result(timeout=15.0)
                    if part is not None:
                        rows.append(part)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SW L1 %s fetch failed/timed out: %s", name, exc)
        if not rows:
            return empty_frame(INDUSTRY_COLUMNS)
        out = pd.concat(rows, ignore_index=True).drop_duplicates("ts_code", keep="first")
        return out[INDUSTRY_COLUMNS]

    # ------------------------------------------------------------------ realtime
    def _normalize_spot_df(self, raw: pd.DataFrame, vol_multiplier: float = 100.0) -> pd.DataFrame:
        if raw is None or raw.empty:
            return empty_frame(SPOT_COLUMNS)
        df = pd.DataFrame()
        df["ts_code"] = _pick_col(raw, ["代码", "symbol", "code"]).astype(str).map(normalize_ts_code)
        df["name"] = _pick_col(raw, ["名称", "name"]).astype(str)
        df["last_price"] = _to_float(_pick_col(raw, ["最新价", "trade", "最新", "close"]))
        df["pre_close"] = _to_float(_pick_col(raw, ["昨收", "settlement", "pre_close"]))
        df["open"] = _to_float(_pick_col(raw, ["今开", "open"]))
        df["high"] = _to_float(_pick_col(raw, ["最高", "high"]))
        df["low"] = _to_float(_pick_col(raw, ["最低", "low"]))
        df["volume"] = _to_float(_pick_col(raw, ["成交量", "volume"])) * vol_multiplier
        df["amount"] = _to_float(_pick_col(raw, ["成交额", "amount"]))
        bid1 = _to_float(_pick_col(raw, ["买一", "buy"]))
        ask1 = _to_float(_pick_col(raw, ["卖一", "sell"]))
        df["bid1"] = bid1.where(bid1.notna() & (bid1 > 0), df["last_price"])
        df["ask1"] = ask1.where(ask1.notna() & (ask1 > 0), df["last_price"])
        df["pct_chg"] = _to_float(_pick_col(raw, ["涨跌幅", "changepercent", "pct_chg"]))
        df["turnover_rate"] = _to_float(_pick_col(raw, ["换手率", "turnoverratio", "turnover"]))
        df["pe_ttm"] = _to_float(_pick_col(raw, ["市盈率-动态", "市盈率", "per", "pe_ttm"]))
        df["pb"] = _to_float(_pick_col(raw, ["市净率", "pb"]))
        df["total_mv"] = _to_float(_pick_col(raw, ["总市值", "mktcap", "total_mv"]))
        df["circ_mv"] = _to_float(_pick_col(raw, ["流通市值", "nmc", "circ_mv"]))
        return df[SPOT_COLUMNS]

    def _spot_tx_batch(self, ts_codes: list[str], max_workers: int = 6) -> pd.DataFrame:
        """Targeted or full-market real-time L1 quote (including bid1/ask1) via Tencent quote endpoint."""
        import concurrent.futures
        import httpx

        prefixes = [ts_code_to_ak_prefix(c) for c in ts_codes]
        chunks = [prefixes[i : i + 60] for i in range(0, len(prefixes), 60)]

        def _fetch_chunk(chunk: list[str]) -> list[dict]:
            out: list[dict] = []
            with httpx.Client(timeout=12.0) as client:
                resp = client.get(f"https://qt.gtimg.cn/q={','.join(chunk)}")
                resp.raise_for_status()
                text = resp.content.decode("gbk", errors="ignore")
                for line in text.strip().splitlines():
                    if '="' not in line:
                        continue
                    lhs, rhs = line.split('="', 1)
                    code_pref = lhs.split("v_")[-1].strip()
                    parts = rhs.rstrip('";').split("~")
                    if len(parts) < 46:
                        continue

                    def _f(idx: int) -> float:
                        try:
                            return float(parts[idx]) if parts[idx] not in ("", "-") else np.nan
                        except (ValueError, IndexError):
                            return np.nan

                    last_p = _f(3)
                    bid1 = _f(9)
                    ask1 = _f(19)
                    out.append({
                        "ts_code": normalize_ts_code(code_pref),
                        "name": parts[1],
                        "last_price": last_p,
                        "pre_close": _f(4),
                        "open": _f(5),
                        "high": _f(33),
                        "low": _f(34),
                        "volume": _f(6) * 100.0,         # 手 -> 股
                        "amount": _f(37) * 10000.0,      # 万元 -> 元
                        "bid1": bid1 if pd.notna(bid1) and bid1 > 0 else last_p,
                        "ask1": ask1 if pd.notna(ask1) and ask1 > 0 else last_p,
                        "pct_chg": _f(32),
                        "turnover_rate": _f(38),
                        "pe_ttm": _f(39),
                        "pb": _f(46) if len(parts) > 46 else np.nan,
                        "total_mv": _f(45) * 1e8,        # 亿元 -> 元
                        "circ_mv": _f(44) * 1e8,
                    })
            return out

        rows: list[dict] = []
        if len(chunks) <= 2:
            for ch in chunks:
                rows.extend(_fetch_chunk(ch))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                for res in pool.map(_fetch_chunk, chunks):
                    rows.extend(res)
        return pd.DataFrame(rows, columns=SPOT_COLUMNS) if rows else empty_frame(SPOT_COLUMNS)

    def get_realtime_spot(self, ts_codes: list[str] | None = None) -> pd.DataFrame:
        # Tencent batch quote returns real L1 bid1/ask1 rapidly for both Top-100 and full-market scans
        if ts_codes:
            try:
                df = self._spot_tx_batch(ts_codes)
                if not df.empty:
                    return df.reset_index(drop=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tencent batch quote failed: %s", exc)

        ak = _ak()
        df = empty_frame(SPOT_COLUMNS)
        if self._em_available():
            self._throttle()
            try:
                raw = ak.stock_zh_a_spot_em()
                df = self._normalize_spot_df(raw, vol_multiplier=100.0)
            except Exception as exc:  # noqa: BLE001
                if _is_em_rate_limit(exc):
                    self._trip_em_cooldown()
                else:
                    logger.warning("AKShare EM spot failed: %s", exc)
        if not df.empty and ts_codes:
            wanted = {normalize_ts_code(c) for c in ts_codes}
            df = df[df["ts_code"].isin(wanted)]
        return df.reset_index(drop=True)

    def get_bid_ask(self, ts_code: str) -> dict[str, float]:
        """Return real level-1 bid1/ask1 for a single stock (used for top-100 intraday matching)."""
        if not self._em_available():
            return {}
        self._throttle()
        try:
            raw = _ak().stock_bid_ask_em(symbol=ts_code_to_symbol(ts_code))
        except Exception as exc:  # noqa: BLE001
            if _is_em_rate_limit(exc):
                self._trip_em_cooldown()
            return {}
        kv = dict(zip(raw["item"].astype(str), _to_float(raw["value"])))
        return {
            "bid1": float(kv.get("buy_1", np.nan)),
            "ask1": float(kv.get("sell_1", np.nan)),
            "last_price": float(kv.get("最新", np.nan)),
            "pre_close": float(kv.get("昨收", np.nan)),
        }

    def health_check(self) -> dict:
        try:
            cal = self.get_trade_calendar(date(2024, 1, 1), date(2024, 1, 31))
            return {"provider": self.name, "ok": len(cal) > 0, "em_push2_ready": self._em_available()}
        except Exception as exc:  # noqa: BLE001
            return {"provider": self.name, "ok": False, "error": str(exc)}
