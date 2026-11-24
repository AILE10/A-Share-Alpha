"""Real data synchronization service: incremental daily sync, full-market batch persistence,
intraday top-100 snapshots, PIT financial derivation and capital-flow degradation tagging.

Primary: AKShare (SSE/SZSE official listing, SW L1 industry, Eastmoney bulk 业绩报表 PIT
financials, Tencent CDN daily OHLCV + hfq & realtime L1 bid1/ask1, Sina indices & calendar).
Backup:  Baostock (fast-failing probe; enriches historical peTTM/pbMRQ/isST when reachable).
Reserved: Tushare Pro (activated when TUSHARE_TOKEN is configured).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from app.core.calendar import today_cn
from app.core.constants import (
    BENCHMARK_CSI1000,
    BENCHMARK_CSI300,
    calc_limit_prices,
    detect_board,
    normalize_ts_code,
)
from app.core.exceptions import DataProviderError
from app.core.logging import get_logger
from app.data.akshare_provider import AKShareProvider
from app.data.base import DataProvider
from app.data.baostock_provider import BaostockProvider
from app.data.tushare_provider import TushareProvider
from app.db.duckdb_client import DuckDBClient, get_duckdb

logger = get_logger(__name__)

ProgressFn = Callable[[str, float], None]


@dataclass
class SyncReport:
    stocks_requested: int = 0
    stocks_synced: int = 0
    bar_rows: int = 0
    index_rows: int = 0
    financial_rows: int = 0
    capital_flow_rows: int = 0
    capital_flow_degraded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def derive_pit_financials(fin: pd.DataFrame, source: str = "akshare_yjbb") -> pd.DataFrame:
    """Derive single-quarter net profit, single-quarter YoY, ROE_TTM, EPS_TTM and BPS.

    Every derived row strictly retains the REAL announcement date (ann_date) of its filing,
    ensuring downstream ASOF JOIN on trade_date > ann_date never leaks future information.
    """
    if fin is None or fin.empty:
        return pd.DataFrame()
    df = fin.copy()
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["ann_date"] = pd.to_datetime(df["ann_date"])
    # Enforce real filing ordering: keep earliest announcement per reporting period
    df = df.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(["ts_code", "end_date"], keep="first")
    df["year"] = df["end_date"].dt.year
    df["quarter"] = df["end_date"].dt.quarter

    out_frames = []
    for ts_code, g in df.groupby("ts_code", sort=False):
        g = g.set_index(["year", "quarter"]).sort_index()
        np_q: dict[tuple[int, int], float] = {}
        for (y, q), row in g.iterrows():
            cum = row.get("net_profit_cum", np.nan)
            if q == 1:
                np_q[(y, q)] = cum
            else:
                prev = g["net_profit_cum"].get((y, q - 1), np.nan)
                np_q[(y, q)] = cum - prev if pd.notna(prev) and pd.notna(cum) else np.nan
        g = g.copy()
        g["net_profit_q"] = [np_q[idx] for idx in g.index]

        yoy, roe_ttm, eps_ttm = [], [], []
        for (y, q), row in g.iterrows():
            last = np_q.get((y - 1, q), np.nan)
            cur = row["net_profit_q"]
            if pd.notna(cur) and pd.notna(last) and abs(last) > 1e-6:
                yoy.append((cur - last) / abs(last))
            elif pd.notna(row.get("np_yoy_cum", np.nan)):
                yoy.append(float(row["np_yoy_cum"]))
            else:
                yoy.append(np.nan)

            # Trailing 4 single quarters -> NP_TTM
            keys = []
            yy, qq = y, q
            for _ in range(4):
                keys.append((yy, qq))
                qq -= 1
                if qq == 0:
                    yy, qq = yy - 1, 4
            vals = [np_q.get(k, np.nan) for k in keys]
            if all(pd.notna(v) for v in vals):
                np_ttm = float(np.sum(vals))
            else:
                # Annualize cumulative net profit when prior 4 quarters not yet fully available
                cum = row.get("net_profit_cum", np.nan)
                np_ttm = float(cum) * (4.0 / q) if pd.notna(cum) else np.nan

            roe_cum = row.get("roe_avg_cum", np.nan)
            cum = row.get("net_profit_cum", np.nan)
            equity = cum / roe_cum if (pd.notna(roe_cum) and abs(roe_cum) > 1e-9 and pd.notna(cum)) else np.nan
            if pd.notna(np_ttm) and pd.notna(equity) and equity > 0:
                roe_ttm.append(np_ttm / equity)
            elif pd.notna(roe_cum):
                roe_ttm.append(float(roe_cum) * (4.0 / q))
            else:
                roe_ttm.append(np.nan)

            tshare = row.get("total_share", np.nan)
            eps_c = row.get("eps_cum", np.nan)
            if pd.notna(np_ttm) and pd.notna(tshare) and tshare > 1e6:
                eps_ttm.append(np_ttm / tshare)
            elif pd.notna(eps_c):
                eps_ttm.append(float(eps_c) * (4.0 / q))
            else:
                eps_ttm.append(np.nan)

        g["np_yoy_q"] = yoy
        g["roe_ttm"] = roe_ttm
        g["eps_ttm"] = eps_ttm
        if "bps" not in g.columns:
            g["bps"] = np.nan
        if "revenue_yoy_cum" in g.columns:
            g["revenue_yoy_q"] = g["revenue_yoy_cum"]
        elif "revenue_yoy_q" not in g.columns:
            g["revenue_yoy_q"] = np.nan
        g = g.reset_index()
        g["ts_code"] = ts_code
        out_frames.append(g)

    out = pd.concat(out_frames, ignore_index=True)
    out["ann_date"] = out["ann_date"].dt.date
    out["end_date"] = out["end_date"].dt.date
    out["source"] = source
    return out[
        [
            "ts_code", "ann_date", "end_date", "net_profit_cum", "net_profit_q",
            "total_share", "bps", "eps_ttm", "roe_ttm", "np_yoy_q", "revenue_yoy_q", "source",
        ]
    ]


def attach_adj_factor(bars: pd.DataFrame, adj_events: pd.DataFrame) -> pd.Series:
    """Forward-fill backward-adjustment factor events onto trading days (as-of join)."""
    if adj_events is None or adj_events.empty:
        return pd.Series(1.0, index=bars.index)
    left = bars[["trade_date"]].copy()
    left["_d"] = pd.to_datetime(left["trade_date"]).astype("datetime64[ns]")
    left["_order"] = np.arange(len(left))
    right = adj_events[["trade_date", "adj_factor"]].copy()
    right["_d"] = pd.to_datetime(right["trade_date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        left.sort_values("_d"),
        right[["_d", "adj_factor"]].sort_values("_d"),
        on="_d",
        direction="backward",
    )
    merged = merged.sort_values("_order")
    vals = merged["adj_factor"].to_numpy()
    first_valid = right["adj_factor"].iloc[0] if len(right) else 1.0
    vals = np.where(np.isnan(vals), first_valid, vals)
    return pd.Series(vals, index=bars.index)


def _quarter_end_dates(start_year: int, end_year: int, up_to: date) -> list[str]:
    out = []
    for y in range(start_year, end_year + 1):
        for mmdd in ("0331", "0630", "0930", "1231"):
            d = date(y, int(mmdd[:2]), int(mmdd[2:]))
            if d <= up_to:
                out.append(f"{y}{mmdd}")
    return out


class DataSyncService:
    """Orchestrates real data acquisition across AKShare (primary) / Baostock (backup) / Tushare (reserved)."""

    def __init__(
        self,
        db: DuckDBClient | None = None,
        primary: DataProvider | None = None,
        backup: BaostockProvider | None = None,
        tushare: TushareProvider | None = None,
    ) -> None:
        self.db = db or get_duckdb()
        self.primary = primary or AKShareProvider()
        self.backup = backup or BaostockProvider()
        self.tushare = tushare or TushareProvider()

    # ------------------------------------------------------------------ calendar & basics
    def sync_calendar(self, start: date, end: date) -> int:
        dates: list[date] = []
        for provider in (self.primary, self.backup):
            try:
                dates = provider.get_trade_calendar(start, end)
                if dates:
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning("calendar via %s failed: %s", provider.name, exc)
        if not dates:
            raise DataProviderError("All providers failed to return trade calendar")
        cal = pd.DataFrame({"trade_date": dates})
        cal["is_open"] = True
        cal["prev_trade_date"] = cal["trade_date"].shift(1)
        cal["next_trade_date"] = cal["trade_date"].shift(-1)
        return self.db.upsert_df("trade_calendar", cal)

    def sync_stock_basic(self, with_industry: bool = True) -> int:
        base = pd.DataFrame()
        try:
            base = self.primary.get_stock_list()
        except Exception as exc:  # noqa: BLE001
            logger.warning("AKShare stock list failed: %s", exc)
        if base.empty:
            try:
                base = self.backup.get_stock_list()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Baostock stock list failed: %s", exc)
        if base.empty:
            raise DataProviderError("All providers failed to return stock list")

        base = base.copy()
        base["is_st_latest"] = base["name"].str.upper().str.contains("ST", na=False)
        base["sw_l1_code"] = None
        base["sw_l1_name"] = None
        if with_industry:
            ind = self.get_industry_map()
            if not ind.empty:
                base = base.drop(columns=["sw_l1_code", "sw_l1_name"]).merge(ind, on="ts_code", how="left")
        base["sw_l1_name"] = base["sw_l1_name"].fillna("未分类")
        base["sw_l1_code"] = base["sw_l1_code"].fillna("000000")
        base["updated_at"] = pd.Timestamp.now()
        return self.db.upsert_df("stock_basic", base)

    def get_industry_map(self) -> pd.DataFrame:
        try:
            ind = self.primary.get_industry_map()
            if ind is not None and len(ind) > 500:
                logger.info("Shenwan L1 industry map loaded via AKShare: %d stocks", len(ind))
                return ind
            logger.warning("AKShare SW L1 map incomplete (%d rows), trying Baostock", 0 if ind is None else len(ind))
        except Exception as exc:  # noqa: BLE001
            logger.warning("AKShare SW industry failed: %s", exc)
        try:
            ind = self.backup.get_industry_map()
            if ind is not None and not ind.empty:
                logger.info("Industry map loaded via Baostock fallback: %d", len(ind))
                return ind
        except Exception as exc:  # noqa: BLE001
            logger.warning("Baostock industry failed: %s", exc)
        return pd.DataFrame()

    # ------------------------------------------------------------------ daily bars
    def _fetch_bars_primary_and_backup(
        self, ts_code: str, start: date, end: date, is_st_hint: bool = False
    ) -> tuple[pd.DataFrame, str]:
        """Primary: AKShare unadjusted + hfq bars; enrich/fallback with Baostock if reachable."""
        ak_bars = pd.DataFrame()
        if hasattr(self.primary, "get_daily_bars_with_hfq"):
            try:
                ak_bars = self.primary.get_daily_bars_with_hfq(ts_code, start, end)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s AKShare TX+hfq failed: %s", ts_code, exc)
        if ak_bars.empty:
            try:
                ak_bars = self.primary.get_daily_bars(ts_code, start, end)
                if not ak_bars.empty and "adj_factor" not in ak_bars.columns:
                    ak_bars["adj_factor"] = 1.0
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s AKShare bars failed: %s", ts_code, exc)

        bs_val = pd.DataFrame()
        if self.backup._probe_server():
            try:
                bs_val = self.backup.get_daily_valuation(ts_code, start, end)
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s Baostock valuation skipped: %s", ts_code, exc)

        if ak_bars.empty and bs_val.empty and self.tushare.enabled:
            df = self.tushare.get_daily_bars(ts_code, start, end)
            df["adj_factor"] = 1.0
            df["pe_ttm"] = np.nan
            df["pb"] = np.nan
            df["is_st"] = is_st_hint
            df["is_suspended"] = df["volume"].fillna(0) <= 0
            return df, "tushare"
        if ak_bars.empty and bs_val.empty:
            raise DataProviderError(f"No real daily bars available for {ts_code}")

        if not ak_bars.empty:
            merged = ak_bars.copy()
            merged["pe_ttm"] = np.nan
            merged["pb"] = np.nan
            merged["is_st"] = is_st_hint
            merged["is_suspended"] = merged["volume"].fillna(0) <= 0
            source = "akshare"
            if not bs_val.empty:
                m_idx = merged.set_index("trade_date")
                b_idx = bs_val.set_index("trade_date")
                common = m_idx.index.intersection(b_idx.index)
                for col in ("pre_close", "pe_ttm", "pb", "is_st", "is_suspended"):
                    m_idx.loc[common, col] = b_idx.loc[common, col]
                merged = m_idx.reset_index()
                source = "akshare+baostock"
        else:
            merged = bs_val.copy()
            merged["amplitude"] = (merged["high"] - merged["low"]) / merged["pre_close"] * 100.0
            try:
                adj = self.backup.get_adj_factor(ts_code, start, end)
                merged["adj_factor"] = attach_adj_factor(merged, adj)
            except Exception:  # noqa: BLE001
                merged["adj_factor"] = 1.0
            source = "baostock"

        merged["ts_code"] = normalize_ts_code(ts_code)
        return merged.sort_values("trade_date").reset_index(drop=True), source

    def sync_daily_bars(
        self,
        ts_codes: list[str],
        start: date,
        end: date,
        progress: ProgressFn | None = None,
        report: SyncReport | None = None,
    ) -> SyncReport:
        report = report or SyncReport()
        report.stocks_requested += len(ts_codes)

        fin = self.db.query_df(
            "SELECT ts_code, ann_date, end_date, total_share, bps, eps_ttm FROM financial_pit ORDER BY ts_code, ann_date, end_date"
        )
        basic_st = self.db.query_df("SELECT ts_code, is_st_latest FROM stock_basic")
        st_map = dict(zip(basic_st["ts_code"], basic_st["is_st_latest"].astype(bool))) if not basic_st.empty else {}

        for i, code in enumerate(ts_codes):
            code = normalize_ts_code(code)
            try:
                bars, source = self._fetch_bars_primary_and_backup(code, start, end, is_st_hint=st_map.get(code, False))
                # Fill first-day pre_close if missing
                if bars["pre_close"].isna().any():
                    bars["pre_close"] = bars["pre_close"].fillna(bars["close"].shift(1)).fillna(bars["open"])
                limits = [calc_limit_prices(pc, code, bool(st)) for pc, st in zip(bars["pre_close"], bars["is_st"])]
                bars["limit_up"] = [x[0] for x in limits]
                bars["limit_down"] = [x[1] for x in limits]

                # Real float market cap from real daily turnover rate
                turn = bars["turnover_rate"].replace(0, np.nan)
                circ_shares = (bars["volume"] / (turn / 100.0)).replace([np.inf, -np.inf], np.nan)
                circ_shares = circ_shares.ffill().bfill()
                bars["circ_mv"] = bars["close"] * circ_shares

                # Strict PIT ASOF merge (trade_date > ann_date -> allow_exact_matches=False)
                # for total_share, bps & eps_ttm -> zero look-ahead daily PE_TTM, PB & total_mv!
                f = fin[fin["ts_code"] == code].copy()
                if not f.empty:
                    f["_d"] = pd.to_datetime(f["ann_date"]).astype("datetime64[ns]")
                    f = f.sort_values("_d").drop_duplicates("_d", keep="last")
                    left = bars[["trade_date"]].copy()
                    left["_d"] = pd.to_datetime(left["trade_date"]).astype("datetime64[ns]")
                    left["_o"] = np.arange(len(left))
                    mm = pd.merge_asof(
                        left.sort_values("_d"),
                        f[["_d", "total_share", "bps", "eps_ttm"]],
                        on="_d",
                        direction="backward",
                        allow_exact_matches=False,
                    ).sort_values("_o")
                    ts_share = mm["total_share"].to_numpy(dtype=float)
                    bps = mm["bps"].to_numpy(dtype=float)
                    eps_ttm = mm["eps_ttm"].to_numpy(dtype=float)
                    close_arr = bars["close"].to_numpy(dtype=float)

                    total_mv = close_arr * ts_share
                    bars["total_mv"] = np.where(np.isnan(total_mv), bars["circ_mv"], total_mv)

                    calc_pb = np.where((~np.isnan(bps)) & (bps > 1e-6), close_arr / bps, np.nan)
                    calc_pe = np.where((~np.isnan(eps_ttm)) & (np.abs(eps_ttm) > 1e-6), close_arr / eps_ttm, np.nan)
                    bars["pb"] = bars["pb"].where(bars["pb"].notna(), calc_pb)
                    bars["pe_ttm"] = bars["pe_ttm"].where(bars["pe_ttm"].notna(), calc_pe)
                else:
                    bars["total_mv"] = bars["circ_mv"]

                bars["source"] = source
                report.bar_rows += self.db.upsert_df("daily_bars", bars)
                report.stocks_synced += 1
                report.sources[source] = report.sources.get(source, 0) + 1
            except Exception as exc:  # noqa: BLE001
                report.failed[code] = str(exc)
                logger.error("sync bars %s failed: %s", code, exc)
            if progress:
                progress(f"daily_bars {code}", (i + 1) / max(1, len(ts_codes)))
        return report

    # ------------------------------------------------------------------ index
    def sync_index(self, start: date, end: date, codes: tuple[str, ...] = (BENCHMARK_CSI1000, BENCHMARK_CSI300)) -> int:
        total = 0
        for code in codes:
            df = pd.DataFrame()
            for provider in (self.primary, self.backup):
                try:
                    df = provider.get_index_daily(code, start, end)
                    if not df.empty:
                        df["source"] = provider.name
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("index %s via %s failed: %s", code, provider.name, exc)
            if df.empty:
                raise DataProviderError(f"Failed to fetch real benchmark index {code}")
            total += self.db.upsert_df("index_daily", df)
        return total

    # ------------------------------------------------------------------ financials (PIT)
    def sync_financials(
        self,
        ts_codes: list[str] | None,
        start_year: int,
        end_year: int,
        progress: ProgressFn | None = None,
    ) -> int:
        """Fetch PIT quarterly financials with real announcement dates.

        When `ts_codes is None`, persists ALL ~5,900 A-shares from Eastmoney bulk 业绩报表 in one pass.
        Primary: AKShare bulk Eastmoney 业绩报表 (stock_yjbb_em) — 1 call per quarter for ALL stocks.
        Backup:  Baostock per-stock query_profit_data.
        """
        wanted = {normalize_ts_code(c) for c in ts_codes} if ts_codes else None
        qdates = _quarter_end_dates(start_year, end_year, today_cn())
        raw_all = pd.DataFrame()
        source = "akshare_yjbb"

        if hasattr(self.primary, "get_bulk_financials_yjbb"):
            try:
                frames = []
                for idx, qd in enumerate(qdates):
                    part = self.primary.get_bulk_financials_yjbb([qd], ts_codes=wanted)  # type: ignore[attr-defined]
                    if not part.empty:
                        frames.append(part)
                    if progress:
                        progress(f"financials_yjbb {qd}", (idx + 1) / max(1, len(qdates)))
                if frames:
                    raw_all = pd.concat(frames, ignore_index=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("AKShare bulk yjbb failed, trying Baostock: %s", exc)

        if raw_all.empty and ts_codes and self.backup._probe_server():
            source = "baostock"
            frames = []
            for i, code in enumerate(ts_codes):
                try:
                    raw = self.backup.get_financials(code, start_year, end_year)
                    if not raw.empty:
                        frames.append(raw)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("baostock financials %s failed: %s", code, exc)
                if progress:
                    progress(f"financials_bs {code}", (i + 1) / max(1, len(ts_codes)))
            if frames:
                raw_all = pd.concat(frames, ignore_index=True)

        if raw_all.empty:
            return 0
        pit = derive_pit_financials(raw_all, source=source)
        return self.db.upsert_df("financial_pit", pit)

    # ------------------------------------------------------------------ capital flow (real multi-source, degradable)
    def _latest_open_trade_date(self) -> date:
        today = today_cn()
        d = self.db.query_scalar(
            "SELECT MAX(trade_date) FROM trade_calendar WHERE is_open AND trade_date <= ?", [today]
        )
        if d is not None:
            return pd.to_datetime(d).date()
        last_bar = self.db.query_scalar("SELECT MAX(trade_date) FROM daily_bars")
        return pd.to_datetime(last_bar).date() if last_bar is not None else today

    def sync_capital_flow_today(
        self,
        ts_codes: list[str] | None = None,
        trade_date: date | None = None,
        report: SyncReport | None = None,
    ) -> SyncReport:
        """Fast single-bulk real capital flow sync for ALL A-shares (Eastmoney rank -> Tonghuashun fallback)."""
        report = report or SyncReport()
        if not hasattr(self.primary, "get_bulk_capital_flow_today"):
            return report
        tdate = trade_date or self._latest_open_trade_date()
        wanted = {normalize_ts_code(c) for c in ts_codes} if ts_codes else None
        try:
            df, src = self.primary.get_bulk_capital_flow_today(trade_date=tdate, ts_codes=wanted)  # type: ignore[attr-defined]
            if not df.empty:
                df = df.copy()
                df["status"] = "OK"
                df["source"] = src
                rows = self.db.upsert_df("capital_flow_daily", df)
                report.capital_flow_rows += rows
                report.sources[src] = report.sources.get(src, 0) + rows
                logger.info("Bulk real capital flow synced: %d stocks for %s via %s", rows, tdate, src)
                return report
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bulk today capital flow failed: %s", exc)
        if ts_codes:
            report.capital_flow_degraded.extend(normalize_ts_code(c) for c in ts_codes)
        return report

    def sync_capital_flow(
        self,
        ts_codes: list[str],
        report: SyncReport | None = None,
        workers: int = 6,
        resume: bool = True,
        min_rows_for_skip: int = 60,
        batch_size: int = 100,
        progress: ProgressFn | None = None,
        include_today_bulk: bool = True,
    ) -> SyncReport:
        """Checkpoint-resumable parallel 120-day real capital flow sync (Eastmoney push2his + bulk EM/THS fallback).

        Never fabricates data: if an interface is unavailable or rate-limited, missing rows remain marked
        and logged cleanly without mocking.
        """
        import concurrent.futures

        report = report or SyncReport()
        norm_codes = [normalize_ts_code(c) for c in ts_codes]
        if not norm_codes:
            return report

        pending = norm_codes
        skipped: list[str] = []
        if resume:
            existing = self.db.query_df(
                "SELECT ts_code, COUNT(*) AS cnt, MAX(trade_date) AS max_d "
                "FROM capital_flow_daily WHERE status = 'OK' GROUP BY ts_code"
            )
            last_bar = self.db.query_scalar("SELECT MAX(trade_date) FROM daily_bars")
            fresh_ref = pd.to_datetime(last_bar).date() if last_bar is not None else today_cn()
            fresh_cutoff = fresh_ref - timedelta(days=3)
            if not existing.empty:
                existing["max_d"] = pd.to_datetime(existing["max_d"]).dt.date
                done_set = set(
                    existing[(existing["cnt"] >= min_rows_for_skip) & (existing["max_d"] >= fresh_cutoff)]["ts_code"].tolist()
                )
                pending = [c for c in norm_codes if c not in done_set]
                skipped = [c for c in norm_codes if c in done_set]

        if pending:
            fast_ak = AKShareProvider(request_interval=0.05) if isinstance(self.primary, AKShareProvider) else self.primary
            buffer_frames: list[pd.DataFrame] = []
            done = 0
            total = len(norm_codes)

            def _fetch_one(code: str) -> tuple[str, pd.DataFrame | None, str]:
                try:
                    cf = fast_ak.get_capital_flow(code)
                    if cf is None or cf.empty:
                        return code, None, "empty capital flow"
                    cf = cf.copy()
                    cf["status"] = "OK"
                    cf["source"] = fast_ak.name
                    return code, cf, ""
                except Exception as exc:  # noqa: BLE001
                    return code, None, str(exc)

            def _flush() -> None:
                if not buffer_frames:
                    return
                combined = pd.concat(buffer_frames, ignore_index=True)
                report.capital_flow_rows += self.db.upsert_df("capital_flow_daily", combined)
                buffer_frames.clear()

            if workers <= 1 or not isinstance(fast_ak, AKShareProvider):
                for code in pending:
                    c, cf, err = _fetch_one(code)
                    done += 1
                    if cf is not None and not cf.empty:
                        buffer_frames.append(cf)
                    else:
                        report.capital_flow_degraded.append(c)
                        logger.debug("capital flow %s unavailable -> %s", c, err)
                    if len(buffer_frames) >= batch_size:
                        _flush()
                    if progress:
                        progress(f"capital_flow ({len(skipped) + done}/{total}) {c}", (len(skipped) + done) / max(1, total))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
                    futs = [pool.submit(_fetch_one, c) for c in pending]
                    for fut in concurrent.futures.as_completed(futs):
                        c, cf, err = fut.result()
                        done += 1
                        if cf is not None and not cf.empty:
                            buffer_frames.append(cf)
                        else:
                            report.capital_flow_degraded.append(c)
                            logger.debug("capital flow %s unavailable -> %s", c, err)
                        if len(buffer_frames) >= batch_size:
                            _flush()
                        if progress:
                            progress(f"capital_flow ({len(skipped) + done}/{total}) {c}", (len(skipped) + done) / max(1, total))
            _flush()
        elif progress:
            progress("capital_flow already up-to-date (resumed)", 1.0)

        if include_today_bulk and isinstance(self.primary, AKShareProvider):
            # Overlay latest trading day's real bulk capital flow (covers any degraded names via EM rank / THS)
            self.sync_capital_flow_today(norm_codes, report=report)

        if report.capital_flow_degraded:
            logger.warning(
                "Capital flow hist unavailable for %d/%d stocks (never mocked)",
                len(report.capital_flow_degraded),
                len(norm_codes),
            )
        return report

    # ------------------------------------------------------------------ sample universe
    def select_sample_universe(self, n: int = 60, min_list_years: float = 2.0) -> list[str]:
        """Deterministically pick real, liquid, mid/small-cap stocks across boards & SW L1 industries."""
        basic = self.db.query_df("SELECT ts_code, name, board, list_date, sw_l1_name FROM stock_basic")
        cutoff = today_cn() - timedelta(days=int(365 * min_list_years))
        basic["list_date"] = pd.to_datetime(basic["list_date"]).dt.date
        basic = basic[basic["list_date"].notna() & (basic["list_date"] <= cutoff)]
        basic = basic[~basic["name"].str.upper().str.contains("ST|退", na=False)]
        basic = basic[basic["board"].isin(["MAIN", "CHINEXT", "STAR"])]
        has_ind = basic[basic["sw_l1_name"].notna() & (basic["sw_l1_name"] != "未分类")]
        if len(has_ind) >= n * 2:
            basic = has_ind

        quotas = {"MAIN": int(n * 0.55), "CHINEXT": int(n * 0.25)}
        quotas["STAR"] = n - quotas["MAIN"] - quotas["CHINEXT"]

        # Stratified real candidate pool verified with live Tencent quotes for liquidity & market cap
        cand_codes: list[str] = []
        for board, quota in quotas.items():
            sub = basic[basic["board"] == board].sort_values("ts_code")
            if sub.empty:
                continue
            take = min(len(sub), max(quota * 8, 120))
            step = max(1, len(sub) // take)
            cand_codes.extend(sub.iloc[::step]["ts_code"].head(take).tolist())

        pool = basic.copy()
        try:
            spot_chunks = []
            for i in range(0, len(cand_codes), 120):
                chunk_df = self.primary.get_realtime_spot(cand_codes[i : i + 120])
                if not chunk_df.empty:
                    spot_chunks.append(chunk_df)
            if spot_chunks:
                spot = pd.concat(spot_chunks, ignore_index=True).drop_duplicates("ts_code")
                merged = basic.merge(spot[["ts_code", "amount", "total_mv", "last_price"]], on="ts_code", how="inner")
                merged = merged[(merged["amount"] > 1.2e8) & merged["total_mv"].notna() & (merged["total_mv"] > 3e9)]
                if len(merged) >= n:
                    lo, hi = merged["total_mv"].quantile([0.10, 0.90])
                    mid = merged[(merged["total_mv"] >= lo) & (merged["total_mv"] <= hi)]
                    pool = (mid if len(mid) >= n else merged).sort_values("amount", ascending=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Candidate liquidity probe failed, using stratified basic list: %s", exc)
            pool = basic.sort_values("ts_code")

        picked: list[str] = []
        ind_count: dict[str, int] = {}
        max_per_ind = max(4, n // 10)
        for board, quota in quotas.items():
            sub = pool[pool["board"] == board]
            if sub.empty:
                continue
            cnt = 0
            for _, row in sub.iterrows():
                ind = row.get("sw_l1_name") or "未分类"
                if ind_count.get(ind, 0) >= max_per_ind:
                    continue
                picked.append(row["ts_code"])
                ind_count[ind] = ind_count.get(ind, 0) + 1
                cnt += 1
                if cnt >= quota:
                    break
        if len(picked) < n:
            for code in pool["ts_code"]:
                if code not in picked:
                    picked.append(code)
                if len(picked) >= n:
                    break
        return picked[:n]

    # ------------------------------------------------------------------ full-market active universe (Plan A)
    def select_active_market_universe(
        self,
        min_amount: float = 40_000_000.0,
        min_list_days: int = 120,
        drop_bottom_mv_pct: float = 0.08,
        max_stocks: int | None = None,
        all_non_st: bool = False,
    ) -> tuple[list[str], dict]:
        """Scan ALL A-shares in stock_basic using fast parallel real-time quotes and select
        every active liquid non-ST stock eligible for full-market cross-sectional ranking.

        Because UniverseFilter excludes <50M daily turnover and bottom-10% micro-cap stocks,
        pre-filtering at >=40M turnover & above bottom-8% market cap preserves all strategy-eligible
        names across the entire A-share market while cutting free-API sync time by ~60%.
        """
        basic = self.db.query_df(
            "SELECT ts_code, name, board, list_date, sw_l1_name, is_st_latest FROM stock_basic"
        )
        if basic.empty:
            self.sync_stock_basic(with_industry=True)
            basic = self.db.query_df(
                "SELECT ts_code, name, board, list_date, sw_l1_name, is_st_latest FROM stock_basic"
            )

        total_listed = len(basic)
        basic["list_date"] = pd.to_datetime(basic["list_date"], errors="coerce").dt.date
        cutoff = today_cn() - timedelta(days=int(min_list_days * 1.45))  # ~120 trading days
        clean = basic[
            (~basic["name"].str.upper().str.contains("ST|退", na=False))
            & (~basic["is_st_latest"].fillna(False).astype(bool))
            & (basic["board"].isin(["MAIN", "CHINEXT", "STAR"]))
            & (basic["list_date"].isna() | (basic["list_date"] <= cutoff))
        ].copy()

        all_codes = clean["ts_code"].tolist()
        if all_non_st:
            picked = all_codes[:max_stocks] if max_stocks else all_codes
            return picked, {
                "mode": "all_non_st",
                "total_listed": total_listed,
                "non_st_mature": len(clean),
                "selected": len(picked),
            }

        # Parallel Tencent batch quote scan across ALL ~4,600 non-ST stocks (~6-10 seconds total)
        spot = self.primary.get_realtime_spot(all_codes)
        if spot.empty:
            raise DataProviderError("Full-market realtime liquidity scan returned empty DataFrame")

        merged = clean.merge(
            spot[["ts_code", "last_price", "amount", "total_mv", "turnover_rate"]],
            on="ts_code",
            how="inner",
        )
        active = merged[merged["last_price"].notna() & (merged["last_price"] > 0)].copy()

        # Market-cap floor (bottom 8% micro-cap excluded; UniverseFilter applies exact 10% daily)
        mv_valid = active["total_mv"].dropna()
        mv_floor = float(mv_valid.quantile(drop_bottom_mv_pct)) if len(mv_valid) > 20 else 0.0
        liquid = active[
            (active["amount"].fillna(0.0) >= min_amount)
            & (active["total_mv"].isna() | (active["total_mv"] >= mv_floor))
        ].sort_values("amount", ascending=False)

        picked = liquid["ts_code"].tolist()
        if max_stocks is not None and max_stocks > 0:
            picked = picked[:max_stocks]

        stats = {
            "mode": "active_market",
            "total_listed": total_listed,
            "non_st_mature": len(clean),
            "quoted_live": len(active),
            "passed_liquidity_and_mcap": len(liquid),
            "min_amount_cny": min_amount,
            "mv_floor_cny": round(mv_floor, 0),
            "selected": len(picked),
            "board_breakdown": liquid[liquid["ts_code"].isin(picked)]["board"].value_counts().to_dict(),
        }
        return picked, stats

    def resolve_aligned_start_date(self, end: date, default_days: int = 730) -> date:
        """Align sync start date to existing DuckDB history floor so newly added stocks
        never create half-history cross-sectional gaps in multi-year Walk-Forward backtests."""
        default_start = end - timedelta(days=default_days)
        min_d = self.db.query_scalar("SELECT MIN(trade_date) FROM daily_bars")
        if min_d is not None:
            existing_floor = pd.to_datetime(min_d).date()
            if existing_floor < default_start:
                return existing_floor
        return default_start

    def get_resume_pending_codes(
        self,
        ts_codes: list[str],
        target_end: date,
        min_bars: int = 120,
        stale_days: int = 5,
        target_start: date | None = None,
    ) -> tuple[list[str], list[str]]:
        """Partition `ts_codes` into (pending_codes, skipped_up_to_date_codes) via DuckDB metadata.

        When `target_start` is provided (e.g. extending history back to 2023-06-01), stocks whose
        stored MIN(trade_date) is still more than 30 days after `target_start` (and listed before
        `target_start`) are treated as pending so historical extension is also checkpoint-resumable.
        """
        if not ts_codes:
            return [], []
        existing = self.db.query_df(
            "SELECT ts_code, COUNT(*) AS cnt, MIN(trade_date) AS min_d, MAX(trade_date) AS max_d "
            "FROM daily_bars GROUP BY ts_code"
        )
        if existing.empty:
            return [normalize_ts_code(c) for c in ts_codes], []

        fresh_cutoff = target_end - timedelta(days=stale_days)
        existing["min_d"] = pd.to_datetime(existing["min_d"]).dt.date
        existing["max_d"] = pd.to_datetime(existing["max_d"]).dt.date
        mask = (existing["cnt"] >= min_bars) & (existing["max_d"] >= fresh_cutoff)
        if target_start is not None:
            start_tol = target_start + timedelta(days=30)
            basic = self.db.query_df("SELECT ts_code, list_date FROM stock_basic")
            list_map: dict[str, date | None] = {}
            if not basic.empty:
                basic["list_date"] = pd.to_datetime(basic["list_date"], errors="coerce").dt.date
                list_map = dict(zip(basic["ts_code"], basic["list_date"]))
            listed_after_start = existing["ts_code"].map(
                lambda c: (list_map.get(c) is not None and pd.notna(list_map.get(c)) and list_map[c] >= target_start)  # type: ignore[operator]
            )
            mask = mask & ((existing["min_d"] <= start_tol) | listed_after_start)

        up_to_date = set(existing[mask]["ts_code"].tolist())
        pending, skipped = [], []
        for raw in ts_codes:
            c = normalize_ts_code(raw)
            if c in up_to_date:
                skipped.append(c)
            else:
                pending.append(c)
        return pending, skipped

    def sync_daily_bars_parallel(
        self,
        ts_codes: list[str],
        start: date,
        end: date,
        workers: int = 4,
        resume: bool = True,
        min_bars_for_skip: int = 120,
        enrich_baostock: bool = False,
        batch_size: int = 20,
        progress: ProgressFn | None = None,
        report: SyncReport | None = None,
    ) -> SyncReport:
        """Multi-threaded, checkpoint-resumable real daily bar sync (Plan A full-market engine).

        Uses AKShare Tencent CDN (`raw` + `hfq` adj_factor) across `workers` threads,
        strictly attaches PIT `financial_pit` (trade_date > ann_date) for daily PE_TTM/PB/total_mv,
        and commits to DuckDB in batches so interrupted runs resume seamlessly.
        """
        import concurrent.futures
        import json
        from app.config import get_settings

        report = report or SyncReport()
        norm_codes = [normalize_ts_code(c) for c in ts_codes]
        report.stocks_requested += len(norm_codes)

        if resume:
            pending, skipped = self.get_resume_pending_codes(
                norm_codes, end, min_bars=min_bars_for_skip, target_start=start
            )
            report.stocks_synced += len(skipped)
            if skipped:
                report.sources["cached_resume"] = report.sources.get("cached_resume", 0) + len(skipped)
        else:
            pending, skipped = norm_codes, []

        if not pending:
            if progress:
                progress("all stocks already up-to-date (resumed)", 1.0)
            return report

        # Pre-load PIT financials & ST map once into memory (dict by ts_code -> O(1) lookup per thread!)
        fin_df = self.db.query_df(
            "SELECT ts_code, ann_date, end_date, total_share, bps, eps_ttm FROM financial_pit ORDER BY ts_code, ann_date, end_date"
        )
        fin_by_code: dict[str, pd.DataFrame] = {}
        if not fin_df.empty:
            fin_df["_d"] = pd.to_datetime(fin_df["ann_date"]).astype("datetime64[ns]")
            for c, grp in fin_df.groupby("ts_code", sort=False):
                fin_by_code[str(c)] = grp[["_d", "total_share", "bps", "eps_ttm"]].drop_duplicates("_d", keep="last")

        basic_st = self.db.query_df("SELECT ts_code, is_st_latest FROM stock_basic")
        st_map = dict(zip(basic_st["ts_code"], basic_st["is_st_latest"].astype(bool))) if not basic_st.empty else {}

        fast_primary = AKShareProvider(request_interval=0.05)

        def _fetch_and_prepare(code: str) -> tuple[str, pd.DataFrame | None, str, str]:
            try:
                is_st_hint = st_map.get(code, False)
                if enrich_baostock:
                    bars, source = self._fetch_bars_primary_and_backup(code, start, end, is_st_hint=is_st_hint)
                else:
                    bars = fast_primary.get_daily_bars_with_hfq(code, start, end)
                    source = "akshare"
                    if bars.empty and self.backup._probe_server():
                        bars, source = self._fetch_bars_primary_and_backup(code, start, end, is_st_hint=is_st_hint)
                    if bars.empty:
                        return code, None, "", "empty bars from providers"
                    bars = bars.copy()
                    bars["pe_ttm"] = np.nan
                    bars["pb"] = np.nan
                    bars["is_st"] = is_st_hint
                    bars["is_suspended"] = bars["volume"].fillna(0) <= 0

                if bars["pre_close"].isna().any():
                    bars["pre_close"] = bars["pre_close"].fillna(bars["close"].shift(1)).fillna(bars["open"])
                limits = [calc_limit_prices(pc, code, bool(st)) for pc, st in zip(bars["pre_close"], bars["is_st"])]
                bars["limit_up"] = [x[0] for x in limits]
                bars["limit_down"] = [x[1] for x in limits]

                turn = bars["turnover_rate"].replace(0, np.nan)
                circ_shares = (bars["volume"] / (turn / 100.0)).replace([np.inf, -np.inf], np.nan).ffill().bfill()
                bars["circ_mv"] = bars["close"] * circ_shares

                f = fin_by_code.get(code)
                if f is not None and not f.empty:
                    left = bars[["trade_date"]].copy()
                    left["_d"] = pd.to_datetime(left["trade_date"]).astype("datetime64[ns]")
                    left["_o"] = np.arange(len(left))
                    mm = pd.merge_asof(
                        left.sort_values("_d"),
                        f,
                        on="_d",
                        direction="backward",
                        allow_exact_matches=False,
                    ).sort_values("_o")
                    ts_share = mm["total_share"].to_numpy(dtype=float)
                    bps = mm["bps"].to_numpy(dtype=float)
                    eps_ttm = mm["eps_ttm"].to_numpy(dtype=float)
                    close_arr = bars["close"].to_numpy(dtype=float)

                    total_mv = close_arr * ts_share
                    bars["total_mv"] = np.where(np.isnan(total_mv), bars["circ_mv"], total_mv)
                    calc_pb = np.where((~np.isnan(bps)) & (bps > 1e-6), close_arr / bps, np.nan)
                    calc_pe = np.where((~np.isnan(eps_ttm)) & (np.abs(eps_ttm) > 1e-6), close_arr / eps_ttm, np.nan)
                    bars["pb"] = bars["pb"].where(bars["pb"].notna(), calc_pb)
                    bars["pe_ttm"] = bars["pe_ttm"].where(bars["pe_ttm"].notna(), calc_pe)
                else:
                    bars["total_mv"] = bars["circ_mv"]

                bars["source"] = source
                bars["ts_code"] = code
                return code, bars, source, ""
            except Exception as exc:  # noqa: BLE001
                return code, None, "", str(exc)

        ckpt_file = get_settings().data_dir / "sync_checkpoint.json"
        buffer_frames: list[pd.DataFrame] = []
        done = 0
        total_pending = len(pending)

        def _flush_buffer() -> None:
            if not buffer_frames:
                return
            combined = pd.concat(buffer_frames, ignore_index=True)
            report.bar_rows += self.db.upsert_df("daily_bars", combined)
            buffer_frames.clear()
            ckpt_file.write_text(
                json.dumps(
                    {
                        "updated_at": pd.Timestamp.now().isoformat(),
                        "total_requested": len(norm_codes),
                        "skipped_cached": len(skipped),
                        "completed_pending": done,
                        "total_pending": total_pending,
                        "failed_count": len(report.failed),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = [pool.submit(_fetch_and_prepare, c) for c in pending]
            for fut in concurrent.futures.as_completed(futs):
                code, bars, source, err = fut.result()
                done += 1
                if bars is not None and not bars.empty:
                    buffer_frames.append(bars)
                    report.stocks_synced += 1
                    report.sources[source] = report.sources.get(source, 0) + 1
                else:
                    report.failed[code] = err or "unknown error"
                    logger.warning("parallel sync %s failed: %s", code, err)
                if len(buffer_frames) >= batch_size:
                    _flush_buffer()
                if progress:
                    progress(f"daily_bars ({len(skipped) + done}/{len(norm_codes)}) {code}", (len(skipped) + done) / max(1, len(norm_codes)))

        _flush_buffer()
        return report

    # ------------------------------------------------------------------ intraday snapshot
    def snapshot_top(self, ts_codes: list[str], max_n: int = 100) -> pd.DataFrame:
        """Fetch realtime snapshot ONLY for candidate top-N + holdings to prevent rate limiting."""
        codes = [normalize_ts_code(c) for c in ts_codes][:max_n]
        if not codes:
            return pd.DataFrame()
        return self.primary.get_realtime_spot(codes)

    # ------------------------------------------------------------------ high level
    def incremental_sync(self, ts_codes: list[str] | None = None, lookback_days: int = 10, workers: int = 4) -> SyncReport:
        """Incremental post-close sync from last stored date across ALL synced stocks in DuckDB."""
        from app.core.calendar import now_cn

        now = now_cn()
        # Before 15:15 CN market close, cap daily bar sync at yesterday to avoid partial intraday bars
        end = (now.date() - timedelta(days=1)) if (now.hour < 15 or (now.hour == 15 and now.minute < 15)) else now.date()
        if ts_codes is None:
            ts_codes = self.db.query_df("SELECT DISTINCT ts_code FROM daily_bars")["ts_code"].tolist()
        last = self.db.query_scalar("SELECT MAX(trade_date) FROM daily_bars")
        start = (pd.to_datetime(last).date() - timedelta(days=lookback_days)) if last is not None else end - timedelta(days=730)
        self.sync_calendar(start - timedelta(days=30), end)
        self.sync_index(start - timedelta(days=120), end)
        report = self.sync_daily_bars_parallel(
            ts_codes, start, end, workers=workers, resume=False, enrich_baostock=False
        )
        # Fast 1-call real bulk capital flow update for today (EM rank -> THS fallback)
        self.sync_capital_flow_today(ts_codes, report=report)
        return report

    def close(self) -> None:
        try:
            self.backup.logout()
        except Exception:  # noqa: BLE001
            pass


def board_of(ts_code: str) -> str:
    return detect_board(ts_code)
