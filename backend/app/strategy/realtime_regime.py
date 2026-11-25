"""Real-time intraday Market Regime service (full-universe live breadth + CSI 1000 live MA20/MA60).

Strictly zero DuckDB write pollution:
  - Pre-caches finalized EOD 19-day stock sums + latest adj_factor + CSI 1000 19d/59d history in memory,
    keyed automatically by DuckDB watermark (so post-close sync invalidates & refreshes cleanly).
  - Pulls real-time Tencent L1 quotes for indices (`sh000852`, `sh000300`, `sh000001`) and all synced
    active A-shares (~2-3 seconds via parallel Tencent batch chunks), computes exact live MA20 breadth
    and live CSI 1000 MA20/MA60/ret_5d, classifies using the shared `classify_regime_point`, caches in
    memory, and broadcasts over WebSocket channel `market_regime`.
  - Never writes unclosed intraday bars into DuckDB `daily_bars` or `market_regime_daily`, preserving
    100% zero-lookahead purity for backtesting and factor research.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import date, datetime

import httpx
import numpy as np
import pandas as pd

from app.api.ws_manager import ws_manager
from app.core.calendar import now_cn, today_cn
from app.core.constants import (
    BENCHMARK_CSI1000,
    BENCHMARK_CSI300,
    BENCHMARK_INDEX_META,
    BENCHMARK_SHCOMP,
)
from app.core.logging import get_logger
from app.data.akshare_provider import AKShareProvider
from app.db.duckdb_client import DuckDBClient, get_duckdb
from app.strategy.regime import RegimeConfig, classify_regime_detail

logger = get_logger(__name__)


@dataclass
class _BaselineCache:
    watermark: tuple
    last_eod_date: date
    codes: list[str]
    sum_19d: dict[str, float]
    cnt_19d: dict[str, int]
    last_adj: dict[str, float]
    idx_sum_19: float
    idx_cnt_19: int
    idx_sum_59: float
    idx_cnt_59: int
    idx_close_5d_ago: float | None
    csi300_sum_19: float = 0.0
    csi300_cnt_19: int = 0
    csi300_sum_59: float = 0.0
    csi300_cnt_59: int = 0
    csi300_close_5d_ago: float | None = None


def fetch_realtime_index_quotes() -> dict[str, dict]:
    """Fetch real-time quotes for CSI1000, CSI300 & SHCOMP in a single Tencent HTTP request (<100ms)."""
    symbols = [BENCHMARK_INDEX_META[c]["ak_symbol"] for c in (BENCHMARK_CSI1000, BENCHMARK_CSI300, BENCHMARK_SHCOMP)]
    sym_to_code = {BENCHMARK_INDEX_META[c]["ak_symbol"]: c for c in (BENCHMARK_CSI1000, BENCHMARK_CSI300, BENCHMARK_SHCOMP)}
    out: dict[str, dict] = {}
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"https://qt.gtimg.cn/q={','.join(symbols)}")
        resp.raise_for_status()
        text = resp.content.decode("gbk", errors="ignore")

    for line in text.strip().splitlines():
        if '="' not in line:
            continue
        lhs, rhs = line.split('="', 1)
        sym = lhs.split("v_")[-1].strip()
        code = sym_to_code.get(sym)
        if not code:
            continue
        parts = rhs.rstrip('";').split("~")
        if len(parts) < 35:
            continue

        def _f(i: int) -> float | None:
            try:
                v = float(parts[i])
                return v if np.isfinite(v) else None
            except (ValueError, IndexError):
                return None

        raw_ts = parts[30].strip() if len(parts) > 30 else ""
        quote_date = ""
        quote_time = ""
        if len(raw_ts) >= 14 and raw_ts[:14].isdigit():
            quote_date = f"{raw_ts[0:4]}-{raw_ts[4:6]}-{raw_ts[6:8]}"
            quote_time = f"{raw_ts[8:10]}:{raw_ts[10:12]}:{raw_ts[12:14]}"
        elif len(raw_ts) >= 8 and raw_ts[:8].isdigit():
            quote_date = f"{raw_ts[0:4]}-{raw_ts[4:6]}-{raw_ts[6:8]}"

        price = _f(3)
        pre_close = _f(4)
        pct_chg = _f(32)
        if price is None or price <= 0:
            continue
        if pct_chg is None and pre_close and pre_close > 0:
            pct_chg = (price / pre_close - 1.0) * 100.0
        out[code] = {
            "index_code": code,
            "name": BENCHMARK_INDEX_META[code]["name"],
            "close": round(float(price), 2),
            "pre_close": round(float(pre_close), 2) if pre_close is not None else None,
            "pct_chg": round(float(pct_chg), 2) if pct_chg is not None else 0.0,
            "quote_date": quote_date,
            "quote_time": quote_time,
        }
    return out


class RealtimeRegimeService:
    """Thread-safe in-memory real-time Market Regime calculator & WebSocket broadcaster."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.cfg = config or RegimeConfig()
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._baseline: _BaselineCache | None = None
        self._latest_snapshot: dict | None = None
        self._latest_mono: float = 0.0

    def clear_cache(self) -> None:
        with self._lock:
            self._baseline = None
            self._latest_snapshot = None
            self._latest_mono = 0.0

    def get_cached(self, max_age_seconds: float | None = 600.0) -> dict | None:
        with self._lock:
            if self._latest_snapshot is None:
                return None
            if max_age_seconds is not None and (time.monotonic() - self._latest_mono) > max_age_seconds:
                return None
            return dict(self._latest_snapshot)

    def _get_watermark(self, db: DuckDBClient) -> tuple:
        row = db.query_df(
            "SELECT MAX(trade_date) AS mx, COUNT(*) AS cnt FROM daily_bars"
        ).iloc[0]
        idx_mx = db.query_scalar("SELECT MAX(trade_date) FROM index_daily WHERE index_code = ?", [BENCHMARK_CSI1000])
        return (
            str(pd.to_datetime(row["mx"]).date()) if pd.notna(row["mx"]) else "",
            int(row["cnt"] or 0),
            str(pd.to_datetime(idx_mx).date()) if idx_mx is not None else "",
        )

    def _ensure_baseline(self, db: DuckDBClient, live_quote_date: date) -> _BaselineCache:
        """Load finalized EOD history strictly BEFORE `live_quote_date` (never double-counts today)."""
        wm = self._get_watermark(db) + (live_quote_date.isoformat(),)
        with self._lock:
            if self._baseline is not None and self._baseline.watermark == wm:
                return self._baseline

        # 1) Finalized CSI 1000 index closes strictly before live_quote_date
        idx_df = db.query_df(
            """
            SELECT trade_date, close
            FROM index_daily
            WHERE index_code = ? AND trade_date < ?
            ORDER BY trade_date DESC
            LIMIT 59
            """,
            [BENCHMARK_CSI1000, live_quote_date],
        )
        if idx_df.empty:
            raise ValueError("No historical CSI 1000 bars in DuckDB to compute realtime MA20/MA60")
        idx_df = idx_df.sort_values("trade_date")
        idx_closes = idx_df["close"].astype(float).to_numpy()
        last_eod_date = pd.to_datetime(idx_df["trade_date"].iloc[-1]).date()

        tail19 = idx_closes[-19:]
        tail59 = idx_closes[-59:]
        close_5d_ago = float(idx_closes[-5]) if len(idx_closes) >= 5 else None

        # 1b) Finalized CSI 300 index closes strictly before live_quote_date (for dual-index style rescue)
        idx300_df = db.query_df(
            """
            SELECT trade_date, close
            FROM index_daily
            WHERE index_code = ? AND trade_date < ?
            ORDER BY trade_date DESC
            LIMIT 59
            """,
            [BENCHMARK_CSI300, live_quote_date],
        )
        if not idx300_df.empty:
            idx300_df = idx300_df.sort_values("trade_date")
            c300_arr = idx300_df["close"].astype(float).to_numpy()
            c300_t19 = c300_arr[-19:]
            c300_t59 = c300_arr[-59:]
            c300_sum_19, c300_cnt_19 = float(np.sum(c300_t19)), int(len(c300_t19))
            c300_sum_59, c300_cnt_59 = float(np.sum(c300_t59)), int(len(c300_t59))
            c300_5d_ago = float(c300_arr[-5]) if len(c300_arr) >= 5 else None
        else:
            c300_sum_19, c300_cnt_19 = 0.0, 0
            c300_sum_59, c300_cnt_59 = 0.0, 0
            c300_5d_ago = None

        # 2) Finalized last 19 trading days per stock strictly before live_quote_date
        dates_df = db.query_df(
            """
            SELECT DISTINCT trade_date
            FROM daily_bars
            WHERE trade_date < ?
            ORDER BY trade_date DESC
            LIMIT 19
            """,
            [live_quote_date],
        )
        if dates_df.empty:
            raise ValueError("No historical stock bars in DuckDB for realtime breadth")
        min_19d = pd.to_datetime(dates_df["trade_date"].min()).date()

        bars_df = db.query_df(
            """
            WITH recent AS (
                SELECT
                    ts_code,
                    trade_date,
                    close * COALESCE(adj_factor, 1.0) AS hfq_close,
                    COALESCE(adj_factor, 1.0) AS adj_factor,
                    ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) AS rn
                FROM daily_bars
                WHERE trade_date >= ? AND trade_date < ? AND close IS NOT NULL AND close > 0
            )
            SELECT
                ts_code,
                SUM(hfq_close) AS sum_19d,
                COUNT(*) AS cnt_19d,
                MAX(CASE WHEN rn = 1 THEN adj_factor END) AS last_adj
            FROM recent
            WHERE rn <= 19
            GROUP BY ts_code
            HAVING COUNT(*) >= 14
            """,
            [min_19d, live_quote_date],
        )
        codes = bars_df["ts_code"].astype(str).tolist()
        sum_19d = dict(zip(codes, bars_df["sum_19d"].astype(float).tolist()))
        cnt_19d = dict(zip(codes, bars_df["cnt_19d"].astype(int).tolist()))
        last_adj = dict(zip(codes, bars_df["last_adj"].astype(float).tolist()))

        cache = _BaselineCache(
            watermark=wm,
            last_eod_date=last_eod_date,
            codes=codes,
            sum_19d=sum_19d,
            cnt_19d=cnt_19d,
            last_adj=last_adj,
            idx_sum_19=float(np.sum(tail19)),
            idx_cnt_19=int(len(tail19)),
            idx_sum_59=float(np.sum(tail59)),
            idx_cnt_59=int(len(tail59)),
            idx_close_5d_ago=close_5d_ago,
            csi300_sum_19=c300_sum_19,
            csi300_cnt_19=c300_cnt_19,
            csi300_sum_59=c300_sum_59,
            csi300_cnt_59=c300_cnt_59,
            csi300_close_5d_ago=c300_5d_ago,
        )
        with self._lock:
            self._baseline = cache
        return cache

    def refresh(
        self,
        db: DuckDBClient | None = None,
        provider: AKShareProvider | None = None,
        broadcast: bool = True,
        max_workers: int = 8,
    ) -> dict:
        """Compute exact full-market real-time Market Regime using live Tencent L1 quotes (~2-3s)."""
        with self._refresh_lock:
            t0 = time.monotonic()
            client_db = db or get_duckdb()
            prov = provider or AKShareProvider()

            idx_quotes = fetch_realtime_index_quotes()
            csi_q = idx_quotes.get(BENCHMARK_CSI1000)
            if not csi_q:
                raise RuntimeError("Failed to fetch realtime CSI 1000 quote from Tencent")

            q_date_str = csi_q.get("quote_date") or today_cn().isoformat()
            q_time_str = csi_q.get("quote_time") or now_cn().strftime("%H:%M:%S")
            live_date = date.fromisoformat(q_date_str)

            # Determine if live_date is strictly newer than latest EOD bar or an active trading day
            latest_db_date_raw = client_db.query_scalar("SELECT MAX(trade_date) FROM daily_bars")
            latest_db_date = pd.to_datetime(latest_db_date_raw).date() if latest_db_date_raw is not None else date(2020, 1, 1)
            is_new_day = live_date > latest_db_date

            base = self._ensure_baseline(client_db, live_date)
            csi_now = float(csi_q["close"])
            ma20_now = (base.idx_sum_19 + csi_now) / (base.idx_cnt_19 + 1) if (base.idx_cnt_19 + 1) >= 15 else np.nan
            ma60_now = (base.idx_sum_59 + csi_now) / (base.idx_cnt_59 + 1) if (base.idx_cnt_59 + 1) >= 30 else np.nan
            ret_5d_now = (
                (csi_now / base.idx_close_5d_ago - 1.0)
                if (base.idx_close_5d_ago is not None and base.idx_close_5d_ago > 0)
                else 0.0
            )

            # Pull exact full-market real-time L1 quotes across all synced active stocks
            spot_df = prov._spot_tx_batch(base.codes, max_workers=max_workers)
            if spot_df.empty:
                raise RuntimeError("Empty full-market realtime spot response")

            above_cnt = 0
            valid_cnt = 0
            advancers = 0
            decliners = 0
            flat = 0
            for code, price, vol, pct in zip(
                spot_df["ts_code"].to_numpy(),
                spot_df["last_price"].to_numpy(dtype=float),
                spot_df["volume"].to_numpy(dtype=float),
                spot_df["pct_chg"].to_numpy(dtype=float),
            ):
                if not np.isfinite(price) or price <= 0 or (np.isfinite(vol) and vol <= 0):
                    continue
                s19 = base.sum_19d.get(code)
                c19 = base.cnt_19d.get(code)
                adj = base.last_adj.get(code)
                if s19 is None or c19 is None or adj is None:
                    continue
                live_hfq = price * adj
                ma20_s = (s19 + live_hfq) / (c19 + 1)
                valid_cnt += 1
                if live_hfq > ma20_s:
                    above_cnt += 1
                if np.isfinite(pct):
                    if pct > 0:
                        advancers += 1
                    elif pct < 0:
                        decliners += 1
                    else:
                        flat += 1

            if valid_cnt < max(10, int(len(base.codes) * 0.3)):
                raise RuntimeError(f"Too few active realtime stock quotes ({valid_cnt}/{len(base.codes)})")

            breadth_now = above_cnt / valid_cnt

            csi300_q = idx_quotes.get(BENCHMARK_CSI300)
            csi300_now = float(csi300_q["close"]) if csi300_q and csi300_q.get("close") else None
            csi300_ma20_now = (
                (base.csi300_sum_19 + csi300_now) / (base.csi300_cnt_19 + 1)
                if (csi300_now is not None and (base.csi300_cnt_19 + 1) >= 15)
                else None
            )
            csi300_ma60_now = (
                (base.csi300_sum_59 + csi300_now) / (base.csi300_cnt_59 + 1)
                if (csi300_now is not None and (base.csi300_cnt_59 + 1) >= 30)
                else None
            )
            csi300_ret_5d_now = (
                (csi300_now / base.csi300_close_5d_ago - 1.0)
                if (csi300_now is not None and base.csi300_close_5d_ago is not None and base.csi300_close_5d_ago > 0)
                else 0.0
            )

            regime_str, target_exp, style_mode, min_mcap_q = classify_regime_detail(
                close=csi_now,
                ma20=float(ma20_now) if pd.notna(ma20_now) else np.nan,
                ma60=float(ma60_now) if pd.notna(ma60_now) else np.nan,
                breadth=breadth_now,
                ret_5d=ret_5d_now,
                config=self.cfg,
                csi300_close=csi300_now,
                csi300_ma20=csi300_ma20_now,
                csi300_ma60=csi300_ma60_now,
                csi300_ret_5d=csi300_ret_5d_now,
            )

            elapsed = round(time.monotonic() - t0, 2)
            snapshot = {
                "trade_date": q_date_str,
                "quote_time": q_time_str,
                "updated_at": f"{q_date_str} {q_time_str}",
                "is_realtime": bool(is_new_day or (now_cn().date() == live_date and now_cn().hour < 15)),
                "regime": regime_str,
                "target_exposure": float(target_exp),
                "style_mode": style_mode,
                "min_mcap_quantile": float(min_mcap_q),
                "csi1000_close": round(csi_now, 2),
                "csi1000_pct_chg": round(float(csi_q.get("pct_chg") or 0.0), 2),
                "csi1000_ma20": round(float(ma20_now), 2) if pd.notna(ma20_now) else None,
                "csi1000_ma60": round(float(ma60_now), 2) if pd.notna(ma60_now) else None,
                "csi300_close": round(csi300_now, 2) if csi300_now is not None else None,
                "csi300_pct_chg": round(float(csi300_q.get("pct_chg") or 0.0), 2) if csi300_q else None,
                "csi300_ma20": round(float(csi300_ma20_now), 2) if csi300_ma20_now is not None else None,
                "csi300_ma60": round(float(csi300_ma60_now), 2) if csi300_ma60_now is not None else None,
                "ret_5d": round(float(ret_5d_now), 4),
                "breadth_above_ma20": round(float(breadth_now), 4),
                "quoted_count": int(valid_cnt),
                "universe_count": int(len(base.codes)),
                "advancers": int(advancers),
                "decliners": int(decliners),
                "flat": int(flat),
                "elapsed_sec": elapsed,
                "indices": idx_quotes,
            }
            with self._lock:
                self._latest_snapshot = snapshot
                self._latest_mono = time.monotonic()

            logger.info(
                "Realtime Regime @ %s %s: %s (%.0f%%) | CSI1000=%.2f (%+.2f%%) MA20=%.2f | Breadth=%.2f%% (%d/%d) | Adv/Dec=%d/%d (%.2fs)",
                q_date_str,
                q_time_str,
                regime_str,
                target_exp * 100.0,
                csi_now,
                snapshot["csi1000_pct_chg"],
                snapshot["csi1000_ma20"] or 0.0,
                breadth_now * 100.0,
                above_cnt,
                valid_cnt,
                advancers,
                decliners,
                elapsed,
            )

            if broadcast:
                ws_manager.broadcast_threadsafe("market_regime", snapshot)

            return dict(snapshot)


_REALTIME_REGIME_SVC = RealtimeRegimeService()


def get_realtime_regime_service() -> RealtimeRegimeService:
    return _REALTIME_REGIME_SVC
