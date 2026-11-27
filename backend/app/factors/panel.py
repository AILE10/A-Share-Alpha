"""Wide-format (date x ts_code) market panel loaded from the real DuckDB store.

Price conventions (guardrail #2):
  * raw_*  : unadjusted exchange prices -> limit-up/down checks, lot sizing, fee & fill prices
  * hfq_*  : backward-adjusted prices (raw * adj_factor) -> returns, momentum, MA, ATR, volatility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from app.core.constants import BENCHMARK_CSI1000, BENCHMARK_CSI300
from app.db.duckdb_client import DuckDBClient, get_duckdb


@dataclass
class MarketPanel:
    dates: pd.DatetimeIndex
    codes: list[str]
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    pre_close: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    turnover: pd.DataFrame
    adj: pd.DataFrame
    is_st: pd.DataFrame
    is_suspended: pd.DataFrame
    pe_ttm: pd.DataFrame
    pb: pd.DataFrame
    total_mv: pd.DataFrame
    limit_up: pd.DataFrame
    limit_down: pd.DataFrame
    index_close: dict[str, pd.Series] = field(default_factory=dict)
    index_bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    basic: pd.DataFrame = field(default_factory=pd.DataFrame)
    capital_flow_ratio: pd.DataFrame | None = None
    fundamentals: dict[str, pd.DataFrame] = field(default_factory=dict)
    split_mult: pd.DataFrame | None = None
    cash_div: pd.DataFrame | None = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # backward-adjusted prices (lazily cached; 100% read-only & deterministic)
    @property
    def hfq_close(self) -> pd.DataFrame:
        c = self._cache.get("hfq_close")
        if c is None:
            c = self.close * self.adj
            self._cache["hfq_close"] = c
        return c

    @property
    def hfq_open(self) -> pd.DataFrame:
        c = self._cache.get("hfq_open")
        if c is None:
            c = self.open * self.adj
            self._cache["hfq_open"] = c
        return c

    @property
    def hfq_high(self) -> pd.DataFrame:
        c = self._cache.get("hfq_high")
        if c is None:
            c = self.high * self.adj
            self._cache["hfq_high"] = c
        return c

    @property
    def hfq_low(self) -> pd.DataFrame:
        c = self._cache.get("hfq_low")
        if c is None:
            c = self.low * self.adj
            self._cache["hfq_low"] = c
        return c

    @property
    def returns(self) -> pd.DataFrame:
        c = self._cache.get("returns")
        if c is None:
            c = self.hfq_close.pct_change(fill_method=None)
            self._cache["returns"] = c
        return c

    def industry_of(self) -> pd.Series:
        if self.basic.empty:
            return pd.Series("未分类", index=self.codes)
        return self.basic.set_index("ts_code").reindex(self.codes)["sw_l1_name"].fillna("未分类")

    def name_of(self) -> pd.Series:
        if self.basic.empty:
            return pd.Series("", index=self.codes)
        return self.basic.set_index("ts_code").reindex(self.codes)["name"].fillna("")

    def slice_until(self, end: pd.Timestamp) -> "MarketPanel":
        """Return a panel truncated at `end` (inclusive) - used by look-ahead tests."""
        mask = self.dates <= end
        d = self.dates[mask]

        def cut(df: pd.DataFrame | None) -> pd.DataFrame | None:
            return None if df is None else df.loc[df.index <= end]

        return MarketPanel(
            dates=d, codes=self.codes,
            open=cut(self.open), high=cut(self.high), low=cut(self.low), close=cut(self.close),
            pre_close=cut(self.pre_close), volume=cut(self.volume), amount=cut(self.amount),
            turnover=cut(self.turnover), adj=cut(self.adj), is_st=cut(self.is_st),
            is_suspended=cut(self.is_suspended), pe_ttm=cut(self.pe_ttm), pb=cut(self.pb),
            total_mv=cut(self.total_mv), limit_up=cut(self.limit_up), limit_down=cut(self.limit_down),
            index_close={k: v.loc[v.index <= end] for k, v in self.index_close.items()},
            index_bars={k: v.loc[v.index <= end] for k, v in self.index_bars.items()},
            basic=self.basic,
            capital_flow_ratio=cut(self.capital_flow_ratio),
            fundamentals={k: cut(v) for k, v in self.fundamentals.items()},
            split_mult=cut(self.split_mult),
            cash_div=cut(self.cash_div),
        )


def _pivot(df: pd.DataFrame, col: str, dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    wide = df.pivot(index="trade_date", columns="ts_code", values=col)
    return wide.reindex(index=dates, columns=codes)


def _to_float64(series: pd.Series) -> np.ndarray:
    try:
        return series.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def _scatter_columns(
    df: pd.DataFrame,
    date_col: str,
    code_col: str,
    val_cols: list[str],
    dates: pd.DatetimeIndex,
    code_to_col: dict[str, int],
    codes_sorted: list[str],
    fill_values: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Single-pass NumPy coordinate scatter replacing repeated pandas pivot_table calls (~10x faster)."""
    n_dates, n_codes = len(dates), len(codes_sorted)
    fills = fill_values or {}
    if df.empty:
        return {
            col: pd.DataFrame(np.full((n_dates, n_codes), fills.get(col, np.nan), dtype=np.float64), index=dates, columns=codes_sorted)
            for col in val_cols
        }

    dt_series = pd.to_datetime(df[date_col])
    ri = dates.get_indexer(dt_series)
    ci = df[code_col].map(code_to_col).fillna(-1).to_numpy(dtype=np.intp)
    valid = (ri >= 0) & (ci >= 0)
    ri_v = ri[valid]
    ci_v = ci[valid]

    out: dict[str, pd.DataFrame] = {}
    for col in val_cols:
        fv = fills.get(col, np.nan)
        mat = np.full((n_dates, n_codes), fv, dtype=np.float64)
        vals = _to_float64(df[col])[valid]
        if np.isnan(fv):
            mat[ri_v, ci_v] = vals
        else:
            ok = ~np.isnan(vals)
            mat[ri_v[ok], ci_v[ok]] = vals[ok]
        out[col] = pd.DataFrame(mat, index=dates, columns=codes_sorted)
    return out


def load_pit_fundamentals(db: DuckDBClient, codes: list[str], start: date, end: date) -> pd.DataFrame:
    """Point-in-time fundamentals via DuckDB ASOF JOIN on the REAL announcement date.

    Strict inequality `trade_date > ann_date`: a report announced on day t (often after close)
    is first usable on the next trading day -> zero look-ahead (guardrail #3). end_date is
    never used for alignment.
    """
    if not codes:
        return pd.DataFrame(columns=["ts_code", "trade_date", "roe_ttm", "np_yoy_q", "ann_date", "end_date"])
    code_list = ",".join(f"'{c}'" for c in codes)
    sql = f"""
        WITH b AS (
            SELECT ts_code, trade_date FROM daily_bars
            WHERE ts_code IN ({code_list}) AND trade_date BETWEEN ? AND ?
        ),
        f_dedup AS (
            SELECT ts_code, ann_date, end_date, roe_ttm, np_yoy_q
            FROM (
                SELECT ts_code, ann_date, end_date, roe_ttm, np_yoy_q,
                       ROW_NUMBER() OVER (PARTITION BY ts_code, ann_date ORDER BY end_date DESC) AS rn
                FROM financial_pit
            )
            WHERE rn = 1
        )
        SELECT b.ts_code, b.trade_date, f.roe_ttm, f.np_yoy_q, f.ann_date, f.end_date
        FROM b
        ASOF LEFT JOIN f_dedup f
          ON b.ts_code = f.ts_code AND b.trade_date > f.ann_date
        ORDER BY b.ts_code, b.trade_date
    """
    return db.query_df(sql, [start, end])


def load_market_panel(
    codes: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    db: DuckDBClient | None = None,
) -> MarketPanel:
    db = db or get_duckdb()
    where, params = ["1=1"], []
    if codes:
        where.append("ts_code IN (" + ",".join(f"'{c}'" for c in codes) + ")")
    if start:
        where.append("trade_date >= ?")
        params.append(start)
    if end:
        where.append("trade_date <= ?")
        params.append(end)
    bars = db.query_df(f"SELECT * FROM daily_bars WHERE {' AND '.join(where)} ORDER BY trade_date, ts_code", params)
    if bars.empty:
        raise ValueError("daily_bars is empty - run real data sync first (scripts/m1_verify_real_data.py)")
    bars["trade_date"] = pd.to_datetime(bars["trade_date"])
    # Guard against incomplete trailing dates (e.g. partial CDN settlement mid-sync)
    day_counts = bars.groupby("trade_date")["ts_code"].nunique().sort_index()
    if len(day_counts) >= 5:
        med_cnt = float(day_counts.median())
        while len(day_counts) > 1 and float(day_counts.iloc[-1]) < 0.30 * med_cnt:
            drop_dt = day_counts.index[-1]
            bars = bars[bars["trade_date"] != drop_dt]
            day_counts = day_counts.iloc[:-1]
    dates = pd.DatetimeIndex(sorted(bars["trade_date"].unique()))
    codes_sorted = sorted(bars["ts_code"].unique().tolist())
    code_to_col = {c: i for i, c in enumerate(codes_sorted)}
    n_dates, n_codes = len(dates), len(codes_sorted)

    # Single-pass NumPy coordinate scatter (replaces 16 separate pandas.pivot calls over 6M+ rows)
    row_idx = dates.get_indexer(bars["trade_date"])
    col_idx = bars["ts_code"].map(code_to_col).to_numpy(dtype=np.intp)

    num_cols = ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover_rate",
                "adj_factor", "pe_ttm", "pb", "total_mv", "limit_up", "limit_down"]
    piv: dict[str, pd.DataFrame] = {}
    for c in num_cols:
        mat = np.full((n_dates, n_codes), np.nan, dtype=np.float64)
        mat[row_idx, col_idx] = _to_float64(bars[c])
        piv[c] = pd.DataFrame(mat, index=dates, columns=codes_sorted)

    st_mat = np.zeros((n_dates, n_codes), dtype=bool)
    st_mat[row_idx, col_idx] = bars["is_st"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    is_st = pd.DataFrame(st_mat, index=dates, columns=codes_sorted)

    # Missing bar on an open day (stock not trading / not yet listed) is treated as suspended
    susp_mat = np.ones((n_dates, n_codes), dtype=bool)
    susp_mat[row_idx, col_idx] = bars["is_suspended"].astype("boolean").fillna(True).to_numpy(dtype=bool)
    is_susp = pd.DataFrame(susp_mat, index=dates, columns=codes_sorted)

    piv["adj_factor"] = piv["adj_factor"].ffill().bfill().fillna(1.0)

    idx_close: dict[str, pd.Series] = {}
    idx_bars: dict[str, pd.DataFrame] = {}
    for code in (BENCHMARK_CSI1000, BENCHMARK_CSI300):
        idf = db.query_df("SELECT * FROM index_daily WHERE index_code = ? ORDER BY trade_date", [code])
        if not idf.empty:
            idf["trade_date"] = pd.to_datetime(idf["trade_date"])
            idf = idf.set_index("trade_date")
            idx_bars[code] = idf.reindex(dates).ffill()
            idx_close[code] = idx_bars[code]["close"].astype(float)

    basic = db.query_df("SELECT * FROM stock_basic WHERE ts_code IN (" + ",".join(f"'{c}'" for c in codes_sorted) + ")")

    cf = db.query_df(
        "SELECT ts_code, trade_date, main_net_inflow_ratio FROM capital_flow_daily WHERE status = 'OK' AND ts_code IN ("
        + ",".join(f"'{c}'" for c in codes_sorted) + ")"
    )
    cf_wide = None
    if not cf.empty:
        cf_wide = _scatter_columns(
            cf, "trade_date", "ts_code", ["main_net_inflow_ratio"], dates, code_to_col, codes_sorted
        )["main_net_inflow_ratio"]

    fund = load_pit_fundamentals(db, codes_sorted, dates[0].date(), dates[-1].date())
    fundamentals: dict[str, pd.DataFrame] = {}
    if not fund.empty:
        fundamentals = _scatter_columns(
            fund, "trade_date", "ts_code", ["roe_ttm", "np_yoy_q"], dates, code_to_col, codes_sorted
        )

    split_mult_df: pd.DataFrame | None = None
    cash_div_df: pd.DataFrame | None = None
    try:
        ca = db.query_df(
            "SELECT ts_code, ex_date, split_mult, cash_div FROM corporate_actions "
            "WHERE ex_date BETWEEN ? AND ? AND ts_code IN ("
            + ",".join(f"'{c}'" for c in codes_sorted)
            + ")",
            [dates[0].date(), dates[-1].date()],
        )
        if not ca.empty:
            ca_piv = _scatter_columns(
                ca,
                "ex_date",
                "ts_code",
                ["split_mult", "cash_div"],
                dates,
                code_to_col,
                codes_sorted,
                fill_values={"split_mult": 1.0, "cash_div": 0.0},
            )
            split_mult_df = ca_piv["split_mult"]
            cash_div_df = ca_piv["cash_div"]
    except Exception:  # noqa: BLE001
        # Backward compatibility for pre-existing test DBs without corporate_actions populated
        pass

    return MarketPanel(
        dates=dates, codes=codes_sorted,
        open=piv["open"], high=piv["high"], low=piv["low"], close=piv["close"],
        pre_close=piv["pre_close"], volume=piv["volume"].fillna(0.0), amount=piv["amount"].fillna(0.0),
        turnover=piv["turnover_rate"], adj=piv["adj_factor"], is_st=is_st, is_suspended=is_susp,
        pe_ttm=piv["pe_ttm"], pb=piv["pb"], total_mv=piv["total_mv"],
        limit_up=piv["limit_up"], limit_down=piv["limit_down"],
        index_close=idx_close, index_bars=idx_bars, basic=basic,
        capital_flow_ratio=cf_wide, fundamentals=fundamentals,
        split_mult=split_mult_df, cash_div=cash_div_df,
    )


def trading_days_since_listing(panel: MarketPanel, all_trade_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Number of exchange trading days since IPO for each (date, code), fully vectorized."""
    cal = all_trade_dates if all_trade_dates is not None else panel.dates
    cal = pd.DatetimeIndex(sorted(cal))
    n_dates, n_codes = len(panel.dates), len(panel.codes)
    mat = np.empty((n_dates, n_codes), dtype=np.float64)

    list_dates: dict[str, pd.Timestamp] = {}
    if not panel.basic.empty:
        list_dates = dict(zip(panel.basic["ts_code"], pd.to_datetime(panel.basic["list_date"])))
    date_pos = cal.searchsorted(panel.dates, side="right").astype(np.float64)

    known_cols: list[int] = []
    known_lpos: list[int] = []
    unknown_cols: list[int] = []
    for idx, code in enumerate(panel.codes):
        ld = list_dates.get(code)
        if ld is None or pd.isna(ld):
            unknown_cols.append(idx)
        else:
            known_cols.append(idx)
            known_lpos.append(int(cal.searchsorted(ld, side="left")))

    if known_cols:
        lpos_arr = np.asarray(known_lpos, dtype=np.float64)
        mat[:, known_cols] = date_pos[:, None] - lpos_arr[None, :]
    if unknown_cols:
        susp_np = panel.is_suspended.to_numpy(dtype=bool)
        mat[:, unknown_cols] = (~susp_np[:, unknown_cols]).cumsum(axis=0, dtype=np.float64)
    return pd.DataFrame(mat, index=panel.dates, columns=panel.codes)


def np_nan_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=df.index, columns=df.columns)
