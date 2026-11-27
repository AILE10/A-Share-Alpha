"""DuckDB analytical store client for A-share time-series, PIT fundamentals and factors."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

DUCKDB_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stock_basic (
    ts_code VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    exchange VARCHAR NOT NULL,
    board VARCHAR NOT NULL,
    list_date DATE,
    delist_date DATE,
    sw_l1_code VARCHAR,
    sw_l1_name VARCHAR,
    is_st_latest BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trade_calendar (
    trade_date DATE PRIMARY KEY,
    is_open BOOLEAN NOT NULL DEFAULT TRUE,
    prev_trade_date DATE,
    next_trade_date DATE
);

CREATE TABLE IF NOT EXISTS daily_bars (
    ts_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    pre_close DOUBLE,
    volume DOUBLE,
    amount DOUBLE,
    turnover_rate DOUBLE,
    amplitude DOUBLE,
    adj_factor DOUBLE DEFAULT 1.0,
    is_st BOOLEAN DEFAULT FALSE,
    is_suspended BOOLEAN DEFAULT FALSE,
    limit_up DOUBLE,
    limit_down DOUBLE,
    pe_ttm DOUBLE,
    pb DOUBLE,
    total_mv DOUBLE,
    circ_mv DOUBLE,
    source VARCHAR DEFAULT 'akshare',
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS index_daily (
    index_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    pre_close DOUBLE,
    volume DOUBLE,
    amount DOUBLE,
    pct_chg DOUBLE,
    source VARCHAR DEFAULT 'akshare',
    PRIMARY KEY (index_code, trade_date)
);

CREATE TABLE IF NOT EXISTS capital_flow_daily (
    ts_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    main_net_inflow DOUBLE,
    main_net_inflow_ratio DOUBLE,
    super_large_net DOUBLE,
    large_net DOUBLE,
    status VARCHAR DEFAULT 'OK',
    source VARCHAR DEFAULT 'akshare',
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS financial_pit (
    ts_code VARCHAR NOT NULL,
    ann_date DATE NOT NULL,
    end_date DATE NOT NULL,
    net_profit_cum DOUBLE,
    net_profit_q DOUBLE,
    total_share DOUBLE,
    bps DOUBLE,
    eps_ttm DOUBLE,
    roe_ttm DOUBLE,
    np_yoy_q DOUBLE,
    revenue_yoy_q DOUBLE,
    source VARCHAR DEFAULT 'akshare_yjbb',
    PRIMARY KEY (ts_code, ann_date, end_date)
);

CREATE TABLE IF NOT EXISTS universe_daily (
    ts_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    is_eligible BOOLEAN NOT NULL,
    exclude_reason VARCHAR DEFAULT '',
    avg_amount_20d DOUBLE,
    mcap_pct_rank DOUBLE,
    list_days INTEGER,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS factor_values (
    ts_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    factor_name VARCHAR NOT NULL,
    raw_value DOUBLE,
    std_value DOUBLE,
    is_degraded BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (ts_code, trade_date, factor_name)
);

CREATE TABLE IF NOT EXISTS factor_ic_daily (
    trade_date DATE NOT NULL,
    factor_name VARCHAR NOT NULL,
    horizon INTEGER NOT NULL DEFAULT 5,
    ic DOUBLE,
    rank_ic DOUBLE,
    sample_size INTEGER,
    q1_ret DOUBLE,
    q2_ret DOUBLE,
    q3_ret DOUBLE,
    q4_ret DOUBLE,
    q5_ret DOUBLE,
    ls_ret DOUBLE,
    PRIMARY KEY (trade_date, factor_name, horizon)
);

CREATE TABLE IF NOT EXISTS market_regime_daily (
    trade_date DATE PRIMARY KEY,
    csi1000_close DOUBLE,
    csi1000_ma20 DOUBLE,
    csi1000_ma60 DOUBLE,
    breadth_above_ma20 DOUBLE,
    regime VARCHAR NOT NULL,
    target_exposure DOUBLE NOT NULL,
    style_mode VARCHAR DEFAULT 'ALL_CAP',
    min_mcap_quantile DOUBLE DEFAULT 0.0,
    csi300_close DOUBLE,
    csi300_ma20 DOUBLE,
    csi300_ma60 DOUBLE
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    ts_code VARCHAR NOT NULL,
    ex_date DATE NOT NULL,
    bonus_ratio DOUBLE DEFAULT 0.0,
    transfer_ratio DOUBLE DEFAULT 0.0,
    cash_div DOUBLE DEFAULT 0.0,
    rights_ratio DOUBLE DEFAULT 0.0,
    rights_price DOUBLE DEFAULT 0.0,
    split_mult DOUBLE DEFAULT 1.0,
    raw_content VARCHAR DEFAULT '',
    source VARCHAR DEFAULT 'tencent',
    PRIMARY KEY (ts_code, ex_date)
);
"""


class DuckDBClient:
    """Thread-safe DuckDB wrapper with idempotent upsert and schema bootstrap."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        settings = get_settings()
        self.db_path = Path(db_path) if db_path is not None else Path(settings.duckdb_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = duckdb.connect(str(self.db_path))
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            self._conn.execute(DUCKDB_SCHEMA_SQL)
            # Safe additive migrations for pre-existing local DuckDB files
            for col_sql in (
                "ALTER TABLE financial_pit ADD COLUMN IF NOT EXISTS net_profit_cum DOUBLE",
                "ALTER TABLE financial_pit ADD COLUMN IF NOT EXISTS net_profit_q DOUBLE",
                "ALTER TABLE financial_pit ADD COLUMN IF NOT EXISTS total_share DOUBLE",
                "ALTER TABLE financial_pit ADD COLUMN IF NOT EXISTS bps DOUBLE",
                "ALTER TABLE financial_pit ADD COLUMN IF NOT EXISTS eps_ttm DOUBLE",
                "ALTER TABLE market_regime_daily ADD COLUMN IF NOT EXISTS style_mode VARCHAR DEFAULT 'ALL_CAP'",
                "ALTER TABLE market_regime_daily ADD COLUMN IF NOT EXISTS min_mcap_quantile DOUBLE DEFAULT 0.0",
                "ALTER TABLE market_regime_daily ADD COLUMN IF NOT EXISTS csi300_close DOUBLE",
                "ALTER TABLE market_regime_daily ADD COLUMN IF NOT EXISTS csi300_ma20 DOUBLE",
                "ALTER TABLE market_regime_daily ADD COLUMN IF NOT EXISTS csi300_ma60 DOUBLE",
            ):
                self._conn.execute(col_sql)

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        with self._lock:
            if params is not None:
                self._conn.execute(sql, list(params))
            else:
                self._conn.execute(sql)

    def query_df(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        with self._lock:
            if params is not None:
                return self._conn.execute(sql, list(params)).df()
            return self._conn.execute(sql).df()

    def query_scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        with self._lock:
            row = (
                self._conn.execute(sql, list(params)).fetchone()
                if params is not None
                else self._conn.execute(sql).fetchone()
            )
            return row[0] if row else None

    def upsert_df(self, table_name: str, df: pd.DataFrame, chunk_size: int = 200_000) -> int:
        """Idempotently upsert DataFrame rows using INSERT OR REPLACE matching table columns.

        Large DataFrames (> chunk_size) are committed in chunks with brief lock yields
        between batches so concurrent read requests (Dashboard / Workflow / APIs) never freeze.
        """
        if df is None or df.empty:
            return 0
        with self._lock:
            table_cols_df = self._conn.execute(f"PRAGMA table_info('{table_name}')").df()
            table_cols = list(table_cols_df["name"])
        use_cols = [c for c in df.columns if c in table_cols]
        if not use_cols:
            raise ValueError(f"No matching columns for table {table_name}: {list(df.columns)}")
        clean_df = df[use_cols].copy()
        cols_sql = ", ".join(use_cols)
        total = len(clean_df)
        step = max(10_000, int(chunk_size))

        for start_i in range(0, total, step):
            chunk = clean_df.iloc[start_i : start_i + step]
            tmp_view = f"tmp_{uuid.uuid4().hex[:12]}"
            with self._lock:
                self._conn.register(tmp_view, chunk)
                try:
                    self._conn.execute(
                        f"INSERT OR REPLACE INTO {table_name} ({cols_sql}) SELECT {cols_sql} FROM {tmp_view}"
                    )
                finally:
                    self._conn.unregister(tmp_view)
            if start_i + step < total:
                # Briefly yield the lock so incoming UI/API read queries never starve
                time.sleep(0.01)
        return total

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


_duckdb_singleton: DuckDBClient | None = None
_singleton_lock = threading.Lock()


def get_duckdb() -> DuckDBClient:
    global _duckdb_singleton
    with _singleton_lock:
        if _duckdb_singleton is None:
            _duckdb_singleton = DuckDBClient()
        return _duckdb_singleton


def reset_duckdb_singleton() -> None:
    global _duckdb_singleton
    with _singleton_lock:
        if _duckdb_singleton is not None:
            _duckdb_singleton.close()
            _duckdb_singleton = None
