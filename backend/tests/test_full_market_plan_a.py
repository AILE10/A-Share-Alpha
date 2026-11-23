"""Plan A full-market active universe, resumable sync checkpoint & Top-100-only quoting tests."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.data.base import DataProvider
from app.data.sync_service import DataSyncService
from app.db.duckdb_client import DuckDBClient


class _StubMarketProvider(DataProvider):
    """Minimal deterministic test double for verifying Plan A filtering & capping logic without network."""

    name = "stub"

    def __init__(self) -> None:
        self.requested_spot_counts: list[int] = []

    def get_stock_list(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_trade_calendar(self, start: date, end: date) -> list[date]:
        return [start]

    def get_daily_bars(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    def get_index_daily(self, index_code: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    def get_realtime_spot(self, ts_codes: list[str] | None = None) -> pd.DataFrame:
        codes = ts_codes or []
        self.requested_spot_counts.append(len(codes))
        rows = []
        for idx, c in enumerate(codes):
            # Half liquid (>= 60M), half illiquid (< 20M)
            amt = 1.2e8 if idx % 2 == 0 else 1.5e7
            mv = 1.0e10 * (idx + 1)
            rows.append({
                "ts_code": c,
                "name": f"S{idx}",
                "last_price": 20.0,
                "pre_close": 20.0,
                "open": 20.0,
                "high": 20.5,
                "low": 19.5,
                "volume": amt / 20.0,
                "amount": amt,
                "bid1": 19.99,
                "ask1": 20.01,
                "pct_chg": 0.0,
                "turnover_rate": 2.0,
                "pe_ttm": 20.0,
                "pb": 2.0,
                "total_mv": mv,
                "circ_mv": mv * 0.8,
            })
        return pd.DataFrame(rows)


def test_select_active_market_universe_filters_st_subnew_and_illiquid(tmp_path):
    db = DuckDBClient(tmp_path / "plan_a.duckdb")
    try:
        old_ipo = date.today() - timedelta(days=900)
        new_ipo = date.today() - timedelta(days=30)
        rows = []
        for i in range(1, 31):
            rows.append({
                "ts_code": f"600{i:03d}.SH",
                "symbol": f"600{i:03d}",
                "name": "*ST风险" if i == 1 else f"活跃股{i}",
                "exchange": "SH",
                "board": "MAIN",
                "list_date": new_ipo if i == 2 else old_ipo,
                "delist_date": None,
                "sw_l1_code": "801010",
                "sw_l1_name": "电子",
                "is_st_latest": i == 1,
            })
        db.upsert_df("stock_basic", pd.DataFrame(rows))

        stub = _StubMarketProvider()
        svc = DataSyncService(db=db, primary=stub)
        picked, stats = svc.select_active_market_universe(min_amount=40_000_000.0)
        # ST (600001.SH) and sub-new (600002.SH) must be excluded before spot scan
        assert "600001.SH" not in picked
        assert "600002.SH" not in picked
        assert stats["non_st_mature"] == 28
        # Only liquid names (>= 40M) & above bottom-8% mcap survive
        assert 10 <= len(picked) <= 15
    finally:
        db.close()


def test_resume_pending_skips_up_to_date_stocks(tmp_path):
    db = DuckDBClient(tmp_path / "resume.duckdb")
    try:
        today = date.today()
        # Stock A has 130 bars up to yesterday -> should be skipped!
        dates_a = [today - timedelta(days=i) for i in range(1, 131)]
        df_a = pd.DataFrame({
            "ts_code": "000001.SZ",
            "trade_date": dates_a,
            "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "pre_close": 10.0,
            "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "amplitude": 5.0,
            "adj_factor": 1.0, "is_st": False, "is_suspended": False,
            "limit_up": 11.0, "limit_down": 9.0,
        })
        # Stock B only has 50 bars -> must remain pending!
        df_b = df_a.head(50).copy()
        df_b["ts_code"] = "000002.SZ"
        db.upsert_df("daily_bars", pd.concat([df_a, df_b], ignore_index=True))

        svc = DataSyncService(db=db, primary=_StubMarketProvider())
        pending, skipped = svc.get_resume_pending_codes(
            ["000001.SZ", "000002.SZ", "300750.SZ"], target_end=today, min_bars=120
        )
        assert skipped == ["000001.SZ"]
        assert pending == ["000002.SZ", "300750.SZ"]
    finally:
        db.close()


def test_snapshot_top_strictly_caps_intraday_requests_at_100(tmp_path):
    db = DuckDBClient(tmp_path / "cap.duckdb")
    try:
        stub = _StubMarketProvider()
        svc = DataSyncService(db=db, primary=stub)
        many_codes = [f"600{i:03d}.SH" for i in range(1, 250)]
        df = svc.snapshot_top(many_codes, max_n=100)
        assert len(df) == 100
        assert stub.requested_spot_counts == [100]
    finally:
        db.close()
