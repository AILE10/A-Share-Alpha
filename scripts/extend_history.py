"""Extend DuckDB historical depth (e.g. back to 2023-06-01) so Purged Walk-Forward
strictly out-of-sample backtests cover 2025-01-02 through today with zero look-ahead.

Why 2023-06-01 is required for a 2025-01-02 OOS start:
  * 60 trading days factor warm-up (ret_60d / price_pct_60d / CSI1000 MA60)
  * 252 trading days Fold-0 training window + 5d label purge + 5d embargo = 262 trading days
  * Total pre-OOS requirement: ~322 trading days (~16 months) before 2025-01-02 -> 2023-06-01!

Usage:
    backend\\.venv\\Scripts\\python.exe -u scripts\\extend_history.py --start 2023-06-01 --workers 6
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd  # noqa: E402

from app.core.calendar import today_cn  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.data.sync_service import DataSyncService  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Extend DuckDB historical market & PIT financial depth")
    ap.add_argument("--start", type=str, default="2023-06-01", help="Target start date YYYY-MM-DD (default 2023-06-01)")
    ap.add_argument("--workers", type=int, default=6, help="Parallel worker threads for AKShare Tencent CDN")
    ap.add_argument("--no-resume", action="store_true", help="Force re-fetch all stocks even if already extended")
    ap.add_argument("--skip-fin", action="store_true", help="Skip bulk PIT financials sync")
    args = ap.parse_args()

    setup_logging("WARNING")
    t_total = time.time()
    start_d = date.fromisoformat(args.start)
    end_d = today_cn()

    svc = DataSyncService()
    try:
        codes = svc.db.query_df("SELECT DISTINCT ts_code FROM daily_bars ORDER BY ts_code")["ts_code"].tolist()
        print(f"=== Extending DuckDB history to {start_d} ~ {end_d} for {len(codes)} A-shares ===", flush=True)

        print("1) Syncing extended trade_calendar & CSI1000/CSI300 indices...", flush=True)
        cal_n = svc.sync_calendar(date(start_d.year - 1, 1, 1), end_d)
        idx_n = svc.sync_index(start_d - timedelta(days=150), end_d)
        print(f"   calendar={cal_n} | index_daily={idx_n}", flush=True)

        if not args.skip_fin:
            print(f"2) Syncing full-market PIT financials ({start_d.year - 1} ~ {end_d.year})...", flush=True)
            t0 = time.time()
            fin_n = svc.sync_financials(
                None,
                start_d.year - 1,
                end_d.year,
                progress=lambda msg, p: print(f"   [{p:6.1%}] {msg}", flush=True),
            )
            print(f"   financial_pit upserted={fin_n} rows in {time.time() - t0:.1f}s", flush=True)

        print(f"3) Checkpoint-resumable parallel daily_bars backfill ({len(codes)} stocks, workers={args.workers})...", flush=True)
        t0 = time.time()
        last_b = [-1]

        def _prog(msg: str, p: float) -> None:
            b = int(p * 20)
            if b > last_b[0] or p >= 1.0:
                last_b[0] = b
                print(f"   [{p:6.1%}] ({time.time() - t0:5.1f}s) {msg}", flush=True)

        rep = svc.sync_daily_bars_parallel(
            codes,
            start_d,
            end_d,
            workers=args.workers,
            resume=not args.no_resume,
            min_bars_for_skip=300,
            batch_size=50,
            progress=_prog,
        )
        print(
            f"   daily_bars done in {time.time() - t0:.1f}s | synced={rep.stocks_synced}/{rep.stocks_requested} "
            f"| upserted_rows={rep.bar_rows} | failed={len(rep.failed)} | sources={rep.sources}",
            flush=True,
        )

        # Print updated DuckDB bounds
        dr = svc.db.query_df(
            "SELECT COUNT(DISTINCT ts_code) stocks, COUNT(*) rows, MIN(trade_date) mn, MAX(trade_date) mx FROM daily_bars"
        ).to_dict("records")[0]
        print(f"\n=== Extension Complete in {time.time() - t_total:.1f}s | daily_bars={dr} ===", flush=True)
    finally:
        svc.close()


if __name__ == "__main__":
    main()
