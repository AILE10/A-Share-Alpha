"""Sync 100% REAL A-share individual stock capital flow (main net inflow & ratio) into DuckDB.

Uses checkpoint-resumable parallel Eastmoney push2his (~120 recent trading days per stock)
plus single-call full-market bulk snapshot (Eastmoney rank -> Tonghuashun instant fallback).
Strictly zero mock data: any unavailable stock remains tagged cleanly without fabrication.

Usage:
    backend\\.venv\\Scripts\\python.exe -u scripts\\sync_capital_flow.py
    backend\\.venv\\Scripts\\python.exe -u scripts\\sync_capital_flow.py --workers 6 --recompute
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.logging import setup_logging  # noqa: E402
from app.data.sync_service import DataSyncService  # noqa: E402
from app.factors.service import FactorEngine  # noqa: E402
from app.strategy.regime import compute_market_regime, persist_market_regime  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync real A-share capital flow into DuckDB")
    ap.add_argument("--workers", type=int, default=6, help="Parallel worker threads for push2his (default 6)")
    ap.add_argument("--no-resume", action="store_true", help="Force re-fetch even for up-to-date stocks")
    ap.add_argument("--today-only", action="store_true", help="Only run fast 1-call bulk snapshot for latest trading day")
    ap.add_argument("--recompute", action="store_true", help="Recompute 16-factor analytics & persist after sync")
    args = ap.parse_args()

    setup_logging("WARNING")
    t0 = time.time()
    svc = DataSyncService()
    try:
        codes = svc.db.query_df("SELECT DISTINCT ts_code FROM daily_bars ORDER BY ts_code")["ts_code"].tolist()
        print(f"=== Syncing REAL capital flow for {len(codes)} A-shares in DuckDB ===", flush=True)

        if args.today_only:
            rep = svc.sync_capital_flow_today(codes)
        else:
            last_bucket = [-1]

            def _prog(msg: str, p: float) -> None:
                b = int(p * 20)
                if b > last_bucket[0] or p >= 1.0:
                    last_bucket[0] = b
                    print(f"  [{p:6.1%}] ({time.time() - t0:5.1f}s) {msg}", flush=True)

            rep = svc.sync_capital_flow(
                codes,
                workers=args.workers,
                resume=not args.no_resume,
                batch_size=100,
                progress=_prog,
                include_today_bulk=True,
            )

        stats = svc.db.query_df(
            "SELECT status, COUNT(*) rows, COUNT(DISTINCT ts_code) stocks, "
            "MIN(trade_date) min_d, MAX(trade_date) max_d FROM capital_flow_daily GROUP BY 1"
        ).to_dict("records")
        print(
            f"=== Capital Flow Sync Finished in {time.time() - t0:.1f}s | "
            f"upserted={rep.capital_flow_rows} | degraded={len(rep.capital_flow_degraded)} ===",
            flush=True,
        )
        print(f"    DuckDB capital_flow_daily: {stats}", flush=True)

        if args.recompute:
            t1 = time.time()
            print("=== Recomputing 4-dimension 16 factors & market regime across FULL DuckDB ===", flush=True)
            eng = FactorEngine(svc.db)
            f_run = eng.build(codes=None, horizon=5)
            persisted = eng.persist(f_run)
            persist_market_regime(compute_market_regime(f_run.panel), svc.db)
            print(
                f"    Recompute done in {time.time() - t1:.1f}s | "
                f"main_net_inflow_ratio status={f_run.factors.status.get('main_net_inflow_ratio')} | "
                f"persisted={persisted}",
                flush=True,
            )
    finally:
        svc.close()


if __name__ == "__main__":
    main()
