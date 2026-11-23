"""Plan A Full-Market Active A-Share Resumable Sync & Analysis Script (Token-Free).

Automatically scans ALL ~4,600+ listed A-shares via fast parallel real-time quotes (<10s),
retains all active liquid non-ST stocks (default >= 40M CNY daily turnover & above bottom 8% mcap),
syncs bulk PIT financials for the entire market, runs multi-threaded resumable daily bar sync
(skipping already up-to-date stocks in DuckDB automatically), and recomputes 16-factor analytics
& market regime across the entire synced DuckDB universe.

Usage:
    # Sync ALL active liquid A-shares (~1,800-2,500 stocks, ~8-12 min, resumable anytime)
    backend\\.venv\\Scripts\\python.exe -u scripts\\sync_full_market.py

    # Fast expansion to Top 300 most liquid active A-shares (~90 seconds)
    backend\\.venv\\Scripts\\python.exe -u scripts\\sync_full_market.py --max-stocks 300

    # Sync literally ALL non-ST mature A-shares (~4,500+ stocks, resumable)
    backend\\.venv\\Scripts\\python.exe -u scripts\\sync_full_market.py --all-market

    # Also retrain LightGBM & refresh recommendations after sync
    backend\\.venv\\Scripts\\python.exe -u scripts\\sync_full_market.py --max-stocks 300 --retrain
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.calendar import today_cn  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.data.sync_service import DataSyncService  # noqa: E402
from app.factors.service import FactorEngine, to_jsonable  # noqa: E402
from app.ml.model_registry import ModelRegistry  # noqa: E402
from app.paper.recommender import RecommendationEngine  # noqa: E402
from app.strategy.regime import compute_market_regime, persist_market_regime  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan A token-free full-market active A-share resumable sync")
    ap.add_argument("--max-stocks", type=int, default=None, help="Cap synced active stocks (default: ALL active liquid A-shares)")
    ap.add_argument("--min-amount", type=float, default=40_000_000.0, help="Pre-filter daily turnover floor in CNY (default 40M)")
    ap.add_argument("--days", type=int, default=730, help="Calendar days of history to sync (default 730 = ~2y)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel worker threads for AKShare Tencent CDN (default 4)")
    ap.add_argument("--all-market", action="store_true", help="Skip 40M pre-filter and sync ALL mature non-ST A-shares")
    ap.add_argument("--force-refresh-fin", action="store_true", help="Force re-downloading bulk Eastmoney PIT financials")
    ap.add_argument("--no-resume", action="store_true", help="Do not skip already up-to-date stocks in DuckDB")
    ap.add_argument("--skip-factors", action="store_true", help="Only sync bars/financials, skip factor recompute")
    ap.add_argument("--retrain", action="store_true", help="Retrain LightGBM snapshot & regenerate recommendations across full DB")
    args = ap.parse_args()

    setup_logging("WARNING")
    t_total = time.time()
    svc = DataSyncService()
    end = today_cn()
    start = end - timedelta(days=args.days)

    try:
        print(f"=== [1/6] Sync trade calendar & full A-share listing ({start} ~ {end}) ===", flush=True)
        cal_n = svc.sync_calendar(start - timedelta(days=400), end)
        basic_cnt = int(svc.db.query_scalar("SELECT COUNT(*) FROM stock_basic") or 0)
        if basic_cnt < 3000:
            basic_cnt = svc.sync_stock_basic(with_industry=True)
        print(f"      calendar={cal_n} | stock_basic={basic_cnt}", flush=True)

        print("=== [2/6] Scan full-market liquidity & select active A-share universe ===", flush=True)
        t0 = time.time()
        codes, scan_stats = svc.select_active_market_universe(
            min_amount=args.min_amount,
            max_stocks=args.max_stocks,
            all_non_st=args.all_market,
        )
        print(f"      scanned in {time.time() - t0:.1f}s -> {json.dumps(scan_stats, ensure_ascii=False)}", flush=True)

        print("=== [3/6] Sync full-market PIT quarterly financials (all A-shares bulk 业绩报表) ===", flush=True)
        fin_stocks = int(svc.db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM financial_pit") or 0)
        if args.force_refresh_fin or fin_stocks < max(1500, int(len(codes) * 0.8)):
            t0 = time.time()
            fin_rows = svc.sync_financials(
                None,
                start.year - 1,
                end.year,
                progress=lambda msg, p: print(f"  [{p:6.1%}] {msg}", flush=True),
            )
            print(f"      financial_pit upserted={fin_rows} rows in {time.time() - t0:.1f}s", flush=True)
        else:
            print(f"      [SKIP] financial_pit already covers {fin_stocks} A-shares (use --force-refresh-fin to refresh)", flush=True)

        print(f"=== [4/6] Resumable parallel daily bars sync ({len(codes)} stocks, workers={args.workers}) ===", flush=True)
        t0 = time.time()
        last_pct_printed = [-1]

        def _prog(msg: str, p: float) -> None:
            bucket = int(p * 20)  # every 5%
            if bucket > last_pct_printed[0] or p >= 1.0:
                last_pct_printed[0] = bucket
                print(f"  [{p:6.1%}] {msg}", flush=True)

        report = svc.sync_daily_bars_parallel(
            codes,
            start,
            end,
            workers=args.workers,
            resume=not args.no_resume,
            progress=_prog,
        )
        print(
            f"      bars done in {time.time() - t0:.1f}s | synced={report.stocks_synced}/{report.stocks_requested} "
            f"| new_rows={report.bar_rows} | failed={len(report.failed)} | sources={report.sources}",
            flush=True,
        )

        print("=== [5/6] Sync benchmark indices & real capital flow ===", flush=True)
        idx_rows = svc.sync_index(start - timedelta(days=120), end)
        cf_rep = svc.sync_capital_flow(
            codes,
            workers=min(6, max(2, args.workers)),
            resume=not args.no_resume,
            include_today_bulk=True,
        )
        print(
            f"      index_daily rows={idx_rows} | capital_flow_rows={cf_rep.capital_flow_rows} "
            f"| cf_degraded={len(cf_rep.capital_flow_degraded)}",
            flush=True,
        )

        if not args.skip_factors:
            print("=== [6/6] Recompute 4-dimension 16 factors & market regime across FULL DuckDB ===", flush=True)
            t0 = time.time()
            eng = FactorEngine(svc.db)
            f_run = eng.build(codes=None, horizon=5)
            persisted = eng.persist(f_run)
            reg_df = compute_market_regime(f_run.panel)
            persist_market_regime(reg_df, svc.db)
            uni = f_run.universe.summary()
            print(
                f"      factors done in {time.time() - t0:.1f}s | universe={uni['eligible']}/{uni['total']} eligible "
                f"| persisted={persisted}",
                flush=True,
            )

            if args.retrain:
                print("=== [Bonus] Retraining LightGBM snapshot & refreshing full-universe recommendations ===", flush=True)
                reg = ModelRegistry()
                saved, _ = reg.retrain_and_register(
                    codes=None,
                    model_type="lambdarank",
                    notes=f"全库扩展重训 ({uni['total']} 只真实 A 股)",
                    duckdb_client=svc.db,
                )
                rec_bundle = RecommendationEngine(svc.db).generate(codes=None, persist=True, broadcast=False)
                print(
                    f"      model={saved['version_id']} (OOS RankIC={saved['oos_rank_ic']:.4f}) "
                    f"| active recommendations={len(rec_bundle.recommendations)}",
                    flush=True,
                )

        # Save summary report
        rep_dir = get_settings().data_dir / "reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        total_in_db = int(svc.db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM daily_bars") or 0)
        summary = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "elapsed_sec": round(time.time() - t_total, 1),
            "scan_stats": scan_stats,
            "sync_report": report.to_dict(),
            "total_stocks_in_duckdb": total_in_db,
        }
        out_path = rep_dir / "full_market_sync_report.json"
        out_path.write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== Plan A Full-Market Sync Finished in {time.time() - t_total:.1f}s | DuckDB stocks={total_in_db} ===", flush=True)
    finally:
        svc.close()


if __name__ == "__main__":
    main()
