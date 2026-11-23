"""M1 acceptance script: sync >= 50 REAL A-shares (~2y) into DuckDB, run the 4-dimension factor
engine and print the real IC / RankIC / ICIR / quintile / correlation report.

Usage (from repo root):
    backend\\.venv\\Scripts\\python.exe scripts\\m1_verify_real_data.py --n 60 --days 800
    backend\\.venv\\Scripts\\python.exe scripts\\m1_verify_real_data.py --skip-sync   # re-run factors only
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
from app.data.sync_service import DataSyncService, SyncReport  # noqa: E402
from app.db.duckdb_client import get_duckdb  # noqa: E402
from app.factors.service import FactorEngine, to_jsonable  # noqa: E402

SAMPLE_FILE = get_settings().data_dir / "sample_universe.json"
REPORT_DIR = get_settings().data_dir / "reports"


def _progress(label: str, pct: float) -> None:
    if int(pct * 100) % 10 == 0 or pct >= 1.0:
        print(f"  [{pct:6.1%}] {label}", flush=True)


def run_sync(n: int, days: int, resume: bool = False) -> list[str]:
    svc = DataSyncService()
    end = today_cn()
    start = end - timedelta(days=days)
    t0 = time.time()
    try:
        if resume and SAMPLE_FILE.exists() and (svc.db.query_scalar("SELECT COUNT(*) FROM financial_pit") or 0) > 100:
            codes = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
            print(f"[RESUME] Using existing trade_calendar, stock_basic, financial_pit & {len(codes)} selected stocks", flush=True)
        else:
            print(f"[1/7] Sync real trade calendar {start} ~ {end}", flush=True)
            print("      rows:", svc.sync_calendar(start - timedelta(days=400), end), flush=True)

            print("[2/7] Sync stock basic list + Shenwan L1 industry map", flush=True)
            print("      rows:", svc.sync_stock_basic(with_industry=True), flush=True)

            print(f"[3/7] Select {n} real liquid mid/small-cap stocks across boards & industries", flush=True)
            codes = svc.select_sample_universe(n=n)
            SAMPLE_FILE.write_text(json.dumps(codes, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"      selected {len(codes)}: {codes[:10]} ...", flush=True)

            print("[4/7] Sync PIT quarterly financials (real announcement dates) via AKShare bulk / Baostock", flush=True)
            fin_rows = svc.sync_financials(codes, start.year - 1, end.year, progress=_progress)
            print("      financial_pit rows:", fin_rows, flush=True)

        print("[5/7] Sync unadjusted daily bars + PE/PB/ST/suspension + adj factor", flush=True)
        report = SyncReport()
        svc.sync_daily_bars(codes, start, end, progress=_progress, report=report)

        print("[6/7] Sync benchmark indices (CSI1000 / CSI300)", flush=True)
        report.index_rows = svc.sync_index(start - timedelta(days=120), end)

        print("[7/7] Sync main-force capital flow (graceful degradation on failure)", flush=True)
        svc.sync_capital_flow(codes, report)
        print("      sync report:", json.dumps(report.to_dict(), ensure_ascii=False, default=str)[:1500], flush=True)
        print(f"      sync finished in {time.time() - t0:.1f}s", flush=True)
        if report.stocks_synced < 50:
            raise SystemExit(f"FAILED: only {report.stocks_synced} stocks synced (< 50)")
        return codes
    finally:
        svc.close()


def db_overview() -> dict:
    db = get_duckdb()
    q = db.query_df
    ov = {
        "stocks_in_daily_bars": int(db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM daily_bars")),
        "daily_bar_rows": int(db.query_scalar("SELECT COUNT(*) FROM daily_bars")),
        "date_range": [str(x) for x in q("SELECT MIN(trade_date) a, MAX(trade_date) b FROM daily_bars").iloc[0]],
        "index_rows": q("SELECT index_code, COUNT(*) n, MAX(trade_date) last FROM index_daily GROUP BY 1").to_dict("records"),
        "financial_pit_rows": int(db.query_scalar("SELECT COUNT(*) FROM financial_pit")),
        "capital_flow_ok_rows": int(db.query_scalar("SELECT COUNT(*) FROM capital_flow_daily WHERE status='OK'")),
        "source_breakdown": q("SELECT source, COUNT(DISTINCT ts_code) n FROM daily_bars GROUP BY 1").to_dict("records"),
        "board_breakdown": q(
            "SELECT s.board, COUNT(DISTINCT b.ts_code) n FROM daily_bars b JOIN stock_basic s USING (ts_code) GROUP BY 1"
        ).to_dict("records"),
        "industry_count": int(db.query_scalar(
            "SELECT COUNT(DISTINCT s.sw_l1_name) FROM daily_bars b JOIN stock_basic s USING (ts_code)"
        )),
    }
    return ov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--days", type=int, default=800, help="calendar days of history to fetch (~1.5y analysis after warm-up)")
    ap.add_argument("--skip-sync", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()
    setup_logging("WARNING")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    if args.skip_sync and SAMPLE_FILE.exists():
        codes = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    else:
        codes = run_sync(args.n, args.days, resume=args.resume)

    ov = db_overview()
    print("\n=== DuckDB real data overview ===")
    print(json.dumps(ov, ensure_ascii=False, indent=1, default=str))

    print("\n=== Factor engine: universe -> 16 factors -> MAD/neutralize/zscore -> IC analysis ===")
    t0 = time.time()
    engine = FactorEngine()
    res = engine.build(codes=codes, horizon=args.horizon)
    persisted = engine.persist(res)
    print(f"factor engine finished in {time.time() - t0:.1f}s, persisted: {persisted}")

    uni = res.universe.summary()
    print("\n--- Universe filter (latest day) ---")
    print(json.dumps(uni, ensure_ascii=False, indent=1))

    coverage = {
        n: round(float(df.where(res.universe.eligible).notna().sum().sum() / max(1, res.universe.eligible.sum().sum())), 4)
        for n, df in res.factors.raw.items()
    }
    print("\n--- Factor coverage on eligible universe & data status ---")
    for n in coverage:
        print(f"  {n:<24} coverage={coverage[n]:.2%}  status={res.factors.status[n]}")

    summ = res.analysis.summary
    cols = ["factor_name", "label", "dimension", "ic_mean", "rank_ic_mean", "icir", "rank_icir", "t_stat",
            "ic_win_rate", "ls_annual_return", "quintile_monotonicity", "decay_status", "n_periods", "data_status"]
    print(f"\n--- Single-factor IC report (horizon={args.horizon}d, non-overlapping) ---")
    print(summ[cols].round(4).to_string(index=False))

    print("\n--- Quintile NAV (final, non-overlapping 5d rebalance) ---")
    nav_final = {n: nav.iloc[-1].round(4).to_dict() for n, nav in res.analysis.quintile_nav.items() if len(nav)}
    print(pd.DataFrame(nav_final).T.to_string())

    corr = res.analysis.correlation
    pairs = []
    names = list(corr.index)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))
    print("\n--- Top-8 most correlated factor pairs (mean cross-sectional Spearman) ---")
    for a, b, c in pairs[:8]:
        print(f"  {a:<24} ~ {b:<24} {c:+.3f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "codes": codes,
        "db_overview": ov,
        "universe": uni,
        "factor_coverage": coverage,
        "factor_status": res.factors.status,
        "summary": summ.to_dict("records"),
        "quintile_nav_final": nav_final,
        "correlation": corr.round(4).to_dict(),
        "persisted": persisted,
    }
    path = REPORT_DIR / "m1_factor_report.json"
    path.write_text(json.dumps(to_jsonable(out), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nM1 report written to {path}")

    checks = {
        ">=50 real stocks in DuckDB": ov["stocks_in_daily_bars"] >= 50,
        ">=1y history": (pd.Timestamp(ov["date_range"][1]) - pd.Timestamp(ov["date_range"][0])).days >= 365,
        "CSI1000 benchmark synced": any(r["index_code"] == "000852.SH" for r in ov["index_rows"]),
        "PIT financials synced": ov["financial_pit_rows"] > 0,
        "16 factors analysed": len(summ) == 16,
        "IC computed": summ["rank_ic_mean"].notna().sum() >= 12,
    }
    print("\n=== M1 acceptance checks ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
