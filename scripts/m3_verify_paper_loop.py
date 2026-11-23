"""M3 acceptance script: verify live paper trading closed loop on REAL A-share data:
  1. ModelRegistry initial training + snapshot persistence + retrain + version rollback
  2. Real-time Top-100 intraday snapshot refresh & explainable recommendation generation
  3. 14:40 rebalance execution through RiskMiddleware -> PaperBrokerAdapter -> SQLite ledger
  4. APScheduler 5 trading jobs registration check
  5. Weekly review (paper vs theoretical backtest tracking error, slippage drag & 20d IC decay alerts)

Usage:
    backend\\.venv\\Scripts\\python.exe scripts\\m3_verify_paper_loop.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.factors.service import to_jsonable  # noqa: E402
from app.ml.model_registry import ModelRegistry  # noqa: E402
from app.paper.reviewer import WeeklyReviewEngine  # noqa: E402
from app.paper.scheduler import StudioScheduler  # noqa: E402
from app.paper.service import PaperTradingService  # noqa: E402

SAMPLE_FILE = get_settings().data_dir / "sample_universe.json"
REPORT_DIR = get_settings().data_dir / "reports"


def main() -> None:
    setup_logging("WARNING")
    codes = json.loads(SAMPLE_FILE.read_text(encoding="utf-8")) if SAMPLE_FILE.exists() else None

    print("=== [1/5] ModelRegistry: train snapshot v1 & v2, verify rollback ===", flush=True)
    registry = ModelRegistry()
    v1, _ = registry.retrain_and_register(codes=codes, model_type="lambdarank", notes="M3 验收初始版本 v1")
    v2, _ = registry.retrain_and_register(codes=codes, model_type="regression", notes="M3 验收回归对照版 v2")
    rolled = registry.activate_version(v1["version_id"])
    print(f"  v1={v1['version_id']} (OOS RankIC={v1['oos_rank_ic']:.4f}), v2={v2['version_id']}")
    print(f"  Rolled back active version -> {rolled['version_id']} (is_active={rolled['is_active']})")

    print("\n=== [2/5] PaperTradingService: 14:40 rebalance + explainable recommendations + real ledger ===", flush=True)
    paper = PaperTradingService()
    rebal = paper.run_rebalance(codes=codes, override_mode="PAPER", fetch_live_quotes=True)
    print(f"  trade_date={rebal['trade_date']} | regime={rebal['regime']['regime']} (exp={rebal['regime']['target_exposure']:.0%})")
    print(f"  recommendations={rebal['recommendations_count']} | orders={len(rebal['orders_executed'])} | positions={rebal['positions_count']}")
    recs = paper.recommender.get_latest_persisted()
    for r in recs[:3]:
        tf = ", ".join(f"{f['label']}={f['std_value']} (SHAP {f['shap']:+.4f})" for f in r["top_factors"][:3])
        print(f"    [{r['action']}] #{r['composite_rank']} {r['ts_code']} {r['name']} wt={r['suggested_weight']:.1%} px={r['ref_price']} stop={r['stop_loss_price']} | {tf}")

    print("\n=== [3/5] Intraday Top-100 snapshot mark-to-market refresh ===", flush=True)
    intra = paper.intraday_refresh(candidate_codes=codes)
    print(f"  quoted={intra['quoted_count']} | equity={intra['account']['total_equity']:.2f} | cash={intra['account']['cash']:.2f} | stop_alerts={len(intra['stop_alerts'])}")

    print("\n=== [4/5] APScheduler trading-session jobs check ===", flush=True)
    sched = StudioScheduler(paper)
    sched.start()
    jobs = sched.list_jobs()
    for j in jobs:
        print(f"  job={j['id']:<26} next={j['next_run_time']}")
    sched.stop()

    print("\n=== [5/5] Weekly Review closed-loop report ===", flush=True)
    review = WeeklyReviewEngine().generate_review()
    print(json.dumps(review, ensure_ascii=False, indent=2))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model_versions": registry.list_versions()[:5],
        "rebalance": rebal,
        "sample_recommendations": recs[:5],
        "intraday_refresh": {"quoted_count": intra["quoted_count"], "account": intra["account"]},
        "scheduler_jobs": jobs,
        "weekly_review": review,
    }
    path = REPORT_DIR / "m3_paper_loop_report.json"
    path.write_text(json.dumps(to_jsonable(out), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nM3 report written to {path}")

    checks = {
        "model snapshot & rollback working": rolled["version_id"] == v1["version_id"] and rolled["is_active"],
        "recommendations carry Top-3 factors & stop-loss": len(recs) > 0 and len(recs[0]["top_factors"]) >= 3 and recs[0]["stop_loss_price"] > 0,
        "paper orders & positions persisted to SQLite": len(rebal["orders_executed"]) > 0 and rebal["positions_count"] > 0,
        "APScheduler registered all 5 jobs": len(jobs) == 5,
        "weekly review generated with advice": bool(review["review_id"]) and len(review["optimization_advice"]) >= 1,
    }
    print("\n=== M3 acceptance checks ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
