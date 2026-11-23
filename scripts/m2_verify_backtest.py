"""M2 acceptance script: Purged Walk-Forward LightGBM (LambdaRank + Regression),
Technical Resonance, Market Regime Timing, Risk Parity vs Equal Weight,
and Event-Driven A-Share Exact Backtest + Walk-Forward Sensitivity Heatmap on REAL data.

Usage:
    backend\\.venv\\Scripts\\python.exe scripts\\m2_verify_backtest.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd  # noqa: E402

from app.backtest.engine import BacktestConfig  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.factors.service import to_jsonable  # noqa: E402
from app.ml.lgbm_scorer import LGBMConfig  # noqa: E402
from app.strategy.alpha_strategy import AlphaStrategyService  # noqa: E402

SAMPLE_FILE = get_settings().data_dir / "sample_universe.json"
REPORT_DIR = get_settings().data_dir / "reports"


def main() -> None:
    setup_logging("WARNING")
    codes = json.loads(SAMPLE_FILE.read_text(encoding="utf-8")) if SAMPLE_FILE.exists() else None
    t0 = time.time()
    svc = AlphaStrategyService()

    print("=== [1/2] Running Composite Strategy (LambdaRank + Risk Parity + Regime + Technical Resonance) ===", flush=True)
    bundle = svc.run_full_pipeline(
        codes=codes,
        lgbm_config=LGBMConfig(mode="lambdarank", train_window=252, embargo_days=5, test_window=63, label_horizon=5),
        bt_config=BacktestConfig(weight_method="risk_parity", rebalance_days=5, target_holdings=10, buffer_rank=25),
        include_ew_compare=True,
        include_heatmap=True,
    )

    m = bundle.backtest.metrics
    ew_m = bundle.equal_weight_comparison["metrics"] if bundle.equal_weight_comparison else {}
    print(f"\nCompleted in {time.time() - t0:.1f}s")
    print(f"Walk-Forward folds: {len(bundle.ml_result.folds)} | OOS IC: {bundle.ml_result.oos_ic_mean:.4f} | OOS RankIC: {bundle.ml_result.oos_rank_ic_mean:.4f} | ICIR: {bundle.ml_result.oos_icir:.4f}")
    print("\n--- Risk Parity (Default) Metrics ---")
    print(json.dumps({k: v for k, v in m.items() if k != "monthly_returns"}, ensure_ascii=False, indent=2))
    print("\n--- Equal Weight Comparison Summary ---")
    print(json.dumps({
        k: ew_m.get(k) for k in ("annualized_return", "excess_annualized_csi1000", "sharpe_ratio", "calmar_ratio", "max_drawdown", "annualized_turnover")
    }, ensure_ascii=False, indent=2))

    print("\n--- Top 6 Feature Importance (TreeSHAP & Gain) ---")
    fi_sorted = sorted(bundle.ml_result.feature_importance.items(), key=lambda kv: kv[1]["shap"], reverse=True)
    for fname, vals in fi_sorted[:6]:
        print(f"  {fname:<24} shap={vals['shap']:.4f}  gain={vals['gain']:.4f}")

    print("\n--- Walk-Forward Parameter Sensitivity Heatmap (Best Cell) ---")
    print(json.dumps(bundle.heatmap["best"], ensure_ascii=False, indent=2))

    # Also verify regression mode briefly
    print("\n=== [2/2] Verifying LightGBM Regression Mode ===", flush=True)
    reg_bundle = svc.run_full_pipeline(
        codes=codes,
        lgbm_config=LGBMConfig(mode="regression", train_window=252, embargo_days=5, test_window=63, label_horizon=5),
        bt_config=BacktestConfig(weight_method="risk_parity", rebalance_days=5, target_holdings=10),
        include_ew_compare=False,
        include_heatmap=False,
        panel=bundle.factor_run.panel,
    )
    print(f"Regression OOS RankIC: {reg_bundle.ml_result.oos_rank_ic_mean:.4f} | Sharpe: {reg_bundle.backtest.metrics['sharpe_ratio']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "ml_summary": {
            "mode": "lambdarank",
            "folds": bundle.ml_result.folds,
            "oos_ic": bundle.ml_result.oos_ic_mean,
            "oos_rank_ic": bundle.ml_result.oos_rank_ic_mean,
            "oos_icir": bundle.ml_result.oos_icir,
            "feature_importance": bundle.ml_result.feature_importance,
        },
        "backtest_metrics": m,
        "equal_weight_metrics": ew_m,
        "regression_mode_metrics": reg_bundle.backtest.metrics,
        "walk_forward_heatmap": bundle.heatmap,
        "trades_count": len(bundle.backtest.trades),
        "rejected_orders_count": len(bundle.backtest.rejected_orders),
        "sample_trades": bundle.backtest.trades[:10],
    }
    path = REPORT_DIR / "m2_backtest_report.json"
    path.write_text(json.dumps(to_jsonable(out), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nM2 report written to {path}")

    required_keys = [
        "annualized_return", "excess_return_csi1000", "excess_return_csi300",
        "sharpe_ratio", "calmar_ratio", "max_drawdown", "mdd_recovery",
        "win_rate", "profit_loss_ratio", "annualized_turnover",
    ]
    checks = {
        "all required backtest metrics present": all(k in m for k in required_keys),
        "real trades executed": len(bundle.backtest.trades) > 0,
        "walk-forward folds >= 1 with embargo >= 5": len(bundle.ml_result.folds) >= 1 and all(f["embargo_days"] >= 5 for f in bundle.ml_result.folds),
        "heatmap generated": bundle.heatmap is not None and len(bundle.heatmap["cells"]) == 9,
        "risk parity vs equal weight compared": bool(ew_m),
    }
    print("\n=== M2 acceptance checks ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
