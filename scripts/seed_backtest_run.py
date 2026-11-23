"""Persist one real-data Purged Walk-Forward backtest run into SQLite for the Backtest Center UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.routers.backtest import BacktestRunRequest, run_backtest  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402


def main() -> None:
    setup_logging("WARNING")
    res = run_backtest(
        BacktestRunRequest(
            name="默认复合策略 (LambdaRank + RiskParity + 大盘择时)",
            model_mode="lambdarank",
            weight_method="risk_parity",
            rebalance_days=5,
            target_holdings=10,
            buffer_rank=25,
            include_heatmap=True,
        )
    )
    print(f"Saved real backtest run {res['id']}: ann={res['metrics']['annualized_return']:.2%}, sharpe={res['metrics']['sharpe_ratio']}")


if __name__ == "__main__":
    main()
