"""Walk-Forward out-of-sample parameter sensitivity grid & heatmap generator."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import pandas as pd

from app.backtest.engine import AShareBacktestEngine, BacktestConfig
from app.factors.panel import MarketPanel
from app.strategy.technical import TechnicalResult


def run_walk_forward_heatmap(
    panel: MarketPanel,
    oos_scores: pd.DataFrame,
    eligible: pd.DataFrame,
    technical: TechnicalResult,
    regime_df: pd.DataFrame,
    base_config: BacktestConfig,
    x_param: str = "rebalance_days",
    x_values: list[Any] | None = None,
    y_param: str = "target_holdings",
    y_values: list[Any] | None = None,
    metric: str = "sharpe_ratio",
    progress_cb: Callable[[str, float], None] | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> dict:
    """Evaluate a 2D parameter grid strictly on out-of-sample Walk-Forward predictions."""
    x_vals = x_values or [3, 5, 10]
    y_vals = y_values or [8, 10, 12]

    cells: list[dict] = []
    best_cell: dict | None = None
    total = max(1, len(x_vals) * len(y_vals))
    step = 0

    for yi, yv in enumerate(y_vals):
        for xi, xv in enumerate(x_vals):
            step += 1
            if progress_cb:
                progress_cb(f"参数敏感性热力图扫描 ({step}/{total}): {x_param}={xv}, {y_param}={yv}", (step - 1) / total)
            cfg = replace(base_config, **{x_param: xv, y_param: yv})
            res = AShareBacktestEngine(cfg).run(
                panel, oos_scores, eligible, technical, regime_df, start_date=start_date, end_date=end_date
            )
            m = res.metrics
            val = float(m.get(metric, 0.0) or 0.0)
            cell = {
                "x_index": xi,
                "y_index": yi,
                "x_value": xv,
                "y_value": yv,
                "value": round(val, 4),
                "annualized_return": m.get("annualized_return", 0.0),
                "excess_annualized_csi1000": m.get("excess_annualized_csi1000", 0.0),
                "sharpe_ratio": m.get("sharpe_ratio", 0.0),
                "max_drawdown": m.get("max_drawdown", 0.0),
                "calmar_ratio": m.get("calmar_ratio", 0.0),
                "annualized_turnover": m.get("annualized_turnover", 0.0),
            }
            cells.append(cell)
            if best_cell is None or val > best_cell["value"]:
                best_cell = cell

    return {
        "x_param": x_param,
        "x_values": x_vals,
        "y_param": y_param,
        "y_values": y_vals,
        "metric": metric,
        "cells": cells,
        "best": best_cell,
    }
