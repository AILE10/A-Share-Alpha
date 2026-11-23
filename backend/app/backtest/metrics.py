"""Comprehensive A-share backtest performance metrics calculator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drawdown_stats(nav: pd.Series) -> tuple[pd.Series, dict]:
    """Compute underwater drawdown series + max drawdown depth, start, trough and recovery days."""
    if nav.empty:
        return pd.Series(dtype=float), {
            "max_drawdown": 0.0,
            "mdd_start": "",
            "mdd_trough": "",
            "mdd_recovery": "",
            "mdd_recovery_days": None,
        }
    peak = nav.cummax()
    dd = nav / peak - 1.0
    trough_dt = dd.idxmin()
    mdd = float(dd.loc[trough_dt])
    pre = nav.loc[:trough_dt]
    start_dt = pre.idxmax()
    peak_val = float(nav.loc[start_dt])
    post = nav.loc[trough_dt:]
    recovered = post[post >= peak_val * (1.0 - 1e-9)]
    if len(recovered) > 0:
        rec_dt = recovered.index[0]
        rec_days = int(nav.index.get_loc(rec_dt) - nav.index.get_loc(trough_dt))
        rec_str = str(pd.Timestamp(rec_dt).date())
    else:
        rec_days = None
        rec_str = "未完全修复"
    return dd, {
        "max_drawdown": round(mdd, 6),
        "mdd_start": str(pd.Timestamp(start_dt).date()),
        "mdd_trough": str(pd.Timestamp(trough_dt).date()),
        "mdd_recovery": rec_str,
        "mdd_recovery_days": rec_days,
    }


def compute_backtest_metrics(
    nav_series: pd.Series,
    bench_1000: pd.Series | None,
    bench_300: pd.Series | None,
    trades: list[dict],
    total_turnover_notional: float,
    total_commission: float,
    total_stamp_duty: float,
    total_transfer_fee: float,
    total_slippage: float,
    rf_annual: float = 0.02,
) -> tuple[dict, pd.Series]:
    """Compute full institutional quant performance report."""
    nav = nav_series.dropna().astype(float)
    if len(nav) < 2:
        return {"error": "insufficient_nav_points"}, pd.Series(dtype=float)

    norm_nav = nav / nav.iloc[0]
    daily_ret = norm_nav.pct_change().dropna()
    n_days = len(daily_ret)
    total_return = float(norm_nav.iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (252.0 / max(n_days, 1)) - 1.0)
    ann_vol = float(daily_ret.std(ddof=1) * np.sqrt(252.0)) if n_days > 1 else 0.0
    sharpe = float((ann_return - rf_annual) / ann_vol) if ann_vol > 1e-12 else 0.0

    down_ret = daily_ret[daily_ret < 0]
    down_vol = float(np.sqrt((down_ret ** 2).mean()) * np.sqrt(252.0)) if len(down_ret) else 0.0
    sortino = float((ann_return - rf_annual) / down_vol) if down_vol > 1e-12 else 0.0

    dd_series, dd_stats = compute_drawdown_stats(norm_nav)
    mdd = abs(dd_stats["max_drawdown"])
    calmar = float(ann_return / mdd) if mdd > 1e-9 else 0.0

    def _bench_stats(bench: pd.Series | None) -> tuple[float, float, float]:
        if bench is None or bench.dropna().empty:
            return (0.0, 0.0, 0.0)
        b = bench.reindex(norm_nav.index).ffill().bfill()
        if b.isna().any() or b.iloc[0] <= 0:
            return (0.0, 0.0, 0.0)
        b_norm = b / b.iloc[0]
        b_tot = float(b_norm.iloc[-1] - 1.0)
        b_ann = float((1.0 + b_tot) ** (252.0 / max(n_days, 1)) - 1.0)
        ex_tot = total_return - b_tot
        ex_ann = ann_return - b_ann
        return (b_tot, ex_tot, ex_ann)

    b1000_tot, ex1000_tot, ex1000_ann = _bench_stats(bench_1000)
    b300_tot, ex300_tot, ex300_ann = _bench_stats(bench_300)

    # Trade-level win rate & profit/loss ratio from closed SELL trades
    sell_trades = [t for t in trades if t.get("side") == "SELL" and t.get("realized_pnl") is not None]
    wins = [t["realized_pnl"] for t in sell_trades if t["realized_pnl"] > 0]
    losses = [abs(t["realized_pnl"]) for t in sell_trades if t["realized_pnl"] < 0]
    win_rate = float(len(wins) / len(sell_trades)) if sell_trades else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    profit_loss_ratio = float(avg_win / avg_loss) if avg_loss > 1e-9 else (float("inf") if avg_win > 0 else 0.0)
    if np.isinf(profit_loss_ratio):
        profit_loss_ratio = 99.0

    # Annualized single-sided turnover: (total traded notional / 2) / avg_equity * (252 / n_days)
    avg_equity = float(nav.mean()) if len(nav) else 1.0
    ann_turnover = float((total_turnover_notional / 2.0) / max(avg_equity, 1.0) * (252.0 / max(n_days, 1)))

    # Monthly returns matrix
    monthly = norm_nav.resample("ME").last().pct_change().dropna()
    monthly_returns = [
        {"month": idx.strftime("%Y-%m"), "return": round(float(val), 6)}
        for idx, val in monthly.items()
    ]

    metrics = {
        "start_date": str(pd.Timestamp(norm_nav.index[0]).date()),
        "end_date": str(pd.Timestamp(norm_nav.index[-1]).date()),
        "trading_days": int(n_days + 1),
        "total_return": round(total_return, 6),
        "annualized_return": round(ann_return, 6),
        "annualized_volatility": round(ann_vol, 6),
        "benchmark_csi1000_return": round(b1000_tot, 6),
        "excess_return_csi1000": round(ex1000_tot, 6),
        "excess_annualized_csi1000": round(ex1000_ann, 6),
        "benchmark_csi300_return": round(b300_tot, 6),
        "excess_return_csi300": round(ex300_tot, 6),
        "excess_annualized_csi300": round(ex300_ann, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        **dd_stats,
        "total_trades": len(trades),
        "closed_sells": len(sell_trades),
        "win_rate": round(win_rate, 4),
        "profit_loss_ratio": round(profit_loss_ratio, 4),
        "annualized_turnover": round(ann_turnover, 4),
        "total_commission": round(total_commission, 2),
        "total_stamp_duty": round(total_stamp_duty, 2),
        "total_transfer_fee": round(total_transfer_fee, 2),
        "total_slippage": round(total_slippage, 2),
        "total_cost": round(total_commission + total_stamp_duty + total_transfer_fee + total_slippage, 2),
        "final_equity": round(float(nav.iloc[-1]), 2),
        "monthly_returns": monthly_returns,
    }
    return metrics, dd_series
