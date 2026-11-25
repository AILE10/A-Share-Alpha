"""Event-driven A-share exact backtest engine.

Timeline on trading day t (strictly zero look-ahead):
  1. Pre-open:
     - Unlock T+1 sellable shares (yesterday's buys become sellable today).
     - Adjust existing positions cleanly on ex-dividend / ex-right days via adj_factor ratio.
  2. Market Open (execution of orders queued at close of t-1):
     - Execute pending sells first (including limit-down retry sells and stop-loss sells), freeing cash.
     - Execute pending buys (subject to limit-up lock, 100-share lots, available cash).
     - Any sell blocked by limit-down lock stays queued for t+1 retry.
  3. Market Close (day t):
     - Mark portfolio to market using raw close prices, compute equity & drawdown.
     - Update per-position highest close since entry; check -8% hard stop & 2.5*ATR trailing stop.
     - Check portfolio drawdown circuit breaker (-8% warning halve exposure, -12% halt buys/defensive).
     - On rebalance dates (every `rebalance_days`), compute target portfolio from day-t OOS scores,
       technical resonance, regime exposure & buffer zone, and queue orders for t+1 open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from app.backtest.matching import BarQuote, match_order, round_buy_lots
from app.backtest.metrics import compute_backtest_metrics
from app.broker.drawdown_fuse import DrawdownFuseConfig, DrawdownFuseState, step_drawdown_fuse
from app.core.constants import (
    BENCHMARK_CSI1000,
    BENCHMARK_CSI300,
    DEFAULT_MIN_PRICE,
    DEFAULT_REBALANCE_DAYS,
    DEFAULT_SLIPPAGE_BPS,
    LOT_SIZE,
)
from app.factors.panel import MarketPanel
from app.strategy.dynamic_holding import (
    DynamicHoldingConfig,
    HeldPositionState,
    compute_dynamic_stop_price,
    select_dynamic_portfolio,
)
from app.strategy.optimizer import PortfolioConfig, WeightMethod, select_portfolio_names, solve_portfolio_weights
from app.strategy.technical import TechnicalResult

HoldingMode = Literal["fixed", "dynamic"]


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    holding_mode: HoldingMode = "dynamic"
    rebalance_days: int = DEFAULT_REBALANCE_DAYS
    min_hold_days: int = 10
    deep_exit_rank: int = 150
    exit_confirm_days: int = 3
    swap_score_gap: float = 0.70
    profit_ride_pct: float = 0.10
    profit_protect_pct: float = 0.15
    profit_atr_mult: float = 2.2
    stop_cooldown_days: int = 5
    target_holdings: int = 10
    top_candidates: int = 30
    buffer_rank: int = 35
    max_single_weight: float = 0.12
    max_industry_weight: float = 0.25
    weight_method: WeightMethod = "risk_parity"
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    hard_stop_pct: float = 0.08
    atr_stop_mult: float = 2.5
    drawdown_warn_pct: float = 0.08
    drawdown_halt_pct: float = 0.12
    warn_exposure_mult: float = 0.50
    rolling_peak_days: int = 90
    warn_stable_days: int = 0
    halt_cooldown_days: int = 20
    enable_regime_timing: bool = True
    enable_technical_filter: bool = True
    min_buy_price: float = DEFAULT_MIN_PRICE  # 选股执行层防面值退市硬底线（不扰动因子标准化与训练样本）
    cash_yield_rate: float = 0.0         # 可选闲置现金货币ETF年化收益率（默认0%=纯股票Alpha，可设0.015/0.018）
    price_mode: Literal["open", "close"] = "open"


@dataclass
class _Position:
    ts_code: str
    total_shares: int
    sellable_shares: int
    avg_cost: float
    entry_date: str
    highest_price: float
    last_adj_factor: float
    entry_day_idx: int = 0
    weak_rank_streak: int = 0


@dataclass
class BacktestResult:
    config: dict
    metrics: dict
    equity_curve: list[dict]
    drawdown_curve: list[dict]
    trades: list[dict]
    rejected_orders: list[dict]
    daily_nav: pd.Series
    regime_history: list[dict] = field(default_factory=list)


class AShareBacktestEngine:
    """Event-driven daily A-share backtest engine."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.cfg = config or BacktestConfig()

    def run(
        self,
        panel: MarketPanel,
        oos_scores: pd.DataFrame,
        eligible: pd.DataFrame,
        technical: TechnicalResult,
        regime_df: pd.DataFrame,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> BacktestResult:
        cfg = self.cfg
        # Restrict simulation to dates where out-of-sample scores actually exist
        scored_mask = oos_scores.notna().any(axis=1)
        valid_dates = panel.dates[scored_mask.reindex(panel.dates).fillna(False)]
        if start_date is not None:
            valid_dates = valid_dates[valid_dates >= pd.Timestamp(start_date)]
        if end_date is not None:
            valid_dates = valid_dates[valid_dates <= pd.Timestamp(end_date)]
        if len(valid_dates) < 5:
            raise ValueError(f"Insufficient scored backtest dates ({len(valid_dates)})")

        # Include one trailing day if available so final queued orders can settle
        sim_dates = valid_dates
        industries = panel.industry_of()
        names = panel.name_of()
        returns_panel = panel.returns
        mcap_pct_rank = panel.total_mv.where(~panel.is_suspended).rank(axis=1, pct=True)

        cash = float(cfg.initial_cash)
        positions: dict[str, _Position] = {}
        pending_sells: dict[str, tuple[int, str]] = {}   # ts_code -> (qty, reason)
        pending_buys: dict[str, tuple[float, str]] = {}  # ts_code -> (target_weight, reason)
        stopped_recently: set[str] = set()

        fuse_cfg = DrawdownFuseConfig(
            warn_pct=cfg.drawdown_warn_pct,
            halt_pct=cfg.drawdown_halt_pct,
            warn_exposure_mult=cfg.warn_exposure_mult,
            halt_cooldown_days=cfg.halt_cooldown_days,
            rolling_peak_days=cfg.rolling_peak_days,
            warn_stable_days=cfg.warn_stable_days,
        )
        fuse_state_obj = DrawdownFuseState(status="NORMAL", peak_equity=cash)
        fuse_state = "NORMAL"

        nav_records: list[dict] = []
        trades: list[dict] = []
        rejected: list[dict] = []
        regime_hist: list[dict] = []

        total_turnover = 0.0
        tot_comm = 0.0
        tot_stamp = 0.0
        tot_transfer = 0.0
        tot_slip = 0.0
        total_cash_interest = 0.0

        port_cfg = PortfolioConfig(
            top_candidates=cfg.top_candidates,
            target_holdings=cfg.target_holdings,
            buffer_rank=cfg.buffer_rank,
            max_single_weight=cfg.max_single_weight,
            max_industry_weight=cfg.max_industry_weight,
            weight_method=cfg.weight_method,
        )
        dyn_cfg = DynamicHoldingConfig(
            min_hold_days=cfg.min_hold_days,
            buffer_rank=cfg.buffer_rank,
            deep_exit_rank=cfg.deep_exit_rank,
            exit_confirm_days=cfg.exit_confirm_days,
            swap_score_gap=cfg.swap_score_gap,
            profit_ride_pct=cfg.profit_ride_pct,
            profit_protect_pct=cfg.profit_protect_pct,
            profit_atr_mult=cfg.profit_atr_mult,
            stop_cooldown_days=cfg.stop_cooldown_days,
        )
        closed_hold_days: list[int] = []
        last_stop_day_idx: int = -999

        def _make_quote(dt: pd.Timestamp, code: str) -> BarQuote:
            def _val(df: pd.DataFrame, default: float = 0.0) -> float:
                v = df.loc[dt, code]
                return float(v) if pd.notna(v) else default

            return BarQuote(
                ts_code=code,
                open=_val(panel.open),
                high=_val(panel.high),
                low=_val(panel.low),
                close=_val(panel.close),
                pre_close=_val(panel.pre_close),
                volume=_val(panel.volume),
                is_st=bool(panel.is_st.loc[dt, code]),
                is_suspended=bool(panel.is_suspended.loc[dt, code]),
                limit_up=_val(panel.limit_up) or None,
                limit_down=_val(panel.limit_down) or None,
            )

        for day_idx, dt in enumerate(sim_dates):
            dt_str = str(dt.date())

            # 1) PRE-OPEN: Optional risk-free money-market yield on overnight idle cash + T+1 unlock + ex-div
            if day_idx > 0 and cfg.cash_yield_rate > 0.0 and cash > 0.0:
                daily_interest = cash * (cfg.cash_yield_rate / 252.0)
                cash += daily_interest
                total_cash_interest += daily_interest
            for code, pos in list(positions.items()):
                pos.sellable_shares = pos.total_shares
                # Detect TRUE ex-dividend / ex-right dates via exchange pre_close vs prior day raw close
                if day_idx > 0:
                    old_close = panel.close.loc[sim_dates[day_idx - 1], code]
                    ex_ref = panel.pre_close.loc[dt, code]
                    if pd.notna(old_close) and pd.notna(ex_ref) and old_close > 0 and ex_ref > 0:
                        rel_gap = abs(float(old_close) - float(ex_ref)) / float(old_close)
                        if rel_gap > 1e-3:
                            ratio = float(old_close) / float(ex_ref)
                            eq_before = pos.total_shares * float(old_close)
                            eq_after = pos.total_shares * float(ex_ref)
                            cash += (eq_before - eq_after)
                            pos.avg_cost = pos.avg_cost / ratio
                            pos.highest_price = pos.highest_price / ratio
                cur_adj = panel.adj.loc[dt, code]
                if pd.notna(cur_adj) and cur_adj > 0:
                    pos.last_adj_factor = float(cur_adj)

            # 2) OPEN: execute pending SELLS first (frees cash & handles limit-down retries)
            retry_sells: dict[str, tuple[int, str]] = {}
            for code, (req_qty, reason) in list(pending_sells.items()):
                pos = positions.get(code)
                if pos is None or pos.total_shares <= 0:
                    continue
                quote = _make_quote(dt, code)
                sell_qty = min(req_qty, pos.total_shares)
                fill = match_order(
                    quote,
                    side="SELL",
                    requested_qty=sell_qty,
                    sellable_shares=pos.sellable_shares,
                    total_shares=pos.total_shares,
                    slippage_bps=cfg.slippage_bps,
                    price_mode=cfg.price_mode,
                )
                if fill.filled:
                    cash += fill.net_cash_delta
                    realized_pnl = (fill.exec_price - pos.avg_cost) * fill.filled_qty - fill.fees.total_fees
                    pos.total_shares -= fill.filled_qty
                    pos.sellable_shares -= fill.filled_qty
                    if pos.total_shares <= 0:
                        closed_hold_days.append(max(1, day_idx - pos.entry_day_idx))
                        del positions[code]
                    total_turnover += fill.gross_amount
                    tot_comm += fill.fees.commission
                    tot_stamp += fill.fees.stamp_duty
                    tot_transfer += fill.fees.transfer_fee
                    tot_slip += fill.fees.slippage_cost
                    trades.append({
                        "trade_date": dt_str,
                        "ts_code": code,
                        "name": str(names.get(code, "")),
                        "side": "SELL",
                        "qty": fill.filled_qty,
                        "price": fill.exec_price,
                        "amount": fill.gross_amount,
                        "commission": fill.fees.commission,
                        "stamp_duty": fill.fees.stamp_duty,
                        "transfer_fee": fill.fees.transfer_fee,
                        "slippage": fill.fees.slippage_cost,
                        "realized_pnl": round(realized_pnl, 2),
                        "reason": reason,
                    })
                else:
                    rejected.append({
                        "trade_date": dt_str,
                        "ts_code": code,
                        "side": "SELL",
                        "requested_qty": sell_qty,
                        "code": fill.reject_code,
                        "reason": fill.reject_reason,
                    })
                    if fill.reject_code in ("LIMIT_DOWN_LOCKED", "SUSPENDED", "T_PLUS_1_LOCKED"):
                        retry_sells[code] = (sell_qty, f"{reason}[顺延]")
            pending_sells = retry_sells

            # 2b) OPEN: execute pending BUYS (unless portfolio fuse is HALTED)
            if pending_buys and fuse_state != "HALTED":
                # Estimate pre-buy total equity at open/pre_close
                est_mv = 0.0
                for c, p in positions.items():
                    px = panel.open.loc[dt, c]
                    if pd.isna(px) or px <= 0:
                        px = panel.pre_close.loc[dt, c]
                    est_mv += p.total_shares * (float(px) if pd.notna(px) else p.avg_cost)
                est_equity = cash + est_mv

                for code, (target_w, reason) in list(pending_buys.items()):
                    if code in pending_sells:
                        continue
                    quote = _make_quote(dt, code)
                    ref_p = quote.open if (cfg.price_mode == "open" and quote.open > 0) else quote.close
                    if ref_p <= 0:
                        continue
                    cur_pos = positions.get(code)
                    cur_val = (cur_pos.total_shares * ref_p) if cur_pos else 0.0
                    target_val = est_equity * min(target_w, cfg.max_single_weight)
                    buy_val = target_val - cur_val
                    if buy_val < ref_p * LOT_SIZE:
                        continue
                    req_qty = round_buy_lots(int(buy_val / (ref_p * (1.0 + cfg.slippage_bps / 10_000.0 + 0.001))))
                    if req_qty < LOT_SIZE:
                        continue
                    fill = match_order(
                        quote,
                        side="BUY",
                        requested_qty=req_qty,
                        available_cash=cash,
                        slippage_bps=cfg.slippage_bps,
                        price_mode=cfg.price_mode,
                    )
                    if fill.filled:
                        cash += fill.net_cash_delta
                        cur_adj = float(panel.adj.loc[dt, code]) if pd.notna(panel.adj.loc[dt, code]) else 1.0
                        if cur_pos is None:
                            positions[code] = _Position(
                                ts_code=code,
                                total_shares=fill.filled_qty,
                                sellable_shares=0,  # T+1 locked today!
                                avg_cost=fill.exec_price,
                                entry_date=dt_str,
                                highest_price=fill.exec_price,
                                last_adj_factor=cur_adj,
                                entry_day_idx=day_idx,
                            )
                        else:
                            new_total = cur_pos.total_shares + fill.filled_qty
                            cur_pos.avg_cost = (
                                cur_pos.avg_cost * cur_pos.total_shares + fill.exec_price * fill.filled_qty
                            ) / new_total
                            cur_pos.total_shares = new_total
                            # sellable_shares unchanged -> newly bought shares are T+1 locked today
                        total_turnover += fill.gross_amount
                        tot_comm += fill.fees.commission
                        tot_transfer += fill.fees.transfer_fee
                        tot_slip += fill.fees.slippage_cost
                        trades.append({
                            "trade_date": dt_str,
                            "ts_code": code,
                            "name": str(names.get(code, "")),
                            "side": "BUY",
                            "qty": fill.filled_qty,
                            "price": fill.exec_price,
                            "amount": fill.gross_amount,
                            "commission": fill.fees.commission,
                            "stamp_duty": 0.0,
                            "transfer_fee": fill.fees.transfer_fee,
                            "slippage": fill.fees.slippage_cost,
                            "realized_pnl": None,
                            "reason": reason,
                        })
                    else:
                        rejected.append({
                            "trade_date": dt_str,
                            "ts_code": code,
                            "side": "BUY",
                            "requested_qty": req_qty,
                            "code": fill.reject_code,
                            "reason": fill.reject_reason,
                        })
            pending_buys = {}

            # 3) CLOSE: Mark-to-market & individual stop-loss + portfolio drawdown fuse
            mv = 0.0
            stopped_today: set[str] = set()
            for code, pos in positions.items():
                px = panel.close.loc[dt, code]
                if pd.isna(px) or px <= 0:
                    px = panel.pre_close.loc[dt, code]
                price = float(px) if pd.notna(px) and px > 0 else pos.avg_cost
                mv += pos.total_shares * price
                if price > pos.highest_price:
                    pos.highest_price = price

                # Hard -8% stop & ATR trailing stop (dynamic mode adds profit-protected breakeven stop)
                atr_val = technical.atr_14_raw.loc[dt, code] if code in technical.atr_14_raw.columns else np.nan
                atr_f = float(atr_val) if pd.notna(atr_val) else 0.0
                if cfg.holding_mode == "dynamic":
                    hard_stop, atr_stop, trail_label = compute_dynamic_stop_price(
                        avg_cost=pos.avg_cost,
                        highest_price=pos.highest_price,
                        atr_raw=atr_f,
                        hard_stop_pct=cfg.hard_stop_pct,
                        atr_stop_mult=cfg.atr_stop_mult,
                        dyn_cfg=dyn_cfg,
                    )
                else:
                    hard_stop = pos.avg_cost * (1.0 - cfg.hard_stop_pct)
                    atr_stop = (pos.highest_price - cfg.atr_stop_mult * atr_f) if atr_f > 0 else 0.0
                    trail_label = f"ATR移动止损({cfg.atr_stop_mult}xATR)"
                if price <= hard_stop:
                    pending_sells[code] = (pos.total_shares, f"个股硬止损(-{cfg.hard_stop_pct:.0%})")
                    stopped_today.add(code)
                elif atr_stop > 0 and price <= atr_stop:
                    pending_sells[code] = (pos.total_shares, trail_label)
                    stopped_today.add(code)

            equity = cash + mv
            reg_row = regime_df.loc[dt] if dt in regime_df.index else None
            regime_label = str(reg_row["regime"]) if reg_row is not None else "NEUTRAL"

            fuse_step = step_drawdown_fuse(fuse_state_obj, equity, regime_label, fuse_cfg)
            fuse_state = fuse_step.status
            if fuse_step.liquidate_all:
                for code, pos in positions.items():
                    pending_sells[code] = (pos.total_shares, f"组合回撤熔断({fuse_step.control_drawdown:.1%})")

            target_exposure = float(reg_row["target_exposure"]) if (reg_row is not None and cfg.enable_regime_timing) else 0.90
            target_exposure *= fuse_step.exposure_mult
            style_mode = (
                str(reg_row["style_mode"])
                if (reg_row is not None and cfg.enable_regime_timing and "style_mode" in reg_row.index and pd.notna(reg_row["style_mode"]))
                else "ALL_CAP"
            )
            min_mcap_q = (
                float(reg_row["min_mcap_quantile"])
                if (reg_row is not None and cfg.enable_regime_timing and "min_mcap_quantile" in reg_row.index and pd.notna(reg_row["min_mcap_quantile"]))
                else 0.0
            )

            nav_records.append({
                "date": dt_str,
                "equity": round(equity, 2),
                "cash": round(cash, 2),
                "market_value": round(mv, 2),
                "exposure": round(mv / equity, 4) if equity > 0 else 0.0,
                "positions": len(positions),
                "regime": regime_label,
                "style_mode": style_mode,
                "fuse": fuse_state,
            })
            regime_hist.append({
                "date": dt_str,
                "regime": regime_label,
                "target_exposure": target_exposure,
                "style_mode": style_mode,
                "min_mcap_quantile": min_mcap_q,
            })

            stopped_recently = stopped_today
            if stopped_today:
                last_stop_day_idx = day_idx

            # 4) REBALANCE DECISION AT CLOSE OF DAY t -> QUEUED FOR DAY t+1 OPEN
            should_eval = (
                (day_idx < len(sim_dates) - 1 and fuse_state != "HALTED")
                and (cfg.holding_mode == "dynamic" or day_idx % cfg.rebalance_days == 0)
            )
            if should_eval:
                day_scores = oos_scores.loc[dt]
                day_elig = eligible.loc[dt]
                if cfg.min_buy_price > 0.0:
                    day_elig = day_elig & (panel.close.loc[dt] >= cfg.min_buy_price)
                if min_mcap_q > 0.0 and dt in mcap_pct_rank.index:
                    day_elig = day_elig & (mcap_pct_rank.loc[dt] >= min_mcap_q)
                day_tech = technical.passed.loc[dt] if cfg.enable_technical_filter else pd.Series(True, index=panel.codes)
                held_set = set(positions.keys()) - set(pending_sells.keys())

                sell_reason_map: dict[str, str] = {}
                buy_reason_map: dict[str, str] = {}
                if cfg.holding_mode == "dynamic":
                    # Update consecutive weak-rank streak for every held position
                    valid_day = day_scores.where(day_elig.reindex(day_scores.index).fillna(False)).dropna().sort_values(ascending=False)
                    day_ranks = {c: idx + 1 for idx, c in enumerate(valid_day.index)}
                    held_states: dict[str, HeldPositionState] = {}
                    for code in held_set:
                        pos = positions[code]
                        rk = day_ranks.get(code, 9999)
                        if rk > dyn_cfg.buffer_rank:
                            pos.weak_rank_streak += 1
                        else:
                            pos.weak_rank_streak = 0
                        px_c = float(panel.close.loc[dt, code]) if pd.notna(panel.close.loc[dt, code]) else pos.avg_cost
                        abv = bool(technical.above_ma20.loc[dt, code]) if code in technical.above_ma20.columns else True
                        div = bool(technical.macd_top_div.loc[dt, code]) if code in technical.macd_top_div.columns else False
                        held_states[code] = HeldPositionState(
                            ts_code=code,
                            hold_days=max(0, day_idx - pos.entry_day_idx),
                            avg_cost=pos.avg_cost,
                            current_price=px_c,
                            highest_price=pos.highest_price,
                            above_ma20=abv,
                            macd_top_div=div,
                            weak_rank_streak=pos.weak_rank_streak,
                        )
                    eff_cooldown = 0 if regime_label == "OFFENSIVE" else dyn_cfg.stop_cooldown_days
                    allow_entries = (day_idx - last_stop_day_idx) >= eff_cooldown
                    dyn_res = select_dynamic_portfolio(
                        scores=day_scores,
                        tech_passed=day_tech,
                        eligible=day_elig,
                        held_states=held_states,
                        stopped_out=stopped_recently,
                        port_cfg=port_cfg,
                        dyn_cfg=dyn_cfg,
                        industries=industries,
                        regime=regime_label,
                        target_exposure=target_exposure,
                        allow_new_entries=allow_entries,
                    )
                    selected = dyn_res.selected
                    rank_map = dyn_res.rank_map
                    sell_reason_map = dyn_res.sell_reasons
                    buy_reason_map = dyn_res.buy_reasons
                    trim_tol = dyn_cfg.rebalance_trim_tol
                    add_tol = dyn_cfg.rebalance_add_tol
                else:
                    selected, rank_map = select_portfolio_names(
                        scores=day_scores,
                        tech_passed=day_tech,
                        eligible=day_elig,
                        current_holdings=held_set,
                        stopped_out=stopped_recently,
                        config=port_cfg,
                        industries=industries,
                    )
                    trim_tol = 0.03
                    add_tol = 0.015

                ret_hist = returns_panel.loc[:dt]
                target_weights = solve_portfolio_weights(
                    selected=selected,
                    returns_window=ret_hist,
                    industries=industries,
                    target_exposure=target_exposure,
                    config=port_cfg,
                )

                # Queue sells for positions dropped from target or needing trim
                cur_total_exp = mv / max(equity, 1.0)
                macro_deleveraging = cur_total_exp > target_exposure + 0.05
                eff_trim_tol = 0.01 if macro_deleveraging else trim_tol
                for code, pos in positions.items():
                    if code in pending_sells:
                        continue
                    tw = target_weights.get(code, 0.0)
                    px = float(panel.close.loc[dt, code]) if pd.notna(panel.close.loc[dt, code]) else pos.avg_cost
                    cur_w = (pos.total_shares * px) / max(equity, 1.0)
                    if tw <= 1e-6:
                        rk = rank_map.get(code, 999)
                        reason = sell_reason_map.get(code, f"定期调仓调出(模型排名#{rk})")
                        pending_sells[code] = (pos.total_shares, reason)
                    elif cur_w - tw > eff_trim_tol:
                        trim_shares = round_buy_lots(int(((cur_w - tw) * equity) / max(px, 0.01)))
                        if trim_shares >= LOT_SIZE:
                            tag = "大盘降仓风控减持" if macro_deleveraging else "仓位再平衡降权"
                            pending_sells[code] = (trim_shares, f"{tag}至{tw:.1%}")

                # Queue buys for new or underweight target names
                for code, tw in target_weights.items():
                    if code in pending_sells:
                        continue
                    pos = positions.get(code)
                    px = float(panel.close.loc[dt, code]) if pd.notna(panel.close.loc[dt, code]) else 0.0
                    cur_w = (pos.total_shares * px) / max(equity, 1.0) if (pos and px > 0) else 0.0
                    if tw - cur_w > add_tol:
                        rk = rank_map.get(code, 0)
                        reason = buy_reason_map.get(code, f"定期调仓入选(排名#{rk},目标{tw:.1%})")
                        pending_buys[code] = (tw, reason)

        # Build benchmark-aligned equity & drawdown curves
        nav_df = pd.DataFrame(nav_records)
        nav_series = pd.Series(nav_df["equity"].to_numpy(dtype=float), index=pd.to_datetime(nav_df["date"]))
        bench1000 = panel.index_close.get(BENCHMARK_CSI1000)
        bench300 = panel.index_close.get(BENCHMARK_CSI300)

        metrics, dd_series = compute_backtest_metrics(
            nav_series=nav_series,
            bench_1000=bench1000,
            bench_300=bench300,
            trades=trades,
            total_turnover_notional=total_turnover,
            total_commission=tot_comm,
            total_stamp_duty=tot_stamp,
            total_transfer_fee=tot_transfer,
            total_slippage=tot_slip,
        )
        metrics["avg_hold_days"] = round(float(np.mean(closed_hold_days)), 1) if closed_hold_days else None
        metrics["holding_mode"] = cfg.holding_mode
        metrics["cash_yield_rate"] = float(cfg.cash_yield_rate)
        metrics["total_cash_interest"] = round(float(total_cash_interest), 2)

        # Normalize curves to 1.0000 for charting
        norm_strat = nav_series / nav_series.iloc[0]
        b1000_n = (
            (bench1000.reindex(nav_series.index).ffill().bfill() / bench1000.reindex(nav_series.index).ffill().bfill().iloc[0])
            if (bench1000 is not None and not bench1000.dropna().empty)
            else pd.Series(1.0, index=nav_series.index)
        )
        b300_n = (
            (bench300.reindex(nav_series.index).ffill().bfill() / bench300.reindex(nav_series.index).ffill().bfill().iloc[0])
            if (bench300 is not None and not bench300.dropna().empty)
            else pd.Series(1.0, index=nav_series.index)
        )

        equity_curve = []
        drawdown_curve = []
        for idx_i, row in nav_df.iterrows():
            ts = nav_series.index[idx_i]
            s_val = float(norm_strat.iloc[idx_i])
            b1_val = float(b1000_n.iloc[idx_i])
            b3_val = float(b300_n.iloc[idx_i])
            equity_curve.append({
                **row,
                "nav": round(s_val, 5),
                "csi1000_nav": round(b1_val, 5),
                "csi300_nav": round(b3_val, 5),
                "excess_1000": round(s_val - b1_val, 5),
            })
            drawdown_curve.append({
                "date": row["date"],
                "drawdown": round(float(dd_series.iloc[idx_i]), 5),
            })

        return BacktestResult(
            config=cfg.__dict__.copy(),
            metrics=metrics,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            trades=trades,
            rejected_orders=rejected,
            daily_nav=nav_series,
            regime_history=regime_hist,
        )
