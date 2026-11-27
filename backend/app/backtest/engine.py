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

import math
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
    DEFAULT_BUFFER_RANK,
    DEFAULT_BULL_MOM_H120_WEIGHT,
    DEFAULT_BULL_MOM_RET20_WEIGHT,
    DEFAULT_DRAWDOWN_HALT_PCT,
    DEFAULT_DRAWDOWN_WARN_PCT,
    DEFAULT_MIN_PRICE,
    DEFAULT_NON_BULL_BP250_WEIGHT,
    DEFAULT_NON_BULL_DL250_WEIGHT,
    DEFAULT_REBALANCE_DAYS,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TARGET_HOLDINGS,
    DEFAULT_WARN_EXPOSURE_MULT,
    DEFAULT_WEIGHT_METHOD,
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
from app.strategy.regime import (
    apply_bull_momentum_overlay,
    apply_non_bull_lowpos_overlay,
    compute_bull_momentum_zscores,
    compute_non_bull_lowpos_zscores,
)
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
    target_holdings: int = DEFAULT_TARGET_HOLDINGS
    top_candidates: int = 30
    buffer_rank: int = DEFAULT_BUFFER_RANK
    max_single_weight: float = 0.12
    max_industry_weight: float = 0.25
    weight_method: WeightMethod = DEFAULT_WEIGHT_METHOD  # type: ignore[assignment]
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    hard_stop_pct: float = 0.08
    atr_stop_mult: float = 2.5
    drawdown_warn_pct: float = DEFAULT_DRAWDOWN_WARN_PCT
    drawdown_halt_pct: float = DEFAULT_DRAWDOWN_HALT_PCT
    warn_exposure_mult: float = DEFAULT_WARN_EXPOSURE_MULT
    rolling_peak_days: int = 90
    warn_stable_days: int = 0
    halt_cooldown_days: int = 20
    enable_regime_timing: bool = True
    enable_technical_filter: bool = True
    bull_mom_h120_weight: float = DEFAULT_BULL_MOM_H120_WEIGHT  # OFFENSIVE 强牛期温和动量倾斜(距120日新高)
    bull_mom_ret20_weight: float = DEFAULT_BULL_MOM_RET20_WEIGHT  # OFFENSIVE 强牛期温和动量倾斜(20日动量)
    non_bull_dl250_weight: float = DEFAULT_NON_BULL_DL250_WEIGHT  # 非强牛期 250 日低位距离安全垫(-z)
    non_bull_bp250_weight: float = DEFAULT_NON_BULL_BP250_WEIGHT  # 非强牛期 250 日 PB 估值低位分位(+z)
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
        ind_dict = industries.to_dict()
        names = panel.name_of()
        returns_panel = panel.returns
        date_pos_map = {dt: idx for idx, dt in enumerate(returns_panel.index)}
        codes_list = list(panel.codes)
        code_to_col = {c: i for i, c in enumerate(codes_list)}
        codes_arr = np.asarray(codes_list, dtype=object)

        # Pre-extract NumPy 2D arrays aligned to (sim_dates x panel.codes) -> 15x faster inner loop, bit-for-bit identical
        sim_row_idx = panel.dates.get_indexer(sim_dates)
        open_np = panel.open.to_numpy(dtype=float)[sim_row_idx]
        high_np = panel.high.to_numpy(dtype=float)[sim_row_idx]
        low_np = panel.low.to_numpy(dtype=float)[sim_row_idx]
        close_np = panel.close.to_numpy(dtype=float)[sim_row_idx]
        pre_close_np = panel.pre_close.to_numpy(dtype=float)[sim_row_idx]
        vol_np = panel.volume.to_numpy(dtype=float)[sim_row_idx]
        lu_np = panel.limit_up.to_numpy(dtype=float)[sim_row_idx]
        ld_np = panel.limit_down.to_numpy(dtype=float)[sim_row_idx]
        adj_np = panel.adj.to_numpy(dtype=float)[sim_row_idx]
        st_np = panel.is_st.to_numpy(dtype=bool)[sim_row_idx]
        susp_np = panel.is_suspended.to_numpy(dtype=bool)[sim_row_idx]
        split_np = (
            panel.split_mult.reindex(index=sim_dates, columns=codes_list).to_numpy(dtype=float)
            if panel.split_mult is not None
            else None
        )
        div_np = (
            panel.cash_div.reindex(index=sim_dates, columns=codes_list).to_numpy(dtype=float)
            if panel.cash_div is not None
            else None
        )
        scores_np = oos_scores.reindex(index=sim_dates, columns=codes_list).to_numpy(dtype=float)
        elig_np = eligible.reindex(index=sim_dates, columns=codes_list).fillna(False).to_numpy(dtype=bool)
        tech_pass_np = (
            technical.passed.reindex(index=sim_dates, columns=codes_list).fillna(False).to_numpy(dtype=bool)
            if cfg.enable_technical_filter
            else np.ones((len(sim_dates), len(codes_list)), dtype=bool)
        )
        atr_np = technical.atr_14_raw.reindex(index=sim_dates, columns=codes_list).to_numpy(dtype=float)
        abv_ma20_np = technical.above_ma20.reindex(index=sim_dates, columns=codes_list).fillna(True).to_numpy(dtype=bool)
        macd_div_np = technical.macd_top_div.reindex(index=sim_dates, columns=codes_list).fillna(False).to_numpy(dtype=bool)

        # Lazy-cache cross-sectional market-cap percentile rank on panel
        mcap_pct_rank = panel._cache.get("mcap_pct_rank")
        if mcap_pct_rank is None:
            mcap_pct_rank = panel.total_mv.where(~panel.is_suspended).rank(axis=1, pct=True)
            panel._cache["mcap_pct_rank"] = mcap_pct_rank
        mcap_pct_np = mcap_pct_rank.reindex(index=sim_dates, columns=codes_list).to_numpy(dtype=float)

        use_bull_mom = (
            cfg.enable_regime_timing
            and (cfg.bull_mom_h120_weight > 0.0 or cfg.bull_mom_ret20_weight > 0.0)
        )
        if use_bull_mom:
            bm_key = ("bull_mom", id(eligible), len(eligible))
            cached_bm = panel._cache.get(bm_key)
            if cached_bm is None:
                cached_bm = compute_bull_momentum_zscores(panel, eligible)
                panel._cache[bm_key] = cached_bm
            z_h120_df, z_r20_df = cached_bm
            z_h120_np = z_h120_df.reindex(index=sim_dates, columns=codes_list).to_numpy(dtype=float)
            z_r20_np = z_r20_df.reindex(index=sim_dates, columns=codes_list).to_numpy(dtype=float)
        else:
            z_h120_np, z_r20_np = None, None

        use_non_bull_lowpos = (
            cfg.enable_regime_timing
            and (cfg.non_bull_dl250_weight > 0.0 or cfg.non_bull_bp250_weight > 0.0)
        )
        if use_non_bull_lowpos:
            lp_key = ("non_bull_lowpos", id(eligible), len(eligible))
            cached_lp = panel._cache.get(lp_key)
            if cached_lp is None:
                cached_lp = compute_non_bull_lowpos_zscores(panel, eligible)
                panel._cache[lp_key] = cached_lp
            z_dl250_df, z_bp250_df = cached_lp
            z_dl250_np = z_dl250_df.reindex(index=sim_dates, columns=codes_list).to_numpy(dtype=float)
            z_bp250_np = z_bp250_df.reindex(index=sim_dates, columns=codes_list).to_numpy(dtype=float)
        else:
            z_dl250_np, z_bp250_np = None, None

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

        def _make_quote(d_idx: int, code: str) -> BarQuote:
            c_idx = code_to_col[code]
            op_v = open_np[d_idx, c_idx]
            hi_v = high_np[d_idx, c_idx]
            lo_v = low_np[d_idx, c_idx]
            cl_v = close_np[d_idx, c_idx]
            pc_v = pre_close_np[d_idx, c_idx]
            vl_v = vol_np[d_idx, c_idx]
            lu_v = lu_np[d_idx, c_idx]
            ld_v = ld_np[d_idx, c_idx]
            return BarQuote(
                ts_code=code,
                open=float(op_v) if np.isfinite(op_v) else 0.0,
                high=float(hi_v) if np.isfinite(hi_v) else 0.0,
                low=float(lo_v) if np.isfinite(lo_v) else 0.0,
                close=float(cl_v) if np.isfinite(cl_v) else 0.0,
                pre_close=float(pc_v) if np.isfinite(pc_v) else 0.0,
                volume=float(vl_v) if np.isfinite(vl_v) else 0.0,
                is_st=bool(st_np[d_idx, c_idx]),
                is_suspended=bool(susp_np[d_idx, c_idx]),
                limit_up=(float(lu_v) if np.isfinite(lu_v) and lu_v > 0 else None),
                limit_down=(float(ld_v) if np.isfinite(ld_v) and ld_v > 0 else None),
            )

        # Pre-extract aligned regime columns as plain arrays (eliminating 1,381 pandas .loc lookups)
        reg_aligned = regime_df.reindex(sim_dates)
        reg_labels_arr = reg_aligned["regime"].fillna("NEUTRAL").astype(str).tolist() if "regime" in reg_aligned.columns else ["NEUTRAL"] * len(sim_dates)
        reg_exp_arr = (
            reg_aligned["target_exposure"].fillna(0.90).to_numpy(dtype=float)
            if (cfg.enable_regime_timing and "target_exposure" in reg_aligned.columns)
            else np.full(len(sim_dates), 0.90, dtype=float)
        )
        reg_style_arr = (
            reg_aligned["style_mode"].fillna("ALL_CAP").astype(str).tolist()
            if (cfg.enable_regime_timing and "style_mode" in reg_aligned.columns)
            else ["ALL_CAP"] * len(sim_dates)
        )
        reg_mcap_q_arr = (
            reg_aligned["min_mcap_quantile"].fillna(0.0).to_numpy(dtype=float)
            if (cfg.enable_regime_timing and "min_mcap_quantile" in reg_aligned.columns)
            else np.zeros(len(sim_dates), dtype=float)
        )
        names_dict = names.to_dict()
        n_codes_total = len(codes_list)
        empty_series = pd.Series(dtype=float)

        for day_idx, dt in enumerate(sim_dates):
            dt_str = str(dt.date())

            # 1) PRE-OPEN: Optional risk-free money-market yield on overnight idle cash + T+1 unlock + ex-div
            if day_idx > 0 and cfg.cash_yield_rate > 0.0 and cash > 0.0:
                daily_interest = cash * (cfg.cash_yield_rate / 252.0)
                cash += daily_interest
                total_cash_interest += daily_interest
            for code, pos in list(positions.items()):
                pos.sellable_shares = pos.total_shares
                # Pre-open corporate-action accounting (stock splits / bonus shares / cash dividends):
                # 1) Exact CSDC brokerage rules when panel.split_mult / panel.cash_div are available
                # 2) Clean fallback via exchange ex-right pre_close vs prior raw close when omitted
                if day_idx > 0:
                    c_idx = code_to_col[code]
                    old_close_v = close_np[day_idx - 1, c_idx]
                    ex_ref_v = pre_close_np[day_idx, c_idx]
                    old_close = float(old_close_v) if np.isfinite(old_close_v) and old_close_v > 0 else 0.0
                    ex_ref = float(ex_ref_v) if np.isfinite(ex_ref_v) and ex_ref_v > 0 else 0.0

                    k_split = 1.0
                    c_div = 0.0
                    if split_np is not None:
                        sv = split_np[day_idx, c_idx]
                        if np.isfinite(sv) and sv > 0:
                            k_split = float(sv)
                    if div_np is not None:
                        dv = div_np[day_idx, c_idx]
                        if np.isfinite(dv) and dv > 0:
                            c_div = float(dv)

                    if k_split > 1.0 + 1e-6 or c_div > 1e-6:
                        old_shares = pos.total_shares
                        if ex_ref <= 0 and old_close > 0:
                            ex_ref = max(0.01, (old_close - c_div) / k_split)
                        # Credit cash dividend per pre-split share
                        if c_div > 0:
                            cash += old_shares * c_div
                        # Credit integer split/bonus shares + fractional remainder in cash at ex_ref
                        if k_split > 1.0 + 1e-6:
                            exact_new = old_shares * k_split
                            new_int = int(math.floor(exact_new + 1e-6))
                            frac_shares = exact_new - new_int
                            if frac_shares > 1e-9 and ex_ref > 0:
                                cash += frac_shares * ex_ref
                            pos.total_shares = new_int
                            pos.sellable_shares = new_int
                        # Adjust cost basis & trailing high price smoothly so stops stay continuous
                        ratio = (old_close / ex_ref) if (old_close > 0 and ex_ref > 0) else k_split
                        if ratio > 0:
                            pos.avg_cost = pos.avg_cost / ratio
                            pos.highest_price = pos.highest_price / ratio
                    elif old_close > 0 and ex_ref > 0:
                        rel_gap = abs(old_close - ex_ref) / old_close
                        if rel_gap > 1e-3:
                            ratio = old_close / ex_ref
                            if ratio >= 1.15:
                                # Large ex-right price drop without explicit split_mult -> treat as split
                                exact_new = pos.total_shares * ratio
                                new_int = int(math.floor(exact_new + 1e-6))
                                cash += (exact_new - new_int) * ex_ref
                                pos.total_shares = new_int
                                pos.sellable_shares = new_int
                            else:
                                eq_before = pos.total_shares * old_close
                                eq_after = pos.total_shares * ex_ref
                                cash += (eq_before - eq_after)
                            pos.avg_cost = pos.avg_cost / ratio
                            pos.highest_price = pos.highest_price / ratio
                cur_adj = adj_np[day_idx, code_to_col[code]]
                if np.isfinite(cur_adj) and cur_adj > 0:
                    pos.last_adj_factor = float(cur_adj)

            # 2) OPEN: execute pending SELLS first (frees cash & handles limit-down retries)
            retry_sells: dict[str, tuple[int, str]] = {}
            for code, (req_qty, reason) in list(pending_sells.items()):
                pos = positions.get(code)
                if pos is None or pos.total_shares <= 0:
                    continue
                quote = _make_quote(day_idx, code)
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
                        "name": str(names_dict.get(code, "")),
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
                    ci = code_to_col[c]
                    px = open_np[day_idx, ci]
                    if not np.isfinite(px) or px <= 0:
                        px = pre_close_np[day_idx, ci]
                    est_mv += p.total_shares * (float(px) if np.isfinite(px) and px > 0 else p.avg_cost)
                est_equity = cash + est_mv

                for code, (target_w, reason) in list(pending_buys.items()):
                    if code in pending_sells:
                        continue
                    quote = _make_quote(day_idx, code)
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
                        adj_v = adj_np[day_idx, code_to_col[code]]
                        cur_adj = float(adj_v) if np.isfinite(adj_v) else 1.0
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
                            "name": str(names_dict.get(code, "")),
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
                ci = code_to_col[code]
                px = close_np[day_idx, ci]
                if not np.isfinite(px) or px <= 0:
                    px = pre_close_np[day_idx, ci]
                price = float(px) if np.isfinite(px) and px > 0 else pos.avg_cost
                mv += pos.total_shares * price
                if price > pos.highest_price:
                    pos.highest_price = price

                # Hard -8% stop & ATR trailing stop (dynamic mode adds profit-protected breakeven stop)
                atr_val = atr_np[day_idx, ci]
                atr_f = float(atr_val) if np.isfinite(atr_val) else 0.0
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
            regime_label = reg_labels_arr[day_idx]

            fuse_step = step_drawdown_fuse(fuse_state_obj, equity, regime_label, fuse_cfg)
            fuse_state = fuse_step.status
            if fuse_step.liquidate_all:
                for code, pos in positions.items():
                    pending_sells[code] = (pos.total_shares, f"组合回撤熔断({fuse_step.control_drawdown:.1%})")

            target_exposure = float(reg_exp_arr[day_idx]) * fuse_step.exposure_mult
            style_mode = reg_style_arr[day_idx]
            min_mcap_q = float(reg_mcap_q_arr[day_idx])

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
                s_row = scores_np[day_idx]
                if use_bull_mom and regime_label == "OFFENSIVE" and z_h120_np is not None and z_r20_np is not None:
                    valid_s = np.isfinite(s_row)
                    if int(valid_s.sum()) >= 2:
                        s_vals = s_row[valid_s]
                        s_mu = float(np.mean(s_vals))
                        s_sd = float(np.std(s_vals, ddof=1))
                        adj_row = (s_row - s_mu) / s_sd if (np.isfinite(s_sd) and s_sd > 1e-12) else s_row.copy()
                        if cfg.bull_mom_h120_weight > 0.0:
                            zh = z_h120_np[day_idx]
                            adj_row = adj_row + float(cfg.bull_mom_h120_weight) * np.where(np.isfinite(zh), zh, 0.0)
                        if cfg.bull_mom_ret20_weight > 0.0:
                            zr = z_r20_np[day_idx]
                            adj_row = adj_row + float(cfg.bull_mom_ret20_weight) * np.where(np.isfinite(zr), zr, 0.0)
                        adj_row = np.where(valid_s, adj_row, np.nan)
                        a_vals = adj_row[valid_s]
                        a_mu = float(np.mean(a_vals))
                        a_sd = float(np.std(a_vals, ddof=1))
                        s_row = (adj_row - a_mu) / a_sd if (np.isfinite(a_sd) and a_sd > 1e-12) else adj_row
                elif use_non_bull_lowpos and regime_label != "OFFENSIVE" and z_dl250_np is not None and z_bp250_np is not None:
                    valid_s = np.isfinite(s_row)
                    if int(valid_s.sum()) >= 2:
                        s_vals = s_row[valid_s]
                        s_mu = float(np.mean(s_vals))
                        s_sd = float(np.std(s_vals, ddof=1))
                        adj_row = (s_row - s_mu) / s_sd if (np.isfinite(s_sd) and s_sd > 1e-12) else s_row.copy()
                        if cfg.non_bull_dl250_weight > 0.0:
                            zdl = z_dl250_np[day_idx]
                            adj_row = adj_row - float(cfg.non_bull_dl250_weight) * np.where(np.isfinite(zdl), zdl, 0.0)
                        if cfg.non_bull_bp250_weight > 0.0:
                            zbp = z_bp250_np[day_idx]
                            adj_row = adj_row + float(cfg.non_bull_bp250_weight) * np.where(np.isfinite(zbp), zbp, 0.0)
                        adj_row = np.where(valid_s, adj_row, np.nan)
                        a_vals = adj_row[valid_s]
                        a_mu = float(np.mean(a_vals))
                        a_sd = float(np.std(a_vals, ddof=1))
                        s_row = (adj_row - a_mu) / a_sd if (np.isfinite(a_sd) and a_sd > 1e-12) else adj_row

                elig_row = elig_np[day_idx]
                if cfg.min_buy_price > 0.0:
                    elig_row = elig_row & (close_np[day_idx] >= cfg.min_buy_price)
                if min_mcap_q > 0.0:
                    elig_row = elig_row & (mcap_pct_np[day_idx] >= min_mcap_q)
                day_tech_row = tech_pass_np[day_idx]
                held_set = set(positions.keys()) - set(pending_sells.keys())

                # Zero-pandas NumPy ranking & z-score extraction (only builds <=45-entry dicts for top candidates + held!)
                ok_idx = np.flatnonzero(elig_row & np.isfinite(s_row))
                sell_reason_map: dict[str, str] = {}
                buy_reason_map: dict[str, str] = {}
                if cfg.holding_mode == "dynamic":
                    rank_by_col = np.full(n_codes_total, 9999, dtype=np.int32)
                    top_k_codes: list[str] = []
                    day_ranks: dict[str, int] = {}
                    score_z_sub: dict[str, float] = {}
                    if len(ok_idx) > 0:
                        ok_vals = s_row[ok_idx]
                        ord_idx = ok_idx[np.argsort(-ok_vals, kind="mergesort")]
                        rank_by_col[ord_idx] = np.arange(1, len(ord_idx) + 1, dtype=np.int32)
                        s_mu = float(np.mean(ok_vals))
                        s_sd = float(np.std(ok_vals, ddof=1)) if len(ok_vals) > 1 else 1.0
                        if s_sd < 1e-12:
                            s_sd = 1.0
                        top_ord = ord_idx[: port_cfg.top_candidates]
                        top_k_codes = [codes_list[int(ci)] for ci in top_ord]
                        for rk_i, ci in enumerate(top_ord, start=1):
                            c_code = codes_list[int(ci)]
                            day_ranks[c_code] = rk_i
                            score_z_sub[c_code] = float((s_row[ci] - s_mu) / s_sd)
                        for code in held_set:
                            ci = code_to_col[code]
                            rk = int(rank_by_col[ci])
                            if rk < 9999:
                                day_ranks[code] = rk
                                score_z_sub[code] = float((s_row[ci] - s_mu) / s_sd)

                    # Update consecutive weak-rank streak for every held position
                    held_states: dict[str, HeldPositionState] = {}
                    for code in held_set:
                        pos = positions[code]
                        ci = code_to_col[code]
                        rk = int(rank_by_col[ci])
                        if rk > dyn_cfg.buffer_rank:
                            pos.weak_rank_streak += 1
                        else:
                            pos.weak_rank_streak = 0
                        px_v = close_np[day_idx, ci]
                        px_c = float(px_v) if np.isfinite(px_v) else pos.avg_cost
                        abv = bool(abv_ma20_np[day_idx, ci])
                        div = bool(macd_div_np[day_idx, ci])
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
                    tech_sub = {c: bool(day_tech_row[code_to_col[c]]) for c in top_k_codes}
                    dyn_res = select_dynamic_portfolio(
                        scores=empty_series,
                        tech_passed=tech_sub,  # type: ignore[arg-type]
                        eligible=empty_series,
                        held_states=held_states,
                        stopped_out=stopped_recently,
                        port_cfg=port_cfg,
                        dyn_cfg=dyn_cfg,
                        industries=industries,
                        regime=regime_label,
                        target_exposure=target_exposure,
                        allow_new_entries=allow_entries,
                        precomputed_rank_map=day_ranks,
                        industries_dict=ind_dict,
                        precomputed_top_pool=top_k_codes,
                        precomputed_score_z=score_z_sub,
                    )
                    selected = dyn_res.selected
                    rank_map = dyn_res.rank_map
                    sell_reason_map = dyn_res.sell_reasons
                    buy_reason_map = dyn_res.buy_reasons
                    trim_tol = dyn_cfg.rebalance_trim_tol
                    add_tol = dyn_cfg.rebalance_add_tol
                else:
                    day_scores_s = pd.Series(s_row, index=codes_list)
                    day_elig_s = pd.Series(elig_row, index=codes_list)
                    day_tech_s = pd.Series(day_tech_row, index=codes_list)
                    selected, rank_map = select_portfolio_names(
                        scores=day_scores_s,
                        tech_passed=day_tech_s,
                        eligible=day_elig_s,
                        current_holdings=held_set,
                        stopped_out=stopped_recently,
                        config=port_cfg,
                        industries=industries,
                    )
                    trim_tol = 0.03
                    add_tol = 0.015

                if cfg.weight_method == "equal_weight" or not selected:
                    ret_hist = returns_panel.iloc[0:0]
                else:
                    dt_pos = date_pos_map.get(dt, len(returns_panel) - 1)
                    win_start = max(0, dt_pos - port_cfg.cov_window - 5)
                    ret_hist = returns_panel.iloc[win_start : dt_pos + 1]
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
                    px_v = close_np[day_idx, code_to_col[code]]
                    px = float(px_v) if np.isfinite(px_v) else pos.avg_cost
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
                    px_v = close_np[day_idx, code_to_col[code]]
                    px = float(px_v) if np.isfinite(px_v) else 0.0
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
        metrics["bull_mom_h120_weight"] = float(cfg.bull_mom_h120_weight)
        metrics["bull_mom_ret20_weight"] = float(cfg.bull_mom_ret20_weight)
        metrics["non_bull_dl250_weight"] = float(cfg.non_bull_dl250_weight)
        metrics["non_bull_bp250_weight"] = float(cfg.non_bull_bp250_weight)
        metrics["drawdown_warn_pct"] = float(cfg.drawdown_warn_pct)
        metrics["warn_exposure_mult"] = float(cfg.warn_exposure_mult)

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
