"""Asymmetric Dynamic Holding Policy (Event-Driven, Start-Date Invariant, Zero Look-Ahead).

Shared identically by AShareBacktestEngine and live RecommendationEngine so backtest
and live paper trading stay 100% aligned:
  1. Daily Risk & Profit Lock: hard stop (-8%) + ATR trailing stop run EVERY trading day;
     once a winner reaches +15% unrealized gain, activates breakeven-floored trailing stop.
  2. Minimum Holding Lock (min_hold_days): shields newly entered stocks from short-term
     A-share washout noise (never overrides hard/ATR stop-loss).
  3. Multi-Day Rank-Decay Confirmation (exit_confirm_days): requires a holding to stay
     outside buffer_rank for consecutive trading days before exiting -> eliminates 1-day
     rank flicker while remaining 100% independent of arbitrary fixed rebalance calendars.
  4. Trend Winner Profit Riding: holdings with >= +10% unrealized gain staying above MA20
     (without MACD top divergence) ride the trend indefinitely until trailing stop triggers.
  5. Score-Advantage Hysteresis + Post-Stop Cooldown: avoids fee-wasting marginal swaps
     and pauses falling-knife refills for `stop_cooldown_days` after stop-outs in weak markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.core.constants import DEFAULT_BUFFER_RANK
from app.strategy.optimizer import PortfolioConfig


@dataclass
class HeldPositionState:
    ts_code: str
    hold_days: int
    avg_cost: float
    current_price: float
    highest_price: float
    above_ma20: bool = True
    macd_top_div: bool = False
    weak_rank_streak: int = 0  # Consecutive trading days ranked outside buffer_rank

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_cost <= 0 or self.current_price <= 0:
            return 0.0
        return self.current_price / self.avg_cost - 1.0


@dataclass
class DynamicHoldingConfig:
    min_hold_days: int = 10                # Min trading days before non-stop rank exit
    regime_adaptive_min_hold: bool = True  # OFFENSIVE -> max(5, min_hold_days - 3)
    buffer_rank: int = DEFAULT_BUFFER_RANK  # Normal keep-holding buffer rank (45 for N=15)
    deep_exit_rank: int = 150              # Deep alpha-decay threshold -> exit after confirm
    exit_confirm_days: int = 3             # Consecutive weak-rank days required to confirm exit
    swap_score_gap: float = 0.70           # Min z-score advantage of replacement to trigger mid-zone swap
    profit_ride_pct: float = 0.10          # >= +10% gain above MA20 rides trend (exempt from rank exit)
    profit_protect_pct: float = 0.15       # >= +15% peak gain activates breakeven-floored trailing stop
    profit_atr_mult: float = 2.2           # Trailing ATR stop multiplier once profit-protect activates
    stop_cooldown_days: int = 5            # Cooldown before refilling slots after stop-out in weak markets
    rebalance_trim_tol: float = 0.06       # Avoid micro-trimming existing positions below 6% drift
    rebalance_add_tol: float = 0.04        # Avoid micro-adding below 4% drift


@dataclass
class DynamicSelectionResult:
    selected: list[str]
    rank_map: dict[str, int]
    hold_reasons: dict[str, str] = field(default_factory=dict)
    sell_reasons: dict[str, str] = field(default_factory=dict)
    buy_reasons: dict[str, str] = field(default_factory=dict)


def compute_dynamic_stop_price(
    avg_cost: float,
    highest_price: float,
    atr_raw: float,
    hard_stop_pct: float = 0.08,
    atr_stop_mult: float = 2.5,
    dyn_cfg: DynamicHoldingConfig | None = None,
) -> tuple[float, float, str]:
    """Return (hard_stop_price, trailing_stop_price, trailing_label) using only info up to date t."""
    dcfg = dyn_cfg or DynamicHoldingConfig()
    hard_stop = avg_cost * (1.0 - hard_stop_pct) if avg_cost > 0 else 0.0
    max_gain = (highest_price / avg_cost - 1.0) if (avg_cost > 0 and highest_price > 0) else 0.0

    if max_gain >= dcfg.profit_protect_pct and avg_cost > 0:
        atr_part = (highest_price - dcfg.profit_atr_mult * atr_raw) if (atr_raw > 0 and np.isfinite(atr_raw)) else 0.0
        breakeven_floor = avg_cost * 1.02
        trail_stop = max(breakeven_floor, atr_part)
        return hard_stop, trail_stop, f"浮盈保护移动止盈({dcfg.profit_atr_mult}xATR/保本)"

    trail_stop = (highest_price - atr_stop_mult * atr_raw) if (atr_raw > 0 and np.isfinite(atr_raw)) else 0.0
    return hard_stop, trail_stop, f"ATR移动止损({atr_stop_mult}xATR)"


def select_dynamic_portfolio(
    scores: pd.Series,
    tech_passed: pd.Series,
    eligible: pd.Series,
    held_states: dict[str, HeldPositionState],
    stopped_out: set[str],
    port_cfg: PortfolioConfig,
    dyn_cfg: DynamicHoldingConfig,
    industries: pd.Series,
    regime: str = "NEUTRAL",
    target_exposure: float = 0.55,
    allow_new_entries: bool = True,
    precomputed_valid: pd.Series | None = None,
    precomputed_rank_map: dict[str, int] | None = None,
    industries_dict: dict[str, str] | None = None,
    precomputed_top_pool: list[str] | None = None,
    precomputed_score_z: dict[str, float] | None = None,
) -> DynamicSelectionResult:
    """Asymmetric dynamic holding & event-driven replacement selection (zero look-ahead)."""
    if precomputed_top_pool is not None and precomputed_rank_map is not None and precomputed_score_z is not None:
        if not precomputed_top_pool and not held_states:
            return DynamicSelectionResult(selected=[], rank_map={})
        rank_map = precomputed_rank_map
        score_z = precomputed_score_z
        top_pool = precomputed_top_pool
    else:
        if precomputed_valid is not None:
            valid = precomputed_valid
        else:
            valid = scores.where(eligible.reindex(scores.index).fillna(False)).dropna().sort_values(ascending=False)
        if valid.empty:
            return DynamicSelectionResult(selected=[], rank_map={})

        rank_map = precomputed_rank_map if precomputed_rank_map is not None else {code: idx + 1 for idx, code in enumerate(valid.index)}
        v_arr = valid.to_numpy(dtype=float)
        s_mean = float(np.mean(v_arr))
        s_std = float(np.std(v_arr, ddof=1)) if len(v_arr) > 1 else 1.0
        if s_std < 1e-12:
            s_std = 1.0
        score_z = dict(zip(valid.index.tolist(), (v_arr - s_mean) / s_std))
        top_pool = list(valid.index[: port_cfg.top_candidates])
    ind_lookup = industries_dict if industries_dict is not None else industries.to_dict()

    if target_exposure <= 1e-6:
        sell_reasons = {
            c: "大盘极端防守(目标仓位0%)动态清仓"
            for c in held_states
            if c not in stopped_out
        }
        return DynamicSelectionResult(selected=[], rank_map=rank_map, sell_reasons=sell_reasons)

    eff_min_hold = (
        max(5, dyn_cfg.min_hold_days - 3)
        if (dyn_cfg.regime_adaptive_min_hold and regime == "OFFENSIVE")
        else dyn_cfg.min_hold_days
    )

    max_names_per_ind = max(
        2, int(np.ceil(port_cfg.max_industry_weight / max(port_cfg.max_single_weight, 1e-6))) + 1
    )
    held_set = set(held_states.keys())

    avail_cands = (
        [
            c
            for c in top_pool
            if c not in held_set and c not in stopped_out and bool(tech_passed.get(c, False))
        ]
        if allow_new_entries
        else []
    )
    used_cands: set[str] = set()

    selected: list[str] = []
    ind_counts: dict[str, int] = {}
    hold_reasons: dict[str, str] = {}
    sell_reasons: dict[str, str] = {}
    buy_reasons: dict[str, str] = {}

    def _can_add(code: str) -> bool:
        ind = str(ind_lookup.get(code, "未分类"))
        return ind_counts.get(ind, 0) < max_names_per_ind

    def _add(code: str) -> None:
        selected.append(code)
        ind = str(ind_lookup.get(code, "未分类"))
        ind_counts[ind] = ind_counts.get(ind, 0) + 1

    def _peek_best_cand() -> str | None:
        for c in avail_cands:
            if c not in used_cands and _can_add(c):
                return c
        return None

    # 1) Dynamically evaluate existing holdings (locked & strongest first)
    active_held = [c for c in held_states if c not in stopped_out]
    active_held.sort(key=lambda c: (0 if held_states[c].hold_days < eff_min_hold else 1, rank_map.get(c, 9999)))

    for code in active_held:
        st = held_states[code]
        rk = rank_map.get(code, 9999)
        z_cur = score_z.get(code, -3.0)
        pnl = st.unrealized_pnl_pct

        if len(selected) >= port_cfg.target_holdings:
            sell_reasons[code] = f"动态控仓(持股达上限{port_cfg.target_holdings}，淘汰较弱排名#{rk})"
            continue

        # Rule 1: Minimum holding lock (filters short-term washout noise)
        if st.hold_days < eff_min_hold:
            _add(code)
            hold_reasons[code] = f"最小持有期保护(已持{st.hold_days}/{eff_min_hold}日，排名#{rk})"
            continue

        # Rule 2: Normal top buffer zone
        if rk <= dyn_cfg.buffer_rank:
            _add(code)
            hold_reasons[code] = f"稳居前{dyn_cfg.buffer_rank}缓冲带(模型排名#{rk}，已持{st.hold_days}日)"
            continue

        # Rule 3: Trend winner profit riding (let strong winners above MA20 run!)
        if pnl >= dyn_cfg.profit_ride_pct and st.above_ma20 and (not st.macd_top_div):
            _add(code)
            hold_reasons[code] = f"浮盈趋势保护({pnl:+.1%}且站稳MA20，已持{st.hold_days}日，让利润奔跑)"
            continue

        # Rule 4: Multi-day confirmation shields against 1-day rank flicker
        if st.weak_rank_streak < dyn_cfg.exit_confirm_days:
            _add(code)
            hold_reasons[code] = f"排名波动观察({st.weak_rank_streak}/{dyn_cfg.exit_confirm_days}日，排名#{rk}暂不急卖)"
            continue

        # Rule 5: Confirmed deep alpha decay -> exit immediately
        if rk > dyn_cfg.deep_exit_rank:
            sell_reasons[code] = (
                f"动态淘汰(已持{st.hold_days}日，连续{st.weak_rank_streak}日走弱至#{rk}跌破#{dyn_cfg.deep_exit_rank})"
            )
            continue

        # Rule 6: Mid-zone score-advantage hysteresis (only swap when replacement is clearly superior)
        best_cand = _peek_best_cand()
        if best_cand is not None:
            gap = score_z.get(best_cand, 0.0) - z_cur
            if gap >= dyn_cfg.swap_score_gap:
                sell_reasons[code] = (
                    f"动态优胜劣汰(已持{st.hold_days}日排名#{rk}，新候选{best_cand}优势+{gap:.2f}σ)"
                )
                used_cands.add(best_cand)
                _add(best_cand)
                buy_reasons[best_cand] = (
                    f"动态换仓入选(排名#{rank_map[best_cand]}，较原持股{code}优势+{gap:.2f}σ)"
                )
                continue

        _add(code)
        hold_reasons[code] = f"动态防换手保留(已持{st.hold_days}日排名#{rk}，换仓优势不足{dyn_cfg.swap_score_gap:.2f}σ)"

    # 2) Refill open slots dynamically when entry cooldown allows
    if allow_new_entries:
        for cand in avail_cands:
            if len(selected) >= port_cfg.target_holdings:
                break
            if cand in selected or not _can_add(cand):
                continue
            _add(cand)
            buy_reasons[cand] = f"动态空槽补位(模型排名#{rank_map[cand]}，通过技术面共振)"

    return DynamicSelectionResult(
        selected=selected,
        rank_map=rank_map,
        hold_reasons=hold_reasons,
        sell_reasons=sell_reasons,
        buy_reasons=buy_reasons,
    )
