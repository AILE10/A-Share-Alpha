"""Unified portfolio drawdown circuit breaker with anti-cash-trap recovery.

Shared by BOTH AShareBacktestEngine (historical simulation) and PaperTradingService /
RiskMiddleware (live paper & broker execution) so backtest and live risk controls remain
100% aligned.

Guardrail guarantees:
  - Hard -12% drawdown redline (HALTED -> liquidate / block BUY + cooldown) is NEVER relaxed.
  - -8% drawdown warning (WARNING -> scale down target exposure & single-stock cap) triggers immediately.
  - Anti-trap recovery prevents an isolated extreme spike followed by a -8%~-12% pullback
    from permanently locking exposure at half-speed for 12+ months during a subsequent bull market:
      1) Rolling control-loop high-watermark (default 120 trading days, ~6 months) ages out ancient spikes.
      2) Stabilization release: after `warn_stable_days` consecutive non-DEFENSIVE days without a
         new equity trough AND with equity >= trailing MA, gracefully re-anchor the control peak and
         exit WARNING back to NORMAL. During an active bear crash (DEFENSIVE or making new lows),
         WARNING remains firmly locked.
  - NOTE: Reported backtest `max_drawdown` is ALWAYS computed on the true all-time equity curve,
    completely independent of the control-loop re-anchoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DrawdownFuseConfig:
    warn_pct: float = 0.08               # -8% triggers WARNING
    halt_pct: float = 0.12               # -12% triggers HALTED (liquidate & block BUY)
    warn_exposure_mult: float = 0.50     # Halve target exposure & single-stock cap during WARNING (matches RiskMiddleware)
    warn_release_ratio: float = 0.60     # Exit WARNING when rolling control_dd > -warn_pct * 0.60 (-4.8%)
    halt_cooldown_days: int = 20         # Minimum trading days in HALTED before resuming
    rolling_peak_days: int = 120         # Trailing 120-day (~6m) high-watermark window (0 = legacy all-time peak)
    warn_stable_days: int = 0            # Optional stabilization release (0 = rely cleanly on rolling_peak_days)
    warn_ma_days: int = 20               # Trailing MA window when warn_stable_days > 0


def compute_rolling_peak(
    daily_equities: list[float],
    current_equity: float,
    stored_peak: float,
    window_days: int = 120,
) -> float:
    """Compute control-loop rolling peak equity shared with PaperBrokerAdapter.

    When fewer than `window_days` daily NAV snapshots exist, the rolling peak equals
    the all-time peak (`max(stored_peak, current_equity)`). Once >= `window_days` snapshots
    exist, peaks older than `window_days` trading days age out cleanly so a single past spike
    cannot trap portfolio exposure permanently.
    """
    cur = max(float(current_equity), 1.0)
    if window_days <= 0 or len(daily_equities) < window_days:
        return max(float(stored_peak or 0.0), cur)
    recent = daily_equities[-window_days:]
    return max(max(float(x) for x in recent), cur)


@dataclass
class DrawdownFuseState:
    status: str = "NORMAL"               # NORMAL / WARNING / HALTED
    peak_equity: float = 0.0             # Control-loop peak equity
    halt_days_elapsed: int = 0
    warn_trough: float = float("inf")
    warn_stable_count: int = 0
    days_since_reanchor: int = 0
    equity_history: list[float] = field(default_factory=list)


@dataclass
class DrawdownFuseStep:
    status: str                          # NORMAL / WARNING / HALTED
    control_peak: float
    control_drawdown: float
    exposure_mult: float                 # 1.0 / warn_exposure_mult / 0.0
    liquidate_all: bool = False
    event: str = ""                      # Optional human-readable transition note


def step_drawdown_fuse(
    state: DrawdownFuseState,
    equity: float,
    regime: str,
    cfg: DrawdownFuseConfig | None = None,
) -> DrawdownFuseStep:
    """Advance the drawdown fuse state machine by one trading day (zero look-ahead)."""
    c = cfg or DrawdownFuseConfig()
    eq = max(float(equity), 1.0)
    state.equity_history.append(eq)
    max_keep = max(c.rolling_peak_days, c.warn_ma_days, 120) + 5
    if len(state.equity_history) > max_keep:
        state.equity_history = state.equity_history[-max_keep:]

    state.days_since_reanchor += 1

    # Compute control-loop high watermark (either rolling window capped by last re-anchor, or legacy all-time)
    if c.rolling_peak_days > 0:
        win_len = min(c.rolling_peak_days, state.days_since_reanchor, len(state.equity_history))
        control_peak = max(state.equity_history[-win_len:])
        state.peak_equity = control_peak
    else:
        if eq > state.peak_equity:
            state.peak_equity = eq
        control_peak = max(state.peak_equity, eq)

    cur_dd = eq / control_peak - 1.0 if control_peak > 0 else 0.0
    liquidate = False
    event = ""

    if state.status == "HALTED":
        state.halt_days_elapsed += 1
        if state.halt_days_elapsed >= c.halt_cooldown_days and regime != "DEFENSIVE":
            state.status = "WARNING"
            state.peak_equity = eq
            state.days_since_reanchor = 1
            state.halt_days_elapsed = 0
            state.warn_trough = eq
            state.warn_stable_count = 0
            control_peak = eq
            cur_dd = 0.0
            event = "HALT_COOLDOWN_RESUME_WARNING"
    elif cur_dd <= -c.halt_pct:
        state.status = "HALTED"
        state.halt_days_elapsed = 0
        state.warn_stable_count = 0
        liquidate = True
        event = f"HALT_TRIGGERED({cur_dd:.1%})"
    elif state.status == "WARNING":
        if eq < state.warn_trough:
            state.warn_trough = eq
            state.warn_stable_count = 0
        elif regime != "DEFENSIVE":
            state.warn_stable_count += 1
        else:
            # Bear market DEFENSIVE regime: do not accumulate stabilization credit
            state.warn_stable_count = 0

        ma_len = min(max(c.warn_ma_days, 1), len(state.equity_history))
        eq_ma = sum(state.equity_history[-ma_len:]) / ma_len

        classic_recovered = cur_dd > -c.warn_pct * c.warn_release_ratio
        stable_released = (
            c.warn_stable_days > 0
            and state.warn_stable_count >= c.warn_stable_days
            and regime != "DEFENSIVE"
            and eq >= eq_ma
        )
        if classic_recovered:
            state.status = "NORMAL"
            state.warn_stable_count = 0
            event = "WARNING_RECOVERED_CLASSIC"
        elif stable_released:
            state.status = "NORMAL"
            state.peak_equity = eq
            state.days_since_reanchor = 1
            state.warn_stable_count = 0
            control_peak = eq
            cur_dd = 0.0
            event = "WARNING_RELEASED_STABILIZED"
    else:
        # NORMAL
        if cur_dd <= -c.warn_pct:
            state.status = "WARNING"
            state.warn_trough = eq
            state.warn_stable_count = 0
            event = f"WARNING_TRIGGERED({cur_dd:.1%})"

    if state.status == "HALTED":
        mult = 0.0
    elif state.status == "WARNING":
        mult = float(c.warn_exposure_mult)
    else:
        mult = 1.0

    return DrawdownFuseStep(
        status=state.status,
        control_peak=control_peak,
        control_drawdown=cur_dd,
        exposure_mult=mult,
        liquidate_all=liquidate,
        event=event,
    )
