"""Portfolio construction: Top-30 resonance selection, Top-25 buffer zone, Risk Parity vs Equal Weight,
and hard constraints (single stock <= 12%, Shenwan L1 industry <= 25%, total <= Regime exposure)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

WeightMethod = Literal["risk_parity", "equal_weight"]


@dataclass
class PortfolioConfig:
    top_candidates: int = 30
    target_holdings: int = 10            # 8~12 只持股
    buffer_rank: int = 25                # 已持仓若仍在模型排名前 25 则保留
    max_single_weight: float = 0.12      # 单票上限 12%
    max_industry_weight: float = 0.25    # 单申万一级行业上限 25%
    weight_method: WeightMethod = "risk_parity"
    cov_window: int = 60


def select_portfolio_names(
    scores: pd.Series,
    tech_passed: pd.Series,
    eligible: pd.Series,
    current_holdings: set[str],
    stopped_out: set[str],
    config: PortfolioConfig,
    industries: pd.Series,
) -> tuple[list[str], dict[str, int]]:
    """Select 8~12 target holdings respecting Top-30 resonance filter, Top-25 buffer zone & industry diversity.

    Returns (selected_codes, rank_map) where rank_map maps ts_code -> 1-based model composite rank.
    """
    valid = scores.where(eligible.reindex(scores.index).fillna(False)).dropna().sort_values(ascending=False)
    if valid.empty:
        return [], {}
    rank_map = {code: idx + 1 for idx, code in enumerate(valid.index)}
    top30 = list(valid.index[: config.top_candidates])

    # Max names allowed from one industry before weight solver saturates the 25% cap
    max_names_per_ind = max(2, int(np.ceil(config.max_industry_weight / max(config.max_single_weight, 1e-6))) + 1)

    selected: list[str] = []
    ind_counts: dict[str, int] = {}

    def _can_add(code: str) -> bool:
        ind = str(industries.get(code, "未分类"))
        return ind_counts.get(ind, 0) < max_names_per_ind

    def _add(code: str) -> None:
        selected.append(code)
        ind = str(industries.get(code, "未分类"))
        ind_counts[ind] = ind_counts.get(ind, 0) + 1

    # 1. Buffer Zone: retain existing holdings ranked <= buffer_rank and not stopped out
    held_ranked = sorted(
        [c for c in current_holdings if c in rank_map and rank_map[c] <= config.buffer_rank and c not in stopped_out],
        key=lambda c: rank_map[c],
    )
    for code in held_ranked:
        if len(selected) >= config.target_holdings:
            break
        if _can_add(code):
            _add(code)

    # 2. Fill remaining slots from Top-30 that pass Technical Resonance filter
    for code in top30:
        if len(selected) >= config.target_holdings:
            break
        if code in selected or code in stopped_out:
            continue
        if bool(tech_passed.get(code, False)) and _can_add(code):
            _add(code)

    # 3. If strict technical resonance yielded fewer than target_holdings in a choppy market,
    #    leave remaining slots unallocated (or hold cash) so we never buy overheated / diverging stocks.
    return selected, rank_map


def _project_constraints(
    raw_weights: np.ndarray,
    industries: list[str],
    target_exposure: float,
    max_single: float,
    max_industry: float,
) -> np.ndarray:
    """Iteratively project non-negative weights onto single-stock, industry, and total exposure caps."""
    n = len(raw_weights)
    if n == 0 or target_exposure <= 1e-9:
        return np.zeros(n, dtype=float)

    w = np.clip(np.asarray(raw_weights, dtype=float), 0.0, None)
    if w.sum() <= 1e-12:
        w = np.ones(n, dtype=float)
    w = w / w.sum() * target_exposure

    ind_arr = np.asarray(industries, dtype=object)
    unique_inds = np.unique(ind_arr)

    for _ in range(40):
        # Cap individual names
        w = np.minimum(w, max_single)
        # Cap industry aggregates proportionally
        for ind in unique_inds:
            mask = ind_arr == ind
            s = w[mask].sum()
            if s > max_industry + 1e-12:
                w[mask] *= max_industry / s
        # Redistribute slack to uncapped names & uncapped industries
        total = w.sum()
        slack = target_exposure - total
        if abs(slack) <= 1e-7:
            break
        if slack < 0:
            w *= target_exposure / total
            continue
        ind_sums = {ind: w[ind_arr == ind].sum() for ind in unique_inds}
        free = np.array(
            [
                (w[i] < max_single - 1e-7) and (ind_sums[ind_arr[i]] < max_industry - 1e-7)
                for i in range(n)
            ],
            dtype=bool,
        )
        if not free.any():
            break
        # Proportional top-up bounded by remaining single & industry headroom
        headroom = np.array(
            [min(max_single - w[i], max_industry - ind_sums[ind_arr[i]]) for i in range(n)],
            dtype=float,
        )
        headroom = np.where(free, np.maximum(headroom, 0.0), 0.0)
        if headroom.sum() <= 1e-12:
            break
        scale = min(1.0, slack / headroom.sum())
        w += headroom * scale

    # Final strict safety clamp
    w = np.minimum(w, max_single)
    for ind in unique_inds:
        mask = ind_arr == ind
        s = w[mask].sum()
        if s > max_industry:
            w[mask] *= max_industry / s
    if w.sum() > target_exposure + 1e-9:
        w *= target_exposure / w.sum()
    return w


def solve_portfolio_weights(
    selected: list[str],
    returns_window: pd.DataFrame,
    industries: pd.Series,
    target_exposure: float,
    config: PortfolioConfig,
) -> dict[str, float]:
    """Compute Risk Parity or Equal Weight allocations strictly satisfying 12% / 25% / exposure caps."""
    n = len(selected)
    if n == 0 or target_exposure <= 1e-6:
        return {}

    ind_list = [str(industries.get(c, "未分类")) for c in selected]

    if config.weight_method == "equal_weight" or n == 1:
        raw = np.ones(n, dtype=float) / n
        w = _project_constraints(raw, ind_list, target_exposure, config.max_single_weight, config.max_industry_weight)
        return {code: round(float(wt), 6) for code, wt in zip(selected, w) if wt > 1e-5}

    # Risk Parity on Ledoit-Wolf shrinkage covariance matrix
    ret = returns_window[selected].tail(config.cov_window).dropna(how="all").fillna(0.0)
    if len(ret) < 15:
        vol = ret.std().replace(0, np.nan).fillna(0.02).to_numpy(dtype=float)
        raw = 1.0 / np.maximum(vol, 1e-4)
        w = _project_constraints(raw, ind_list, target_exposure, config.max_single_weight, config.max_industry_weight)
        return {code: round(float(wt), 6) for code, wt in zip(selected, w) if wt > 1e-5}

    try:
        cov = LedoitWolf().fit(ret.to_numpy(dtype=float)).covariance_
        cov = cov + np.eye(n) * 1e-8
    except Exception:  # noqa: BLE001
        cov = np.cov(ret.to_numpy(dtype=float), rowvar=False) + np.eye(n) * 1e-6

    inv_vol = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-8))
    w0 = inv_vol / inv_vol.sum()

    def rc_objective(w: np.ndarray) -> float:
        port_var = float(w @ cov @ w)
        if port_var <= 1e-16:
            return 0.0
        mrc = cov @ w
        rc = w * mrc / port_var
        target_rc = 1.0 / n
        return float(np.sum((rc - target_rc) ** 2))

    # Solve normalized simplex risk-parity first, then project cleanly to exact caps + target exposure
    res = minimize(
        rc_objective,
        w0,
        method="SLSQP",
        bounds=[(1e-4, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    base_w = res.x if res.success and np.all(np.isfinite(res.x)) else w0
    w = _project_constraints(base_w, ind_list, target_exposure, config.max_single_weight, config.max_industry_weight)
    return {code: round(float(wt), 6) for code, wt in zip(selected, w) if wt > 1e-5}
