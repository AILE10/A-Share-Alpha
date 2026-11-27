"""Single-factor analytics: IC / RankIC / ICIR, quintile layered backtest, correlation and decay."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.core.constants import FACTOR_META_MAP


def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank().to_numpy()


def cross_ic(x: np.ndarray, y: np.ndarray, min_n: int = 10) -> tuple[float, float, int]:
    mask = ~(np.isnan(x) | np.isnan(y))
    n = int(mask.sum())
    if n < min_n:
        return (np.nan, np.nan, n)
    xs, ys = x[mask], y[mask]
    if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return (np.nan, np.nan, n)
    ic = float(np.corrcoef(xs, ys)[0, 1])
    ric = float(np.corrcoef(_rank(xs), _rank(ys))[0, 1])
    return (ic, ric, n)


@dataclass
class FactorAnalysis:
    ic_daily: pd.DataFrame                      # long: trade_date, factor_name, ic, rank_ic, sample_size, q1..q5, ls
    summary: pd.DataFrame                       # one row per factor
    quintile_nav: dict[str, pd.DataFrame]       # factor -> date x [Q1..Q5, LS]
    correlation: pd.DataFrame                   # factor x factor mean cross-sectional Spearman
    meta: dict = field(default_factory=dict)


def quintile_returns(score: np.ndarray, fwd: np.ndarray, n_groups: int = 5, min_n: int = 15) -> list[float]:
    mask = ~(np.isnan(score) | np.isnan(fwd))
    n = int(mask.sum())
    if n < min_n:
        return [np.nan] * n_groups
    s, r = score[mask], fwd[mask]
    # Stable mergesort reproduces pd.Series.rank(method="first") identically without Series overhead
    order = np.argsort(s, kind="mergesort")
    r_sorted = r[order]
    ranks = np.arange(1, n + 1, dtype=float)
    groups = np.clip(np.ceil(ranks / n * n_groups).astype(int), 1, n_groups)
    return [float(r_sorted[groups == g].mean()) if np.any(groups == g) else np.nan for g in range(1, n_groups + 1)]


def _vectorized_row_pearson(
    a: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray,
    ns: np.ndarray,
    min_n: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact per-row Pearson correlation & non-constant validity mask in vectorized NumPy."""
    safe_n = np.maximum(ns, 1).astype(float)
    a0 = np.where(valid, a, 0.0)
    b0 = np.where(valid, b, 0.0)
    ma = a0.sum(axis=1, keepdims=True) / safe_n[:, None]
    mb = b0.sum(axis=1, keepdims=True) / safe_n[:, None]
    da = np.where(valid, a - ma, 0.0)
    db = np.where(valid, b - mb, 0.0)
    va = (da * da).sum(axis=1) / safe_n
    vb = (db * db).sum(axis=1) / safe_n
    ok = (ns >= min_n) & (np.sqrt(np.maximum(va, 0.0)) >= 1e-12) & (np.sqrt(np.maximum(vb, 0.0)) >= 1e-12)
    cov = (da * db).sum(axis=1) / safe_n
    denom = np.sqrt(np.maximum(va * vb, 0.0))
    corr = np.where(ok & (denom > 0.0), cov / np.where(denom > 0.0, denom, 1.0), np.nan)
    return np.clip(corr, -1.0, 1.0), ok


def _vectorized_cross_ic(
    arr: np.ndarray,
    fwd_arr: np.ndarray,
    valid_label_dates: np.ndarray,
    min_n: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized cross-sectional Pearson IC & Spearman RankIC matching cross_ic() identically."""
    valid = (~np.isnan(arr)) & (~np.isnan(fwd_arr)) & valid_label_dates[:, None]
    ns = valid.sum(axis=1).astype(int)
    ics, ok_raw = _vectorized_row_pearson(arr, fwd_arr, valid, ns, min_n=min_n)

    # Rank only non-empty valid rows via Cython DataFrame.rank(axis=1)
    rics = np.full(len(ns), np.nan, dtype=float)
    if np.any(ok_raw):
        sub_idx = np.flatnonzero(ok_raw)
        v_sub = valid[sub_idx]
        ra = pd.DataFrame(np.where(v_sub, arr[sub_idx], np.nan)).rank(axis=1, method="average").to_numpy(dtype=float)
        rb = pd.DataFrame(np.where(v_sub, fwd_arr[sub_idx], np.nan)).rank(axis=1, method="average").to_numpy(dtype=float)
        r_sub, _ = _vectorized_row_pearson(ra, rb, v_sub, ns[sub_idx], min_n=min_n)
        rics[sub_idx] = r_sub
    return ics, rics, ns


def analyze_factors(
    std: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    eligible: pd.DataFrame,
    horizon: int = 5,
    decay_window: int = 20,
    status: dict[str, str] | None = None,
) -> FactorAnalysis:
    dates = eligible.index
    fwd = fwd_ret.reindex_like(eligible).where(eligible)
    fwd_arr = fwd.to_numpy(dtype=float)
    valid_label_dates = ~np.isnan(fwd_arr).all(axis=1)
    rebalance_idx = np.arange(0, len(dates), horizon)

    # Align and mask each standardized factor matrix ONCE (reused for IC, quintiles, and correlation)
    aligned_arrs: dict[str, np.ndarray] = {
        name: df.reindex_like(eligible).where(eligible).to_numpy(dtype=float)
        for name, df in std.items()
    }

    rows = []
    navs: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for name, arr in aligned_arrs.items():
        ics, rics, ns = _vectorized_cross_ic(arr, fwd_arr, valid_label_dates)
        qrets = np.full((len(dates), 5), np.nan)
        for i in rebalance_idx:
            if valid_label_dates[i]:
                qrets[i] = quintile_returns(arr[i], fwd_arr[i])
        ls = qrets[:, 4] - qrets[:, 0]
        part = pd.DataFrame({
            "trade_date": dates.date, "factor_name": name, "horizon": horizon,
            "ic": ics, "rank_ic": rics, "sample_size": ns,
            "q1_ret": qrets[:, 0], "q2_ret": qrets[:, 1], "q3_ret": qrets[:, 2],
            "q4_ret": qrets[:, 3], "q5_ret": qrets[:, 4], "ls_ret": ls,
        })
        rows.append(part)

        # Quintile NAV on non-overlapping rebalance dates
        q_df = pd.DataFrame(qrets[rebalance_idx], index=dates[rebalance_idx], columns=["Q1", "Q2", "Q3", "Q4", "Q5"])
        q_df["LS"] = q_df["Q5"] - q_df["Q1"]
        q_df = q_df.dropna(how="all")
        nav = (1.0 + q_df.fillna(0.0)).cumprod()
        navs[name] = nav

        ic_s = pd.Series(ics, index=dates)
        ric_s = pd.Series(rics, index=dates)
        # Non-overlapping sample for unbiased ICIR / t-stat
        ic_nov = ic_s.iloc[rebalance_idx].dropna()
        ric_nov = ric_s.iloc[rebalance_idx].dropna()
        ic_mean = float(ic_nov.mean()) if len(ic_nov) else np.nan
        ric_mean = float(ric_nov.mean()) if len(ric_nov) else np.nan
        ic_std = float(ic_nov.std(ddof=1)) if len(ic_nov) > 1 else np.nan
        ric_std = float(ric_nov.std(ddof=1)) if len(ric_nov) > 1 else np.nan
        periods_per_year = 252.0 / horizon
        icir = ic_mean / ic_std if ic_std and ic_std > 1e-12 else np.nan
        ricir = ric_mean / ric_std if ric_std and ric_std > 1e-12 else np.nan
        t_stat = ricir * np.sqrt(len(ric_nov)) if not np.isnan(ricir) else np.nan
        win = float((ric_nov > 0).mean()) if len(ric_nov) else np.nan
        recent = ric_s.dropna().iloc[-decay_window:]
        recent_mean = float(recent.mean()) if len(recent) else np.nan
        if np.isnan(ric_mean) or np.isnan(recent_mean):
            decay = "INSUFFICIENT"
        elif np.sign(recent_mean) != np.sign(ric_mean) and abs(ric_mean) > 0.01:
            decay = "REVERSED"
        elif abs(recent_mean) < 0.5 * abs(ric_mean):
            decay = "DECAYING"
        else:
            decay = "STABLE"
        ls_series = q_df["LS"].dropna()
        ls_ann = float((1 + ls_series).prod() ** (periods_per_year / max(1, len(ls_series))) - 1) if len(ls_series) else np.nan
        q_means = q_df[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean()
        mono = float(pd.Series(q_means.to_numpy()).corr(pd.Series(np.arange(5)), method="spearman")) if q_means.notna().all() else np.nan
        meta = FACTOR_META_MAP.get(name, {})
        summary_rows.append({
            "factor_name": name,
            "label": meta.get("label", name),
            "dimension": meta.get("dimension", ""),
            "expected_direction": meta.get("direction", 1),
            "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
            "rank_ic_mean": ric_mean, "rank_ic_std": ric_std, "rank_icir": ricir,
            "rank_ic_ann_icir": ricir * np.sqrt(periods_per_year) if not np.isnan(ricir) else np.nan,
            "t_stat": t_stat, "ic_win_rate": win,
            "recent_rank_ic_20d": recent_mean, "decay_status": decay,
            "ls_annual_return": ls_ann, "quintile_monotonicity": mono,
            "n_periods": int(len(ric_nov)),
            "avg_sample_size": float(np.nanmean(ns)) if len(ns) else 0.0,
            "data_status": (status or {}).get(name, "OK"),
        })

    # Mean cross-sectional Spearman correlation matrix (per-pair valid counts so partial factors stay exact)
    names = list(aligned_arrs.keys())
    k = len(names)
    corr_acc = np.zeros((k, k), dtype=float)
    pair_cnt = np.zeros((k, k), dtype=float)
    for i in rebalance_idx:
        mat = np.column_stack([aligned_arrs[n][i] for n in names])
        # drop all-NaN rows (ineligible stocks)
        active_rows = ~np.isnan(mat).all(axis=1)
        if active_rows.sum() < 10:
            continue
        sub = pd.DataFrame(mat[active_rows], columns=names)
        # treat constant-zero degraded columns on date i as NaN so they don't dilute average correlation
        col_stds = sub.std(numeric_only=True)
        const_cols = col_stds[col_stds < 1e-12].index
        if len(const_cols) > 0:
            sub[const_cols] = np.nan
        c = sub.rank().corr().to_numpy(dtype=float)
        valid = ~np.isnan(c)
        corr_acc[valid] += c[valid]
        pair_cnt[valid] += 1.0
    mean_corr = np.where(pair_cnt > 0, corr_acc / np.maximum(pair_cnt, 1.0), 0.0)
    corr = pd.DataFrame(mean_corr, index=names, columns=names)

    return FactorAnalysis(
        ic_daily=pd.concat(rows, ignore_index=True),
        summary=pd.DataFrame(summary_rows).sort_values("rank_icir", key=lambda s: s.abs(), ascending=False).reset_index(drop=True),
        quintile_nav=navs,
        correlation=corr,
        meta={"horizon": horizon, "n_dates": len(dates), "n_rebalance": len(rebalance_idx)},
    )
