"""Cross-sectional standardization pipeline:
median fill -> MAD winsorization -> industry + log market-cap OLS neutralization -> Z-Score.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826


def median_fill(values: np.ndarray, industries: np.ndarray) -> np.ndarray:
    """Fill NaN with industry median first, then cross-sectional median. Column-wise."""
    out = values.copy()
    nan_mat = np.isnan(out)
    if not nan_mat.any():
        return out

    # Fast integer path when caller passes pre-encoded int32 industry IDs (releases GIL cleanly)
    if np.issubdtype(industries.dtype, np.integer):
        n_groups = int(industries.max()) + 1 if len(industries) else 0
        inv = industries
    else:
        _, inv = np.unique(industries, return_inverse=True)
        n_groups = int(inv.max()) + 1 if len(inv) else 0
    order = np.argsort(inv, kind="mergesort")
    counts = np.bincount(inv, minlength=n_groups)
    splits = np.cumsum(counts)[:-1]
    groups = np.split(order, splits) if n_groups > 0 else []

    for j in range(out.shape[1]):
        nan_mask = nan_mat[:, j]
        if not nan_mask.any() or nan_mask.all():
            continue  # fully valid or fully missing (degraded -> neutral 0 later)
        col = out[:, j]
        for g_idx in groups:
            if len(g_idx) == 0:
                continue
            g_nan = nan_mask[g_idx]
            if not g_nan.any():
                continue
            g_valid = col[g_idx[~g_nan]]
            if len(g_valid):
                col[g_idx[g_nan]] = float(np.median(g_valid))
        still = np.isnan(col)
        if still.any():
            col[still] = float(np.median(col[~still]))
        out[:, j] = col
    return out


def mad_winsorize(values: np.ndarray, n_mad: float = 3.0) -> np.ndarray:
    out = values.copy()
    valid_cols = ~np.isnan(out).all(axis=0)
    if valid_cols.any():
        sub = out[:, valid_cols]
        med = np.nanmedian(sub, axis=0)
        mad = np.nanmedian(np.abs(sub - med), axis=0) * MAD_SCALE
        # If >50% of values are tied (e.g. partial-coverage factor median-filled), MAD==0 -> fallback to std
        std_fb = np.nanstd(sub, axis=0)
        scale = np.where(mad > 1e-12, mad, std_fb)
        has_scale = scale > 1e-12
        out[:, valid_cols] = np.where(has_scale, np.clip(sub, med - n_mad * scale, med + n_mad * scale), sub)
    return out


def build_neutral_design(
    industries: np.ndarray,
    log_mv: np.ndarray,
    min_group: int = 3,
    other_code: int | None = None,
) -> np.ndarray:
    """Design matrix [1, ln(mv), industry dummies]; sparse industries (<min_group) merged into '其他'."""
    n = len(industries)
    if np.issubdtype(industries.dtype, np.integer):
        max_id = int(industries.max()) + 1 if n else 0
        counts_full = np.bincount(industries, minlength=max_id)
        if (counts_full[industries] < min_group).any():
            fallback_id = other_code if other_code is not None else max_id
            merged = np.where(counts_full[industries] < min_group, fallback_id, industries)
            uniq, inv = np.unique(merged, return_inverse=True)
        else:
            uniq, inv = np.unique(industries, return_inverse=True)
    else:
        uniq, inv, counts = np.unique(industries, return_inverse=True, return_counts=True)
        if (counts < min_group).any():
            sparse_mask = counts < min_group
            ind = np.where(sparse_mask[inv], "其他", industries)
            uniq, inv = np.unique(ind, return_inverse=True)
    n_cats = len(uniq)
    n_cols = 2 + max(0, n_cats - 1)
    if n_cols >= n - 5:
        return np.column_stack([np.ones(n, dtype=np.float64), log_mv])
    design = np.zeros((n, n_cols), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1] = log_mv
    if n_cats > 1:
        nonzero = inv > 0
        design[np.flatnonzero(nonzero), 1 + inv[nonzero]] = 1.0
    return design


def neutralize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """OLS residuals of every factor column on the design matrix (exact vectorized lstsq)."""
    y = values.copy()
    valid_cols = ~np.isnan(y).any(axis=0)
    if valid_cols.any():
        coef, *_ = np.linalg.lstsq(design, y[:, valid_cols], rcond=None)
        y[:, valid_cols] = y[:, valid_cols] - design @ coef
    return y


def zscore(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    valid_cols = ~np.isnan(values).all(axis=0)
    if valid_cols.any():
        sub = values[:, valid_cols]
        mu = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0, ddof=1)
        sd = np.where(sd > 1e-12, sd, np.nan)
        z = (sub - mu) / sd
        out[:, valid_cols] = np.where(np.isnan(z), 0.0, z)
    return out


def standardize_cross_section(
    values: np.ndarray,
    industries: np.ndarray,
    total_mv: np.ndarray,
    neutralize_on: bool = True,
    other_code: int | None = None,
) -> np.ndarray:
    filled = median_fill(values, industries)
    clipped = mad_winsorize(filled)
    if neutralize_on:
        log_mv = np.log(np.where(total_mv > 0, total_mv, np.nan))
        log_mv = np.where(np.isnan(log_mv), np.nanmedian(log_mv), log_mv)
        design = build_neutral_design(industries, log_mv, other_code=other_code)
        resid = neutralize(clipped, design)
    else:
        resid = clipped
    return zscore(resid)


def standardize_panel(
    raw: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    industry: pd.Series,
    total_mv: pd.DataFrame,
    min_names: int = 10,
) -> dict[str, pd.DataFrame]:
    """Apply the standardization pipeline on each date over the eligible cross-section (multi-threaded)."""
    names = list(raw.keys())
    dates = eligible.index
    codes = list(eligible.columns)
    stacked = np.stack([raw[n].reindex(index=dates, columns=codes).to_numpy(dtype=float) for n in names], axis=2)
    ind_str = industry.reindex(codes).fillna("未分类").astype(str).to_numpy(dtype=object)
    # Pre-encode sorted industry labels (including '其他' at its exact Unicode position) to int32 once
    # -> eliminates 4.5M Python string comparisons & GIL locks while keeping exact bit-for-bit design matrix order
    vocab = np.unique(np.append(ind_str, "其他"))
    ind_codes = np.searchsorted(vocab, ind_str).astype(np.int32)
    other_code = int(np.searchsorted(vocab, "其他"))
    mv_arr = total_mv.reindex(index=dates, columns=codes).to_numpy(dtype=float)
    elig_arr = eligible.to_numpy(dtype=bool)
    result = np.full_like(stacked, np.nan)
    n_dates = len(dates)

    def _run_chunk(idx_range: range) -> None:
        for i in idx_range:
            sel = elig_arr[i]
            if int(sel.sum()) < min_names:
                continue
            vals = stacked[i, sel, :]
            # skip dates where factor windows are not yet warmed up for most names
            if np.isnan(vals).mean() > 0.6:
                continue
            result[i, sel, :] = standardize_cross_section(
                vals, ind_codes[sel], mv_arr[i, sel], other_code=other_code
            )

    workers = min(8, os.cpu_count() or 4)
    if n_dates >= 64 and workers > 1:
        chunk_sz = max(16, (n_dates + workers - 1) // workers)
        ranges = [range(s, min(n_dates, s + chunk_sz)) for s in range(0, n_dates, chunk_sz)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_run_chunk, ranges))
    else:
        _run_chunk(range(n_dates))

    return {n: pd.DataFrame(result[:, :, k], index=dates, columns=codes) for k, n in enumerate(names)}
