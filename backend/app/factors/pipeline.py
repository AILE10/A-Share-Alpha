"""Cross-sectional standardization pipeline:
median fill -> MAD winsorization -> industry + log market-cap OLS neutralization -> Z-Score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826


def median_fill(values: np.ndarray, industries: np.ndarray) -> np.ndarray:
    """Fill NaN with industry median first, then cross-sectional median. Column-wise."""
    out = values.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        if nan_mask.all():
            continue  # fully missing (degraded) -> handled later as neutral 0
        for ind in np.unique(industries[nan_mask]):
            sel = industries == ind
            med = np.nanmedian(col[sel]) if np.any(~np.isnan(col[sel])) else np.nan
            fill = sel & nan_mask
            col[fill] = med
        still = np.isnan(col)
        if still.any():
            col[still] = np.nanmedian(col)
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


def build_neutral_design(industries: np.ndarray, log_mv: np.ndarray, min_group: int = 3) -> np.ndarray:
    """Design matrix [1, ln(mv), industry dummies]; sparse industries (<min_group) merged into '其他'."""
    n = len(industries)
    ind = industries.astype(object).copy()
    uniq, counts = np.unique(ind, return_counts=True)
    sparse = set(uniq[counts < min_group])
    ind = np.array(["其他" if x in sparse else x for x in ind], dtype=object)
    cats = sorted(set(ind))
    cols = [np.ones(n), log_mv]
    # drop first category to avoid collinearity with the intercept
    for c in cats[1:]:
        cols.append((ind == c).astype(float))
    design = np.column_stack(cols)
    # keep regression well-posed for small cross-sections
    if design.shape[1] >= n - 5:
        design = np.column_stack([np.ones(n), log_mv])
    return design


def neutralize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """OLS residuals of every factor column on the design matrix (vectorized lstsq)."""
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
) -> np.ndarray:
    filled = median_fill(values, industries)
    clipped = mad_winsorize(filled)
    if neutralize_on:
        log_mv = np.log(np.where(total_mv > 0, total_mv, np.nan))
        log_mv = np.where(np.isnan(log_mv), np.nanmedian(log_mv), log_mv)
        design = build_neutral_design(industries, log_mv)
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
    """Apply the standardization pipeline on each date over the eligible cross-section."""
    names = list(raw.keys())
    dates = eligible.index
    codes = list(eligible.columns)
    out = {n: pd.DataFrame(np.nan, index=dates, columns=codes) for n in names}
    stacked = np.stack([raw[n].reindex(index=dates, columns=codes).to_numpy(dtype=float) for n in names], axis=2)
    ind_arr = industry.reindex(codes).fillna("未分类").to_numpy(dtype=object)
    mv_arr = total_mv.reindex(index=dates, columns=codes).to_numpy(dtype=float)
    elig_arr = eligible.to_numpy(dtype=bool)
    result = np.full_like(stacked, np.nan)
    for i in range(len(dates)):
        sel = elig_arr[i]
        if sel.sum() < min_names:
            continue
        vals = stacked[i, sel, :]
        # skip dates where factor windows are not yet warmed up for most names
        if np.isnan(vals).mean() > 0.6:
            continue
        std = standardize_cross_section(vals, ind_arr[sel], mv_arr[i, sel])
        result[i, sel, :] = std
    for k, n in enumerate(names):
        out[n] = pd.DataFrame(result[:, :, k], index=dates, columns=codes)
    return out
