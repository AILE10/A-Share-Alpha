"""LightGBM cross-sectional regression/rank scorer with Purged Walk-Forward,
6-Seed Bagging Ensemble variance reduction & exact additive TreeSHAP attribution."""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.core.constants import ALL_FACTOR_NAMES, MIN_EMBARGO_DAYS
from app.core.logging import get_logger
from app.factors.analyzer import cross_ic
from app.ml.splitter import PurgedWalkForwardSplitter, WalkForwardFold

logger = get_logger(__name__)

ModelMode = Literal["regression", "lambdarank"]
DEFAULT_ENSEMBLE_SEEDS: tuple[int, ...] = (42, 1024, 2026, 7, 99, 777)


class EnsembleBooster:
    """Multi-seed LightGBM ensemble wrapper duck-typing lgb.Booster.

    Combines per-seed predictions via strictly causal per-cross-section z-scoring
    so variance from feature/row subsampling cancels out while preserving exact
    additive TreeSHAP attributions: SHAP_ens = (1/M) * sum_k (SHAP_k / std_k).
    """

    def __init__(self, boosters: list[lgb.Booster], seeds: tuple[int, ...] | list[int] | None = None) -> None:
        if not boosters:
            raise ValueError("EnsembleBooster requires at least one booster")
        self.boosters: list[lgb.Booster] = list(boosters)
        self.seeds: tuple[int, ...] = tuple(seeds) if seeds is not None else tuple(range(len(boosters)))

    @property
    def primary(self) -> lgb.Booster:
        return self.boosters[0]

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        accs = [np.asarray(b.feature_importance(importance_type=importance_type), dtype=float) for b in self.boosters]
        return np.mean(accs, axis=0)

    def predict(self, X: np.ndarray, pred_contrib: bool = False) -> np.ndarray:
        if len(self.boosters) == 1:
            return np.asarray(self.boosters[0].predict(X, pred_contrib=pred_contrib), dtype=float)
        raw_preds = [np.asarray(b.predict(X), dtype=float) for b in self.boosters]
        if not pred_contrib:
            z_preds = []
            for p in raw_preds:
                sd = float(np.std(p, ddof=1)) if len(p) > 1 else 0.0
                if sd > 1e-12:
                    z_preds.append((p - float(np.mean(p))) / sd)
                else:
                    z_preds.append(p)
            return np.mean(z_preds, axis=0)

        # Exact linear scaling of per-seed TreeSHAP under cross-sectional z-score averaging
        contribs = []
        for b, p in zip(self.boosters, raw_preds):
            c = np.asarray(b.predict(X, pred_contrib=True), dtype=float)
            sd = float(np.std(p, ddof=1)) if len(p) > 1 else 0.0
            if sd > 1e-12:
                c = c / sd
            contribs.append(c)
        return np.mean(contribs, axis=0)

    def save_model(self, path: str) -> None:
        self.primary.save_model(path)

    def __getstate__(self) -> dict:
        return {
            "model_strs": [b.model_to_string() for b in self.boosters],
            "seeds": self.seeds,
        }

    def __setstate__(self, state: dict) -> None:
        self.boosters = [lgb.Booster(model_str=s) for s in state["model_strs"]]
        self.seeds = tuple(state["seeds"])


@dataclass
class LGBMConfig:
    mode: ModelMode = "regression"
    train_window: int = 252
    embargo_days: int = 10
    test_window: int = 63
    label_horizon: int = 10
    num_leaves: int = 15
    max_depth: int = 4
    learning_rate: float = 0.04
    n_estimators: int = 120
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 5.0
    random_state: int = 42
    ensemble_seeds: tuple[int, ...] = DEFAULT_ENSEMBLE_SEEDS
    feature_names: list[str] = field(default_factory=lambda: list(ALL_FACTOR_NAMES))

    def effective_seeds(self) -> tuple[int, ...]:
        if self.ensemble_seeds:
            return tuple(int(s) for s in self.ensemble_seeds)
        return (int(self.random_state),)


@dataclass
class WalkForwardMLResult:
    oos_scores: pd.DataFrame                       # date x ts_code strictly out-of-sample scores
    oos_shap: dict[pd.Timestamp, pd.DataFrame]     # date -> ts_code x feature_name TreeSHAP contrib
    folds: list[dict]
    oos_ic_series: pd.Series
    oos_rank_ic_series: pd.Series
    oos_ic_mean: float
    oos_rank_ic_mean: float
    oos_icir: float
    feature_importance: dict[str, dict[str, float]]  # factor -> {gain, shap}
    latest_booster: lgb.Booster | EnsembleBooster | None = None
    latest_train_range: tuple[str, str] = ("", "")
    feature_names: list[str] = field(default_factory=list)
    label_horizon: int = 10
    ensemble_seeds: tuple[int, ...] = DEFAULT_ENSEMBLE_SEEDS


def build_ml_dataset(
    std_factors: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    fwd_excess: pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    dates_filter: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Stack wide standardized factors + eligibility (+ optional forward excess return label) to long panel."""
    feats = feature_names or list(std_factors.keys())
    dates = eligible.index if dates_filter is None else eligible.index.intersection(dates_filter)
    codes = list(eligible.columns)

    elig_sub = eligible.reindex(index=dates, columns=codes).fillna(False)
    elig_mask = elig_sub.to_numpy(dtype=bool)
    row_idx, col_idx = np.where(elig_mask)
    if len(row_idx) == 0:
        return pd.DataFrame(columns=["trade_date", "ts_code", *feats, "label"])

    data: dict[str, np.ndarray] = {
        "trade_date": dates.to_numpy()[row_idx],
        "ts_code": np.asarray(codes, dtype=object)[col_idx],
    }
    for name in feats:
        arr = std_factors[name].reindex(index=dates, columns=codes).to_numpy(dtype=float)
        data[name] = arr[row_idx, col_idx]

    if fwd_excess is not None:
        y_arr = fwd_excess.reindex(index=dates, columns=codes).to_numpy(dtype=float)
        data["label"] = y_arr[row_idx, col_idx]

    df = pd.DataFrame(data)
    # Require at least half the features non-NaN
    valid_feat = df[feats].notna().mean(axis=1) >= 0.5
    df = df[valid_feat].copy()
    df[feats] = df[feats].fillna(0.0)
    if "label" in df.columns:
        df = df[df["label"].notna()].copy()
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _make_rank_labels(df: pd.DataFrame, n_bins: int = 10) -> np.ndarray:
    """Convert daily continuous excess returns into 0..(n_bins-1) integer relevance grades per date."""
    out = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("trade_date", sort=False).groups.items():
        vals = df.loc[idx, "label"].to_numpy(dtype=float)
        ranks = pd.Series(vals).rank(method="first").to_numpy()
        grades = np.floor((ranks - 1) / max(len(vals), 1) * n_bins).astype(int).clip(0, n_bins - 1)
        out[df.index.get_indexer(idx)] = grades
    return out


def _make_reg_labels(df: pd.DataFrame) -> np.ndarray:
    """Cross-sectional rank-centered excess return target in [-1, 1] per date (outlier-robust)."""
    out = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("trade_date", sort=False).groups.items():
        vals = df.loc[idx, "label"].to_numpy(dtype=float)
        pct = pd.Series(vals).rank(pct=True).to_numpy()
        out[df.index.get_indexer(idx)] = pct * 2.0 - 1.0
    return out


def _zscore_by_date(df: pd.DataFrame, raw_pred: np.ndarray) -> np.ndarray:
    """Strictly causal per-trade_date cross-sectional z-score normalization."""
    if "trade_date" not in df.columns or len(raw_pred) <= 1:
        sd = float(np.std(raw_pred, ddof=1)) if len(raw_pred) > 1 else 0.0
        return (raw_pred - float(np.mean(raw_pred))) / sd if sd > 1e-12 else raw_pred
    s = pd.Series(raw_pred, index=df.index, dtype=float)
    grp = s.groupby(df["trade_date"], sort=False)
    mu = grp.transform("mean")
    sd = grp.transform("std").fillna(0.0)
    out = np.where(sd.to_numpy() > 1e-12, (raw_pred - mu.to_numpy()) / np.where(sd.to_numpy() > 1e-12, sd.to_numpy(), 1.0), raw_pred)
    return np.asarray(out, dtype=float)


def _zscore_by_group_idx(group_ids: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
    """Fast vectorized per-date cross-sectional z-score normalization using integer date indices."""
    n = len(raw_pred)
    if n <= 1:
        return raw_pred
    uniq, inv, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
    if len(uniq) == 1:
        sd = float(np.std(raw_pred, ddof=1))
        return (raw_pred - float(np.mean(raw_pred))) / sd if sd > 1e-12 else raw_pred
    sums = np.bincount(inv, weights=raw_pred, minlength=len(uniq))
    means = sums / np.maximum(counts, 1)
    diff = raw_pred - means[inv]
    sq_sums = np.bincount(inv, weights=diff * diff, minlength=len(uniq))
    denom = np.maximum(counts - 1, 1)
    stds = np.sqrt(np.where(counts > 1, sq_sums / denom, 0.0))
    sd_row = stds[inv]
    return np.where(sd_row > 1e-12, diff / np.where(sd_row > 1e-12, sd_row, 1.0), raw_pred)


def _atomic_pickle_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:  # noqa: BLE001
            pass


def _precompute_daily_labels(
    fwd_np: np.ndarray,
    valid_mask: np.ndarray,
    mode: ModelMode,
    n_bins: int = 10,
) -> np.ndarray:
    """Precompute cross-sectional per-date target labels once across the entire panel."""
    n_dates, n_codes = fwd_np.shape
    out = np.full((n_dates, n_codes), np.nan, dtype=np.float64)
    for i in range(n_dates):
        sel = valid_mask[i]
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        vals = fwd_np[i, sel]
        if mode == "lambdarank":
            ranks = pd.Series(vals).rank(method="first").to_numpy()
            out[i, sel] = np.floor((ranks - 1) / max(cnt, 1) * n_bins).clip(0, n_bins - 1)
        else:
            pct = pd.Series(vals).rank(pct=True).to_numpy()
            out[i, sel] = pct * 2.0 - 1.0
    return out


class LGBMScorer:
    """LightGBM cross-sectional model supporting Huber regression & LambdaRank + Bagging + TreeSHAP."""

    def __init__(self, config: LGBMConfig | None = None) -> None:
        self.config = config or LGBMConfig()
        self.booster: lgb.Booster | EnsembleBooster | None = None

    def _fit_params_for_seed(self, n_rows: int, seed: int) -> dict:
        cfg = self.config
        common_params = {
            "num_leaves": cfg.num_leaves,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "min_child_samples": min(cfg.min_child_samples, max(5, n_rows // 20)),
            "subsample": cfg.subsample,
            "subsample_freq": 1,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_alpha": cfg.reg_alpha,
            "reg_lambda": cfg.reg_lambda,
            "random_state": int(seed),
            "n_jobs": -1,
            "verbose": -1,
            "deterministic": True,
            "force_row_wise": True,
        }
        if cfg.mode == "lambdarank":
            return {
                **common_params,
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [10, 20],
                "label_gain": list(range(10)),
            }
        return {
            **common_params,
            "objective": "huber",
            "metric": "l2",
        }

    def _fit_arrays(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None) -> lgb.Booster | EnsembleBooster:
        cfg = self.config
        feats = cfg.feature_names
        n_rows = len(X)
        if n_rows == 0:
            raise ValueError("Cannot train LightGBM on empty dataset")

        seeds = cfg.effective_seeds()
        grp_arg = groups if cfg.mode == "lambdarank" else None
        if len(seeds) == 1:
            dtrain = lgb.Dataset(X, label=y, group=grp_arg, feature_name=feats, free_raw_data=False)
            self.booster = lgb.train(
                self._fit_params_for_seed(n_rows, seeds[0]),
                dtrain,
                num_boost_round=cfg.n_estimators,
            )
        else:
            boosters = []
            for s in seeds:
                # Per-seed Dataset lets LightGBM bind data_random_seed=seed for exact bin-sampling bagging diversity
                dtrain = lgb.Dataset(X, label=y, group=grp_arg, feature_name=feats, free_raw_data=False)
                boosters.append(
                    lgb.train(
                        self._fit_params_for_seed(n_rows, s),
                        dtrain,
                        num_boost_round=cfg.n_estimators,
                    )
                )
            self.booster = EnsembleBooster(boosters, seeds=seeds)
        return self.booster

    def fit(self, train_df: pd.DataFrame) -> lgb.Booster | EnsembleBooster:
        cfg = self.config
        feats = cfg.feature_names
        if train_df.empty:
            raise ValueError("Cannot train LightGBM on empty dataset")

        ordered = train_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        X = ordered[feats].to_numpy(dtype=float)
        if cfg.mode == "lambdarank":
            y = _make_rank_labels(ordered, n_bins=10)
            groups = ordered.groupby("trade_date", sort=True).size().to_numpy()
        else:
            y = _make_reg_labels(ordered)
            groups = None

        return self._fit_arrays(X, y, groups)

    def _predict_arrays(
        self,
        X: np.ndarray,
        group_ids: np.ndarray,
        booster: lgb.Booster | EnsembleBooster | None = None,
    ) -> np.ndarray:
        b = booster or self.booster
        if b is None:
            raise ValueError("Model not fitted")
        if len(X) == 0:
            return np.array([], dtype=float)
        if isinstance(b, EnsembleBooster) and len(b.boosters) > 1:
            z_preds = [_zscore_by_group_idx(group_ids, np.asarray(m.predict(X), dtype=float)) for m in b.boosters]
            return np.mean(z_preds, axis=0)
        return np.asarray(b.predict(X), dtype=float)

    def _predict_shap_arrays(
        self,
        X: np.ndarray,
        group_ids: np.ndarray,
        booster: lgb.Booster | EnsembleBooster | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        b = booster or self.booster
        if b is None:
            raise ValueError("Model not fitted")
        n_feats = len(self.config.feature_names)
        if len(X) == 0:
            return np.array([], dtype=float), np.empty((0, n_feats), dtype=float)
        if isinstance(b, EnsembleBooster) and len(b.boosters) > 1:
            z_preds: list[np.ndarray] = []
            shap_list: list[np.ndarray] = []
            uniq, inv, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
            for m in b.boosters:
                raw_p = np.asarray(m.predict(X), dtype=float)
                c = np.asarray(m.predict(X, pred_contrib=True), dtype=float)[:, :n_feats]
                sums = np.bincount(inv, weights=raw_p, minlength=len(uniq))
                means = sums / np.maximum(counts, 1)
                diff = raw_p - means[inv]
                sq_sums = np.bincount(inv, weights=diff * diff, minlength=len(uniq))
                stds = np.sqrt(np.where(counts > 1, sq_sums / np.maximum(counts - 1, 1), 0.0))
                sd_row = stds[inv]
                valid_sd = sd_row > 1e-12
                scale = np.where(valid_sd, sd_row, 1.0)
                z_p = np.where(valid_sd, diff / scale, raw_p)
                z_preds.append(z_p)
                shap_list.append(c / scale[:, None])
            return np.mean(z_preds, axis=0), np.mean(shap_list, axis=0)

        scores = np.asarray(b.predict(X), dtype=float)
        contrib = np.asarray(b.predict(X, pred_contrib=True), dtype=float)[:, :n_feats]
        return scores, contrib

    def predict(self, df: pd.DataFrame, booster: lgb.Booster | EnsembleBooster | None = None) -> np.ndarray:
        b = booster or self.booster
        if b is None:
            raise ValueError("Model not fitted")
        if df.empty:
            return np.array([], dtype=float)
        X = df[self.config.feature_names].to_numpy(dtype=float)
        if isinstance(b, EnsembleBooster) and len(b.boosters) > 1:
            z_preds = [_zscore_by_date(df, np.asarray(m.predict(X), dtype=float)) for m in b.boosters]
            return np.mean(z_preds, axis=0)
        return np.asarray(b.predict(X), dtype=float)

    def predict_with_shap(
        self, df: pd.DataFrame, booster: lgb.Booster | EnsembleBooster | None = None
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Return (scores, exact additive TreeSHAP contribution DataFrame aligned to df.index)."""
        b = booster or self.booster
        if b is None:
            raise ValueError("Model not fitted")
        feats = self.config.feature_names
        if df.empty:
            return np.array([], dtype=float), pd.DataFrame(columns=feats)
        X = df[feats].to_numpy(dtype=float)

        if isinstance(b, EnsembleBooster) and len(b.boosters) > 1:
            z_preds: list[np.ndarray] = []
            shap_list: list[np.ndarray] = []
            has_date = "trade_date" in df.columns
            for m in b.boosters:
                raw_p = np.asarray(m.predict(X), dtype=float)
                c = np.asarray(m.predict(X, pred_contrib=True), dtype=float)[:, : len(feats)]
                if has_date:
                    s_p = pd.Series(raw_p, index=df.index, dtype=float)
                    grp = s_p.groupby(df["trade_date"], sort=False)
                    mu = grp.transform("mean").to_numpy()
                    sd = grp.transform("std").fillna(0.0).to_numpy()
                    valid_sd = sd > 1e-12
                    scale = np.where(valid_sd, sd, 1.0)
                    z_p = np.where(valid_sd, (raw_p - mu) / scale, raw_p)
                    c_scaled = c / scale[:, None]
                else:
                    sd_val = float(np.std(raw_p, ddof=1)) if len(raw_p) > 1 else 0.0
                    if sd_val > 1e-12:
                        z_p = (raw_p - float(np.mean(raw_p))) / sd_val
                        c_scaled = c / sd_val
                    else:
                        z_p = raw_p
                        c_scaled = c
                z_preds.append(z_p)
                shap_list.append(c_scaled)
            scores = np.mean(z_preds, axis=0)
            shap_arr = np.mean(shap_list, axis=0)
            return scores, pd.DataFrame(shap_arr, index=df.index, columns=feats)

        scores = np.asarray(b.predict(X), dtype=float)
        contrib = np.asarray(b.predict(X, pred_contrib=True), dtype=float)
        shap_df = pd.DataFrame(contrib[:, : len(feats)], index=df.index, columns=feats)
        return scores, shap_df

    def save(self, path: Path | str, booster: lgb.Booster | EnsembleBooster | None = None) -> Path:
        b = booster or self.booster
        if b is None:
            raise ValueError("No fitted booster to save")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        b.save_model(str(p))
        return p

    @staticmethod
    def load(path: Path | str) -> lgb.Booster:
        return lgb.Booster(model_file=str(path))

    def run_walk_forward(
        self,
        std_factors: dict[str, pd.DataFrame],
        fwd_excess: pd.DataFrame,
        eligible: pd.DataFrame,
        adaptive: bool = True,
        progress_cb: Callable[[str, float], None] | None = None,
        fold_cache_dir: Path | None = None,
    ) -> WalkForwardMLResult:
        """Run strict Purged Walk-Forward training & out-of-sample prediction across the entire panel.

        Optimized with:
          * Single pre-stacked 3D NumPy feature tensor & precomputed daily cross-sectional labels
            (eliminating 44x DataFrame re-stacking & 1,381x per-day fwd_excess reindexing).
          * Optional cryptographic per-fold disk cache (`fold_cache_dir`): when DuckDB adds new
            trailing bars, all unchanged historical folds reuse cached OOS scores/SHAP in milliseconds
            while any altered fold automatically misses and retrains cleanly.
        """
        cfg = self.config
        feats = cfg.feature_names
        # Enforce embargo >= max(MIN_EMBARGO_DAYS, label_horizon) so zero label overlap can occur
        effective_embargo = max(int(cfg.embargo_days), int(MIN_EMBARGO_DAYS), int(cfg.label_horizon))
        sample_df = next(iter(std_factors.values()))
        warm_dates = sample_df.index[sample_df.notna().sum(axis=1) >= 10]
        if len(warm_dates) < 60:
            raise ValueError(f"Too few warmed-up dates ({len(warm_dates)}) for walk-forward LightGBM")

        splitter = PurgedWalkForwardSplitter(
            train_window=cfg.train_window,
            embargo_days=effective_embargo,
            test_window=cfg.test_window,
            label_horizon=cfg.label_horizon,
            adaptive=adaptive,
        )
        folds: list[WalkForwardFold] = splitter.split(warm_dates)
        if not folds:
            raise ValueError("PurgedWalkForwardSplitter generated zero folds")

        dates = eligible.index
        codes = list(eligible.columns)
        codes_arr = np.asarray(codes, dtype=object)
        n_dates, n_codes, n_feats = len(dates), len(codes), len(feats)

        # Pre-stack all features, eligibility, forward labels & daily rank targets ONCE
        elig_all = eligible.fillna(False).to_numpy(dtype=bool)
        X_all = np.empty((n_dates, n_codes, n_feats), dtype=np.float64)
        for k, name in enumerate(feats):
            X_all[:, :, k] = std_factors[name].reindex(index=dates, columns=codes).to_numpy(dtype=np.float64)
        valid_feat_all = elig_all & (np.isfinite(X_all).mean(axis=2) >= 0.5)
        X_all = np.where(np.isfinite(X_all), X_all, 0.0)

        fwd_np = fwd_excess.reindex(index=dates, columns=codes).to_numpy(dtype=np.float64)
        train_valid_all = valid_feat_all & np.isfinite(fwd_np)
        y_all = _precompute_daily_labels(fwd_np, train_valid_all, cfg.mode)

        oos_scores_np = np.full((n_dates, n_codes), np.nan, dtype=np.float64)
        oos_shap: dict[pd.Timestamp, pd.DataFrame] = {}
        fold_records: list[dict] = []
        gain_acc = np.zeros(n_feats, dtype=float)
        shap_acc = np.zeros(n_feats, dtype=float)
        n_fitted = 0
        fold_cache_hits = 0
        latest_booster: lgb.Booster | EnsembleBooster | None = None
        latest_train_range = ("", "")
        total_steps = len(folds) + 1
        n_seeds = len(cfg.effective_seeds())

        # Deterministic config signature for per-fold disk cache safety
        cfg_sig = repr((
            "v5",
            cfg.mode,
            cfg.label_horizon,
            cfg.num_leaves,
            cfg.max_depth,
            cfg.learning_rate,
            cfg.n_estimators,
            cfg.min_child_samples,
            cfg.subsample,
            cfg.colsample_bytree,
            cfg.reg_alpha,
            cfg.reg_lambda,
            cfg.effective_seeds(),
            tuple(feats),
        )).encode("utf-8")

        def _extract_slice(date_slice: pd.DatetimeIndex, with_labels: bool):
            d_idx = dates.get_indexer(date_slice)
            d_idx = d_idx[d_idx >= 0]
            if len(d_idx) == 0:
                return np.empty((0, n_feats), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
            mask = train_valid_all[d_idx] if with_labels else valid_feat_all[d_idx]
            r_loc, c_loc = np.where(mask)
            if len(r_loc) == 0:
                return np.empty((0, n_feats), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
            r_abs = d_idx[r_loc]
            X_sub = np.ascontiguousarray(X_all[r_abs, c_loc])
            y_sub = np.ascontiguousarray(y_all[r_abs, c_loc]) if with_labels else np.empty(0, dtype=np.float64)
            return X_sub, y_sub, r_abs, c_loc

        for f_idx, fold in enumerate(folds):
            X_tr, y_tr, tr_r, tr_c = _extract_slice(fold.train_dates, with_labels=True)
            X_te, _, te_r, te_c = _extract_slice(fold.test_dates, with_labels=False)
            if len(X_tr) < 50 or len(X_te) == 0:
                continue

            # Representative dates for TreeSHAP (every ~15th test day + fold end)
            te_uniq_r = np.unique(te_r)
            shap_r_set = set(int(x) for x in te_uniq_r[::15])
            shap_r_set.add(int(te_uniq_r[-1]))

            fold_cache_path: Path | None = None
            cached_fold = None
            if fold_cache_dir is not None:
                h = hashlib.blake2b(digest_size=16)
                h.update(cfg_sig)
                h.update(f"|{fold.train_start.date()}|{fold.train_end.date()}|{fold.test_start.date()}|{fold.test_end.date()}|".encode("utf-8"))
                h.update(X_tr.tobytes())
                h.update(y_tr.tobytes())
                h.update(X_te.tobytes())
                h.update(te_r.tobytes())
                h.update(te_c.tobytes())
                fold_cache_path = fold_cache_dir / f"fold_{h.hexdigest()}.pkl"
                if fold_cache_path.exists():
                    try:
                        with open(fold_cache_path, "rb") as fh:
                            cached_fold = pickle.load(fh)
                    except Exception:  # noqa: BLE001
                        cached_fold = None

            if cached_fold is not None:
                fold_cache_hits += 1
                if progress_cb:
                    progress_cb(
                        f"命中磁盘 Fold 缓存 ({fold_cache_hits}/{f_idx + 1}) · Fold {f_idx + 1}/{len(folds)} "
                        f"[{fold.test_start.date()} ~ {fold.test_end.date()}]",
                        f_idx / total_steps,
                    )
                scores = cached_fold["scores"]
                shap_by_r = cached_fold["shap_by_r"]
                gain_norm = cached_fold["gain_norm"]
                shap_norm = cached_fold["shap_norm"]
                oos_scores_np[te_r, te_c] = scores
                for r_i, (c_arr_i, s_mat_i) in shap_by_r.items():
                    oos_shap[dates[r_i]] = pd.DataFrame(s_mat_i, index=codes_arr[c_arr_i], columns=feats)
            else:
                if progress_cb:
                    progress_cb(
                        f"LightGBM ({cfg.mode} · {n_seeds}xBag · H{cfg.label_horizon}d) Fold {f_idx + 1}/{len(folds)} "
                        f"[{fold.test_start.date()} ~ {fold.test_end.date()}]",
                        f_idx / total_steps,
                    )
                if cfg.mode == "lambdarank":
                    _, groups_tr = np.unique(tr_r, return_counts=True)
                    y_fit = y_tr.astype(np.int32)
                else:
                    groups_tr = None
                    y_fit = y_tr
                booster = self._fit_arrays(X_tr, y_fit, groups_tr)
                scores = self._predict_arrays(X_te, te_r, booster)
                oos_scores_np[te_r, te_c] = scores

                shap_mask = np.isin(te_r, list(shap_r_set))
                X_sh = X_te[shap_mask]
                sh_r = te_r[shap_mask]
                sh_c = te_c[shap_mask]
                _, shap_arr = self._predict_shap_arrays(X_sh, sh_r, booster)

                shap_by_r: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                for r_i in sorted(shap_r_set):
                    sel_r = sh_r == r_i
                    if sel_r.any():
                        c_arr_i = sh_c[sel_r]
                        s_mat_i = shap_arr[sel_r]
                        shap_by_r[r_i] = (c_arr_i, s_mat_i)
                        oos_shap[dates[r_i]] = pd.DataFrame(s_mat_i, index=codes_arr[c_arr_i], columns=feats)

                gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
                gain_norm = gain / max(gain.sum(), 1e-12)
                shap_abs = np.abs(shap_arr).mean(axis=0) if len(shap_arr) else np.zeros(n_feats, dtype=float)
                shap_norm = shap_abs / max(shap_abs.sum(), 1e-12)
                latest_booster = booster
                latest_train_range = (str(fold.train_start.date()), str(fold.train_end.date()))

                if fold_cache_path is not None:
                    _atomic_pickle_dump(
                        {
                            "scores": scores,
                            "shap_by_r": shap_by_r,
                            "gain_norm": gain_norm,
                            "shap_norm": shap_norm,
                        },
                        fold_cache_path,
                    )

            # Fast zero-reindex Fold OOS IC / RankIC directly from NumPy rows
            f_ics, f_rics = [], []
            for r_i in te_uniq_r:
                ic, ric, _ = cross_ic(oos_scores_np[r_i], fwd_np[r_i], min_n=10)
                if not np.isnan(ic):
                    f_ics.append(ic)
                    f_rics.append(ric)

            gain_acc += gain_norm
            shap_acc += shap_norm
            n_fitted += 1
            if not latest_train_range[0]:
                latest_train_range = (str(fold.train_start.date()), str(fold.train_end.date()))

            rec = fold.to_dict()
            rec["oos_ic"] = float(np.mean(f_ics)) if f_ics else None
            rec["oos_rank_ic"] = float(np.mean(f_rics)) if f_rics else None
            fold_records.append(rec)

        if fold_cache_hits > 0:
            logger.info("Walk-Forward per-fold disk cache: %d/%d folds reused", fold_cache_hits, len(folds))

        # Fit (or load cached) production-ready latest booster using the most recent labeled window
        # (ending strictly label_horizon days before the latest date so every label is realized)
        if progress_cb:
            progress_cb(f"LightGBM ({cfg.mode} · {n_seeds}xBag) 拟合最新滚动快照与样本外 IC 评估...", len(folds) / total_steps)
        labeled_dates = warm_dates[:-cfg.label_horizon] if len(warm_dates) > cfg.label_horizon + 40 else warm_dates
        prod_window = labeled_dates[-min(cfg.train_window, len(labeled_dates)):]
        X_prod, y_prod, prod_r, _ = _extract_slice(prod_window, with_labels=True)
        if len(X_prod) >= 50:
            prod_cache_path: Path | None = None
            cached_prod = None
            if fold_cache_dir is not None:
                hp = hashlib.blake2b(digest_size=16)
                hp.update(b"prod|")
                hp.update(cfg_sig)
                hp.update(f"|{prod_window[0].date()}|{prod_window[-1].date()}|".encode("utf-8"))
                hp.update(X_prod.tobytes())
                hp.update(y_prod.tobytes())
                prod_cache_path = fold_cache_dir / f"prod_{hp.hexdigest()}.pkl"
                if prod_cache_path.exists():
                    try:
                        with open(prod_cache_path, "rb") as fh:
                            cached_prod = pickle.load(fh)
                    except Exception:  # noqa: BLE001
                        cached_prod = None

            if cached_prod is not None:
                latest_booster = cached_prod
            else:
                if cfg.mode == "lambdarank":
                    _, prod_groups = np.unique(prod_r, return_counts=True)
                    y_pfit = y_prod.astype(np.int32)
                else:
                    prod_groups = None
                    y_pfit = y_prod
                latest_booster = self._fit_arrays(X_prod, y_pfit, prod_groups)
                if prod_cache_path is not None:
                    _atomic_pickle_dump(latest_booster, prod_cache_path)

            latest_train_range = (str(prod_window[0].date()), str(prod_window[-1].date()))
            # Fill ONLY trailing dates strictly AFTER the last walk-forward fold's test_end
            last_fold_end = max(f.test_end for f in folds)
            warm_idx = dates.get_indexer(warm_dates)
            has_any_oos = np.isfinite(oos_scores_np).any(axis=1)
            trailing = warm_dates[(warm_dates > last_fold_end) & (~has_any_oos[warm_idx])]
            if len(trailing):
                X_live, _, live_r, live_c = _extract_slice(trailing, with_labels=False)
                if len(X_live):
                    l_scores, l_shap = self._predict_shap_arrays(X_live, live_r, latest_booster)
                    oos_scores_np[live_r, live_c] = l_scores
                    for r_i in np.unique(live_r):
                        sel_r = live_r == r_i
                        oos_shap[dates[r_i]] = pd.DataFrame(l_shap[sel_r], index=codes_arr[live_c[sel_r]], columns=feats)

        oos_scores = pd.DataFrame(oos_scores_np, index=dates, columns=codes)

        # Overall OOS IC time series (fast direct NumPy row scans)
        ic_list, ric_list, ic_dates = [], [], []
        for r_i in range(n_dates):
            s_row = oos_scores_np[r_i]
            if not np.isfinite(s_row).any():
                continue
            ic, ric, _ = cross_ic(s_row, fwd_np[r_i], min_n=10)
            if not np.isnan(ic):
                ic_dates.append(dates[r_i])
                ic_list.append(ic)
                ric_list.append(ric)

        ic_s = pd.Series(ic_list, index=pd.DatetimeIndex(ic_dates), dtype=float)
        ric_s = pd.Series(ric_list, index=pd.DatetimeIndex(ic_dates), dtype=float)
        nov = ric_s.iloc[::cfg.label_horizon] if len(ric_s) else ric_s
        oos_ic_mean = float(ic_s.iloc[::cfg.label_horizon].mean()) if len(ic_s) else 0.0
        oos_ric_mean = float(nov.mean()) if len(nov) else 0.0
        oos_ric_std = float(nov.std(ddof=1)) if len(nov) > 1 else 0.0
        oos_icir = float(oos_ric_mean / oos_ric_std) if oos_ric_std > 1e-12 else 0.0

        fi = {}
        denom = max(n_fitted, 1)
        for i, name in enumerate(feats):
            fi[name] = {
                "gain": round(float(gain_acc[i] / denom), 6),
                "shap": round(float(shap_acc[i] / denom), 6),
            }

        self.booster = latest_booster
        return WalkForwardMLResult(
            oos_scores=oos_scores,
            oos_shap=oos_shap,
            folds=fold_records,
            oos_ic_series=ic_s,
            oos_rank_ic_series=ric_s,
            oos_ic_mean=oos_ic_mean,
            oos_rank_ic_mean=oos_ric_mean,
            oos_icir=oos_icir,
            feature_importance=fi,
            latest_booster=latest_booster,
            latest_train_range=latest_train_range,
            feature_names=list(feats),
            label_horizon=int(cfg.label_horizon),
            ensemble_seeds=cfg.effective_seeds(),
        )
