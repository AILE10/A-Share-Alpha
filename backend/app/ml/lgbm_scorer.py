"""LightGBM cross-sectional regression/rank scorer with Purged Walk-Forward,
3-Seed Bagging Ensemble variance reduction & exact additive TreeSHAP attribution."""

from __future__ import annotations

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
DEFAULT_ENSEMBLE_SEEDS: tuple[int, ...] = (42, 1024, 2026)


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


class LGBMScorer:
    """LightGBM cross-sectional model supporting Huber regression & LambdaRank + Bagging + TreeSHAP."""

    def __init__(self, config: LGBMConfig | None = None) -> None:
        self.config = config or LGBMConfig()
        self.booster: lgb.Booster | EnsembleBooster | None = None

    def _fit_single(self, ordered: pd.DataFrame, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, seed: int) -> lgb.Booster:
        cfg = self.config
        feats = cfg.feature_names
        common_params = {
            "num_leaves": cfg.num_leaves,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "min_child_samples": min(cfg.min_child_samples, max(5, len(ordered) // 20)),
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
            params = {
                **common_params,
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [10, 20],
                "label_gain": list(range(10)),
            }
            dtrain = lgb.Dataset(X, label=y, group=groups, feature_name=feats, free_raw_data=False)
        else:
            params = {
                **common_params,
                "objective": "huber",
                "metric": "l2",
            }
            dtrain = lgb.Dataset(X, label=y, feature_name=feats, free_raw_data=False)

        return lgb.train(params, dtrain, num_boost_round=cfg.n_estimators)

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

        seeds = cfg.effective_seeds()
        if len(seeds) == 1:
            self.booster = self._fit_single(ordered, X, y, groups, seeds[0])
        else:
            boosters = [self._fit_single(ordered, X, y, groups, s) for s in seeds]
            self.booster = EnsembleBooster(boosters, seeds=seeds)
        return self.booster

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
    ) -> WalkForwardMLResult:
        """Run strict Purged Walk-Forward training & out-of-sample prediction across the entire panel."""
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

        codes = list(eligible.columns)
        oos_scores = pd.DataFrame(np.nan, index=eligible.index, columns=codes)
        oos_shap: dict[pd.Timestamp, pd.DataFrame] = {}
        fold_records: list[dict] = []
        gain_acc = np.zeros(len(feats), dtype=float)
        shap_acc = np.zeros(len(feats), dtype=float)
        n_fitted = 0
        latest_booster: lgb.Booster | EnsembleBooster | None = None
        latest_train_range = ("", "")
        total_steps = len(folds) + 1
        n_seeds = len(cfg.effective_seeds())

        for f_idx, fold in enumerate(folds):
            if progress_cb:
                progress_cb(
                    f"LightGBM ({cfg.mode} · {n_seeds}xBag · H{cfg.label_horizon}d) Fold {f_idx + 1}/{len(folds)} "
                    f"[{fold.test_start.date()} ~ {fold.test_end.date()}]",
                    f_idx / total_steps,
                )
            train_df = build_ml_dataset(std_factors, eligible, fwd_excess, feats, dates_filter=fold.train_dates)
            test_df = build_ml_dataset(std_factors, eligible, fwd_excess=None, feature_names=feats, dates_filter=fold.test_dates)
            if len(train_df) < 50 or test_df.empty:
                continue

            booster = self.fit(train_df)
            scores = self.predict(test_df, booster)
            test_df = test_df.copy()
            test_df["score"] = scores

            # Scatter OOS scores across every test date
            for dt, grp in test_df.groupby("trade_date", sort=True):
                ts = pd.Timestamp(dt)
                oos_scores.loc[ts, grp["ts_code"].to_numpy()] = grp["score"].to_numpy()

            # Compute exact TreeSHAP on representative fold dates (every ~15th day + fold end)
            # to keep full-market 4,111-stock Walk-Forward ~4x faster with unbiased SHAP attribution
            fold_unique_dates = pd.DatetimeIndex(sorted(test_df["trade_date"].unique()))
            shap_dates = set(fold_unique_dates[::15])
            shap_dates.add(fold_unique_dates[-1])
            shap_sub = test_df[test_df["trade_date"].isin(shap_dates)]
            _, shap_df = self.predict_with_shap(shap_sub, booster)
            for dt, grp in shap_sub.groupby("trade_date", sort=True):
                ts = pd.Timestamp(dt)
                s_mat = shap_df.loc[grp.index].copy()
                s_mat.index = grp["ts_code"].to_numpy()
                oos_shap[ts] = s_mat

            # Fold OOS IC / RankIC
            f_ics, f_rics = [], []
            for dt in fold.test_dates:
                s_row = oos_scores.loc[dt].to_numpy(dtype=float)
                y_row = fwd_excess.reindex(index=eligible.index, columns=codes).loc[dt].to_numpy(dtype=float)
                ic, ric, _ = cross_ic(s_row, y_row, min_n=10)
                if not np.isnan(ic):
                    f_ics.append(ic)
                    f_rics.append(ric)

            gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            gain_norm = gain / max(gain.sum(), 1e-12)
            shap_abs = shap_df.abs().mean(axis=0).to_numpy(dtype=float)
            shap_norm = shap_abs / max(shap_abs.sum(), 1e-12)
            gain_acc += gain_norm
            shap_acc += shap_norm
            n_fitted += 1
            latest_booster = booster
            latest_train_range = (str(fold.train_start.date()), str(fold.train_end.date()))

            rec = fold.to_dict()
            rec["oos_ic"] = float(np.mean(f_ics)) if f_ics else None
            rec["oos_rank_ic"] = float(np.mean(f_rics)) if f_rics else None
            fold_records.append(rec)

        # Fit a production-ready latest booster using the most recent labeled window
        # (ending strictly label_horizon days before the latest date so every label is realized)
        if progress_cb:
            progress_cb(f"LightGBM ({cfg.mode} · {n_seeds}xBag) 拟合最新滚动快照与样本外 IC 评估...", len(folds) / total_steps)
        labeled_dates = warm_dates[:-cfg.label_horizon] if len(warm_dates) > cfg.label_horizon + 40 else warm_dates
        prod_window = labeled_dates[-min(cfg.train_window, len(labeled_dates)):]
        prod_df = build_ml_dataset(std_factors, eligible, fwd_excess, feats, dates_filter=prod_window)
        if len(prod_df) >= 50:
            latest_booster = self.fit(prod_df)
            latest_train_range = (str(prod_window[0].date()), str(prod_window[-1].date()))
            # Fill ONLY trailing dates strictly AFTER the last walk-forward fold's test_end
            # (NEVER backfill pre-OOS training dates before folds[0].test_start -> strict zero look-ahead!)
            last_fold_end = max(f.test_end for f in folds)
            trailing = warm_dates[(warm_dates > last_fold_end) & oos_scores.loc[warm_dates].isna().all(axis=1)]
            if len(trailing):
                live_df = build_ml_dataset(std_factors, eligible, None, feats, dates_filter=trailing)
                if not live_df.empty:
                    l_scores, l_shap = self.predict_with_shap(live_df, latest_booster)
                    live_df = live_df.copy()
                    live_df["score"] = l_scores
                    for dt, grp in live_df.groupby("trade_date", sort=True):
                        ts = pd.Timestamp(dt)
                        oos_scores.loc[ts, grp["ts_code"].to_numpy()] = grp["score"].to_numpy()
                        s_mat = l_shap.loc[grp.index].copy()
                        s_mat.index = grp["ts_code"].to_numpy()
                        oos_shap[ts] = s_mat

        # Overall OOS IC time series
        ic_list, ric_list, ic_dates = [], [], []
        fwd_aligned = fwd_excess.reindex_like(oos_scores)
        for dt in oos_scores.index:
            s_row = oos_scores.loc[dt].to_numpy(dtype=float)
            if np.isnan(s_row).all():
                continue
            y_row = fwd_aligned.loc[dt].to_numpy(dtype=float)
            ic, ric, _ = cross_ic(s_row, y_row, min_n=10)
            if not np.isnan(ic):
                ic_dates.append(dt)
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
