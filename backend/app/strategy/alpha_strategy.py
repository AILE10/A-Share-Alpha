"""Default Composite Strategy Orchestrator:
Mid/Small-Cap Multi-Factor Enhancement + LightGBM Cross-Sectional Ranking +
Technical Resonance + Market Regime Timing + Event-Driven A-Share Backtest.

Includes a thread-safe, DuckDB-watermark-keyed Two-Level Research Cache:
  * Level 1: (DuckDB watermark, date range, horizon) -> MarketPanel + 16 Factors + Technical + Regime
  * Level 2: (Level-1 key, model_mode, window params) -> Purged Walk-Forward OOS Scores & SHAP

Whenever DuckDB receives new bars / financials / capital flow, the watermark changes and
invalidates cached objects automatically. Repeated parameter-sweep & ablation experiments
against the same 3,200+ stock dataset reuse Level-1/Level-2 artifacts for ~15-30x speedup
with bit-for-bit identical out-of-sample results.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from app.backtest.engine import AShareBacktestEngine, BacktestConfig, BacktestResult
from app.backtest.walk_forward_opt import run_walk_forward_heatmap
from app.config import get_settings
from app.core.logging import get_logger
from app.db.duckdb_client import DuckDBClient, get_duckdb
from app.factors.definitions import FactorSet, forward_excess_returns
from app.factors.panel import MarketPanel
from app.factors.service import FactorEngine, FactorRunResult
from app.factors.universe import UniverseResult
from app.ml.lgbm_scorer import LGBMConfig, LGBMScorer, WalkForwardMLResult
from app.strategy.regime import RegimeConfig, compute_market_regime, persist_market_regime
from app.strategy.technical import TechnicalResult, compute_technical_signals

logger = get_logger(__name__)
ProgressCallback = Callable[[str, float], None]

_DISK_CACHE_VERSION = "v6"


@dataclass
class StrategyBundle:
    factor_run: FactorRunResult
    ml_result: WalkForwardMLResult
    technical: TechnicalResult
    regime_df: pd.DataFrame
    backtest: BacktestResult
    equal_weight_comparison: dict | None = None
    heatmap: dict | None = None
    cache_info: dict = field(default_factory=dict)


def _key_hash(key: tuple) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update((_DISK_CACHE_VERSION + "|" + repr(key)).encode("utf-8"))
    return h.hexdigest()


class _DiskResearchCache:
    """Bounded persistent disk cache (L2) under `data/cache/research/` with atomic writes & pruning.

    Stores:
      * `btctx_<hash>.pkl` : compact backtest-ready context (panel, eligible, tech, regime_df)
      * `ml_<mode>_<hash>.pkl` : full WalkForwardMLResult (oos_scores, oos_shap, folds, latest_booster)
      * `folds/fold_<hash>.pkl` : cryptographic per-fold LightGBM cache reused across daily syncs
    """

    def __init__(self, max_ctx_files: int = 2, max_ml_files: int = 4, max_fold_files: int = 90) -> None:
        self._max_ctx = max_ctx_files
        self._max_ml = max_ml_files
        self._max_folds = max_fold_files

    @property
    def root(self) -> Path:
        p = get_settings().data_dir / "cache" / "research"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def folds_dir(self) -> Path:
        p = self.root / "folds"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _ctx_path(self, factor_key: tuple) -> Path:
        return self.root / f"btctx_{_key_hash(factor_key)}.pkl"

    def _ml_path(self, ml_key: tuple) -> Path:
        mode = str(ml_key[5]) if len(ml_key) > 5 else "model"
        return self.root / f"ml_{mode}_{_key_hash(ml_key)}.pkl"

    @staticmethod
    def _atomic_write(obj, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Disk cache write skipped (%s): %s", path.name, exc)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _prune(directory: Path, pattern: str, keep: int) -> None:
        try:
            files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            for stale in files[keep:]:
                try:
                    stale.unlink()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

    def has_ctx_and_ml(self, factor_key: tuple, ml_key: tuple) -> bool:
        return self._ctx_path(factor_key).exists() and self._ml_path(ml_key).exists()

    def load_ctx(self, factor_key: tuple):
        p = self._ctx_path(factor_key)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and obj.get("factor_key") == factor_key:
                return obj["value"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Corrupt/incompatible disk ctx cache %s ignored: %s", p.name, exc)
        return None

    def save_ctx(self, factor_key: tuple, panel: MarketPanel, eligible: pd.DataFrame, tech: TechnicalResult, reg_df: pd.DataFrame) -> None:
        # Build lean backtest-only snapshot (~60% smaller -> ~2s disk reload instead of 8s!)
        empty_df = pd.DataFrame()
        lean_panel = MarketPanel(
            dates=panel.dates,
            codes=panel.codes,
            open=panel.open,
            high=panel.high,
            low=panel.low,
            close=panel.close,
            pre_close=panel.pre_close,
            volume=panel.volume,
            amount=empty_df,
            turnover=empty_df,
            adj=panel.adj,
            is_st=panel.is_st,
            is_suspended=panel.is_suspended,
            pe_ttm=empty_df,
            pb=panel.pb,
            total_mv=panel.total_mv,
            limit_up=panel.limit_up,
            limit_down=panel.limit_down,
            index_close=panel.index_close,
            index_bars={},
            basic=panel.basic,
            capital_flow_ratio=None,
            fundamentals={},
            split_mult=panel.split_mult,
            cash_div=panel.cash_div,
        )
        lean_tech = TechnicalResult(
            passed=tech.passed,
            overheated=empty_df,
            macd_top_div=tech.macd_top_div,
            above_ma20=tech.above_ma20,
            bull_ma=empty_df,
            pullback_vol=empty_df,
            macd_golden=empty_df,
            atr_14_raw=tech.atr_14_raw,
            ma5=empty_df,
            ma10=empty_df,
            ma20=empty_df,
            ma60=empty_df,
            macd_dif=empty_df,
            macd_dea=empty_df,
            macd_hist=empty_df,
        )
        self._atomic_write(
            {"factor_key": factor_key, "value": (lean_panel, eligible, lean_tech, reg_df)},
            self._ctx_path(factor_key),
        )
        self._prune(self.root, "btctx_*.pkl", self._max_ctx)

    def load_ml(self, ml_key: tuple) -> WalkForwardMLResult | None:
        p = self._ml_path(ml_key)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and obj.get("ml_key") == ml_key:
                return obj["value"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Corrupt/incompatible disk ml cache %s ignored: %s", p.name, exc)
        return None

    def save_ml(self, ml_key: tuple, val: WalkForwardMLResult) -> None:
        self._atomic_write({"ml_key": ml_key, "value": val}, self._ml_path(ml_key))
        self._prune(self.root, "ml_*.pkl", self._max_ml)
        self._prune(self.folds_dir, "fold_*.pkl", self._max_folds)
        self._prune(self.folds_dir, "prod_*.pkl", 8)

    def disk_status(self, current_watermark: tuple) -> dict:
        disk_ctx = False
        disk_ml_modes: set[str] = set()
        fold_count = 0
        try:
            if self.root.exists():
                # Fast metadata index file avoids reading large pickles on every status poll
                meta_path = self.root / "index_meta.pkl"
                if meta_path.exists():
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                    if isinstance(meta, dict):
                        for fk in meta.get("ctx_keys", []):
                            if fk and fk[0] == current_watermark and self._ctx_path(fk).exists():
                                disk_ctx = True
                        for mk in meta.get("ml_keys", []):
                            if mk and mk[0] == current_watermark and self._ml_path(mk).exists():
                                if len(mk) > 5:
                                    disk_ml_modes.add(str(mk[5]))
            if self.folds_dir.exists():
                fold_count = len(list(self.folds_dir.glob("fold_*.pkl")))
        except Exception:  # noqa: BLE001
            pass
        return {
            "disk_factor_cached": disk_ctx,
            "disk_ml_modes": sorted(disk_ml_modes),
            "fold_cache_entries": fold_count,
        }

    def record_meta(self, factor_key: tuple | None = None, ml_key: tuple | None = None) -> None:
        meta_path = self.root / "index_meta.pkl"
        meta = {"ctx_keys": [], "ml_keys": []}
        try:
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    meta = loaded
        except Exception:  # noqa: BLE001
            meta = {"ctx_keys": [], "ml_keys": []}
        if factor_key is not None:
            ck = [k for k in meta.get("ctx_keys", []) if k != factor_key and self._ctx_path(k).exists()]
            ck.append(factor_key)
            meta["ctx_keys"] = ck[-self._max_ctx:]
        if ml_key is not None:
            mk = [k for k in meta.get("ml_keys", []) if k != ml_key and self._ml_path(k).exists()]
            mk.append(ml_key)
            meta["ml_keys"] = mk[-self._max_ml:]
        self._atomic_write(meta, meta_path)

    def clear(self) -> int:
        removed = 0
        try:
            if self.root.exists():
                for p in list(self.root.glob("*.pkl")) + list(self.root.glob("*.tmp.*")):
                    try:
                        p.unlink()
                        removed += 1
                    except Exception:  # noqa: BLE001
                        pass
            if self.folds_dir.exists():
                for p in list(self.folds_dir.glob("*.pkl")) + list(self.folds_dir.glob("*.tmp.*")):
                    try:
                        p.unlink()
                        removed += 1
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
        return removed


class _BacktestResearchCache:
    """Bounded thread-safe two-level in-memory cache (L1 RAM + L2 Disk) keyed by live DuckDB watermark."""

    def __init__(self, max_factor_entries: int = 2, max_ml_entries: int = 2) -> None:
        self._lock = threading.Lock()
        self._factor_cache: OrderedDict[tuple, tuple[FactorRunResult, pd.DataFrame, TechnicalResult, pd.DataFrame]] = OrderedDict()
        self._ctx_cache: OrderedDict[tuple, tuple[MarketPanel, pd.DataFrame, TechnicalResult, pd.DataFrame]] = OrderedDict()
        self._ml_cache: OrderedDict[tuple, WalkForwardMLResult] = OrderedDict()
        self._max_factor = max_factor_entries
        self._max_ml = max_ml_entries
        self._last_watermark: tuple | None = None
        self.disk = _DiskResearchCache()

    def get_factor(self, key: tuple):
        with self._lock:
            val = self._factor_cache.get(key)
            if val is not None:
                self._factor_cache.move_to_end(key)
            return val

    def put_factor(self, key: tuple, val, watermark: tuple) -> None:
        with self._lock:
            self._last_watermark = watermark
            self._factor_cache[key] = val
            self._factor_cache.move_to_end(key)
            while len(self._factor_cache) > self._max_factor:
                self._factor_cache.popitem(last=False)

    def get_ctx(self, key: tuple):
        with self._lock:
            val = self._ctx_cache.get(key)
            if val is not None:
                self._ctx_cache.move_to_end(key)
            return val

    def put_ctx(self, key: tuple, val, watermark: tuple) -> None:
        with self._lock:
            self._last_watermark = watermark
            self._ctx_cache[key] = val
            self._ctx_cache.move_to_end(key)
            while len(self._ctx_cache) > self._max_factor:
                self._ctx_cache.popitem(last=False)

    def get_ml(self, key: tuple) -> WalkForwardMLResult | None:
        with self._lock:
            val = self._ml_cache.get(key)
            if val is not None:
                self._ml_cache.move_to_end(key)
            return val

    def put_ml(self, key: tuple, val: WalkForwardMLResult, watermark: tuple) -> None:
        with self._lock:
            self._last_watermark = watermark
            self._ml_cache[key] = val
            self._ml_cache.move_to_end(key)
            while len(self._ml_cache) > self._max_ml:
                self._ml_cache.popitem(last=False)

    def clear(self, include_disk: bool = False) -> dict:
        with self._lock:
            n_f = len(self._factor_cache) + len(self._ctx_cache)
            n_m = len(self._ml_cache)
            self._factor_cache.clear()
            self._ctx_cache.clear()
            self._ml_cache.clear()
            self._last_watermark = None
        disk_removed = self.disk.clear() if include_disk else 0
        return {
            "cleared_factor_entries": n_f,
            "cleared_ml_entries": n_m,
            "cleared_disk_files": disk_removed,
        }

    def status(self, current_watermark: tuple) -> dict:
        with self._lock:
            ram_factors = [k for k in self._factor_cache.keys() if k[0] == current_watermark]
            ram_ctx = [k for k in self._ctx_cache.keys() if k[0] == current_watermark]
            ram_ml_modes = {str(k[5]) for k in self._ml_cache.keys() if k[0] == current_watermark}
        dstat = self.disk.disk_status(current_watermark)
        all_ml_modes = sorted(ram_ml_modes | set(dstat["disk_ml_modes"]))
        return {
            "factor_panel_cached": (len(ram_factors) > 0 or len(ram_ctx) > 0 or dstat["disk_factor_cached"]),
            "ram_cached": len(ram_factors) > 0 or len(ram_ctx) > 0,
            "disk_factor_cached": dstat["disk_factor_cached"],
            "cached_ml_modes": all_ml_modes,
            "disk_ml_modes": dstat["disk_ml_modes"],
            "fold_cache_entries": dstat["fold_cache_entries"],
            "stocks_count": current_watermark[2],
            "earliest_bar_date": current_watermark[5] if len(current_watermark) > 5 else "",
            "latest_bar_date": current_watermark[1],
            "capital_flow_ok_rows": current_watermark[3],
        }


_RESEARCH_CACHE = _BacktestResearchCache()


def get_duckdb_watermark(db: DuckDBClient) -> tuple:
    """Fast (<3ms) DuckDB content signature for automatic cache invalidation."""
    row = db.query_df(
        "SELECT COUNT(*) AS cnt, MIN(trade_date) AS mn, MAX(trade_date) AS mx, COUNT(DISTINCT ts_code) AS n_stocks FROM daily_bars"
    ).iloc[0]
    cf_cnt = int(db.query_scalar("SELECT COUNT(*) FROM capital_flow_daily WHERE status='OK'") or 0)
    fin_cnt = int(db.query_scalar("SELECT COUNT(*) FROM financial_pit") or 0)
    mn = str(pd.to_datetime(row["mn"]).date()) if pd.notna(row["mn"]) else ""
    mx = str(pd.to_datetime(row["mx"]).date()) if pd.notna(row["mx"]) else ""
    return (int(row["cnt"] or 0), mx, int(row["n_stocks"] or 0), cf_cnt, fin_cnt, mn)


def get_research_cache_status(db: DuckDBClient | None = None) -> dict:
    client = db or get_duckdb()
    return _RESEARCH_CACHE.status(get_duckdb_watermark(client))


def clear_research_cache(include_disk: bool = False) -> dict:
    return _RESEARCH_CACHE.clear(include_disk=include_disk)


class AlphaStrategyService:
    """End-to-end orchestrator for the default A-Share Alpha Studio strategy."""

    def __init__(self, db: DuckDBClient | None = None) -> None:
        self.db = db or get_duckdb()

    def run_full_pipeline(
        self,
        codes: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        lgbm_config: LGBMConfig | None = None,
        bt_config: BacktestConfig | None = None,
        regime_config: RegimeConfig | None = None,
        include_ew_compare: bool = True,
        include_heatmap: bool = False,
        panel: MarketPanel | None = None,
        use_cache: bool = True,
        progress_cb: ProgressCallback | None = None,
    ) -> StrategyBundle:
        l_cfg = lgbm_config or LGBMConfig()
        b_cfg = bt_config or BacktestConfig()

        def _emit(stage: str, pct: float) -> None:
            if progress_cb:
                progress_cb(stage, max(0.0, min(1.0, float(pct))))

        # Bypass shared cache when caller supplies an explicit custom panel (e.g. unit tests)
        # IMPORTANT: Walk-Forward training ALWAYS uses full history up to `end` (panel_start=None) so
        # requested `start` (e.g. 2025-01-01) slices ONLY the strictly out-of-sample simulation window
        # without starving Fold 0 of its 252-day + 60-day warm-up training history!
        cache_enabled = use_cache and (panel is None)
        wm = get_duckdb_watermark(self.db) if cache_enabled else None
        codes_key = "ALL" if codes is None else (len(codes), tuple(sorted(codes)))
        reg_key = None if regime_config is None else tuple(sorted(regime_config.__dict__.items()))
        factor_key = (wm, codes_key, "FULL", str(end), l_cfg.label_horizon, reg_key)
        ml_key = (
            wm,
            codes_key,
            "FULL",
            str(end),
            l_cfg.label_horizon,
            l_cfg.mode,
            l_cfg.embargo_days,
            l_cfg.train_window,
            l_cfg.test_window,
            l_cfg.effective_seeds(),
            tuple(l_cfg.feature_names),
        )

        factor_cache_hit = False
        ml_cache_hit = False
        disk_cache_hit = False
        disk_cache_enabled = cache_enabled and (codes is None)

        f_run: FactorRunResult | None = None
        fwd_excess: pd.DataFrame | None = None
        tech: TechnicalResult | None = None
        reg_df: pd.DataFrame | None = None
        ml_res: WalkForwardMLResult | None = None

        # Ultra-fast L1/L2 shortcut when BOTH backtest context AND ML OOS scores are already cached
        # (survives backend restarts via data/cache/research/*.pkl -> ~1-2s cold reload!)
        if cache_enabled:
            cached_ml_ram = _RESEARCH_CACHE.get_ml(ml_key)
            cached_ctx_ram = _RESEARCH_CACHE.get_ctx(factor_key)
            cached_f_ram = _RESEARCH_CACHE.get_factor(factor_key)
            if cached_ml_ram is not None and (cached_f_ram is not None or cached_ctx_ram is not None):
                ml_res = cached_ml_ram
                ml_cache_hit = True
                factor_cache_hit = True
                if cached_f_ram is not None:
                    f_run, fwd_excess, tech, reg_df = cached_f_ram
                else:
                    bt_panel, bt_elig, tech, reg_df = cached_ctx_ram
                    empty_df = pd.DataFrame()
                    f_run = FactorRunResult(
                        panel=bt_panel,
                        universe=UniverseResult(bt_elig, empty_df, empty_df, empty_df, empty_df),
                        factors=FactorSet({}, {}, {}),
                        std={},
                        analysis=None,
                    )
                _emit(f"命中全市场 {len(f_run.panel.codes)} 股内存二级缓存 (跳过因子与模型，秒级进入撮合)", 0.66)
            elif disk_cache_enabled and _RESEARCH_CACHE.disk.has_ctx_and_ml(factor_key, ml_key):
                _emit("命中本地磁盘二级研究缓存，正在极速载入全市场行情与样本外预测...", 0.18)
                loaded_ctx = _RESEARCH_CACHE.disk.load_ctx(factor_key)
                loaded_ml = _RESEARCH_CACHE.disk.load_ml(ml_key)
                if loaded_ctx is not None and loaded_ml is not None:
                    bt_panel, bt_elig, tech, reg_df = loaded_ctx
                    ml_res = loaded_ml
                    factor_cache_hit = True
                    ml_cache_hit = True
                    disk_cache_hit = True
                    empty_df = pd.DataFrame()
                    f_run = FactorRunResult(
                        panel=bt_panel,
                        universe=UniverseResult(bt_elig, empty_df, empty_df, empty_df, empty_df),
                        factors=FactorSet({}, {}, {}),
                        std={},
                        analysis=None,
                    )
                    if wm is not None:
                        _RESEARCH_CACHE.put_ctx(factor_key, loaded_ctx, wm)
                        _RESEARCH_CACHE.put_ml(ml_key, ml_res, wm)
                    _emit(f"磁盘二级缓存载入完成 ({len(bt_panel.codes)} 股，秒级进入撮合回测)", 0.66)

        # Stage 1: Market Panel, 20 Factors, Technical Resonance & Market Regime (when full std needed)
        if f_run is None or tech is None or reg_df is None:
            cached_f = _RESEARCH_CACHE.get_factor(factor_key) if cache_enabled else None
            if cached_f is not None:
                f_run, fwd_excess, tech, reg_df = cached_f
                factor_cache_hit = True
                _emit(f"命中全市场 {len(f_run.panel.codes)} 只股票因子与技术面缓存 (秒级跳过)", 0.22)
            else:
                _emit("正在加载全市场行情面板并计算四维 20 因子与中性化...", 0.06)
                f_engine = FactorEngine(self.db)
                f_run = f_engine.build(
                    codes=codes,
                    start=None,
                    end=end,
                    horizon=l_cfg.label_horizon,
                    panel=panel,
                    compute_analysis=False,
                )
                _emit(f"正在计算 {len(f_run.panel.codes)} 只股票技术共振与大盘择时状态...", 0.18)
                fwd_excess = forward_excess_returns(f_run.panel, horizon=l_cfg.label_horizon)
                tech = compute_technical_signals(f_run.panel)
                reg_df = compute_market_regime(f_run.panel, regime_config)
                persist_market_regime(reg_df, self.db)
                if cache_enabled and wm is not None:
                    _RESEARCH_CACHE.put_factor(factor_key, (f_run, fwd_excess, tech, reg_df), wm)
                    if disk_cache_enabled:
                        _RESEARCH_CACHE.disk.save_ctx(factor_key, f_run.panel, f_run.universe.eligible, tech, reg_df)
                        _RESEARCH_CACHE.disk.record_meta(factor_key=factor_key)

        # Stage 2: Purged Walk-Forward LightGBM OOS Predictions (with per-fold incremental disk reuse)
        if ml_res is None:
            cached_ml = _RESEARCH_CACHE.get_ml(ml_key) if cache_enabled else None
            if cached_ml is not None:
                ml_res = cached_ml
                ml_cache_hit = True
                _emit(f"命中 {l_cfg.mode} 样本外滚动预测缓存 (跳过模型重训，进入秒级回测)", 0.66)
            else:
                _emit(f"正在执行 Purged Walk-Forward 滚动训练 ({l_cfg.mode}, Embargo={l_cfg.embargo_days}日)...", 0.24)
                scorer = LGBMScorer(l_cfg)
                fold_dir = _RESEARCH_CACHE.disk.folds_dir if disk_cache_enabled else None
                ml_res = scorer.run_walk_forward(
                    f_run.std,
                    fwd_excess,
                    f_run.universe.eligible,
                    adaptive=True,
                    progress_cb=lambda msg, p: _emit(msg, 0.24 + 0.42 * p),
                    fold_cache_dir=fold_dir,
                )
                if cache_enabled and wm is not None:
                    _RESEARCH_CACHE.put_ml(ml_key, ml_res, wm)
                    if disk_cache_enabled:
                        _RESEARCH_CACHE.disk.save_ml(ml_key, ml_res)
                        _RESEARCH_CACHE.disk.record_meta(ml_key=ml_key)

        sim_start = pd.Timestamp(start) if start is not None else None
        sim_end = pd.Timestamp(end) if end is not None else None

        # Stage 3: Primary Event-Driven A-Share Backtest
        _emit("正在执行 A 股精确撮合回测 (T+1 / 涨跌停 / 百股 / 真实费用滑点)...", 0.70)
        bt_res = AShareBacktestEngine(b_cfg).run(
            panel=f_run.panel,
            oos_scores=ml_res.oos_scores,
            eligible=f_run.universe.eligible,
            technical=tech,
            regime_df=reg_df,
            start_date=sim_start,
            end_date=sim_end,
        )

        # Stage 4: Risk Parity vs Equal Weight Comparison
        ew_comp = None
        if include_ew_compare:
            alt_method = "equal_weight" if b_cfg.weight_method == "risk_parity" else "risk_parity"
            _emit(f"正在运行对照组合回测 ({alt_method})...", 0.78)
            alt_cfg = BacktestConfig(**{**b_cfg.__dict__, "weight_method": alt_method})
            alt_res = AShareBacktestEngine(alt_cfg).run(
                panel=f_run.panel,
                oos_scores=ml_res.oos_scores,
                eligible=f_run.universe.eligible,
                technical=tech,
                regime_df=reg_df,
                start_date=sim_start,
                end_date=sim_end,
            )
            ew_comp = {
                "method": alt_method,
                "metrics": alt_res.metrics,
                "equity_curve": alt_res.equity_curve,
            }

        # Stage 5: Optional 3x3 Walk-Forward Parameter Sensitivity Heatmap
        heatmap = None
        if include_heatmap:
            _emit("正在扫描 3×3 样本外参数敏感性热力图...", 0.84)
            hm_x_param = "min_hold_days" if b_cfg.holding_mode == "dynamic" else "rebalance_days"
            hm_x_vals = [5, 10, 12] if b_cfg.holding_mode == "dynamic" else [3, 5, 10]
            heatmap = run_walk_forward_heatmap(
                panel=f_run.panel,
                oos_scores=ml_res.oos_scores,
                eligible=f_run.universe.eligible,
                technical=tech,
                regime_df=reg_df,
                base_config=b_cfg,
                x_param=hm_x_param,
                x_values=hm_x_vals,
                y_param="target_holdings",
                y_values=[8, 10, 12],
                progress_cb=lambda msg, p: _emit(msg, 0.84 + 0.14 * p),
                start_date=sim_start,
                end_date=sim_end,
            )

        _emit("回测完成，正在持久化报告...", 0.99)
        return StrategyBundle(
            factor_run=f_run,
            ml_result=ml_res,
            technical=tech,
            regime_df=reg_df,
            backtest=bt_res,
            equal_weight_comparison=ew_comp,
            heatmap=heatmap,
            cache_info={
                "factor_cache_hit": factor_cache_hit,
                "ml_cache_hit": ml_cache_hit,
                "disk_cache_hit": disk_cache_hit,
                "stocks_count": len(f_run.panel.codes),
            },
        )
