"""Model version snapshot manager: persist LightGBM artifacts + OOS metrics + feature importance,
activate / rollback versions, and run one-click rolling retraining."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import lightgbm as lgb

from app.config import get_settings
from app.core.calendar import now_cn
from app.core.logging import get_logger
from app.db.duckdb_client import DuckDBClient, get_duckdb
from app.db.models import ModelVersion
from app.db.sqlite_client import SQLiteDatabase, get_sqlite_db
from app.factors.definitions import forward_excess_returns
from app.factors.service import FactorEngine, to_jsonable
from app.ml.lgbm_scorer import DEFAULT_ENSEMBLE_SEEDS, EnsembleBooster, LGBMConfig, LGBMScorer, WalkForwardMLResult

logger = get_logger(__name__)


class ModelRegistry:
    """Manages versioned LightGBM boosters on disk + metadata in SQLite."""

    def __init__(
        self,
        sqlite_db: SQLiteDatabase | None = None,
        models_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.sqlite = sqlite_db or get_sqlite_db()
        self.models_dir = Path(models_dir or settings.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_version(
        self,
        ml_result: WalkForwardMLResult,
        model_type: str = "regression",
        embargo_days: int = 10,
        activate: bool = True,
        notes: str = "",
        version_id: str | None = None,
    ) -> dict:
        if ml_result.latest_booster is None:
            raise ValueError("WalkForwardMLResult has no trained latest_booster to persist")
        vid = version_id or f"lgbm_{model_type}_{now_cn().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        vdir = self.models_dir / vid
        vdir.mkdir(parents=True, exist_ok=True)
        model_path = vdir / "model.txt"
        ml_result.latest_booster.save_model(str(model_path))

        ensemble_files: list[str] = []
        ensemble_seeds: list[int] = list(getattr(ml_result, "ensemble_seeds", (42,)))
        if isinstance(ml_result.latest_booster, EnsembleBooster) and len(ml_result.latest_booster.boosters) > 1:
            ensemble_seeds = list(ml_result.latest_booster.seeds)
            for idx, b_member in enumerate(ml_result.latest_booster.boosters):
                fname = f"model_seed_{idx}.txt"
                b_member.save_model(str(vdir / fname))
                ensemble_files.append(fname)

        meta_path = vdir / "metadata.json"
        meta_path.write_text(
            json.dumps(
                to_jsonable({
                    "version_id": vid,
                    "model_type": model_type,
                    "train_start": ml_result.latest_train_range[0],
                    "train_end": ml_result.latest_train_range[1],
                    "embargo_days": embargo_days,
                    "label_horizon": getattr(ml_result, "label_horizon", 10),
                    "ensemble_seeds": ensemble_seeds,
                    "ensemble_files": ensemble_files,
                    "feature_names": ml_result.feature_names,
                    "oos_ic": ml_result.oos_ic_mean,
                    "oos_rank_ic": ml_result.oos_rank_ic_mean,
                    "oos_icir": ml_result.oos_icir,
                    "feature_importance": ml_result.feature_importance,
                    "folds": ml_result.folds,
                }),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        with self.sqlite.session_scope() as session:
            if activate:
                session.query(ModelVersion).update({ModelVersion.is_active: False})
            row = ModelVersion(
                version_id=vid,
                model_type=model_type,
                train_start=ml_result.latest_train_range[0],
                train_end=ml_result.latest_train_range[1],
                embargo_days=embargo_days,
                artifact_path=str(model_path),
                feature_list_json=json.dumps(ml_result.feature_names, ensure_ascii=False),
                oos_ic=round(float(ml_result.oos_ic_mean), 6),
                oos_rank_ic=round(float(ml_result.oos_rank_ic_mean), 6),
                oos_icir=round(float(ml_result.oos_icir), 6),
                feature_importance_json=json.dumps(to_jsonable(ml_result.feature_importance), ensure_ascii=False),
                is_active=activate,
                notes=notes,
            )
            session.add(row)
            session.flush()
            return self._row_to_dict(row)

    def list_versions(self) -> list[dict]:
        with self.sqlite.session_scope() as session:
            rows = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
            return [self._row_to_dict(r) for r in rows]

    def get_active(self) -> dict | None:
        with self.sqlite.session_scope() as session:
            row = session.query(ModelVersion).filter(ModelVersion.is_active.is_(True)).first()
            if row is None:
                row = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()
            return self._row_to_dict(row) if row else None

    def activate_version(self, version_id: str) -> dict:
        with self.sqlite.session_scope() as session:
            target = session.get(ModelVersion, version_id)
            if target is None:
                raise KeyError(f"Model version not found: {version_id}")
            if not Path(target.artifact_path).exists():
                raise FileNotFoundError(f"Model artifact missing on disk: {target.artifact_path}")
            session.query(ModelVersion).update({ModelVersion.is_active: False})
            target.is_active = True
            session.flush()
            logger.info("Activated model version %s", version_id)
            return self._row_to_dict(target)

    def load_active_booster(self) -> tuple[lgb.Booster | EnsembleBooster, dict]:
        meta = self.get_active()
        if meta is None:
            raise FileNotFoundError("No trained model version found in registry; run retrain first.")
        artifact = Path(meta["artifact_path"])
        meta_file = artifact.parent / "metadata.json"
        if meta_file.exists():
            try:
                disk_meta = json.loads(meta_file.read_text(encoding="utf-8"))
                ens_files = disk_meta.get("ensemble_files") or []
                if len(ens_files) > 1 and all((artifact.parent / fn).exists() for fn in ens_files):
                    boosters = [LGBMScorer.load(artifact.parent / fn) for fn in ens_files]
                    seeds = disk_meta.get("ensemble_seeds") or list(range(len(boosters)))
                    return EnsembleBooster(boosters, seeds=seeds), meta
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load ensemble members for %s; falling back to primary model.txt", artifact)
        booster = LGBMScorer.load(artifact)
        return booster, meta

    def retrain_and_register(
        self,
        codes: list[str] | None = None,
        model_type: str = "regression",
        feature_names: list[str] | None = None,
        notes: str = "一键滚动重训",
        duckdb_client: DuckDBClient | None = None,
        label_horizon: int = 10,
        ensemble_seeds: tuple[int, ...] = DEFAULT_ENSEMBLE_SEEDS,
    ) -> tuple[dict, WalkForwardMLResult]:
        db = duckdb_client or get_duckdb()
        f_run = FactorEngine(db).build(codes=codes, horizon=label_horizon)
        fwd_ex = forward_excess_returns(f_run.panel, horizon=label_horizon)
        embargo = max(5, int(label_horizon))
        cfg = LGBMConfig(
            mode=model_type,  # type: ignore[arg-type]
            label_horizon=label_horizon,
            embargo_days=embargo,
            ensemble_seeds=ensemble_seeds,
            feature_names=feature_names if feature_names else list(f_run.std.keys()),
        )
        scorer = LGBMScorer(cfg)
        ml_res = scorer.run_walk_forward(f_run.std, fwd_ex, f_run.universe.eligible, adaptive=True)
        saved = self.save_version(ml_res, model_type=model_type, embargo_days=embargo, activate=True, notes=notes)
        return saved, ml_res

    @staticmethod
    def _row_to_dict(row: ModelVersion) -> dict:
        return {
            "version_id": row.version_id,
            "created_at": row.created_at.isoformat() if isinstance(row.created_at, datetime) else str(row.created_at),
            "model_type": row.model_type,
            "train_start": row.train_start,
            "train_end": row.train_end,
            "embargo_days": row.embargo_days,
            "artifact_path": row.artifact_path,
            "feature_list": json.loads(row.feature_list_json or "[]"),
            "oos_ic": row.oos_ic,
            "oos_rank_ic": row.oos_rank_ic,
            "oos_icir": row.oos_icir,
            "feature_importance": json.loads(row.feature_importance_json or "{}"),
            "is_active": bool(row.is_active),
            "notes": row.notes,
        }
