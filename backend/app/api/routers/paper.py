"""Paper Trading ledger, Weekly Review attribution, Model Version Registry & Scheduler endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.broker.gateway import BrokerGateway
from app.config import get_settings
from app.db.duckdb_client import get_duckdb
from app.db.sqlite_client import get_sqlite_db
from app.factors.service import to_jsonable
from app.ml.model_registry import ModelRegistry
from app.paper.reviewer import WeeklyReviewEngine
from app.paper.service import PaperTradingService

router = APIRouter(tags=["Paper Trading, Review & Models"])


class RetrainRequest(BaseModel):
    model_type: str = "lambdarank"
    notes: str = "工作台一键滚动重训"


class RebalanceRequest(BaseModel):
    fetch_live_quotes: bool = True


class ResetPaperRequest(BaseModel):
    initial_cash: float = 300_000.0
    clear_reviews: bool = True
    clear_signals: bool = True
    clear_audit_logs: bool = True


def _sample_codes() -> list[str] | None:
    """Operate across ALL stocks synced in DuckDB."""
    return None


@router.get("/paper/account")
def get_paper_account() -> dict:
    gw = BrokerGateway(get_sqlite_db())
    return to_jsonable(gw.paper.get_account_state())


@router.get("/paper/nav")
def get_paper_nav() -> list[dict]:
    paper = PaperTradingService(get_duckdb(), get_sqlite_db())
    return to_jsonable(paper.get_nav_history())


@router.post("/paper/intraday-refresh")
def trigger_intraday_refresh() -> dict:
    paper = PaperTradingService(get_duckdb(), get_sqlite_db())
    return to_jsonable(paper.intraday_refresh(candidate_codes=_sample_codes()))


@router.post("/paper/rebalance")
def trigger_paper_rebalance(req: RebalanceRequest = RebalanceRequest()) -> dict:
    paper = PaperTradingService(get_duckdb(), get_sqlite_db())
    return to_jsonable(paper.run_rebalance(codes=_sample_codes(), fetch_live_quotes=req.fetch_live_quotes))


@router.post("/paper/reset")
def reset_paper_account(req: ResetPaperRequest = ResetPaperRequest()) -> dict:
    paper = PaperTradingService(get_duckdb(), get_sqlite_db())
    try:
        return to_jsonable(
            paper.reset_account(
                initial_cash=req.initial_cash,
                clear_reviews=req.clear_reviews,
                clear_signals=req.clear_signals,
                clear_audit_logs=req.clear_audit_logs,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/paper/reviews")
def list_weekly_reviews(limit: int = 20) -> list[dict]:
    eng = WeeklyReviewEngine(get_duckdb(), get_sqlite_db())
    items = eng.list_reviews(limit=limit)
    return to_jsonable(items)


@router.post("/paper/reviews/generate")
def generate_weekly_review() -> dict:
    eng = WeeklyReviewEngine(get_duckdb(), get_sqlite_db())
    return to_jsonable(eng.generate_review())


@router.get("/models/versions")
def list_model_versions() -> list[dict]:
    reg = ModelRegistry(get_sqlite_db())
    return to_jsonable(reg.list_versions())


@router.post("/models/retrain")
def retrain_model(req: RetrainRequest) -> dict:
    reg = ModelRegistry(get_sqlite_db())
    try:
        saved, _ = reg.retrain_and_register(
            codes=_sample_codes(),
            model_type=req.model_type,
            notes=req.notes,
            duckdb_client=get_duckdb(),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"模型重训失败: {exc}") from exc
    return to_jsonable(saved)


@router.post("/models/{version_id}/activate")
def activate_model_version(version_id: str) -> dict:
    reg = ModelRegistry(get_sqlite_db())
    try:
        return to_jsonable(reg.activate_version(version_id))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
