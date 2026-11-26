"""Paper Trading ledger, Weekly Review attribution, Model Version Registry & Scheduler endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.broker.gateway import BrokerGateway
from app.config import get_settings
from app.core.task_manager import action_task_manager
from app.db.duckdb_client import get_duckdb
from app.db.sqlite_client import get_sqlite_db
from app.factors.service import to_jsonable
from app.ml.model_registry import ModelRegistry
from app.paper.reviewer import WeeklyReviewEngine
from app.paper.service import PaperTradingService
from app.paper.workflow import get_workflow_tracker

router = APIRouter(tags=["Paper Trading, Review & Models"])


class RetrainRequest(BaseModel):
    model_type: str = "regression"
    notes: str = "工作台一键滚动重训 (Huber 10d · 6-Seed Bagging)"


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


@router.post("/paper/pre-market-unlock-async")
def trigger_pre_market_unlock_async() -> dict:
    def _run(emit):
        wf = get_workflow_tracker()
        wf.mark_running("pre_market_unlock", "正在执行盘前 T+1 持仓解锁...", trigger_source="MANUAL")
        emit("正在解锁昨日买入 T+1 持仓...", 0.35)
        try:
            paper = PaperTradingService(get_duckdb(), get_sqlite_db())
            unlocked = paper.pre_market_unlock()
            emit(f"已完成 T+1 解锁（共解锁 {unlocked} 只持仓）", 1.0)
            wf.mark_done(
                "pre_market_unlock",
                f"已手动完成 T+1 解锁（解锁 {unlocked} 只持仓）",
                trigger_source="MANUAL",
                detail={"unlocked": unlocked},
            )
            return {"ok": True, "unlocked": unlocked}
        except Exception as exc:  # noqa: BLE001
            wf.mark_failed("pre_market_unlock", str(exc), trigger_source="MANUAL")
            raise

    return action_task_manager.submit(
        kind="pre_market_unlock",
        title="盘前 T+1 持仓解锁",
        fn=_run,
    )


@router.post("/paper/intraday-refresh")
def trigger_intraday_refresh() -> dict:
    wf = get_workflow_tracker()
    wf.mark_running("intraday_monitor", "正在刷新盘中实时快照与大盘状态...", trigger_source="MANUAL")
    try:
        paper = PaperTradingService(get_duckdb(), get_sqlite_db())
        res = paper.intraday_refresh(candidate_codes=_sample_codes())
        quoted = int(res.get("quoted_count", 0))
        alerts = len(res.get("stop_alerts", []) or [])
        wf.mark_done(
            "intraday_monitor",
            f"已手动刷新盯市（行情 {quoted} 股，止损告警 {alerts} 条）",
            trigger_source="MANUAL",
            detail={"quoted_count": quoted, "stop_alerts": alerts},
        )
        return to_jsonable(res)
    except Exception as exc:  # noqa: BLE001
        wf.mark_failed("intraday_monitor", str(exc), trigger_source="MANUAL")
        raise


@router.post("/paper/intraday-refresh-async")
def trigger_intraday_refresh_async() -> dict:
    def _run(emit):
        wf = get_workflow_tracker()
        wf.mark_running("intraday_monitor", "正在刷新盘中实时快照与大盘状态...", trigger_source="MANUAL")
        try:
            paper = PaperTradingService(get_duckdb(), get_sqlite_db())
            res = paper.intraday_refresh(candidate_codes=_sample_codes(), progress_cb=emit)
            quoted = int(res.get("quoted_count", 0))
            alerts = len(res.get("stop_alerts", []) or [])
            rt = res.get("realtime_regime") or {}
            reg_cn = {"OFFENSIVE": "强势进攻", "NEUTRAL": "震荡中性", "DEFENSIVE": "弱势防守"}.get(
                str(rt.get("regime", "")), ""
            )
            extra = f" · 大盘 {reg_cn}" if reg_cn else ""
            wf.mark_done(
                "intraday_monitor",
                f"已手动刷新盯市（行情 {quoted} 股，止损告警 {alerts} 条{extra}）",
                trigger_source="MANUAL",
                detail={"quoted_count": quoted, "stop_alerts": alerts},
            )
            return res
        except Exception as exc:  # noqa: BLE001
            wf.mark_failed("intraday_monitor", str(exc), trigger_source="MANUAL")
            raise

    return action_task_manager.submit(
        kind="paper_intraday",
        title="盘中快照盯市与大盘刷新",
        fn=_run,
    )


@router.post("/paper/rebalance")
def trigger_paper_rebalance(req: RebalanceRequest = RebalanceRequest()) -> dict:
    wf = get_workflow_tracker()
    wf.mark_running("rebalance_1440", "正在重算全市场选股并执行调仓...", trigger_source="MANUAL")
    try:
        paper = PaperTradingService(get_duckdb(), get_sqlite_db())
        res = paper.run_rebalance(codes=_sample_codes(), fetch_live_quotes=req.fetch_live_quotes)
        n_ord = len(res.get("orders_executed", []) or [])
        n_pos = int(res.get("positions_count", 0))
        wf.mark_done(
            "rebalance_1440",
            f"已手动完成调仓（执行 {n_ord} 笔，当前持仓 {n_pos} 只）",
            trigger_source="MANUAL",
            detail={"orders": n_ord, "positions": n_pos},
        )
        return to_jsonable(res)
    except Exception as exc:  # noqa: BLE001
        wf.mark_failed("rebalance_1440", str(exc), trigger_source="MANUAL")
        raise


@router.post("/paper/rebalance-async")
def trigger_paper_rebalance_async(req: RebalanceRequest = RebalanceRequest()) -> dict:
    def _run(emit):
        wf = get_workflow_tracker()
        wf.mark_running("rebalance_1440", "正在重算全市场选股并执行模拟盘调仓...", trigger_source="MANUAL")
        try:
            paper = PaperTradingService(get_duckdb(), get_sqlite_db())
            res = paper.run_rebalance(
                codes=_sample_codes(),
                fetch_live_quotes=req.fetch_live_quotes,
                progress_cb=emit,
            )
            n_ord = len(res.get("orders_executed", []) or [])
            n_pos = int(res.get("positions_count", 0))
            reg = res.get("regime") or {}
            reg_cn = {"OFFENSIVE": "强势进攻", "NEUTRAL": "震荡中性", "DEFENSIVE": "弱势防守"}.get(
                str(reg.get("regime", "")), ""
            )
            wf.mark_done(
                "rebalance_1440",
                f"已手动完成调仓（执行 {n_ord} 笔，当前持仓 {n_pos} 只 · {reg_cn}）",
                trigger_source="MANUAL",
                detail={"orders": n_ord, "positions": n_pos, "regime": reg.get("regime")},
            )
            return res
        except Exception as exc:  # noqa: BLE001
            wf.mark_failed("rebalance_1440", str(exc), trigger_source="MANUAL")
            raise

    return action_task_manager.submit(
        kind="paper_rebalance",
        title="14:40 全市场选股与模拟盘调仓",
        fn=_run,
    )


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
    wf = get_workflow_tracker()
    wf.mark_running("weekly_review", "正在生成周度复盘与因子衰减报告...", trigger_source="MANUAL")
    try:
        eng = WeeklyReviewEngine(get_duckdb(), get_sqlite_db())
        rev = eng.generate_review()
        wf.mark_done(
            "weekly_review",
            f"已手动生成周度复盘（{rev.get('week_start', '')} ~ {rev.get('week_end', '')}）",
            trigger_source="MANUAL",
            detail={"review_id": rev.get("review_id", "")},
        )
        return to_jsonable(rev)
    except Exception as exc:  # noqa: BLE001
        wf.mark_failed("weekly_review", str(exc), trigger_source="MANUAL")
        raise


@router.post("/paper/reviews/generate-async")
def generate_weekly_review_async() -> dict:
    def _run(emit):
        wf = get_workflow_tracker()
        wf.mark_running("weekly_review", "正在生成周度复盘与 20 日因子衰减归因...", trigger_source="MANUAL")
        try:
            eng = WeeklyReviewEngine(get_duckdb(), get_sqlite_db())
            rev = eng.generate_review(progress_cb=emit)
            wf.mark_done(
                "weekly_review",
                f"已手动生成周度复盘（{rev.get('week_start', '')} ~ {rev.get('week_end', '')}）",
                trigger_source="MANUAL",
                detail={"review_id": rev.get("review_id", "")},
            )
            return rev
        except Exception as exc:  # noqa: BLE001
            wf.mark_failed("weekly_review", str(exc), trigger_source="MANUAL")
            raise

    return action_task_manager.submit(
        kind="weekly_review",
        title="周度复盘与 20 日因子衰减归因",
        fn=_run,
    )


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


@router.post("/models/retrain-async")
def retrain_model_async(req: RetrainRequest) -> dict:
    def _run(emit):
        reg = ModelRegistry(get_sqlite_db())
        saved, _ = reg.retrain_and_register(
            codes=_sample_codes(),
            model_type=req.model_type,
            notes=req.notes,
            duckdb_client=get_duckdb(),
            progress_cb=emit,
        )
        return saved

    return action_task_manager.submit(
        kind="model_retrain",
        title=f"全市场 Purged Walk-Forward 滚动重训 ({req.model_type})",
        fn=_run,
    )


@router.post("/models/{version_id}/activate")
def activate_model_version(version_id: str) -> dict:
    reg = ModelRegistry(get_sqlite_db())
    try:
        return to_jsonable(reg.activate_version(version_id))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
