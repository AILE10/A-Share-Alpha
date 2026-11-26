"""Daily Trading Workflow Flowchart & Live Execution Status Tracker.

Tracks all 5 fixed A-share trading workflow steps (both APScheduler automatic runs and
manual workbench button runs) in SQLite `daily_workflow_runs`, overlays real-time running
progress from `action_task_manager`, reconciles genuine DB artifacts on cold start, and
broadcasts instant step transitions over WebSocket channel `workflow_step`.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta

import pandas as pd

from app.api.ws_manager import ws_manager
from app.core.calendar import now_cn, today_cn
from app.core.task_manager import action_task_manager
from app.db.duckdb_client import DuckDBClient, get_duckdb
from app.db.models import (
    DailyWorkflowRun,
    PaperNavDaily,
    PaperOrder,
    SignalRecommendation,
    WeeklyReview,
)
from app.db.sqlite_client import SQLiteDatabase, get_sqlite_db
from app.factors.service import to_jsonable
from app.strategy.realtime_regime import get_realtime_regime_service

WORKFLOW_STEP_SPECS: list[dict] = [
    {
        "step_key": "pre_market_unlock",
        "order": 1,
        "schedule_label": "09:26",
        "title": "盘前 T+1 解锁",
        "short_desc": "昨日买入持仓解锁为可卖，准备当日监控标的",
        "action_kind": "pre_market_unlock",
        "action_label": "手动解锁",
        "frequency": "DAILY",
    },
    {
        "step_key": "intraday_monitor",
        "order": 2,
        "schedule_label": "09:30~14:55",
        "title": "盘中盯市与大盘扫描",
        "short_desc": "实时拉取 L1 盘口检查止损，每 3 分钟刷新全市场大盘状态",
        "action_kind": "paper_intraday",
        "action_label": "立即盯市",
        "frequency": "DAILY",
    },
    {
        "step_key": "rebalance_1440",
        "order": 3,
        "schedule_label": "14:40",
        "title": "14:40 选股与调仓",
        "short_desc": "重算 6-Seed 模型推荐清单，经前置风控校验后自动先卖后买",
        "action_kind": "paper_rebalance",
        "action_label": "立即调仓",
        "frequency": "DAILY",
    },
    {
        "step_key": "post_market_sync",
        "order": 4,
        "schedule_label": "15:30",
        "title": "收盘同步与净值记账",
        "short_desc": "增量同步 4,111 股日线/指数/资金流，重算因子并记录当日净值",
        "action_kind": "data_sync_incremental",
        "action_label": "收盘同步",
        "frequency": "DAILY",
    },
    {
        "step_key": "weekly_review",
        "order": 5,
        "schedule_label": "周五 16:30",
        "title": "周度复盘与因子体检",
        "short_desc": "对比模拟实盘与理论回测误差、滑点损耗及 20 日因子衰减",
        "action_kind": "weekly_review",
        "action_label": "立即复盘",
        "frequency": "WEEKLY_FRI",
    },
]

STEP_SPEC_MAP: dict[str, dict] = {s["step_key"]: s for s in WORKFLOW_STEP_SPECS}
ACTION_KIND_TO_STEP: dict[str, str] = {
    "pre_market_unlock": "pre_market_unlock",
    "paper_intraday": "intraday_monitor",
    "regime_refresh": "intraday_monitor",
    "paper_rebalance": "rebalance_1440",
    "data_sync_incremental": "post_market_sync",
    "weekly_review": "weekly_review",
}


def _now_naive_cn() -> datetime:
    return now_cn().replace(tzinfo=None)


def _fmt_hhmmss(dt: datetime | None) -> str:
    return dt.strftime("%H:%M:%S") if dt is not None else ""


def _fmt_dt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt is not None else ""


class DailyWorkflowTracker:
    """Thread-safe persistent tracker for the 5-step daily trading pipeline."""

    def __init__(self, sqlite_db: SQLiteDatabase | None = None) -> None:
        self._sqlite = sqlite_db
        self._lock = threading.Lock()

    @property
    def sqlite(self) -> SQLiteDatabase:
        return self._sqlite or get_sqlite_db()

    def _upsert(
        self,
        step_key: str,
        status: str,
        summary: str = "",
        trigger_source: str = "AUTO",
        detail: dict | None = None,
        trade_date: str | None = None,
        broadcast: bool = True,
    ) -> dict:
        td = trade_date or str(today_cn())
        ts = _now_naive_cn()
        with self._lock:
            with self.sqlite.session_scope() as session:
                row = (
                    session.query(DailyWorkflowRun)
                    .filter(
                        DailyWorkflowRun.trade_date == td,
                        DailyWorkflowRun.step_key == step_key,
                    )
                    .first()
                )
                if row is None:
                    row = DailyWorkflowRun(trade_date=td, step_key=step_key)
                    session.add(row)
                row.status = status
                row.trigger_source = trigger_source
                if status == "RUNNING":
                    row.started_at = ts
                    row.finished_at = None
                elif status in ("DONE", "FAILED"):
                    if row.started_at is None:
                        row.started_at = ts
                    row.finished_at = ts
                if summary:
                    row.summary = summary
                if detail is not None:
                    row.detail_json = json.dumps(to_jsonable(detail), ensure_ascii=False)
                row.updated_at = ts
                session.flush()
                payload = {
                    "trade_date": row.trade_date,
                    "step_key": row.step_key,
                    "status": row.status,
                    "trigger_source": row.trigger_source,
                    "started_at": _fmt_dt(row.started_at),
                    "finished_at": _fmt_dt(row.finished_at),
                    "time_label": _fmt_hhmmss(row.finished_at or row.started_at),
                    "summary": row.summary,
                }
        if broadcast:
            ws_manager.broadcast_threadsafe("workflow_step", to_jsonable(payload))
        return payload

    def mark_running(
        self,
        step_key: str,
        summary: str = "正在执行...",
        trigger_source: str = "AUTO",
        trade_date: str | None = None,
    ) -> dict:
        return self._upsert(
            step_key=step_key,
            status="RUNNING",
            summary=summary,
            trigger_source=trigger_source,
            trade_date=trade_date,
        )

    def mark_done(
        self,
        step_key: str,
        summary: str,
        trigger_source: str = "AUTO",
        detail: dict | None = None,
        trade_date: str | None = None,
    ) -> dict:
        return self._upsert(
            step_key=step_key,
            status="DONE",
            summary=summary,
            trigger_source=trigger_source,
            detail=detail,
            trade_date=trade_date,
        )

    def mark_failed(
        self,
        step_key: str,
        error: str,
        trigger_source: str = "AUTO",
        trade_date: str | None = None,
    ) -> dict:
        return self._upsert(
            step_key=step_key,
            status="FAILED",
            summary=f"执行异常: {str(error)[:140]}",
            trigger_source=trigger_source,
            detail={"error": str(error)},
            trade_date=trade_date,
        )

    def _is_trading_day(self, db: DuckDBClient, d) -> bool:
        try:
            val = db.query_scalar("SELECT is_open FROM trade_calendar WHERE trade_date = ?", [d])
            if val is not None:
                return bool(val)
        except Exception:  # noqa: BLE001
            pass
        return d.weekday() < 5

    def _resolve_current_stage(self, now_dt: datetime, is_trading_day: bool) -> tuple[str, str]:
        if not is_trading_day:
            return ("weekly_review", "今日为非交易日休市，可查阅历史复盘或手动演练")
        hm = now_dt.hour * 60 + now_dt.minute
        if hm < 9 * 60 + 30:
            return ("pre_market_unlock", "盘前准备时段（09:26 自动 T+1 持仓解锁）")
        if hm < 14 * 60 + 35:
            return ("intraday_monitor", "盘中实时盯市时段（自动监测止损与全市场大盘）")
        if hm < 15 * 60 + 5:
            return ("rebalance_1440", "尾盘核心调仓时段（14:40 自动重算清单并执行调仓）")
        if now_dt.weekday() == 4 and hm >= 16 * 60 + 15:
            return ("weekly_review", "周五盘后周度复盘时段（16:30 自动归因体检）")
        return ("post_market_sync", "盘后清算时段（15:30 自动同步全市场日线并记账）")

    def get_workflow_status(
        self,
        db: DuckDBClient | None = None,
        sqlite_db: SQLiteDatabase | None = None,
    ) -> dict:
        """Assemble live 5-step daily workflow status backed strictly by real DB records & active tasks."""
        duck = db or get_duckdb()
        sq = sqlite_db or self.sqlite
        now_dt = now_cn()
        today = today_cn()
        today_str = str(today)
        is_open = self._is_trading_day(duck, today)
        is_friday = today.weekday() == 4
        week_monday = today - timedelta(days=today.weekday())
        week_monday_dt = datetime(week_monday.year, week_monday.month, week_monday.day)

        # 1) Load persisted DailyWorkflowRun records for today (plus current week for weekly_review)
        persisted: dict[str, DailyWorkflowRun] = {}
        with sq.session_scope() as session:
            rows = (
                session.query(DailyWorkflowRun)
                .filter(DailyWorkflowRun.trade_date == today_str)
                .all()
            )
            for r in rows:
                persisted[r.step_key] = r

            if "weekly_review" not in persisted:
                wk_row = (
                    session.query(DailyWorkflowRun)
                    .filter(
                        DailyWorkflowRun.step_key == "weekly_review",
                        DailyWorkflowRun.status == "DONE",
                        DailyWorkflowRun.trade_date >= str(week_monday),
                    )
                    .order_by(DailyWorkflowRun.updated_at.desc())
                    .first()
                )
                if wk_row is not None:
                    persisted["weekly_review"] = wk_row

            # Genuine real-data artifacts for cold-start inference (never fabricated)
            today_start = datetime(today.year, today.month, today.day)
            today_orders_cnt = int(
                session.query(PaperOrder)
                .filter(PaperOrder.created_at >= today_start)
                .count()
            )
            latest_sig = (
                session.query(SignalRecommendation)
                .filter(SignalRecommendation.trade_date == today_str)
                .order_by(SignalRecommendation.generated_at.desc())
                .first()
            )
            today_nav = (
                session.query(PaperNavDaily)
                .filter(PaperNavDaily.trade_date == today)
                .first()
            )
            week_review = (
                session.query(WeeklyReview)
                .filter(WeeklyReview.created_at >= week_monday_dt)
                .order_by(WeeklyReview.created_at.desc())
                .first()
            )

        # DuckDB real-data checks (complete EOD bars for today?)
        today_bars_complete = False
        today_bars_stocks = 0
        try:
            today_bars_stocks = int(
                duck.query_scalar(
                    "SELECT COUNT(DISTINCT ts_code) FROM daily_bars WHERE trade_date = ?", [today]
                )
                or 0
            )
            total_synced = int(duck.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM daily_bars") or 0)
            today_bars_complete = today_bars_stocks >= max(1, int(total_synced * 0.30)) and total_synced > 0
        except Exception:  # noqa: BLE001
            today_bars_complete = False

        rt_snap = get_realtime_regime_service().get_cached(max_age_seconds=3600.0)

        # 2) Overlay active running background tasks from action_task_manager
        running_by_step: dict[str, dict] = {}
        try:
            for t in action_task_manager.list_tasks(active_only=True):
                sk = ACTION_KIND_TO_STEP.get(str(t.get("kind", "")))
                if sk:
                    running_by_step[sk] = t
        except Exception:  # noqa: BLE001
            pass

        current_stage_key, phase_hint = self._resolve_current_stage(now_dt, is_open)
        hm = now_dt.hour * 60 + now_dt.minute
        steps_out: list[dict] = []
        done_daily = 0
        total_daily = 0

        for spec in WORKFLOW_STEP_SPECS:
            sk = spec["step_key"]
            rec = persisted.get(sk)
            active_task = running_by_step.get(sk)

            status = "PENDING"
            trigger_source = ""
            time_label = ""
            summary = ""
            pct = 0.0

            if active_task is not None:
                status = "RUNNING"
                trigger_source = "MANUAL"
                pct = float(active_task.get("pct") or 0.1)
                summary = str(active_task.get("stage") or "正在执行中...")
                time_label = _fmt_hhmmss(_now_naive_cn())
            elif rec is not None:
                status = rec.status
                trigger_source = rec.trigger_source
                time_label = _fmt_hhmmss(rec.finished_at or rec.started_at)
                summary = rec.summary
                pct = 1.0 if status == "DONE" else (0.5 if status == "RUNNING" else 0.0)
            else:
                # Reconcile genuine persisted evidence if backend restarted mid-day
                if sk == "intraday_monitor" and rt_snap and rt_snap.get("trade_date") == today_str:
                    status = "DONE"
                    trigger_source = "AUTO"
                    time_label = str(rt_snap.get("quote_time") or str(rt_snap.get("updated_at", ""))[-8:])
                    reg_cn = {"OFFENSIVE": "强势进攻", "NEUTRAL": "震荡中性", "DEFENSIVE": "弱势防守"}.get(
                        str(rt_snap.get("regime", "")), str(rt_snap.get("regime", ""))
                    )
                    exp_pct = int(round(float(rt_snap.get("target_exposure") or 0.0) * 100))
                    q_cnt = int(rt_snap.get("quoted_count") or 0)
                    summary = f"实时大盘 {reg_cn} ({exp_pct}%) · 已扫描 {q_cnt} 股"
                    pct = 1.0
                elif sk == "rebalance_1440" and (today_orders_cnt > 0 or latest_sig is not None):
                    status = "DONE"
                    trigger_source = "AUTO"
                    ts_ref = latest_sig.generated_at if latest_sig is not None else None
                    time_label = _fmt_hhmmss(ts_ref)
                    summary = (
                        f"今日已生成信号并执行 {today_orders_cnt} 笔委托"
                        if today_orders_cnt > 0
                        else "今日已完成选股评估（当前防守空仓，未产生新买单）"
                    )
                    pct = 1.0
                elif sk == "post_market_sync" and today_bars_complete:
                    status = "DONE"
                    trigger_source = "AUTO"
                    summary = (
                        f"已入库今日 {today_bars_stocks} 只股票完整日线并记录净值"
                        if today_nav is not None
                        else f"已入库今日 {today_bars_stocks} 只股票完整日线"
                    )
                    pct = 1.0
                elif sk == "weekly_review" and week_review is not None:
                    status = "DONE"
                    trigger_source = "AUTO"
                    time_label = week_review.created_at.strftime("%m-%d %H:%M") if week_review.created_at else ""
                    summary = f"本周复盘已生成（周期 {week_review.week_start} ~ {week_review.week_end}）"
                    pct = 1.0

            # Clear human-readable placeholder when still pending
            if status == "PENDING":
                if not is_open and sk != "weekly_review":
                    status = "SKIPPED"
                    summary = "今日为非交易日（休市），无需自动执行"
                elif sk == "weekly_review" and not is_friday:
                    status = "SCHEDULED"
                    summary = "每周五 16:30 自动执行（平时也可随时点击手动生成）"
                elif sk == "pre_market_unlock" and hm >= 9 * 60 + 30:
                    summary = "若服务在 09:26 后启动，可点击右侧按钮手动解锁"
                elif sk == "rebalance_1440" and hm >= 15 * 60:
                    summary = "已过 14:40 自动调仓窗口，可点击「立即调仓」按收盘价补跑"
                elif sk == "post_market_sync" and hm >= 15 * 60 + 35:
                    summary = "已到收盘同步时间，可点击「收盘同步」补齐今日最新日线"
                else:
                    summary = f"等待 {spec['schedule_label']} 自动触发，也可随时手动执行"

            if spec["frequency"] == "DAILY" or is_friday:
                total_daily += 1
                if status == "DONE":
                    done_daily += 1

            steps_out.append({
                "step_key": sk,
                "order": spec["order"],
                "schedule_label": spec["schedule_label"],
                "title": spec["title"],
                "short_desc": spec["short_desc"],
                "action_kind": spec["action_kind"],
                "action_label": spec["action_label"],
                "frequency": spec["frequency"],
                "status": status,  # PENDING / SCHEDULED / RUNNING / DONE / FAILED / SKIPPED
                "trigger_source": trigger_source,  # AUTO / MANUAL / ""
                "time_label": time_label,
                "summary": summary,
                "pct": round(pct, 4),
                "is_current_stage": sk == current_stage_key,
            })

        return to_jsonable({
            "trade_date": today_str,
            "weekday_cn": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][today.weekday()],
            "server_time": now_dt.strftime("%H:%M:%S"),
            "is_trading_day": is_open,
            "current_stage_key": current_stage_key,
            "phase_hint": phase_hint,
            "completed_count": done_daily,
            "total_due_count": total_daily,
            "steps": steps_out,
        })


_TRACKER_LOCK = threading.Lock()
_TRACKER_SINGLETON: DailyWorkflowTracker | None = None


def get_workflow_tracker() -> DailyWorkflowTracker:
    global _TRACKER_SINGLETON
    with _TRACKER_LOCK:
        if _TRACKER_SINGLETON is None:
            _TRACKER_SINGLETON = DailyWorkflowTracker()
        return _TRACKER_SINGLETON
