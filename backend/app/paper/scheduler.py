"""APScheduler trading-session scheduler (Asia/Shanghai timezone).

Jobs:
  1. 09:26 Mon-Fri: Pre-market T+1 unlock & candidate watchlist prep
  2. 09:30-11:30 & 13:00-14:55 (every N sec): Intraday Top-100 snapshot refresh & stop-loss guard
  3. 09:30-11:30 & 13:00-15:00 (every 180s): Full-market real-time Market Regime & breadth WebSocket push
  4. 14:40 Mon-Fri: Rebalance signal generation (with 14:40 live regime), explainable recommendation push & execution
  5. 15:30 Mon-Fri: Post-market incremental DuckDB sync, EOD regime refresh & daily NAV reconciliation
  6. 16:30 Fri: Weekly review (paper vs theoretical backtest, slippage drag, 20d IC decay alerts)
"""

from __future__ import annotations

import threading

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.api.ws_manager import ws_manager
from app.config import get_settings
from app.core.calendar import is_trading_session, today_cn
from app.core.logging import get_logger
from app.data.sync_service import DataSyncService
from app.db.duckdb_client import get_duckdb
from app.factors.service import FactorEngine, to_jsonable
from app.paper.reviewer import WeeklyReviewEngine
from app.paper.service import PaperTradingService
from app.paper.workflow import get_workflow_tracker
from app.strategy.realtime_regime import get_realtime_regime_service
from app.strategy.regime import compute_market_regime, persist_market_regime

logger = get_logger(__name__)


class StudioScheduler:
    def __init__(self, paper_service: PaperTradingService | None = None) -> None:
        self.settings = get_settings()
        self.paper = paper_service or PaperTradingService()
        self.reviewer = WeeklyReviewEngine()
        self.scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
        self._started = False

    def _is_open_today(self) -> bool:
        try:
            db = get_duckdb()
            val = db.query_scalar("SELECT is_open FROM trade_calendar WHERE trade_date = ?", [today_cn()])
            if val is not None:
                return bool(val)
        except Exception:  # noqa: BLE001
            pass
        return today_cn().weekday() < 5

    def _job_pre_market(self) -> None:
        if not self._is_open_today():
            return
        wf = get_workflow_tracker()
        wf.mark_running("pre_market_unlock", "09:26 自动执行盘前 T+1 持仓解锁...", trigger_source="AUTO")
        try:
            unlocked = self.paper.pre_market_unlock()
            wf.mark_done(
                "pre_market_unlock",
                f"已完成盘前 T+1 解锁（解锁 {unlocked} 只持仓）",
                trigger_source="AUTO",
                detail={"unlocked": unlocked},
            )
        except Exception as exc:  # noqa: BLE001
            wf.mark_failed("pre_market_unlock", str(exc), trigger_source="AUTO")
            logger.error("pre_market job failed: %s", exc)

    def _job_intraday(self) -> None:
        if not (is_trading_session() and self._is_open_today()):
            return
        wf = get_workflow_tracker()
        try:
            res = self.paper.intraday_refresh()
            quoted = int(res.get("quoted_count", 0))
            alerts = len(res.get("stop_alerts", []) or [])
            rt = res.get("realtime_regime") or {}
            reg_cn = {"OFFENSIVE": "强势进攻", "NEUTRAL": "震荡中性", "DEFENSIVE": "弱势防守"}.get(
                str(rt.get("regime", "")), ""
            )
            extra = f" · 大盘 {reg_cn}" if reg_cn else ""
            wf.mark_done(
                "intraday_monitor",
                f"实时盯市已刷新（行情 {quoted} 股，止损告警 {alerts} 条{extra}）",
                trigger_source="AUTO",
                detail={"quoted_count": quoted, "stop_alerts": alerts},
            )
        except Exception as exc:  # noqa: BLE001
            wf.mark_failed("intraday_monitor", str(exc), trigger_source="AUTO")
            logger.error("intraday_refresh job failed: %s", exc)

    def _job_realtime_regime(self) -> None:
        if not (is_trading_session() and self._is_open_today()):
            return
        try:
            snap = get_realtime_regime_service().refresh(db=self.paper.db, broadcast=True)
            if snap:
                reg_cn = {"OFFENSIVE": "强势进攻", "NEUTRAL": "震荡中性", "DEFENSIVE": "弱势防守"}.get(
                    str(snap.get("regime", "")), str(snap.get("regime", ""))
                )
                exp_pct = int(round(float(snap.get("target_exposure") or 0.0) * 100))
                q_cnt = int(snap.get("quoted_count") or 0)
                get_workflow_tracker().mark_done(
                    "intraday_monitor",
                    f"实时大盘 {reg_cn} ({exp_pct}%) · 已扫描 {q_cnt} 股",
                    trigger_source="AUTO",
                    detail={"regime": snap.get("regime"), "quoted_count": q_cnt},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("realtime_regime job failed: %s", exc)

    def _job_rebalance_1440(self) -> None:
        if not self._is_open_today():
            return
        wf = get_workflow_tracker()
        wf.mark_running("rebalance_1440", "14:40 自动重算全市场选股清单并执行调仓...", trigger_source="AUTO")
        try:
            res = self.paper.run_rebalance()
            n_ord = len(res.get("orders_executed", []) or [])
            n_pos = int(res.get("positions_count", 0))
            reg = res.get("regime") or {}
            reg_cn = {"OFFENSIVE": "强势进攻", "NEUTRAL": "震荡中性", "DEFENSIVE": "弱势防守"}.get(
                str(reg.get("regime", "")), ""
            )
            wf.mark_done(
                "rebalance_1440",
                f"已完成 14:40 调仓（执行 {n_ord} 笔，当前持仓 {n_pos} 只 · {reg_cn}）",
                trigger_source="AUTO",
                detail={"orders": n_ord, "positions": n_pos, "regime": reg.get("regime")},
            )
        except Exception as exc:  # noqa: BLE001
            wf.mark_failed("rebalance_1440", str(exc), trigger_source="AUTO")
            logger.error("rebalance_1440 job failed: %s", exc)

    def _job_post_market(self) -> None:
        if not self._is_open_today():
            return
        wf = get_workflow_tracker()
        wf.mark_running("post_market_sync", "15:30 自动增量同步收盘日线、重算因子并记账...", trigger_source="AUTO")
        try:
            db = get_duckdb()
            svc = DataSyncService(db)
            try:
                rep = svc.incremental_sync()
                eng = FactorEngine(db)
                f_run = eng.build(codes=None, compute_analysis=True)
                # Daily incremental sync only needs to persist the trailing 5 trading days of
                # universe_daily & factor_values (full factor_ic_daily is always persisted),
                # avoiding rewriting 8M+ historical rows and starving UI reads.
                eng.persist(f_run, factor_values_tail_days=5, universe_tail_days=5)
                reg_df = compute_market_regime(f_run.panel)
                persist_market_regime(reg_df, db)
                get_realtime_regime_service().clear_cache()
                if not reg_df.empty:
                    last = reg_df.iloc[-1]
                    ws_manager.broadcast_threadsafe(
                        "market_regime",
                        to_jsonable({
                            "trade_date": str(last["trade_date"]),
                            "updated_at": str(last["trade_date"]),
                            "is_realtime": False,
                            "regime": str(last["regime"]),
                            "target_exposure": float(last["target_exposure"]),
                            "style_mode": str(last["style_mode"]) if "style_mode" in last.index and last["style_mode"] == last["style_mode"] else "ALL_CAP",
                            "min_mcap_quantile": float(last["min_mcap_quantile"]) if "min_mcap_quantile" in last.index and last["min_mcap_quantile"] == last["min_mcap_quantile"] else 0.0,
                            "csi1000_close": round(float(last["csi1000_close"]), 2),
                            "csi1000_ma20": round(float(last["csi1000_ma20"]), 2) if last["csi1000_ma20"] == last["csi1000_ma20"] else None,
                            "csi1000_ma60": round(float(last["csi1000_ma60"]), 2) if last["csi1000_ma60"] == last["csi1000_ma60"] else None,
                            "csi300_close": round(float(last["csi300_close"]), 2) if "csi300_close" in last.index and last["csi300_close"] == last["csi300_close"] else None,
                            "csi300_ma20": round(float(last["csi300_ma20"]), 2) if "csi300_ma20" in last.index and last["csi300_ma20"] == last["csi300_ma20"] else None,
                            "csi300_ma60": round(float(last["csi300_ma60"]), 2) if "csi300_ma60" in last.index and last["csi300_ma60"] == last["csi300_ma60"] else None,
                            "breadth_above_ma20": round(float(last["breadth_above_ma20"]), 4),
                        }),
                    )
            finally:
                svc.close()
            self.paper.record_daily_nav()
            synced = getattr(rep, "stocks_synced", 0)
            wf.mark_done(
                "post_market_sync",
                f"已完成收盘同步（更新 {synced} 股）、因子重算与当日净值记账",
                trigger_source="AUTO",
                detail={"stocks_synced": synced},
            )
        except Exception as exc:  # noqa: BLE001
            wf.mark_failed("post_market_sync", str(exc), trigger_source="AUTO")
            logger.error("post_market sync job failed: %s", exc)

    def _job_weekly_review(self) -> None:
        wf = get_workflow_tracker()
        wf.mark_running("weekly_review", "周五 16:30 自动生成周度复盘与因子衰减体检...", trigger_source="AUTO")
        try:
            rev = self.reviewer.generate_review()
            wf.mark_done(
                "weekly_review",
                f"周度复盘已生成（{rev.get('week_start', '')} ~ {rev.get('week_end', '')}）",
                trigger_source="AUTO",
                detail={"review_id": rev.get("review_id", "")},
            )
        except Exception as exc:  # noqa: BLE001
            wf.mark_failed("weekly_review", str(exc), trigger_source="AUTO")
            logger.error("weekly_review job failed: %s", exc)

    def start(self) -> None:
        if self._started:
            return
        self.scheduler.add_job(
            self._job_pre_market,
            CronTrigger(day_of_week="mon-fri", hour=9, minute=26, timezone="Asia/Shanghai"),
            id="pre_market_unlock",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._job_intraday,
            IntervalTrigger(seconds=max(30, self.settings.intraday_refresh_seconds)),
            id="intraday_top100_refresh",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._job_realtime_regime,
            IntervalTrigger(seconds=180),
            id="realtime_regime_refresh",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._job_rebalance_1440,
            CronTrigger(day_of_week="mon-fri", hour=14, minute=40, timezone="Asia/Shanghai"),
            id="rebalance_1440",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._job_post_market,
            CronTrigger(day_of_week="mon-fri", hour=15, minute=30, timezone="Asia/Shanghai"),
            id="post_market_sync",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._job_weekly_review,
            CronTrigger(day_of_week="fri", hour=16, minute=30, timezone="Asia/Shanghai"),
            id="weekly_review",
            replace_existing=True,
        )
        if self.settings.scheduler_enabled:
            self.scheduler.start()
            self._started = True
            logger.info("StudioScheduler started with %d jobs", len(self.scheduler.get_jobs()))
            # Always warm up the latest full-market live/closing Market Regime asynchronously on startup
            # (when DuckDB has synced index history) so pre-market, lunch, post-close and weekend views
            # never stay stuck on an older DuckDB date.
            has_idx = False
            try:
                has_idx = int(self.paper.db.query_scalar("SELECT COUNT(*) FROM index_daily") or 0) > 0
            except Exception:  # noqa: BLE001
                has_idx = False
            if has_idx:
                def _startup_regime_warmup() -> None:
                    try:
                        if not self._started:
                            return
                        get_realtime_regime_service().refresh(db=self.paper.db, broadcast=True)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Startup realtime regime warmup skipped: %s", exc)

                threading.Thread(target=_startup_regime_warmup, daemon=True, name="regime_warmup").start()

    def list_jobs(self) -> list[dict]:
        return [
            {
                "id": j.id,
                "next_run_time": j.next_run_time.isoformat() if j.next_run_time else None,
                "trigger": str(j.trigger),
            }
            for j in self.scheduler.get_jobs()
        ]

    def stop(self) -> None:
        if self._started:
            self.scheduler.shutdown(wait=False)
            self._started = False
