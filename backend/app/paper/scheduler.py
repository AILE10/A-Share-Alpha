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
        try:
            self.paper.pre_market_unlock()
        except Exception as exc:  # noqa: BLE001
            logger.error("pre_market job failed: %s", exc)

    def _job_intraday(self) -> None:
        if not (self._is_open_today() and is_trading_session()):
            return
        try:
            self.paper.intraday_refresh()
        except Exception as exc:  # noqa: BLE001
            logger.error("intraday_refresh job failed: %s", exc)

    def _job_realtime_regime(self) -> None:
        if not (self._is_open_today() and is_trading_session()):
            return
        try:
            get_realtime_regime_service().refresh(db=self.paper.db, broadcast=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("realtime_regime job failed: %s", exc)

    def _job_rebalance_1440(self) -> None:
        if not self._is_open_today():
            return
        try:
            self.paper.run_rebalance()
        except Exception as exc:  # noqa: BLE001
            logger.error("rebalance_1440 job failed: %s", exc)

    def _job_post_market(self) -> None:
        if not self._is_open_today():
            return
        try:
            db = get_duckdb()
            svc = DataSyncService(db)
            try:
                svc.incremental_sync()
                eng = FactorEngine(db)
                f_run = eng.build(codes=None, compute_analysis=True)
                eng.persist(f_run)
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
        except Exception as exc:  # noqa: BLE001
            logger.error("post_market sync job failed: %s", exc)

    def _job_weekly_review(self) -> None:
        try:
            self.reviewer.generate_review()
        except Exception as exc:  # noqa: BLE001
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
