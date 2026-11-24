"""Live paper trading service: T+1 unlock, Top-100 intraday mark-to-market & stop-loss monitor,
14:40 automated rebalance execution through BrokerGateway (RiskMiddleware -> Paper/Assisted/QMT),
and daily NAV ledger persistence."""

from __future__ import annotations

import pandas as pd

from app.api.ws_manager import ws_manager
from app.backtest.matching import round_buy_lots
from app.broker.base import BrokerMode, OrderRequest
from app.broker.gateway import BrokerGateway
from app.broker.paper_adapter import PaperBrokerAdapter
from app.core.calendar import today_cn
from app.core.constants import BENCHMARK_CSI1000, BENCHMARK_CSI300, LOT_SIZE
from app.core.logging import get_logger
from app.data.sync_service import DataSyncService
from app.db.duckdb_client import DuckDBClient, get_duckdb
from app.db.models import (
    PaperAccount,
    PaperNavDaily,
    PaperOrder,
    PaperPosition,
    RiskAuditLog,
    SignalRecommendation,
    WeeklyReview,
)
from app.db.sqlite_client import SQLiteDatabase, get_sqlite_db
from app.factors.service import to_jsonable
from app.paper.recommender import RecommendationEngine

logger = get_logger(__name__)


class PaperTradingService:
    def __init__(
        self,
        duckdb_client: DuckDBClient | None = None,
        sqlite_db: SQLiteDatabase | None = None,
        gateway: BrokerGateway | None = None,
        sync_service: DataSyncService | None = None,
    ) -> None:
        self.db = duckdb_client or get_duckdb()
        self.sqlite = sqlite_db or get_sqlite_db()
        self.gateway = gateway or BrokerGateway(
            self.sqlite,
            on_stage_order=lambda payload: ws_manager.broadcast_threadsafe("assisted_order", payload),
        )
        self.sync_service = sync_service or DataSyncService(self.db)
        self.recommender = RecommendationEngine(self.db, self.sqlite)

    def reset_account(
        self,
        initial_cash: float = 300_000.0,
        account_id: str = "default_paper",
        clear_reviews: bool = True,
        clear_signals: bool = True,
        clear_audit_logs: bool = True,
    ) -> dict:
        """Reset paper trading ledger cleanly to a fresh formal starting capital (e.g., 300,000 CNY)."""
        import json

        cash_val = round(float(initial_cash), 2)
        if cash_val < 10_000.0:
            raise ValueError("初始本金不能低于 10,000 元")

        with self.sqlite.session_scope() as session:
            acct = session.get(PaperAccount, account_id)
            if acct is None:
                acct = PaperAccount(account_id=account_id, name="A-Share Alpha 默认模拟账户", broker_mode="PAPER")
                session.add(acct)

            deleted_positions = session.query(PaperPosition).filter(PaperPosition.account_id == account_id).delete()
            deleted_orders = session.query(PaperOrder).filter(PaperOrder.account_id == account_id).delete()
            deleted_nav = session.query(PaperNavDaily).filter(PaperNavDaily.account_id == account_id).delete()
            deleted_signals = session.query(SignalRecommendation).delete() if clear_signals else 0
            deleted_reviews = session.query(WeeklyReview).delete() if clear_reviews else 0
            deleted_logs = session.query(RiskAuditLog).delete() if clear_audit_logs else 0

            acct.initial_cash = cash_val
            acct.cash = cash_val
            acct.total_equity = cash_val
            acct.peak_equity = cash_val
            acct.current_drawdown = 0.0
            acct.fuse_status = "NORMAL"

            session.add(
                RiskAuditLog(
                    order_id="",
                    ts_code="",
                    side="RESET",
                    rule_name="PAPER_ACCOUNT_RESET",
                    decision="PASS",
                    detail_json=json.dumps(
                        {
                            "account_id": account_id,
                            "initial_cash": cash_val,
                            "start_date": today_cn().isoformat(),
                            "cleared": {
                                "positions": deleted_positions,
                                "orders": deleted_orders,
                                "nav": deleted_nav,
                                "signals": deleted_signals,
                                "reviews": deleted_reviews,
                                "audit_logs": deleted_logs,
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            )

        state = self.gateway.paper.get_account_state(account_id)
        ws_manager.broadcast_threadsafe("paper_nav", to_jsonable(state["account"]))
        logger.info("Paper account %s cleanly reset to %.2f CNY", account_id, cash_val)
        return {
            "ok": True,
            "account": state["account"],
            "cleared": {
                "positions": deleted_positions,
                "orders": deleted_orders,
                "nav": deleted_nav,
                "signals": deleted_signals,
                "reviews": deleted_reviews,
                "audit_logs": deleted_logs,
            },
        }

    def pre_market_unlock(self, account_id: str = "default_paper") -> int:
        """09:26 job: unlock T+1 sellable shares for all positions bought before today."""
        unlocked = 0
        with self.sqlite.session_scope() as session:
            positions = session.query(PaperPosition).filter(PaperPosition.account_id == account_id).all()
            for p in positions:
                if p.sellable_shares < p.total_shares:
                    p.sellable_shares = p.total_shares
                    unlocked += 1
        logger.info("Pre-market T+1 unlock completed: %d positions unlocked", unlocked)
        return unlocked

    def intraday_refresh(self, account_id: str = "default_paper", candidate_codes: list[str] | None = None) -> dict:
        """Intraday job: fetch real quotes ONLY for held positions + candidate Top 100,
        mark-to-market positions, detect hard/ATR stop-loss triggers, update drawdown & fuse."""
        with self.sqlite.session_scope() as session:
            positions = session.query(PaperPosition).filter(PaperPosition.account_id == account_id).all()
            held_codes = [p.ts_code for p in positions if p.total_shares > 0]

        rec_codes = [r["ts_code"] for r in self.recommender.get_latest_persisted()]
        watch_codes = list(dict.fromkeys([*held_codes, *rec_codes, *(candidate_codes or [])]))[:100]
        if not watch_codes:
            df_c = self.db.query_df(
                "SELECT ts_code FROM daily_bars GROUP BY ts_code ORDER BY MAX(amount) DESC LIMIT 100"
            )
            watch_codes = df_c["ts_code"].tolist()

        quotes_df = pd.DataFrame()
        if watch_codes:
            try:
                quotes_df = self.sync_service.snapshot_top(watch_codes, max_n=100)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Intraday snapshot_top failed, using latest DuckDB close: %s", exc)

        quote_map = {str(r["ts_code"]): r.to_dict() for _, r in quotes_df.iterrows()} if not quotes_df.empty else {}
        risk_cfg = self.gateway.risk.get_config()
        stop_alerts: list[dict] = []

        with self.sqlite.session_scope() as session:
            acct = session.get(PaperAccount, account_id)
            if acct is None:
                return {"ok": False, "error": "account_not_found"}
            positions = session.query(PaperPosition).filter(PaperPosition.account_id == account_id).all()
            for p in positions:
                q = quote_map.get(p.ts_code)
                px = float(q["last_price"]) if (q and pd.notna(q.get("last_price")) and float(q["last_price"]) > 0) else 0.0
                if px <= 0:
                    last_c = self.db.query_scalar(
                        "SELECT close FROM daily_bars WHERE ts_code = ? ORDER BY trade_date DESC LIMIT 1", [p.ts_code]
                    )
                    px = float(last_c) if last_c else p.current_price
                if px > 0:
                    p.current_price = round(px, 2)
                    if px > p.highest_price_since_entry:
                        p.highest_price_since_entry = round(px, 2)
                    hard_stop = round(p.avg_cost * (1.0 - risk_cfg["stock_hard_stop_pct"]), 2)
                    p.stop_loss_price = hard_stop
                    if px <= hard_stop or (p.atr_stop_price > 0 and px <= p.atr_stop_price):
                        stop_alerts.append({
                            "ts_code": p.ts_code,
                            "name": p.name,
                            "current_price": px,
                            "stop_loss_price": max(hard_stop, p.atr_stop_price),
                            "sellable_shares": p.sellable_shares,
                        })

            PaperBrokerAdapter._recompute_account_equity(session, acct)
            if acct.current_drawdown <= -risk_cfg["drawdown_halt_pct"]:
                acct.fuse_status = "HALTED"
            elif acct.current_drawdown <= -risk_cfg["drawdown_warn_pct"]:
                acct.fuse_status = "WARNING"
            else:
                acct.fuse_status = "NORMAL"

        state = self.gateway.paper.get_account_state(account_id)
        self.record_daily_nav(account_id)

        # Also refresh full-market real-time Market Regime if stale (>90s) during live refresh
        realtime_regime = None
        try:
            from app.strategy.realtime_regime import get_realtime_regime_service

            rt_svc = get_realtime_regime_service()
            realtime_regime = rt_svc.get_cached(max_age_seconds=90.0)
            if realtime_regime is None:
                realtime_regime = rt_svc.refresh(db=self.db, broadcast=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Optional realtime regime refresh skipped during intraday_refresh: %s", exc)

        ws_manager.broadcast_threadsafe("paper_nav", to_jsonable({**state["account"], "stop_alerts": stop_alerts}))
        if stop_alerts:
            ws_manager.broadcast_threadsafe("risk_alert", {"type": "STOP_LOSS_TRIGGERED", "alerts": stop_alerts})
        return {
            "ok": True,
            "quoted_count": len(quote_map),
            "account": state["account"],
            "positions": state["positions"],
            "stop_alerts": stop_alerts,
            "realtime_regime": realtime_regime,
        }

    def run_rebalance(
        self,
        account_id: str = "default_paper",
        codes: list[str] | None = None,
        override_mode: BrokerMode | None = None,
        fetch_live_quotes: bool = True,
    ) -> dict:
        """14:40 rebalance workflow:
        1. Score ALL stocks synced in DuckDB (`codes=None` -> full synced universe) to find Top-100 candidates.
        2. Fetch real-time L1 quotes ONLY for held positions + Top-100 candidates (strictly <= 100 to prevent rate limits).
        3. Re-generate final recommendations with live prices & pass EVERY order through RiskMiddleware -> BrokerAdapter."""
        # Step 1: Full-universe cross-sectional scoring
        pre_bundle = self.recommender.generate(
            account_id=account_id,
            codes=codes,
            live_quotes=None,
            persist=not fetch_live_quotes,
            broadcast=not fetch_live_quotes,
        )

        state_pre = self.gateway.paper.get_account_state(account_id)
        held_codes = [p["ts_code"] for p in state_pre.get("positions", []) if p.get("total_shares", 0) > 0]

        # Step 2: Targeted Top-100 + holdings live L1 quote snapshot
        live_quotes = pd.DataFrame()
        if fetch_live_quotes:
            watch_top100 = list(dict.fromkeys([*held_codes, *pre_bundle.top_candidates_codes]))[:100]
            if watch_top100:
                try:
                    live_quotes = self.sync_service.snapshot_top(watch_top100, max_n=100)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Rebalance live quote snapshot failed, falling back to latest DuckDB bars: %s", exc)

        if fetch_live_quotes:
            bundle = self.recommender.generate(
                account_id=account_id,
                codes=codes,
                live_quotes=live_quotes,
                persist=True,
                broadcast=True,
            )
        else:
            bundle = pre_bundle

        quote_map = {str(r["ts_code"]): r.to_dict() for _, r in live_quotes.iterrows()} if not live_quotes.empty else {}
        state = self.gateway.paper.get_account_state(account_id)
        total_equity = float(state["account"]["total_equity"])
        pos_map = {p["ts_code"]: p for p in state["positions"]}

        def _quote_fields(ts_code: str, fallback_px: float) -> dict:
            q = quote_map.get(ts_code, {})
            if q and pd.notna(q.get("last_price")) and float(q["last_price"]) > 0:
                return {
                    "ref_price": float(q["last_price"]),
                    "pre_close": float(q["pre_close"]) if pd.notna(q.get("pre_close")) else float(q["last_price"]),
                    "ask1": float(q["ask1"]) if pd.notna(q.get("ask1")) else float(q["last_price"]),
                    "bid1": float(q["bid1"]) if pd.notna(q.get("bid1")) else float(q["last_price"]),
                    "open_price": float(q["open"]) if pd.notna(q.get("open")) else float(q["last_price"]),
                    "high_price": float(q["high"]) if pd.notna(q.get("high")) else float(q["last_price"]),
                    "low_price": float(q["low"]) if pd.notna(q.get("low")) else float(q["last_price"]),
                    "volume": float(q["volume"]) if pd.notna(q.get("volume")) and float(q["volume"]) > 0 else 100_000.0,
                }
            row = self.db.query_df(
                "SELECT open, high, low, close, pre_close, volume, is_st, is_suspended, limit_up, limit_down "
                "FROM daily_bars WHERE ts_code = ? ORDER BY trade_date DESC LIMIT 1",
                [ts_code],
            )
            if not row.empty:
                r = row.iloc[0]
                px = float(r["close"]) if pd.notna(r["close"]) else fallback_px
                return {
                    "ref_price": px,
                    "pre_close": float(r["pre_close"]) if pd.notna(r["pre_close"]) else px,
                    "ask1": px,
                    "bid1": px,
                    "open_price": float(r["open"]) if pd.notna(r["open"]) else px,
                    "high_price": float(r["high"]) if pd.notna(r["high"]) else px,
                    "low_price": float(r["low"]) if pd.notna(r["low"]) else px,
                    "volume": float(r["volume"]) if pd.notna(r["volume"]) else 100_000.0,
                    "is_st": bool(r["is_st"]),
                    "is_suspended": bool(r["is_suspended"]),
                    "limit_up": float(r["limit_up"]) if pd.notna(r["limit_up"]) else 0.0,
                    "limit_down": float(r["limit_down"]) if pd.notna(r["limit_down"]) else 0.0,
                }
            return {"ref_price": fallback_px, "pre_close": fallback_px, "ask1": fallback_px, "bid1": fallback_px}

        receipts: list[dict] = []

        # 1) Execute SELLs first to release cash
        for rec in [r for r in bundle.recommendations if r["action"] == "SELL"]:
            pos = pos_map.get(rec["ts_code"])
            if not pos or pos["total_shares"] <= 0:
                continue
            qf = _quote_fields(rec["ts_code"], rec["ref_price"])
            req = OrderRequest(
                account_id=account_id,
                ts_code=rec["ts_code"],
                name=rec["name"],
                sw_l1_name=rec["sw_l1_name"],
                side="SELL",
                qty=int(pos["total_shares"]),
                signal_id=rec["signal_id"],
                stop_loss_price=rec["stop_loss_price"],
                take_profit_price=rec["take_profit_price"],
                suggested_weight=0.0,
                reason_summary=rec["rationale"],
                top_factors=rec["top_factors"],
                tech_patterns=rec["tech_patterns"],
                **qf,
            )
            receipt = self.gateway.submit_order(req, override_mode=override_mode)
            receipts.append(receipt.to_dict())

        # Refresh equity after full SELLs
        state = self.gateway.paper.get_account_state(account_id)
        total_equity = float(state["account"]["total_equity"])
        pos_map = {p["ts_code"]: p for p in state["positions"]}

        # 1b) Macro-deleveraging trim for HOLD positions when Regime drops sharply (cur_exp > target_exp + 5%)
        cur_mv = sum(float(p["market_value"]) for p in pos_map.values())
        cur_total_exp = cur_mv / max(total_equity, 1.0)
        target_exp = float(bundle.regime.get("target_exposure", 0.55))
        macro_deleveraging = cur_total_exp > target_exp + 0.05
        eff_trim_tol = 0.01 if macro_deleveraging else 0.06
        for rec in [r for r in bundle.recommendations if r["action"] == "HOLD"]:
            pos = pos_map.get(rec["ts_code"])
            if not pos or pos["sellable_shares"] < LOT_SIZE:
                continue
            tw = float(rec["suggested_weight"])
            cur_w = float(pos["market_value"]) / max(total_equity, 1.0)
            if cur_w - tw > eff_trim_tol:
                qf = _quote_fields(rec["ts_code"], rec["ref_price"])
                px = qf["ref_price"]
                trim_qty = min(int(pos["sellable_shares"]), round_buy_lots(int(((cur_w - tw) * total_equity) / max(px, 0.01))))
                if trim_qty >= LOT_SIZE:
                    tag = "大盘降仓风控减持" if macro_deleveraging else "仓位再平衡降权"
                    req = OrderRequest(
                        account_id=account_id,
                        ts_code=rec["ts_code"],
                        name=rec["name"],
                        sw_l1_name=rec["sw_l1_name"],
                        side="SELL",
                        qty=trim_qty,
                        signal_id=rec["signal_id"],
                        stop_loss_price=rec["stop_loss_price"],
                        take_profit_price=rec["take_profit_price"],
                        suggested_weight=tw,
                        reason_summary=f"{tag}至{tw:.1%}",
                        top_factors=rec["top_factors"],
                        tech_patterns=rec["tech_patterns"],
                        **qf,
                    )
                    receipt = self.gateway.submit_order(req, override_mode=override_mode)
                    receipts.append(receipt.to_dict())

        # Refresh equity after trims
        state = self.gateway.paper.get_account_state(account_id)
        total_equity = float(state["account"]["total_equity"])
        pos_map = {p["ts_code"]: p for p in state["positions"]}

        # 2) Execute BUYs (new entries or significantly underweight targets)
        for rec in [r for r in bundle.recommendations if r["action"] in ("BUY", "HOLD")]:
            tw = float(rec["suggested_weight"])
            if tw <= 0:
                continue
            qf = _quote_fields(rec["ts_code"], rec["ref_price"])
            px = qf["ref_price"]
            if px <= 0:
                continue
            is_held = rec["ts_code"] in pos_map
            cur_val = float(pos_map[rec["ts_code"]]["market_value"]) if is_held else 0.0
            cur_w = cur_val / max(total_equity, 1.0)
            # Avoid micro-churn top-ups on already-held stocks
            if is_held and (tw - cur_w) < 0.03:
                continue
            delta_val = total_equity * tw - cur_val
            if delta_val < px * LOT_SIZE:
                continue
            qty = round_buy_lots(int(delta_val / (px * 1.002)))
            if qty < LOT_SIZE:
                continue
            req = OrderRequest(
                account_id=account_id,
                ts_code=rec["ts_code"],
                name=rec["name"],
                sw_l1_name=rec["sw_l1_name"],
                side="BUY",
                qty=qty,
                signal_id=rec["signal_id"],
                stop_loss_price=rec["stop_loss_price"],
                take_profit_price=rec["take_profit_price"],
                suggested_weight=tw,
                reason_summary=rec["rationale"],
                top_factors=rec["top_factors"],
                tech_patterns=rec["tech_patterns"],
                **qf,
            )
            receipt = self.gateway.submit_order(req, override_mode=override_mode)
            receipts.append(receipt.to_dict())

        nav_point = self.record_daily_nav(account_id, regime=bundle.regime["regime"])
        final_state = self.gateway.paper.get_account_state(account_id)
        ws_manager.broadcast_threadsafe("paper_nav", to_jsonable(final_state["account"]))

        return {
            "trade_date": bundle.trade_date,
            "model_version_id": bundle.model_version_id,
            "regime": bundle.regime,
            "recommendations_count": len(bundle.recommendations),
            "orders_executed": receipts,
            "account": final_state["account"],
            "positions_count": len(final_state["positions"]),
            "nav_snapshot": nav_point,
        }

    def record_daily_nav(self, account_id: str = "default_paper", regime: str | None = None) -> dict:
        """Persist daily NAV snapshot into SQLite paper_nav_daily aligned with real benchmark indices."""
        latest_db_date = self.db.query_scalar("SELECT MAX(trade_date) FROM daily_bars")
        if regime is None:
            reg_val = self.db.query_scalar("SELECT regime FROM market_regime_daily ORDER BY trade_date DESC LIMIT 1")
            regime = str(reg_val) if reg_val else "NEUTRAL"
        today = today_cn()
        is_open_today = self.db.query_scalar("SELECT is_open FROM trade_calendar WHERE trade_date = ?", [today])
        if is_open_today:
            nav_date = today
        elif latest_db_date is not None:
            nav_date = pd.to_datetime(latest_db_date).date()
        else:
            nav_date = today

        with self.sqlite.session_scope() as session:
            acct = session.get(PaperAccount, account_id)
            if acct is None:
                return {}
            prev_rows = (
                session.query(PaperNavDaily)
                .filter(PaperNavDaily.account_id == account_id, PaperNavDaily.trade_date < nav_date)
                .order_by(PaperNavDaily.trade_date.asc())
                .all()
            )
            prev_eq = prev_rows[-1].total_equity if prev_rows else acct.initial_cash
            first_date = prev_rows[0].trade_date if prev_rows else nav_date

            daily_ret = round(acct.total_equity / max(prev_eq, 1.0) - 1.0, 6)
            cum_ret = round(acct.total_equity / max(acct.initial_cash, 1.0) - 1.0, 6)

            def _bench_cum(index_code: str) -> float:
                df = self.db.query_df(
                    "SELECT trade_date, close FROM index_daily WHERE index_code = ? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
                    [index_code, first_date, nav_date],
                )
                if len(df) >= 2 and df["close"].iloc[0] > 0:
                    return round(float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0), 6)
                return 0.0

            b1000 = _bench_cum(BENCHMARK_CSI1000)
            b300 = _bench_cum(BENCHMARK_CSI300)

            existing = (
                session.query(PaperNavDaily)
                .filter(PaperNavDaily.account_id == account_id, PaperNavDaily.trade_date == nav_date)
                .first()
            )
            if existing is None:
                existing = PaperNavDaily(account_id=account_id, trade_date=nav_date)
                session.add(existing)
            existing.total_equity = acct.total_equity
            existing.cash = acct.cash
            existing.market_value = round(acct.total_equity - acct.cash, 2)
            existing.daily_return = daily_ret
            existing.cum_return = cum_ret
            existing.csi1000_cum_return = b1000
            existing.csi300_cum_return = b300
            existing.excess_return_1000 = round(cum_ret - b1000, 6)
            existing.drawdown = acct.current_drawdown
            existing.regime = regime
            session.flush()
            return {
                "trade_date": nav_date.isoformat(),
                "total_equity": existing.total_equity,
                "cum_return": existing.cum_return,
                "csi1000_cum_return": existing.csi1000_cum_return,
                "csi300_cum_return": existing.csi300_cum_return,
                "excess_return_1000": existing.excess_return_1000,
                "drawdown": existing.drawdown,
                "regime": existing.regime,
            }

    def get_nav_history(self, account_id: str = "default_paper") -> list[dict]:
        with self.sqlite.session_scope() as session:
            rows = (
                session.query(PaperNavDaily)
                .filter(PaperNavDaily.account_id == account_id)
                .order_by(PaperNavDaily.trade_date.asc())
                .all()
            )
            return [
                {
                    "trade_date": r.trade_date.isoformat(),
                    "total_equity": r.total_equity,
                    "cash": r.cash,
                    "market_value": r.market_value,
                    "daily_return": r.daily_return,
                    "cum_return": r.cum_return,
                    "nav": round(1.0 + r.cum_return, 5),
                    "csi1000_nav": round(1.0 + r.csi1000_cum_return, 5),
                    "csi300_nav": round(1.0 + r.csi300_cum_return, 5),
                    "excess_return_1000": r.excess_return_1000,
                    "drawdown": r.drawdown,
                    "regime": r.regime,
                }
                for r in rows
            ]
