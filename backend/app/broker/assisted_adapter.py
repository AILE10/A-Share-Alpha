"""Semi-automatic AssistedSignalAdapter (UI confirmation modal + WeCom/DingTalk webhook + manual fill reconciliation)."""

from __future__ import annotations

import uuid
from typing import Callable

from app.backtest.matching import compute_fees
from app.broker.base import BaseBrokerAdapter, OrderReceipt, OrderRequest
from app.broker.paper_adapter import PaperBrokerAdapter
from app.broker.webhook import format_trade_ticket_markdown, send_dingtalk_markdown, send_wecom_markdown
from app.core.calendar import today_cn
from app.db.models import BrokerConfig, PaperAccount, PaperOrder, PaperPosition
from app.db.sqlite_client import SQLiteDatabase, get_sqlite_db


class AssistedSignalAdapter(BaseBrokerAdapter):
    """Stages compliant trade tickets for manual brokerage execution and reconciles real fills back to ledger."""

    mode = "ASSISTED"

    def __init__(
        self,
        sqlite_db: SQLiteDatabase | None = None,
        on_stage_order: Callable[[dict], None] | None = None,
    ) -> None:
        self.sqlite = sqlite_db or get_sqlite_db()
        self.on_stage_order = on_stage_order

    def submit_order(self, req: OrderRequest) -> OrderReceipt:
        order_id = f"ast_{uuid.uuid4().hex[:12]}"
        ticket_md = format_trade_ticket_markdown(req, order_id)

        with self.sqlite.session_scope() as session:
            bcfg = session.get(BrokerConfig, 1)
            wecom_url = bcfg.wecom_webhook_url if bcfg else ""
            dt_url = bcfg.dingtalk_webhook_url if bcfg else ""
            dt_secret = bcfg.dingtalk_secret if bcfg else ""

            order_row = PaperOrder(
                order_id=order_id,
                account_id=req.account_id,
                ts_code=req.ts_code,
                name=req.name,
                sw_l1_name=req.sw_l1_name,
                side=req.side,
                order_type="ASSISTED_LIMIT",
                request_qty=req.qty,
                filled_qty=0,
                limit_price=req.ref_price,
                avg_fill_price=0.0,
                status="PENDING_CONFIRM",
                broker_mode=self.mode,
                signal_id=req.signal_id,
                reason_summary=req.reason_summary,
            )
            session.add(order_row)

        notifications = []
        if wecom_url:
            notifications.append(send_wecom_markdown(wecom_url, ticket_md))
        if dt_url:
            notifications.append(send_dingtalk_markdown(dt_url, f"辅助交易单 {req.side} {req.ts_code}", ticket_md, dt_secret))

        payload = {
            "order_id": order_id,
            "account_id": req.account_id,
            "ts_code": req.ts_code,
            "name": req.name,
            "sw_l1_name": req.sw_l1_name,
            "side": req.side,
            "request_qty": req.qty,
            "limit_price": req.ref_price,
            "stop_loss_price": req.stop_loss_price,
            "take_profit_price": req.take_profit_price,
            "suggested_weight": req.suggested_weight,
            "top_factors": req.top_factors,
            "tech_patterns": req.tech_patterns,
            "reason_summary": req.reason_summary,
            "markdown": ticket_md,
            "notifications": notifications,
        }
        if self.on_stage_order is not None:
            try:
                self.on_stage_order(payload)
            except Exception:  # noqa: BLE001
                pass

        return OrderReceipt(
            order_id=order_id,
            ts_code=req.ts_code,
            side=req.side,
            broker_mode=self.mode,
            status="PENDING_CONFIRM",
            requested_qty=req.qty,
            extra={"markdown": ticket_md, "notifications": notifications},
        )

    def confirm_manual_fill(
        self,
        order_id: str,
        filled_qty: int,
        avg_fill_price: float,
        commission: float | None = None,
        stamp_duty: float | None = None,
        transfer_fee: float | None = None,
    ) -> dict:
        """Write real manually-executed brokerage fill back into local positions & cash ledger."""
        if filled_qty <= 0 or avg_fill_price <= 0:
            raise ValueError("filled_qty and avg_fill_price must be positive")

        with self.sqlite.session_scope() as session:
            order = session.get(PaperOrder, order_id)
            if order is None:
                raise KeyError(f"Order not found: {order_id}")
            if order.status != "PENDING_CONFIRM":
                raise ValueError(f"Order {order_id} status is {order.status}, cannot confirm fill")

            acct = session.get(PaperAccount, order.account_id)
            if acct is None:
                raise KeyError(f"Account not found: {order.account_id}")

            gross = round(filled_qty * avg_fill_price, 2)
            auto_fees = compute_fees(order.side, gross)  # type: ignore[arg-type]
            comm = round(commission, 2) if commission is not None else auto_fees.commission
            stamp = round(stamp_duty, 2) if stamp_duty is not None else auto_fees.stamp_duty
            xfer = round(transfer_fee, 2) if transfer_fee is not None else auto_fees.transfer_fee
            slip = round(abs(avg_fill_price - order.limit_price) * filled_qty, 2) if order.limit_price > 0 else 0.0
            total_fees = comm + stamp + xfer

            order.filled_qty = filled_qty
            order.avg_fill_price = avg_fill_price
            order.commission = comm
            order.stamp_duty = stamp
            order.transfer_fee = xfer
            order.slippage_cost = slip
            order.status = "FILLED"

            pos = (
                session.query(PaperPosition)
                .filter(PaperPosition.account_id == order.account_id, PaperPosition.ts_code == order.ts_code)
                .first()
            )
            if order.side == "BUY":
                acct.cash = round(float(acct.cash) - gross - total_fees, 2)
                if pos is None:
                    pos = PaperPosition(
                        account_id=order.account_id,
                        ts_code=order.ts_code,
                        name=order.name,
                        sw_l1_name=order.sw_l1_name,
                        total_shares=filled_qty,
                        sellable_shares=0,  # T+1
                        avg_cost=avg_fill_price,
                        entry_date=today_cn().isoformat(),
                        highest_price_since_entry=avg_fill_price,
                        current_price=avg_fill_price,
                        stop_loss_price=round(avg_fill_price * 0.92, 2),
                        atr_stop_price=round(avg_fill_price * 0.92, 2),
                        market_value=gross,
                    )
                    session.add(pos)
                else:
                    new_total = pos.total_shares + filled_qty
                    pos.avg_cost = round((pos.avg_cost * pos.total_shares + gross) / new_total, 4)
                    pos.total_shares = new_total
                    pos.current_price = avg_fill_price
            else:
                acct.cash = round(float(acct.cash) + gross - total_fees, 2)
                if pos is not None:
                    pos.total_shares = max(0, pos.total_shares - filled_qty)
                    pos.sellable_shares = max(0, pos.sellable_shares - filled_qty)
                    if pos.total_shares == 0:
                        session.delete(pos)
                    else:
                        pos.current_price = avg_fill_price

            PaperBrokerAdapter._recompute_account_equity(session, acct)
            return {
                "ok": True,
                "order_id": order.order_id,
                "ts_code": order.ts_code,
                "side": order.side,
                "filled_qty": filled_qty,
                "avg_fill_price": avg_fill_price,
                "total_fees": round(total_fees, 2),
                "slippage_cost": slip,
                "account_cash": acct.cash,
                "account_equity": acct.total_equity,
            }

    def list_pending(self, account_id: str = "default_paper") -> list[dict]:
        with self.sqlite.session_scope() as session:
            rows = (
                session.query(PaperOrder)
                .filter(PaperOrder.account_id == account_id, PaperOrder.status == "PENDING_CONFIRM")
                .order_by(PaperOrder.created_at.desc())
                .all()
            )
            return [
                {
                    "order_id": r.order_id,
                    "ts_code": r.ts_code,
                    "name": r.name,
                    "sw_l1_name": r.sw_l1_name,
                    "side": r.side,
                    "request_qty": r.request_qty,
                    "limit_price": r.limit_price,
                    "reason_summary": r.reason_summary,
                    "status": r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                }
                for r in rows
            ]

    def cancel_order(self, order_id: str) -> dict:
        with self.sqlite.session_scope() as session:
            row = session.get(PaperOrder, order_id)
            if row is None:
                return {"ok": False, "reason": "订单不存在"}
            if row.status != "PENDING_CONFIRM":
                return {"ok": False, "reason": f"订单状态为 {row.status}，无法撤单"}
            row.status = "CANCELLED"
            return {"ok": True, "order_id": order_id, "status": "CANCELLED"}

    def get_account_state(self, account_id: str = "default_paper") -> dict:
        return PaperBrokerAdapter(self.sqlite).get_account_state(account_id)

    def health_check(self) -> dict:
        with self.sqlite.session_scope() as session:
            cfg = session.get(BrokerConfig, 1)
            has_wecom = bool(cfg and cfg.wecom_webhook_url)
            has_dt = bool(cfg and cfg.dingtalk_webhook_url)
        return {
            "adapter": "AssistedSignalAdapter",
            "mode": self.mode,
            "ok": True,
            "wecom_configured": has_wecom,
            "dingtalk_configured": has_dt,
            "detail": "半自动辅助模式就绪（支持前端确认弹窗、Webhook 推送与实盘成交回写）",
        }
