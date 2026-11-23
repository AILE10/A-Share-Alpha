"""Local high-precision PaperBrokerAdapter matching against real bid1/ask1 + slippage & exact A-share rules."""

from __future__ import annotations

import uuid

from app.backtest.matching import BarQuote, match_order
from app.broker.base import BaseBrokerAdapter, OrderReceipt, OrderRequest
from app.core.calendar import today_cn
from app.core.constants import DEFAULT_SLIPPAGE_BPS
from app.db.models import PaperAccount, PaperOrder, PaperPosition
from app.db.sqlite_client import SQLiteDatabase, get_sqlite_db


class PaperBrokerAdapter(BaseBrokerAdapter):
    mode = "PAPER"

    def __init__(self, sqlite_db: SQLiteDatabase | None = None, slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> None:
        self.sqlite = sqlite_db or get_sqlite_db()
        self.slippage_bps = slippage_bps

    def submit_order(self, req: OrderRequest) -> OrderReceipt:
        order_id = f"ppr_{uuid.uuid4().hex[:12]}"
        quote = BarQuote(
            ts_code=req.ts_code,
            open=req.open_price or req.ref_price,
            high=req.high_price or req.ref_price,
            low=req.low_price or req.ref_price,
            close=req.ref_price,
            pre_close=req.pre_close or req.ref_price,
            volume=req.volume,
            is_st=req.is_st,
            is_suspended=req.is_suspended,
            limit_up=req.limit_up or None,
            limit_down=req.limit_down or None,
            ask1=req.ask1 or req.ref_price,
            bid1=req.bid1 or req.ref_price,
        )

        with self.sqlite.session_scope() as session:
            acct = session.get(PaperAccount, req.account_id)
            if acct is None:
                raise ValueError(f"PaperAccount not found: {req.account_id}")
            pos = (
                session.query(PaperPosition)
                .filter(PaperPosition.account_id == req.account_id, PaperPosition.ts_code == req.ts_code)
                .first()
            )
            fill = match_order(
                quote=quote,
                side=req.side,
                requested_qty=req.qty,
                available_cash=float(acct.cash),
                sellable_shares=pos.sellable_shares if pos else 0,
                total_shares=pos.total_shares if pos else 0,
                slippage_bps=self.slippage_bps,
                price_mode="intraday",
            )

            status = "FILLED" if fill.filled else ("REJECTED_LIMIT" if "LIMIT" in fill.reject_code else "REJECTED_RISK")
            order_row = PaperOrder(
                order_id=order_id,
                account_id=req.account_id,
                ts_code=req.ts_code,
                name=req.name,
                sw_l1_name=req.sw_l1_name,
                side=req.side,
                order_type="INTRADAY_L1",
                request_qty=req.qty,
                filled_qty=fill.filled_qty,
                limit_price=req.ref_price,
                avg_fill_price=fill.exec_price,
                commission=fill.fees.commission,
                stamp_duty=fill.fees.stamp_duty,
                transfer_fee=fill.fees.transfer_fee,
                slippage_cost=fill.fees.slippage_cost,
                status=status,
                broker_mode=self.mode,
                signal_id=req.signal_id,
                reject_reason=fill.reject_reason,
                reason_summary=req.reason_summary,
            )
            session.add(order_row)

            if fill.filled:
                acct.cash = round(float(acct.cash) + fill.net_cash_delta, 2)
                if req.side == "BUY":
                    if pos is None:
                        pos = PaperPosition(
                            account_id=req.account_id,
                            ts_code=req.ts_code,
                            name=req.name,
                            sw_l1_name=req.sw_l1_name,
                            total_shares=fill.filled_qty,
                            sellable_shares=0,  # T+1 locked today!
                            avg_cost=fill.exec_price,
                            entry_date=today_cn().isoformat(),
                            highest_price_since_entry=fill.exec_price,
                            current_price=fill.exec_price,
                            stop_loss_price=req.stop_loss_price or round(fill.exec_price * 0.92, 2),
                            atr_stop_price=req.stop_loss_price or round(fill.exec_price * 0.92, 2),
                            market_value=round(fill.filled_qty * fill.exec_price, 2),
                            unrealized_pnl=0.0,
                            unrealized_pnl_pct=0.0,
                        )
                        session.add(pos)
                    else:
                        new_total = pos.total_shares + fill.filled_qty
                        pos.avg_cost = round(
                            (pos.avg_cost * pos.total_shares + fill.exec_price * fill.filled_qty) / new_total, 4
                        )
                        pos.total_shares = new_total
                        pos.current_price = fill.exec_price
                        pos.market_value = round(pos.total_shares * fill.exec_price, 2)
                else:
                    assert pos is not None
                    pos.total_shares -= fill.filled_qty
                    pos.sellable_shares = max(0, pos.sellable_shares - fill.filled_qty)
                    if pos.total_shares <= 0:
                        session.delete(pos)
                    else:
                        pos.current_price = fill.exec_price
                        pos.market_value = round(pos.total_shares * fill.exec_price, 2)

                self._recompute_account_equity(session, acct)

            return OrderReceipt(
                order_id=order_id,
                ts_code=req.ts_code,
                side=req.side,
                broker_mode=self.mode,
                status=status,
                requested_qty=req.qty,
                filled_qty=fill.filled_qty,
                avg_fill_price=fill.exec_price,
                commission=fill.fees.commission,
                stamp_duty=fill.fees.stamp_duty,
                transfer_fee=fill.fees.transfer_fee,
                slippage_cost=fill.fees.slippage_cost,
                reject_reason=fill.reject_reason,
            )

    @staticmethod
    def _recompute_account_equity(session, acct: PaperAccount) -> None:
        from app.broker.drawdown_fuse import compute_rolling_peak
        from app.db.models import PaperNavDaily

        # Flush newly added / deleted positions before querying (SessionLocal uses autoflush=False)
        session.flush()
        positions = session.query(PaperPosition).filter(PaperPosition.account_id == acct.account_id).all()
        mv = sum(p.total_shares * (p.current_price or p.avg_cost) for p in positions)
        acct.total_equity = round(float(acct.cash) + mv, 2)
        nav_rows = (
            session.query(PaperNavDaily.total_equity)
            .filter(PaperNavDaily.account_id == acct.account_id)
            .order_by(PaperNavDaily.trade_date.desc())
            .limit(120)
            .all()
        )
        hist_eq = [float(r[0]) for r in reversed(nav_rows)]
        acct.peak_equity = round(
            compute_rolling_peak(hist_eq, acct.total_equity, acct.peak_equity, window_days=120),
            2,
        )
        acct.current_drawdown = round(acct.total_equity / max(acct.peak_equity, 1.0) - 1.0, 6)
        for p in positions:
            p.market_value = round(p.total_shares * (p.current_price or p.avg_cost), 2)
            cost_basis = p.total_shares * p.avg_cost
            p.unrealized_pnl = round(p.market_value - cost_basis, 2)
            p.unrealized_pnl_pct = round(p.unrealized_pnl / max(cost_basis, 1.0), 6)
            p.weight = round(p.market_value / max(acct.total_equity, 1.0), 6)

    def cancel_order(self, order_id: str) -> dict:
        with self.sqlite.session_scope() as session:
            row = session.get(PaperOrder, order_id)
            if row is None:
                return {"ok": False, "reason": "订单不存在"}
            if row.status in ("FILLED", "CANCELLED"):
                return {"ok": False, "reason": f"订单状态为 {row.status}，无法撤单"}
            row.status = "CANCELLED"
            return {"ok": True, "order_id": order_id, "status": "CANCELLED"}

    def get_account_state(self, account_id: str = "default_paper") -> dict:
        with self.sqlite.session_scope() as session:
            acct = session.get(PaperAccount, account_id)
            if acct is None:
                return {}
            positions = (
                session.query(PaperPosition)
                .filter(PaperPosition.account_id == account_id)
                .order_by(PaperPosition.market_value.desc())
                .all()
            )
            orders = (
                session.query(PaperOrder)
                .filter(PaperOrder.account_id == account_id)
                .order_by(PaperOrder.created_at.desc())
                .limit(100)
                .all()
            )
            return {
                "account": {
                    "account_id": acct.account_id,
                    "name": acct.name,
                    "broker_mode": acct.broker_mode,
                    "initial_cash": acct.initial_cash,
                    "cash": acct.cash,
                    "total_equity": acct.total_equity,
                    "market_value": round(acct.total_equity - acct.cash, 2),
                    "total_return": round(acct.total_equity / max(acct.initial_cash, 1.0) - 1.0, 6),
                    "peak_equity": acct.peak_equity,
                    "current_drawdown": acct.current_drawdown,
                    "fuse_status": acct.fuse_status,
                    "updated_at": acct.updated_at.isoformat() if acct.updated_at else "",
                },
                "positions": [
                    {
                        "ts_code": p.ts_code,
                        "name": p.name,
                        "sw_l1_name": p.sw_l1_name,
                        "total_shares": p.total_shares,
                        "sellable_shares": p.sellable_shares,
                        "avg_cost": p.avg_cost,
                        "entry_date": p.entry_date,
                        "highest_price_since_entry": p.highest_price_since_entry,
                        "current_price": p.current_price,
                        "stop_loss_price": p.stop_loss_price,
                        "atr_stop_price": p.atr_stop_price,
                        "market_value": p.market_value,
                        "unrealized_pnl": p.unrealized_pnl,
                        "unrealized_pnl_pct": p.unrealized_pnl_pct,
                        "weight": p.weight,
                    }
                    for p in positions
                ],
                "orders": [
                    {
                        "order_id": o.order_id,
                        "ts_code": o.ts_code,
                        "name": o.name,
                        "sw_l1_name": o.sw_l1_name,
                        "side": o.side,
                        "request_qty": o.request_qty,
                        "filled_qty": o.filled_qty,
                        "limit_price": o.limit_price,
                        "avg_fill_price": o.avg_fill_price,
                        "commission": o.commission,
                        "stamp_duty": o.stamp_duty,
                        "transfer_fee": o.transfer_fee,
                        "slippage_cost": o.slippage_cost,
                        "status": o.status,
                        "broker_mode": o.broker_mode,
                        "reject_reason": o.reject_reason,
                        "reason_summary": o.reason_summary,
                        "created_at": o.created_at.isoformat() if o.created_at else "",
                    }
                    for o in orders
                ],
            }

    def health_check(self) -> dict:
        return {"adapter": "PaperBrokerAdapter", "mode": self.mode, "ok": True, "detail": "本地高精度 A 股模拟撮合引擎就绪"}
