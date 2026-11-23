"""Compliant QMT (迅投 MiniQMT xtquant) / PTrade bridge skeleton (Guardrail #4).

STRICT COMPLIANCE NOTICE:
  - Reserved exclusively for individual/institutional investors who have officially applied
    for and received brokerage-authorized programmatic trading access (MiniQMT / PTrade).
  - Contains ZERO reverse-engineering, packet sniffing, or GUI auto-clicking code.
  - Defaults to `qmt_dry_run=True` unless explicitly connected to an authorized MiniQMT session.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

from app.broker.base import BaseBrokerAdapter, OrderReceipt, OrderRequest
from app.broker.paper_adapter import PaperBrokerAdapter
from app.db.models import BrokerConfig, PaperOrder
from app.db.sqlite_client import SQLiteDatabase, get_sqlite_db


class CompliantQMTAdapter(BaseBrokerAdapter):
    mode = "QMT"

    def __init__(self, sqlite_db: SQLiteDatabase | None = None) -> None:
        self.sqlite = sqlite_db or get_sqlite_db()

    def _load_cfg(self) -> tuple[str, str, bool]:
        with self.sqlite.session_scope() as session:
            cfg = session.get(BrokerConfig, 1)
            if cfg is None:
                return ("", "", True)
            return (cfg.qmt_userdata_path or "", cfg.qmt_account_id or "", bool(cfg.qmt_dry_run))

    def _xtquant_installed(self) -> bool:
        return importlib.util.find_spec("xtquant") is not None

    def health_check(self) -> dict:
        userdata_path, account_id, dry_run = self._load_cfg()
        xt_ok = self._xtquant_installed()
        path_ok = bool(userdata_path) and Path(userdata_path).exists()
        acct_ok = bool(account_id)
        ready_live = xt_ok and path_ok and acct_ok and (not dry_run)
        issues = []
        if not xt_ok:
            issues.append("未检测到官方 xtquant SDK（需向开户券商申请迅投 MiniQMT 合规客户端及 xtquant 包）")
        if not path_ok:
            issues.append("未配置有效的 MiniQMT userdata_mini 路径")
        if not acct_ok:
            issues.append("未配置券商资金账号")
        return {
            "adapter": "CompliantQMTAdapter",
            "mode": self.mode,
            "ok": True if dry_run else ready_live,
            "ready_for_live": ready_live,
            "dry_run": dry_run,
            "checks": {
                "xtquant_installed": xt_ok,
                "userdata_path_exists": path_ok,
                "account_id_configured": acct_ok,
            },
            "issues": issues,
            "compliance_note": "仅通过券商官方授权 MiniQMT/PTrade 接口桥接，严禁任何模拟点击或逆向抓包。",
            "guide_doc": "docs/COMPLIANT_QMT_GUIDE.md",
        }

    def submit_order(self, req: OrderRequest) -> OrderReceipt:
        userdata_path, account_id, dry_run = self._load_cfg()
        order_id = f"qmt_{uuid.uuid4().hex[:12]}"

        if dry_run or not self._xtquant_installed() or not userdata_path or not account_id:
            status = "SUBMITTED" if dry_run else "REJECTED_RISK"
            reason = (
                "QMT Dry-Run 合规演练模式：已通过前置风控并生成标准 xtquant 委托载荷，未向实盘发送"
                if dry_run
                else "未满足实盘条件（缺少 xtquant SDK / userdata 路径 / 资金账号），已安全拦截"
            )
            with self.sqlite.session_scope() as session:
                session.add(
                    PaperOrder(
                        order_id=order_id,
                        account_id=req.account_id,
                        ts_code=req.ts_code,
                        name=req.name,
                        sw_l1_name=req.sw_l1_name,
                        side=req.side,
                        order_type="QMT_DRY_RUN" if dry_run else "QMT_LIVE",
                        request_qty=req.qty,
                        filled_qty=0,
                        limit_price=req.ref_price,
                        status=status,
                        broker_mode=self.mode,
                        signal_id=req.signal_id,
                        reject_reason="" if dry_run else reason,
                        reason_summary=reason,
                    )
                )
            return OrderReceipt(
                order_id=order_id,
                ts_code=req.ts_code,
                side=req.side,
                broker_mode=self.mode,
                status=status,
                requested_qty=req.qty,
                reject_reason="" if dry_run else reason,
                extra={
                    "dry_run": dry_run,
                    "xtquant_payload": {
                        "stock_code": req.ts_code,
                        "order_type": "STOCK_BUY" if req.side == "BUY" else "STOCK_SELL",
                        "order_volume": req.qty,
                        "price_type": "FIX_PRICE",
                        "price": req.ref_price,
                        "strategy_name": "A-Share-Alpha-Studio",
                    },
                },
            )

        # Official authorized MiniQMT xtquant.xttrader bridge (executed only when dry_run=False & SDK present)
        from xtquant import xtconstant  # type: ignore[import-not-found]
        from xtquant.xttrader import XtQuantTrader  # type: ignore[import-not-found]
        from xtquant.xttype import StockAccount  # type: ignore[import-not-found]

        session_id = int(uuid.uuid4().int % 1000000)
        trader = XtQuantTrader(userdata_path, session_id)
        trader.start()
        if trader.connect() != 0:
            return OrderReceipt(
                order_id=order_id, ts_code=req.ts_code, side=req.side, broker_mode=self.mode,
                status="REJECTED_RISK", requested_qty=req.qty, reject_reason="MiniQMT 客户端连接失败",
            )
        acct = StockAccount(account_id)
        otype = xtconstant.STOCK_BUY if req.side == "BUY" else xtconstant.STOCK_SELL
        ext_id = trader.order_stock(
            acct, req.ts_code, otype, req.qty, xtconstant.FIX_PRICE, req.ref_price, "A-Share-Alpha-Studio", order_id
        )
        return OrderReceipt(
            order_id=order_id, ts_code=req.ts_code, side=req.side, broker_mode=self.mode,
            status="SUBMITTED" if ext_id > 0 else "REJECTED_RISK",
            requested_qty=req.qty,
            extra={"qmt_order_id": ext_id},
        )

    def cancel_order(self, order_id: str) -> dict:
        with self.sqlite.session_scope() as session:
            row = session.get(PaperOrder, order_id)
            if row:
                row.status = "CANCELLED"
        return {"ok": True, "order_id": order_id, "mode": self.mode}

    def get_account_state(self, account_id: str = "default_paper") -> dict:
        return PaperBrokerAdapter(self.sqlite).get_account_state(account_id)
