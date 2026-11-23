"""Unified BrokerGateway orchestrating RiskMiddleware + Paper/Assisted/QMT adapters."""

from __future__ import annotations

from typing import Callable

from app.broker.assisted_adapter import AssistedSignalAdapter
from app.broker.base import BaseBrokerAdapter, BrokerMode, OrderReceipt, OrderRequest
from app.broker.paper_adapter import PaperBrokerAdapter
from app.broker.qmt_adapter import CompliantQMTAdapter
from app.broker.risk_middleware import RiskMiddleware
from app.broker.webhook import send_dingtalk_markdown, send_wecom_markdown
from app.db.models import BrokerConfig, PaperAccount
from app.db.sqlite_client import SQLiteDatabase, get_sqlite_db


class BrokerGateway:
    """Single entry point for order execution. Guarantees RiskMiddleware wraps EVERY broker mode."""

    def __init__(
        self,
        sqlite_db: SQLiteDatabase | None = None,
        risk: RiskMiddleware | None = None,
        on_stage_order: Callable[[dict], None] | None = None,
    ) -> None:
        self.sqlite = sqlite_db or get_sqlite_db()
        self.risk = risk or RiskMiddleware(self.sqlite)
        self.paper = PaperBrokerAdapter(self.sqlite)
        self.assisted = AssistedSignalAdapter(self.sqlite, on_stage_order=on_stage_order)
        self.qmt = CompliantQMTAdapter(self.sqlite)

    def get_config(self) -> dict:
        with self.sqlite.session_scope() as session:
            cfg = session.get(BrokerConfig, 1)
            if cfg is None:
                cfg = BrokerConfig(id=1)
                session.add(cfg)
                session.flush()
            return {
                "active_mode": cfg.active_mode,
                "wecom_webhook_url": cfg.wecom_webhook_url,
                "dingtalk_webhook_url": cfg.dingtalk_webhook_url,
                "dingtalk_secret": "***" if cfg.dingtalk_secret else "",
                "has_dingtalk_secret": bool(cfg.dingtalk_secret),
                "qmt_userdata_path": cfg.qmt_userdata_path,
                "qmt_account_id": cfg.qmt_account_id,
                "qmt_dry_run": bool(cfg.qmt_dry_run),
                "tushare_token_configured": bool(cfg.tushare_token_configured),
                "adapters_health": {
                    "PAPER": self.paper.health_check(),
                    "ASSISTED": self.assisted.health_check(),
                    "QMT": self.qmt.health_check(),
                },
            }

    def update_config(self, updates: dict) -> dict:
        with self.sqlite.session_scope() as session:
            cfg = session.get(BrokerConfig, 1)
            if cfg is None:
                cfg = BrokerConfig(id=1)
                session.add(cfg)
            if "active_mode" in updates and updates["active_mode"] in ("PAPER", "ASSISTED", "QMT"):
                cfg.active_mode = updates["active_mode"]
                acct = session.get(PaperAccount, "default_paper")
                if acct is not None:
                    acct.broker_mode = cfg.active_mode
            for key in ("wecom_webhook_url", "dingtalk_webhook_url", "qmt_userdata_path", "qmt_account_id", "qmt_dry_run"):
                if key in updates and updates[key] is not None:
                    setattr(cfg, key, updates[key])
            if "dingtalk_secret" in updates and updates["dingtalk_secret"] not in (None, "***"):
                cfg.dingtalk_secret = updates["dingtalk_secret"]
            session.flush()
        return self.get_config()

    def active_adapter(self, override_mode: BrokerMode | None = None) -> BaseBrokerAdapter:
        mode = override_mode or self.get_config()["active_mode"]
        if mode == "ASSISTED":
            return self.assisted
        if mode == "QMT":
            return self.qmt
        return self.paper

    def submit_order(self, req: OrderRequest, override_mode: BrokerMode | None = None) -> OrderReceipt:
        adapter = self.active_adapter(override_mode)
        return self.risk.execute(adapter, req)

    def test_webhooks(self) -> list[dict]:
        with self.sqlite.session_scope() as session:
            cfg = session.get(BrokerConfig, 1)
            wecom_url = cfg.wecom_webhook_url if cfg else ""
            dt_url = cfg.dingtalk_webhook_url if cfg else ""
            dt_secret = cfg.dingtalk_secret if cfg else ""
        msg = "### A-Share Alpha Studio 连通性测试\n- Webhook 通道正常，后续辅助交易单将实时推送至此群。"
        return [
            send_wecom_markdown(wecom_url, msg),
            send_dingtalk_markdown(dt_url, "A-Share Alpha Studio 测试", msg, dt_secret),
        ]
