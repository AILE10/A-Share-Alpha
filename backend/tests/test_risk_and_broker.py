"""M5 tests: Unified RiskMiddleware interceptions (single stock <=12%, SW industry <=25%,
portfolio drawdown circuit breaker, order rate limiter, lot size), AssistedSignalAdapter
staging & manual fill reconciliation, and CompliantQMTAdapter dry-run & health diagnostics."""

from __future__ import annotations

import pytest

from app.broker.base import OrderRequest
from app.broker.gateway import BrokerGateway
from app.db.models import PaperAccount
from app.db.sqlite_client import SQLiteDatabase


def _make_req(**overrides) -> OrderRequest:
    base = dict(
        account_id="default_paper",
        ts_code="000001.SZ",
        name="平安银行",
        sw_l1_name="银行",
        side="BUY",
        qty=1000,
        ref_price=10.0,
        pre_close=10.0,
        ask1=10.0,
        bid1=9.99,
    )
    base.update(overrides)
    return OrderRequest(**base)


class TestRiskMiddleware:
    def test_single_stock_12pct_cap_blocks_oversized_buy(self, tmp_path):
        sqlite = SQLiteDatabase(tmp_path / "risk1.db", initial_cash=1_000_000.0)
        try:
            gw = BrokerGateway(sqlite)
            # Initial equity = 1,000,000 -> 12% cap = 120,000. Try buying 150,000 -> REJECT!
            res = gw.submit_order(_make_req(qty=15_000, ref_price=10.0), override_mode="PAPER")
            assert res.status == "REJECTED_RISK"
            assert "SINGLE_STOCK_CAP" in res.reject_reason
            logs = gw.risk.get_audit_logs()
            assert any(l["rule_name"] == "SINGLE_STOCK_CAP" and l["decision"] == "REJECT" for l in logs)
        finally:
            sqlite.close()

    def test_industry_25pct_cap_blocks_concentration(self, tmp_path):
        sqlite = SQLiteDatabase(tmp_path / "risk2.db", initial_cash=1_000_000.0)
        try:
            gw = BrokerGateway(sqlite)
            # Buy 100k + 100k in same industry "电子" -> 20% OK; third stock 80k -> 28% > 25% REJECT!
            r1 = gw.submit_order(_make_req(ts_code="002475.SZ", sw_l1_name="电子", qty=10_000, ref_price=10.0), "PAPER")
            r2 = gw.submit_order(_make_req(ts_code="002241.SZ", sw_l1_name="电子", qty=10_000, ref_price=10.0), "PAPER")
            assert r1.status == "FILLED" and r2.status == "FILLED"
            r3 = gw.submit_order(_make_req(ts_code="300750.SZ", sw_l1_name="电子", qty=8_000, ref_price=10.0), "PAPER")
            assert r3.status == "REJECTED_RISK"
            assert "INDUSTRY_CONCENTRATION_CAP" in r3.reject_reason
        finally:
            sqlite.close()

    def test_drawdown_circuit_breaker_halts_buys(self, tmp_path):
        sqlite = SQLiteDatabase(tmp_path / "risk3.db", initial_cash=1_000_000.0)
        try:
            with sqlite.session_scope() as s:
                acct = s.get(PaperAccount, "default_paper")
                acct.current_drawdown = -0.14
                acct.fuse_status = "HALTED"
            gw = BrokerGateway(sqlite)
            res = gw.submit_order(_make_req(qty=1000, ref_price=10.0), "PAPER")
            assert res.status == "REJECTED_RISK"
            assert "DRAWDOWN_CIRCUIT_BREAKER" in res.reject_reason
        finally:
            sqlite.close()

    def test_order_rate_limiter_blocks_burst(self, tmp_path):
        sqlite = SQLiteDatabase(tmp_path / "risk4.db", initial_cash=1_000_000.0)
        try:
            gw = BrokerGateway(sqlite)
            gw.risk.update_config({"max_orders_per_min": 2})
            r1 = gw.submit_order(_make_req(ts_code="000001.SZ", qty=100, ref_price=10.0), "PAPER")
            r2 = gw.submit_order(_make_req(ts_code="000002.SZ", qty=100, ref_price=10.0), "PAPER")
            r3 = gw.submit_order(_make_req(ts_code="000004.SZ", qty=100, ref_price=10.0), "PAPER")
            assert r1.status == "FILLED" and r2.status == "FILLED"
            assert r3.status == "REJECTED_RISK" and "RATE_LIMIT_MINUTE" in r3.reject_reason
        finally:
            sqlite.close()


class TestAssistedAndQMTAdapters:
    def test_assisted_stage_and_manual_fill_reconciliation(self, tmp_path):
        sqlite = SQLiteDatabase(tmp_path / "assist.db", initial_cash=1_000_000.0)
        staged = []
        try:
            gw = BrokerGateway(sqlite, on_stage_order=lambda p: staged.append(p))
            gw.update_config({"active_mode": "ASSISTED"})
            receipt = gw.submit_order(_make_req(qty=2000, ref_price=12.5, sw_l1_name="银行"))
            assert receipt.status == "PENDING_CONFIRM"
            assert len(staged) == 1
            pending = gw.assisted.list_pending()
            assert len(pending) == 1 and pending[0]["order_id"] == receipt.order_id

            # User executes manually in broker app at 12.55 and reconciles fill back
            rec = gw.assisted.confirm_manual_fill(receipt.order_id, filled_qty=2000, avg_fill_price=12.55)
            assert rec["ok"]
            assert len(gw.assisted.list_pending()) == 0
            state = gw.assisted.get_account_state()
            assert len(state["positions"]) == 1
            assert state["positions"][0]["total_shares"] == 2000
            assert state["positions"][0]["avg_cost"] == pytest.approx(12.55)
            assert state["account"]["cash"] < 1_000_000.0
        finally:
            sqlite.close()

    def test_compliant_qmt_dry_run_and_health(self, tmp_path):
        sqlite = SQLiteDatabase(tmp_path / "qmt.db", initial_cash=1_000_000.0)
        try:
            gw = BrokerGateway(sqlite)
            gw.update_config({"active_mode": "QMT", "qmt_dry_run": True})
            health = gw.qmt.health_check()
            assert health["ok"] is True and health["dry_run"] is True
            assert "严禁" in health["compliance_note"]

            res = gw.submit_order(_make_req(qty=1000, ref_price=10.0))
            assert res.status == "SUBMITTED"
            assert res.extra["dry_run"] is True
            assert res.extra["xtquant_payload"]["stock_code"] == "000001.SZ"
        finally:
            sqlite.close()
