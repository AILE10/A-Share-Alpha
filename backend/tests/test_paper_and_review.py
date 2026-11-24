"""M3 tests: Paper trading matching & T+1 unlock, explainable recommendation attribution,
weekly review report generation, APScheduler job registration, and ModelRegistry snapshot/rollback."""

from __future__ import annotations

import pytest

from app.broker.base import OrderRequest
from app.broker.gateway import BrokerGateway
from app.db.duckdb_client import DuckDBClient
from app.db.sqlite_client import SQLiteDatabase
from app.ml.model_registry import ModelRegistry
from app.paper.recommender import RecommendationEngine
from app.paper.reviewer import WeeklyReviewEngine
from app.paper.scheduler import StudioScheduler
from app.paper.service import PaperTradingService


def test_paper_t_plus_1_and_unlock_cycle(tmp_path):
    sqlite = SQLiteDatabase(tmp_path / "m3_test.db", initial_cash=1_000_000.0)
    try:
        gw = BrokerGateway(sqlite)
        req_buy = OrderRequest(
            account_id="default_paper",
            ts_code="600519.SH",
            name="贵州茅台",
            sw_l1_name="食品饮料",
            side="BUY",
            qty=100,
            ref_price=500.0,
            pre_close=500.0,
            ask1=500.0,
            bid1=499.9,
        )
        r_buy = gw.submit_order(req_buy, override_mode="PAPER")
        assert r_buy.status == "FILLED" and r_buy.filled_qty == 100
        # Guard against autoflush=False missing newly added position in total_equity & weight
        st_after_buy = gw.paper.get_account_state("default_paper")
        assert abs(st_after_buy["account"]["total_equity"] - (st_after_buy["account"]["cash"] + st_after_buy["account"]["market_value"])) < 0.02
        assert st_after_buy["account"]["market_value"] > 49_000.0
        assert st_after_buy["positions"][0]["weight"] > 0.04

        # Immediate same-day sell MUST be rejected by T+1
        req_sell = OrderRequest(
            account_id="default_paper",
            ts_code="600519.SH",
            name="贵州茅台",
            sw_l1_name="食品饮料",
            side="SELL",
            qty=100,
            ref_price=505.0,
            pre_close=500.0,
            bid1=505.0,
        )
        r_sell_blocked = gw.submit_order(req_sell, override_mode="PAPER")
        assert r_sell_blocked.status == "REJECTED_RISK"
        assert "T_PLUS_1" in r_sell_blocked.reject_reason

        # Pre-market unlock -> next day sell fills cleanly
        tmp_duck = DuckDBClient(tmp_path / "m3_duck.duckdb")
        try:
            paper_svc = PaperTradingService(duckdb_client=tmp_duck, sqlite_db=sqlite, gateway=gw)
            unlocked = paper_svc.pre_market_unlock()
            assert unlocked == 1
            r_sell_ok = gw.submit_order(req_sell, override_mode="PAPER")
            assert r_sell_ok.status == "FILLED" and r_sell_ok.filled_qty == 100
            assert r_sell_ok.stamp_duty > 0
        finally:
            tmp_duck.close()
    finally:
        sqlite.close()


def test_scheduler_registers_five_trading_jobs(tmp_path):
    sqlite = SQLiteDatabase(tmp_path / "sched_test.db")
    tmp_duck = DuckDBClient(tmp_path / "sched_duck.duckdb")
    try:
        paper_svc = PaperTradingService(duckdb_client=tmp_duck, sqlite_db=sqlite)
        sched = StudioScheduler(paper_svc)
        sched.start()
        ids = {j["id"] for j in sched.list_jobs()}
        sched.stop()
        assert ids == {
            "pre_market_unlock",
            "intraday_top100_refresh",
            "realtime_regime_refresh",
            "rebalance_1440",
            "post_market_sync",
            "weekly_review",
        }
    finally:
        tmp_duck.close()
        sqlite.close()


def test_real_data_recommender_model_rollback_and_weekly_review(tmp_path, real_db, sample_codes):
    sqlite = SQLiteDatabase(tmp_path / "m3_real.db")
    try:
        reg = ModelRegistry(sqlite_db=sqlite, models_dir=tmp_path / "models")
        v1, _ = reg.retrain_and_register(codes=sample_codes, model_type="lambdarank", notes="v1", duckdb_client=real_db)
        v2, _ = reg.retrain_and_register(codes=sample_codes, model_type="regression", notes="v2", duckdb_client=real_db)
        assert reg.get_active()["version_id"] == v2["version_id"]
        reg.activate_version(v1["version_id"])
        assert reg.get_active()["version_id"] == v1["version_id"]

        rec_engine = RecommendationEngine(duckdb_client=real_db, sqlite_db=sqlite, registry=reg)
        bundle = rec_engine.generate(codes=sample_codes, persist=True, broadcast=False)
        assert len(bundle.recommendations) > 0
        first = bundle.recommendations[0]
        assert len(first["top_factors"]) >= 3
        assert first["stop_loss_price"] > 0
        assert first["suggested_weight"] <= 0.12 + 1e-5

        paper_svc = PaperTradingService(duckdb_client=real_db, sqlite_db=sqlite)
        paper_svc.recommender = rec_engine
        rebal = paper_svc.run_rebalance(codes=sample_codes, override_mode="PAPER", fetch_live_quotes=False)
        target_exp = float(bundle.regime.get("target_exposure", 0.0))
        if target_exp > 1e-6:
            assert len(rebal["orders_executed"]) > 0
            assert rebal["positions_count"] > 0
        else:
            # On real DEFENSIVE (0%) extreme-bear dates, cash-protection MUST block all new BUYs
            assert rebal["positions_count"] == 0

        review = WeeklyReviewEngine(duckdb_client=real_db, sqlite_db=sqlite).generate_review()
        assert review["review_id"]
        assert len(review["optimization_advice"]) >= 1

        # Clean reset to 300,000 CNY wipes positions, orders, nav, signals & reviews
        reset_res = paper_svc.reset_account(initial_cash=300_000.0)
        assert reset_res["ok"] is True
        assert reset_res["account"]["initial_cash"] == pytest.approx(300_000.0)
        assert reset_res["account"]["cash"] == pytest.approx(300_000.0)
        assert reset_res["account"]["total_equity"] == pytest.approx(300_000.0)
        if target_exp > 1e-6:
            assert reset_res["cleared"]["positions"] >= 1
        assert reset_res["cleared"]["signals"] >= 1
        assert len(paper_svc.gateway.paper.get_account_state()["positions"]) == 0
        assert len(paper_svc.get_nav_history()) == 0
    finally:
        sqlite.close()
