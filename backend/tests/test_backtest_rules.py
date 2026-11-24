"""M2 tests: A-share matching rules (T+1, lots, price limits, fees), Purged Walk-Forward embargo,
zero look-ahead verification, hard portfolio constraints, technical resonance & real-data backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import AShareBacktestEngine, BacktestConfig
from app.backtest.matching import BarQuote, compute_fees, match_order
from app.core.exceptions import LookAheadBiasError
from app.factors.definitions import forward_excess_returns
from app.ml.lgbm_scorer import LGBMConfig, LGBMScorer
from app.ml.splitter import PurgedWalkForwardSplitter
from app.strategy.optimizer import PortfolioConfig, select_portfolio_names, solve_portfolio_weights
from app.strategy.regime import compute_market_regime
from app.strategy.technical import compute_technical_signals


# --------------------------------------------------------------------------- matching rules
class TestAShareMatchingRules:
    def test_t_plus_1_blocks_same_day_sell(self):
        q = BarQuote("600519.SH", open=100.0, high=102.0, low=99.0, close=101.0, pre_close=100.0, volume=100000)
        buy = match_order(q, "BUY", 200, available_cash=50_000.0, slippage_bps=0.0)
        assert buy.filled and buy.filled_qty == 200
        # Bought today -> sellable_shares == 0 -> MUST reject sell!
        sell_same_day = match_order(q, "SELL", 200, sellable_shares=0, total_shares=200, slippage_bps=0.0)
        assert not sell_same_day.filled
        assert sell_same_day.reject_code == "T_PLUS_1_LOCKED"
        # Next trading day unlock -> sellable_shares == 200 -> fills cleanly
        sell_next_day = match_order(q, "SELL", 200, sellable_shares=200, total_shares=200, slippage_bps=0.0)
        assert sell_next_day.filled and sell_next_day.filled_qty == 200

    def test_100_share_lot_size_enforcement(self):
        q = BarQuote("000001.SZ", open=10.0, high=10.5, low=9.8, close=10.2, pre_close=10.0, volume=100000)
        r_small = match_order(q, "BUY", 99, available_cash=100_000.0, slippage_bps=0.0)
        assert not r_small.filled and r_small.reject_code == "BELOW_LOT_SIZE"
        r_round = match_order(q, "BUY", 275, available_cash=100_000.0, slippage_bps=0.0)
        assert r_round.filled and r_round.filled_qty == 200
        # Cash only allows 230 shares -> scales down to 200 shares
        r_cash = match_order(q, "BUY", 500, available_cash=2350.0, slippage_bps=0.0)
        assert r_cash.filled and r_cash.filled_qty == 200

    def test_price_limit_up_and_down_locks(self):
        # Main board 10%: pre_close=10.00 -> limit_up=11.00, limit_down=9.00
        q_up = BarQuote("600000.SH", open=11.00, high=11.00, low=11.00, close=11.00, pre_close=10.00, volume=50000)
        buy_up = match_order(q_up, "BUY", 100, available_cash=50_000.0)
        assert not buy_up.filled and buy_up.reject_code == "LIMIT_UP_LOCKED"

        q_down = BarQuote("600000.SH", open=9.00, high=9.00, low=9.00, close=9.00, pre_close=10.00, volume=50000)
        sell_down = match_order(q_down, "SELL", 100, sellable_shares=100, total_shares=100)
        assert not sell_down.filled and sell_down.reject_code == "LIMIT_DOWN_LOCKED"

        # ChiNext 20%: pre_close=10.00 -> 11.50 is NOT limit up (limit_up = 12.00)
        q_gem_ok = BarQuote("300750.SZ", open=11.50, high=11.80, low=11.20, close=11.60, pre_close=10.00, volume=50000)
        assert match_order(q_gem_ok, "BUY", 100, available_cash=50_000.0).filled
        q_gem_up = BarQuote("300750.SZ", open=12.00, high=12.00, low=12.00, close=12.00, pre_close=10.00, volume=50000)
        assert match_order(q_gem_up, "BUY", 100, available_cash=50_000.0).reject_code == "LIMIT_UP_LOCKED"

        # BSE 30%: pre_close=10.00 -> limit_up=13.00
        q_bse_ok = BarQuote("830799.BJ", open=12.50, high=12.80, low=12.00, close=12.50, pre_close=10.00, volume=50000)
        assert match_order(q_bse_ok, "BUY", 100, available_cash=50_000.0).filled
        q_bse_up = BarQuote("830799.BJ", open=13.00, high=13.00, low=13.00, close=13.00, pre_close=10.00, volume=50000)
        assert match_order(q_bse_up, "BUY", 100, available_cash=50_000.0).reject_code == "LIMIT_UP_LOCKED"

        # Suspended blocks both
        q_susp = BarQuote("600000.SH", open=10.0, high=10.0, low=10.0, close=10.0, pre_close=10.0, volume=0, is_suspended=True)
        assert match_order(q_susp, "BUY", 100, available_cash=50_000.0).reject_code == "SUSPENDED"
        assert match_order(q_susp, "SELL", 100, sellable_shares=100, total_shares=100).reject_code == "SUSPENDED"

    def test_exact_a_share_fees(self):
        # Small trade 10,000 CNY: min 5 CNY commission floor, 0.001% transfer, 0.05% sell stamp duty
        f_buy_small = compute_fees("BUY", 10_000.0)
        assert f_buy_small.commission == 5.0
        assert f_buy_small.stamp_duty == 0.0
        assert f_buy_small.transfer_fee == 0.10

        f_sell_small = compute_fees("SELL", 10_000.0)
        assert f_sell_small.commission == 5.0
        assert f_sell_small.stamp_duty == 5.0
        assert f_sell_small.transfer_fee == 0.10

        # Large trade 100,000 CNY: 0.015% = 15.0 CNY (> 5 floor), sell stamp = 50.0 CNY, transfer = 1.0 CNY
        f_sell_big = compute_fees("SELL", 100_000.0)
        assert f_sell_big.commission == 15.0
        assert f_sell_big.stamp_duty == 50.0
        assert f_sell_big.transfer_fee == 1.0


# --------------------------------------------------------------------------- purged walk-forward & no look-ahead
class TestPurgedWalkForwardAndNoLookAhead:
    def test_embargo_less_than_5_raises(self):
        with pytest.raises(LookAheadBiasError):
            PurgedWalkForwardSplitter(train_window=252, embargo_days=4, test_window=63)

    def test_strict_purging_and_embargo_gap(self):
        dates = pd.bdate_range("2023-01-01", periods=450)
        splitter = PurgedWalkForwardSplitter(train_window=252, embargo_days=5, test_window=63, label_horizon=5)
        folds = splitter.split(dates)
        assert len(folds) >= 2
        for fold in folds:
            assert len(fold.embargo_dates) == 5
            assert len(fold.purged_dates) == 5
            train_end_pos = dates.get_loc(fold.train_end)
            test_start_pos = dates.get_loc(fold.test_start)
            # Even the label end date (train_end_pos + 5) is strictly >= 5 trading days before test_start
            assert test_start_pos - (train_end_pos + 5) >= 5
            assert set(fold.train_dates).isdisjoint(set(fold.test_dates))

    def test_technical_and_regime_zero_lookahead_on_real_data(self, real_panel):
        cut = real_panel.dates[-30]
        full_tech = compute_technical_signals(real_panel)
        trunc_tech = compute_technical_signals(real_panel.slice_until(cut))
        pd.testing.assert_series_equal(full_tech.passed.loc[cut], trunc_tech.passed.loc[cut], check_names=False)
        pd.testing.assert_series_equal(full_tech.macd_top_div.loc[cut], trunc_tech.macd_top_div.loc[cut], check_names=False)
        pd.testing.assert_series_equal(full_tech.atr_14_raw.loc[cut], trunc_tech.atr_14_raw.loc[cut], check_names=False)

        full_reg = compute_market_regime(real_panel)
        trunc_reg = compute_market_regime(real_panel.slice_until(cut))
        assert full_reg.loc[cut, "regime"] == trunc_reg.loc[cut, "regime"]
        assert full_reg.loc[cut, "target_exposure"] == pytest.approx(trunc_reg.loc[cut, "target_exposure"])


# --------------------------------------------------------------------------- portfolio optimizer & buffer zone
class TestPortfolioConstraints:
    def test_single_stock_and_industry_caps_and_buffer_zone(self):
        codes = [f"0000{i:02d}.SZ" for i in range(1, 35)]
        scores = pd.Series(np.linspace(34, 1, len(codes)), index=codes)
        tech = pd.Series(True, index=codes)
        elig = pd.Series(True, index=codes)
        # Put first 6 stocks in same industry to stress-test 25% industry cap
        inds = pd.Series(["电子"] * 6 + [f"行业{i // 2}" for i in range(len(codes) - 6)], index=codes)
        cfg = PortfolioConfig(target_holdings=10, buffer_rank=25, max_single_weight=0.12, max_industry_weight=0.25)

        # Hold stock ranked #20 -> buffer zone must retain it!
        held = {codes[19]}
        selected, rank_map = select_portfolio_names(scores, tech, elig, held, set(), cfg, inds)
        assert codes[19] in selected
        assert len(selected) == 10

        # Solve both equal weight and risk parity against real-like cov
        rng = np.random.default_rng(7)
        ret_df = pd.DataFrame(rng.normal(0, 0.02, size=(60, len(codes))), columns=codes)
        for method in ("risk_parity", "equal_weight"):
            cfg_m = PortfolioConfig(
                target_holdings=10, max_single_weight=0.12, max_industry_weight=0.25, weight_method=method
            )
            w = solve_portfolio_weights(selected, ret_df, inds, target_exposure=0.90, config=cfg_m)
            assert sum(w.values()) <= 0.90 + 1e-5
            assert all(wt <= 0.12 + 1e-5 for wt in w.values())
            ind_totals: dict[str, float] = {}
            for c, wt in w.items():
                ind_totals[inds[c]] = ind_totals.get(inds[c], 0.0) + wt
            assert all(v <= 0.25 + 1e-5 for v in ind_totals.values())


# --------------------------------------------------------------------------- real-data M2 integration
class TestRealDataBacktestIntegration:
    def test_lgbm_walk_forward_and_backtest_on_real_data(self, factor_run):
        fwd_ex = forward_excess_returns(factor_run.panel, horizon=5)
        scorer = LGBMScorer(LGBMConfig(mode="lambdarank", train_window=252, embargo_days=5, test_window=63))
        ml = scorer.run_walk_forward(factor_run.std, fwd_ex, factor_run.universe.eligible, adaptive=True)
        assert len(ml.folds) >= 1
        assert all(f["embargo_days"] >= 5 for f in ml.folds)
        assert ml.oos_scores.notna().sum().sum() > 0
        # Guardrail #3: ZERO scores allowed before first fold's out-of-sample test_start (no training backfill leakage!)
        first_oos = pd.Timestamp(ml.folds[0]["test_start"])
        pre_oos = ml.oos_scores.loc[ml.oos_scores.index < first_oos]
        assert pre_oos.isna().all().all(), "Look-ahead leakage: pre-OOS training dates must remain NaN!"
        assert len(ml.oos_shap) > 0
        assert ml.latest_booster is not None

        tech = compute_technical_signals(factor_run.panel)
        reg = compute_market_regime(factor_run.panel)
        bt = AShareBacktestEngine(BacktestConfig(weight_method="risk_parity")).run(
            factor_run.panel, ml.oos_scores, factor_run.universe.eligible, tech, reg
        )
        assert len(bt.equity_curve) > 10
        assert len(bt.trades) > 0
        assert bt.metrics["total_commission"] > 0
        assert "sharpe_ratio" in bt.metrics and "max_drawdown" in bt.metrics
        # All executed buy lots must be multiples of 100
        for tr in bt.trades:
            if tr["side"] == "BUY":
                assert tr["qty"] % 100 == 0 and tr["qty"] >= 100

    def test_two_level_research_cache_accelerates_parameter_sweeps(self, real_db, sample_codes):
        from app.strategy.alpha_strategy import (
            AlphaStrategyService,
            clear_research_cache,
            get_research_cache_status,
        )

        clear_research_cache()
        svc = AlphaStrategyService(real_db)
        sub_codes = sample_codes[:35]

        b1 = svc.run_full_pipeline(
            codes=sub_codes,
            lgbm_config=LGBMConfig(mode="lambdarank", n_estimators=40),
            bt_config=BacktestConfig(weight_method="risk_parity", enable_regime_timing=True),
            include_ew_compare=False,
            include_heatmap=False,
            use_cache=True,
        )
        assert b1.cache_info["factor_cache_hit"] is False
        assert b1.cache_info["ml_cache_hit"] is False

        # Second experiment changes portfolio/timing params -> MUST hit BOTH Level-1 & Level-2 cache!
        b2 = svc.run_full_pipeline(
            codes=sub_codes,
            lgbm_config=LGBMConfig(mode="lambdarank", n_estimators=40),
            bt_config=BacktestConfig(weight_method="equal_weight", enable_regime_timing=False),
            include_ew_compare=False,
            include_heatmap=False,
            use_cache=True,
        )
        assert b2.cache_info["factor_cache_hit"] is True
        assert b2.cache_info["ml_cache_hit"] is True
        pd.testing.assert_frame_equal(b1.ml_result.oos_scores, b2.ml_result.oos_scores)

        st = get_research_cache_status(real_db)
        assert st["factor_panel_cached"] is True
        assert "lambdarank" in st["cached_ml_modes"]
        clear_research_cache()

    def test_dynamic_holding_policy_rules(self):
        from app.strategy.dynamic_holding import (
            DynamicHoldingConfig,
            HeldPositionState,
            compute_dynamic_stop_price,
            select_dynamic_portfolio,
        )

        codes = [f"0000{i:02d}.SZ" for i in range(1, 201)]
        scores = pd.Series(np.linspace(5.0, -5.0, len(codes)), index=codes)
        tech = pd.Series(True, index=codes)
        elig = pd.Series(True, index=codes)
        inds = pd.Series([f"行业{i % 10}" for i in range(len(codes))], index=codes)
        pcfg = PortfolioConfig(target_holdings=5, top_candidates=30, buffer_rank=35)
        dcfg = DynamicHoldingConfig(min_hold_days=10, buffer_rank=35, deep_exit_rank=150, exit_confirm_days=3, swap_score_gap=0.7)

        # Case 1: Position ranked #180 but hold_days=3 (<10) -> protected by min_hold_days
        # Case 2: Position ranked #190, hold_days=15, pnl=+12% & above_ma20 -> kept by profit riding!
        # Case 3: Position ranked #185, hold_days=15, pnl=-2%, weak_rank_streak=3 -> dynamically exited!
        # Case 4: Position ranked #186, hold_days=15, pnl=-2%, weak_rank_streak=1 -> observed, NOT exited yet!
        held = {
            codes[179]: HeldPositionState(codes[179], hold_days=3, avg_cost=10.0, current_price=9.8, highest_price=10.2),
            codes[189]: HeldPositionState(codes[189], hold_days=15, avg_cost=10.0, current_price=11.2, highest_price=11.5, above_ma20=True, weak_rank_streak=5),
            codes[184]: HeldPositionState(codes[184], hold_days=15, avg_cost=10.0, current_price=9.8, highest_price=10.1, weak_rank_streak=3),
            codes[185]: HeldPositionState(codes[185], hold_days=15, avg_cost=10.0, current_price=9.8, highest_price=10.1, weak_rank_streak=1),
        }
        res = select_dynamic_portfolio(scores, tech, elig, held, set(), pcfg, dcfg, inds, regime="NEUTRAL", target_exposure=0.55, allow_new_entries=True)
        assert codes[179] in res.selected and "最小持有期保护" in res.hold_reasons[codes[179]]
        assert codes[189] in res.selected and "浮盈趋势保护" in res.hold_reasons[codes[189]]
        assert codes[185] in res.selected and "排名波动观察" in res.hold_reasons[codes[185]]
        assert codes[184] not in res.selected and codes[184] in res.sell_reasons
        assert len(res.selected) == 5  # remaining slots dynamically refilled

        # Post-stop cooldown blocks new entries while still allowing exits
        res_cd = select_dynamic_portfolio(scores, tech, elig, held, set(), pcfg, dcfg, inds, regime="DEFENSIVE", target_exposure=0.15, allow_new_entries=False)
        assert codes[184] not in res_cd.selected
        assert len(res_cd.buy_reasons) == 0

        # Profit-protected trailing stop locks above breakeven (+2%) once peak gain >= +15%
        hard, trail, label = compute_dynamic_stop_price(avg_cost=10.0, highest_price=11.6, atr_raw=0.4, dyn_cfg=dcfg)
        assert hard == pytest.approx(9.2)
        assert trail >= 10.2 and "浮盈保护" in label

    def test_start_date_slices_simulation_without_starving_walk_forward(self, real_db, sample_codes):
        from app.strategy.alpha_strategy import AlphaStrategyService, clear_research_cache

        clear_research_cache()
        svc = AlphaStrategyService(real_db)
        sub_codes = sample_codes[:35]

        b_full = svc.run_full_pipeline(
            codes=sub_codes,
            lgbm_config=LGBMConfig(mode="lambdarank", n_estimators=40),
            bt_config=BacktestConfig(weight_method="risk_parity", holding_mode="dynamic"),
            include_ew_compare=False,
            include_heatmap=False,
            use_cache=True,
        )
        assert len(b_full.backtest.equity_curve) >= 20
        mid_idx = len(b_full.backtest.equity_curve) // 2
        mid_date = pd.Timestamp(b_full.backtest.equity_curve[mid_idx]["date"]).date()

        # Passing start=mid_date must slice ONLY the simulation window, reusing full-history
        # Walk-Forward training folds and hitting both Level-1 & Level-2 caches!
        b_slice = svc.run_full_pipeline(
            codes=sub_codes,
            start=mid_date,
            lgbm_config=LGBMConfig(mode="lambdarank", n_estimators=40),
            bt_config=BacktestConfig(weight_method="risk_parity", holding_mode="dynamic"),
            include_ew_compare=False,
            include_heatmap=False,
            use_cache=True,
        )
        assert b_slice.cache_info["factor_cache_hit"] is True
        assert b_slice.cache_info["ml_cache_hit"] is True
        assert b_slice.ml_result.folds == b_full.ml_result.folds
        assert b_slice.backtest.metrics["start_date"] >= str(mid_date)
        assert len(b_slice.backtest.equity_curve) < len(b_full.backtest.equity_curve)
        first_fold_start = b_slice.ml_result.folds[0]["test_start"]
        assert b_slice.backtest.metrics["start_date"] >= first_fold_start
        clear_research_cache()

    def test_drawdown_fuse_rolling_peak_prevents_cash_trap_while_preserving_redlines(self):
        from app.broker.drawdown_fuse import (
            DrawdownFuseConfig,
            DrawdownFuseState,
            compute_rolling_peak,
            step_drawdown_fuse,
        )

        cfg = DrawdownFuseConfig(warn_pct=0.08, halt_pct=0.12, warn_exposure_mult=0.50, rolling_peak_days=120)
        st = DrawdownFuseState(status="NORMAL", peak_equity=100.0)

        # 1) Isolated spike to 123.0 followed by -11.2% pullback to 109.2 -> MUST trigger WARNING (0.5x) immediately
        step_drawdown_fuse(st, 123.0, "OFFENSIVE", cfg)
        res_warn = step_drawdown_fuse(st, 109.2, "OFFENSIVE", cfg)
        assert res_warn.status == "WARNING"
        assert res_warn.exposure_mult == pytest.approx(0.50)
        assert res_warn.liquidate_all is False

        # 2) While the 123.0 spike ages inside the 120d window, WARNING remains active
        for _ in range(118):
            r = step_drawdown_fuse(st, 111.0, "OFFENSIVE", cfg)
        assert r.status == "WARNING"

        # 3) Once the ancient 123.0 spike rolls out past 120 trading days, control peak rolls to 111.0 -> exits WARNING!
        res_rel = step_drawdown_fuse(st, 111.0, "OFFENSIVE", cfg)
        assert res_rel.status == "NORMAL"
        assert res_rel.exposure_mult == pytest.approx(1.0)

        # 4) Hard -12% redline MUST still immediately trigger HALTED + liquidate_all=True
        res_halt = step_drawdown_fuse(st, 96.0, "DEFENSIVE", cfg)
        assert res_halt.status == "HALTED"
        assert res_halt.exposure_mult == 0.0
        assert res_halt.liquidate_all is True

        # 5) Live paper helper parity: <120 rows keeps stored_peak; >=120 rows ages out ancient spike
        assert compute_rolling_peak([123.0] + [110.0] * 50, 111.0, 123.0, window_days=120) == pytest.approx(123.0)
        assert compute_rolling_peak([123.0] + [110.0] * 120, 111.0, 123.0, window_days=120) == pytest.approx(111.0)

    def test_huber_10d_bag3_ensemble_and_antichase_guard(self, factor_run, tmp_path):
        from app.db.sqlite_client import SQLiteDatabase
        from app.ml.lgbm_scorer import EnsembleBooster
        from app.ml.model_registry import ModelRegistry

        # 1) Anti-Chase guard: default require_trend=False blocks overheated & macd_top_div
        #    while NOT discarding healthy below-MA20 pullback stocks
        tech_ac = compute_technical_signals(factor_run.panel)
        tech_tr = compute_technical_signals(factor_run.panel, require_trend=True)
        assert not (tech_ac.passed & tech_ac.overheated).any().any()
        assert not (tech_ac.passed & tech_ac.macd_top_div).any().any()
        assert tech_ac.passed.sum().sum() >= tech_tr.passed.sum().sum()

        # 2) Default Huber 10d + 3-Seed Bagging + strict embargo >= 10 zero look-ahead
        fwd_ex10 = forward_excess_returns(factor_run.panel, horizon=10)
        cfg = LGBMConfig(n_estimators=30)
        assert cfg.mode == "regression"
        assert cfg.label_horizon == 10
        assert cfg.embargo_days >= 10
        assert len(cfg.effective_seeds()) == 3

        scorer = LGBMScorer(cfg)
        ml = scorer.run_walk_forward(factor_run.std, fwd_ex10, factor_run.universe.eligible, adaptive=True)
        assert isinstance(ml.latest_booster, EnsembleBooster)
        assert len(ml.latest_booster.boosters) == 3
        assert all(f["embargo_days"] >= 10 for f in ml.folds)
        first_oos = pd.Timestamp(ml.folds[0]["test_start"])
        assert ml.oos_scores.loc[ml.oos_scores.index < first_oos].isna().all().all()

        # 3) ModelRegistry save & reload preserves full 3-seed EnsembleBooster & identical predictions/SHAP
        sqlite = SQLiteDatabase(tmp_path / "reg_test.db")
        sqlite.init_schema()
        reg = ModelRegistry(sqlite_db=sqlite, models_dir=tmp_path / "models")
        saved = reg.save_version(ml, model_type="regression", embargo_days=10, activate=True, notes="bag3 test")
        loaded_booster, meta = reg.load_active_booster()
        assert meta["version_id"] == saved["version_id"]
        assert isinstance(loaded_booster, EnsembleBooster)
        assert len(loaded_booster.boosters) == 3

        from app.ml.lgbm_scorer import build_ml_dataset

        last_dt = factor_run.panel.dates[-1]
        latest_df = build_ml_dataset(
            factor_run.std,
            factor_run.universe.eligible,
            fwd_excess=None,
            feature_names=cfg.feature_names,
            dates_filter=pd.DatetimeIndex([last_dt]),
        )
        s1, sh1 = scorer.predict_with_shap(latest_df, ml.latest_booster)
        s2, sh2 = scorer.predict_with_shap(latest_df, loaded_booster)
        np.testing.assert_allclose(s1, s2, atol=1e-6)
        np.testing.assert_allclose(sh1.to_numpy(), sh2.to_numpy(), atol=1e-6)
