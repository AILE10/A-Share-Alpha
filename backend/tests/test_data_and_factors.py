"""M1 tests: A-share code/board/limit rules, PIT anti-lookahead, universe filter, factor pipeline.

Rule tests use exact exchange arithmetic; data tests run on the REAL synced DuckDB store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.constants import (
    ALL_FACTOR_NAMES,
    calc_limit_prices,
    detect_board,
    get_price_limit_pct,
    normalize_ts_code,
)
from app.data.sync_service import derive_pit_financials
from app.factors.definitions import compute_factors
from app.factors.pipeline import build_neutral_design, mad_winsorize, median_fill, standardize_cross_section


# --------------------------------------------------------------------------- rules
class TestCodeAndBoardRules:
    @pytest.mark.parametrize(
        "raw,expected",
        [("600519", "600519.SH"), ("sh.600519", "600519.SH"), ("SZ000001", "000001.SZ"),
         ("300750.SZ", "300750.SZ"), ("688981", "688981.SH"), ("830799", "830799.BJ"), ("920118", "920118.BJ")],
    )
    def test_normalize(self, raw, expected):
        assert normalize_ts_code(raw) == expected

    def test_board_and_limit_pct(self):
        assert detect_board("600519.SH") == "MAIN" and get_price_limit_pct("600519.SH") == 0.10
        assert get_price_limit_pct("600519.SH", is_st=True) == 0.05
        assert detect_board("300750.SZ") == "CHINEXT" and get_price_limit_pct("300750.SZ") == 0.20
        assert detect_board("688981.SH") == "STAR" and get_price_limit_pct("688981.SH") == 0.20
        assert detect_board("830799.BJ") == "BSE" and get_price_limit_pct("830799.BJ") == 0.30

    def test_limit_price_exchange_rounding(self):
        assert calc_limit_prices(10.00, "600000.SH") == (11.00, 9.00)
        assert calc_limit_prices(12.35, "000001.SZ") == (13.59, 11.12)   # 13.585 -> 13.59 half-up
        assert calc_limit_prices(25.55, "300001.SZ") == (30.66, 20.44)
        assert calc_limit_prices(8.88, "600001.SH", is_st=True) == (9.32, 8.44)


# --------------------------------------------------------------------------- PIT
class TestPointInTime:
    def test_derive_single_quarter_and_keep_ann_date(self, real_db, sample_codes):
        fin = real_db.query_df(
            "SELECT * FROM financial_pit WHERE ts_code = ? ORDER BY end_date", [sample_codes[0]]
        )
        if fin.empty:
            pytest.skip("no PIT financials for first sample stock")
        # announcement date is always after the report period end (no report published before quarter end)
        assert (pd.to_datetime(fin["ann_date"]) > pd.to_datetime(fin["end_date"])).all()
        # Q1 single-quarter profit equals cumulative profit
        q1 = fin[pd.to_datetime(fin["end_date"]).dt.quarter == 1].dropna(subset=["net_profit_cum", "net_profit_q"])
        if not q1.empty:
            assert np.allclose(q1["net_profit_cum"], q1["net_profit_q"])

    def test_asof_join_strictly_uses_past_announcements(self, real_db, sample_codes):
        from app.factors.panel import load_pit_fundamentals

        cal = real_db.query_df("SELECT MIN(trade_date) a, MAX(trade_date) b FROM daily_bars").iloc[0]
        df = load_pit_fundamentals(real_db, sample_codes, cal["a"], cal["b"])
        joined = df.dropna(subset=["ann_date"])
        assert len(joined) > 0
        # guardrail #3: only reports announced STRICTLY before trade_date are visible
        assert (pd.to_datetime(joined["ann_date"]) < pd.to_datetime(joined["trade_date"])).all()

    def test_report_invisible_before_announcement(self, real_db, sample_codes):
        """For each report, the trading day ON the announcement date must not see it."""
        from app.factors.panel import load_pit_fundamentals

        fin = real_db.query_df("SELECT ts_code, ann_date, end_date FROM financial_pit")
        bars_range = real_db.query_df("SELECT MIN(trade_date) a, MAX(trade_date) b FROM daily_bars").iloc[0]
        df = load_pit_fundamentals(real_db, sample_codes, bars_range["a"], bars_range["b"])
        merged = df.merge(fin, left_on=["ts_code", "trade_date"], right_on=["ts_code", "ann_date"], suffixes=("", "_ann"))
        if merged.empty:
            pytest.skip("no announcement fell on a trading day in range")
        # on the announcement day itself, any visible filing must have been announced STRICTLY earlier
        # (restatements of older quarters can carry newer ann_date, so check ann_date strictly < trade_date)
        visible_ann = pd.to_datetime(merged["ann_date"])
        trade_d = pd.to_datetime(merged["trade_date"])
        assert ((visible_ann < trade_d) | visible_ann.isna()).all()

    def test_derive_pit_uses_cumulative_diff(self):
        # pure arithmetic check of the derivation logic (statement structure, not market prices)
        fin = pd.DataFrame({
            "ts_code": ["X"] * 8,
            "end_date": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
                                        "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]),
            "ann_date": pd.to_datetime(["2023-04-28", "2023-08-30", "2023-10-30", "2024-04-20",
                                        "2024-04-29", "2024-08-29", "2024-10-30", "2025-04-25"]),
            "net_profit_cum": [10.0, 25.0, 45.0, 60.0, 12.0, 30.0, 50.0, 80.0],
            "roe_avg_cum": [0.01, 0.025, 0.045, 0.06, 0.012, 0.03, 0.05, 0.08],
            "total_share": [1e8] * 8, "revenue_cum": [np.nan] * 8,
        })
        out = derive_pit_financials(fin).set_index("end_date")
        q = out["net_profit_q"].to_numpy()
        assert np.allclose(q, [10, 15, 20, 15, 12, 18, 20, 30])
        yoy_2024q2 = out.loc[pd.Timestamp("2024-06-30").date(), "np_yoy_q"]
        assert yoy_2024q2 == pytest.approx((18 - 15) / 15)
        # ann_date preserved verbatim for every row
        assert list(pd.to_datetime(out["ann_date"])) == list(fin["ann_date"])


# --------------------------------------------------------------------------- universe
class TestUniverse:
    def test_universe_rules_hold_on_real_data(self, factor_run):
        u = factor_run.universe
        panel = factor_run.panel
        elig = u.eligible
        assert elig.sum().sum() > 0
        assert not (elig & panel.is_st).any().any(), "ST stock entered universe"
        assert not (elig & panel.is_suspended).any().any(), "suspended stock entered universe"
        assert (u.avg_amount_20d[elig].stack().dropna() >= 5e7).all()
        assert (u.list_days[elig].stack().dropna() >= 120).all()
        assert (u.mcap_pct_rank[elig].stack().dropna() >= 0.10).all()

    def test_universe_summary(self, factor_run):
        s = factor_run.universe.summary()
        assert s["total"] >= 50
        assert s["eligible"] > 0


# --------------------------------------------------------------------------- factors
class TestFactors:
    def test_all_16_factors_present(self, factor_run):
        assert set(factor_run.factors.raw) == set(ALL_FACTOR_NAMES)
        assert len(ALL_FACTOR_NAMES) == 16

    def test_factor_formulas_on_real_prices(self, real_panel, factor_run):
        p = real_panel
        hc = p.hfq_close
        code = p.codes[0]
        t = p.dates[-10]
        i = p.dates.get_loc(t)
        if p.is_suspended.loc[t, code]:
            pytest.skip("first stock suspended on test date")
        exp_ret20 = hc[code].iloc[i] / hc[code].iloc[i - 20] - 1
        assert factor_run.factors.raw["ret_20d"].loc[t, code] == pytest.approx(exp_ret20)
        exp_rev5 = -(hc[code].iloc[i] / hc[code].iloc[i - 5] - 1)
        assert factor_run.factors.raw["rev_5d"].loc[t, code] == pytest.approx(exp_rev5)
        hi, lo, cl = p.hfq_high.loc[t, code], p.hfq_low.loc[t, code], hc.loc[t, code]
        if hi - lo > 1e-9:
            assert factor_run.factors.raw["intraday_pos"].loc[t, code] == pytest.approx((cl - lo) / (hi - lo))
        pe = p.pe_ttm.loc[t, code]
        if pd.notna(pe) and abs(pe) > 1e-6:
            assert factor_run.factors.raw["ep_ttm"].loc[t, code] == pytest.approx(1 / pe)

    def test_no_lookahead_in_factors(self, real_panel):
        """Factors at date T must be identical whether or not data after T exists."""
        p = real_panel
        cut = p.dates[-40]
        full = compute_factors(p)
        trunc = compute_factors(p.slice_until(cut))
        for name in ALL_FACTOR_NAMES:
            a = full.raw[name].loc[cut]
            b = trunc.raw[name].loc[cut]
            pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-9, atol=1e-12)

    def test_capital_flow_degradation_never_crashes(self, real_panel):
        import dataclasses

        p = dataclasses.replace(real_panel, capital_flow_ratio=None)
        fs = compute_factors(p)
        assert fs.status["main_net_inflow_ratio"] == "DEGRADED_MISSING"
        assert fs.degraded["main_net_inflow_ratio"].all().all()

    def test_parse_cn_amount_and_sparse_mad_fallback(self):
        from app.data.akshare_provider import parse_cn_amount

        assert parse_cn_amount("1.46亿") == pytest.approx(1.46e8)
        assert parse_cn_amount("-9068.88万") == pytest.approx(-9068.88e4)
        assert parse_cn_amount("20.00%") == pytest.approx(20.0)
        assert np.isnan(parse_cn_amount("-"))

        # When >50% of values are tied (median-filled partial factor), mad_winsorize must not crush dispersion to 0
        arr = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [-2.5], [1.8], [3.2], [-1.1]])
        clipped = mad_winsorize(arr)
        assert np.std(clipped) > 0.5


# --------------------------------------------------------------------------- pipeline
class TestPipeline:
    def test_standardized_output_properties_on_real_data(self, factor_run):
        elig = factor_run.universe.eligible
        checked = 0
        for name, df in factor_run.std.items():
            for t in df.index[-30::10]:
                row = df.loc[t][elig.loc[t]].dropna()
                if len(row) < 10 or row.abs().sum() < 1e-9:
                    continue
                assert abs(row.mean()) < 1e-6
                assert row.std(ddof=1) == pytest.approx(1.0, abs=1e-6)
                checked += 1
        assert checked > 0

    def test_neutralized_residual_orthogonal_to_size(self, factor_run):
        p = factor_run.panel
        elig = factor_run.universe.eligible
        t = elig.index[-5]
        sel = elig.loc[t]
        log_mv = np.log(p.total_mv.loc[t][sel])
        for name in ("ret_20d", "amihud_20d", "ep_ttm"):
            z = factor_run.std[name].loc[t][sel]
            if z.abs().sum() < 1e-9:
                continue
            assert abs(np.corrcoef(z, log_mv)[0, 1]) < 1e-6

    def test_median_fill_and_mad(self):
        vals = np.array([[1.0, np.nan], [2.0, 5.0], [np.nan, 7.0], [100.0, 6.0], [4.0, np.nan]])
        inds = np.array(["A", "A", "B", "B", "C"], dtype=object)
        filled = median_fill(vals, inds)
        assert filled[2, 0] == pytest.approx(100.0)       # industry B median
        assert filled[0, 1] == pytest.approx(5.0)         # industry A median (row 1 = 5.0)
        assert filled[4, 1] == pytest.approx(5.5)         # industry C has no value -> global median of [5,5,7,6] = 5.5
        clipped = mad_winsorize(np.array([[1.0], [2.0], [3.0], [4.0], [1000.0]]))
        assert clipped.max() < 1000.0

    def test_sparse_industry_merge_keeps_design_full_rank(self):
        inds = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 1 + ["D"] * 2, dtype=object)
        log_mv = np.linspace(20, 25, len(inds))
        X = build_neutral_design(inds, log_mv)
        assert np.linalg.matrix_rank(X) == X.shape[1]
        z = standardize_cross_section(np.column_stack([log_mv * 2 + 1]), inds, np.exp(log_mv))
        assert np.allclose(z, 0.0, atol=1e-6)               # pure size exposure fully neutralized


# --------------------------------------------------------------------------- analytics
class TestAnalytics:
    def test_ic_summary_complete(self, factor_run):
        s = factor_run.analysis.summary
        assert len(s) == 16
        assert s["rank_ic_mean"].notna().sum() >= 12
        assert s["rank_ic_mean"].abs().max() < 0.6     # sanity: real cross-sectional IC magnitude

    def test_quintile_nav_and_correlation(self, factor_run):
        nav = factor_run.analysis.quintile_nav["ret_20d"]
        assert list(nav.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5", "LS"]
        assert (nav > 0).all().all()
        corr = factor_run.analysis.correlation
        assert corr.shape == (16, 16)
        valid = [n for n in corr.index if corr.loc[n, n] != 0]
        assert np.allclose(np.diag(corr.loc[valid, valid]), 1.0)
