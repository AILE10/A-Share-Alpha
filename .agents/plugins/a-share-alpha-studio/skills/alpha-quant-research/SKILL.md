---
name: alpha-quant-research
description: >-
  Extend and evaluate A-Share Alpha Studio quantitative strategies: add new orthogonal factors,
  run single-factor IC/ICIR/quintile diagnostics, tune Purged Walk-Forward LightGBM models,
  compare Risk Parity vs Equal Weight backtests, run weekly review attribution, and verify
  A-share guardrails & RiskMiddleware rules. Use for quant R&D and strategy iteration.
---

# A-Share Alpha Studio — Quant Research & Strategy Iteration Workflow

Use this skill when modifying factors, tuning LightGBM models, adjusting technical resonance / regime rules, running walk-forward optimization, or iterating after weekly paper reviews.

## 1. Architecture Map (Where to Edit)

| Layer | File Path | Responsibility |
| :--- | :--- | :--- |
| Constants & A-Share Rules | `backend/app/core/constants.py` | Board limits, exact fee rates, `FACTOR_DEFINITIONS` (16 factors) |
| DuckDB Panel & PIT Join | `backend/app/factors/panel.py` | Wide `MarketPanel`, raw vs `hfq` prices, strict `trade_date > ann_date` ASOF join |
| Universe Filter | `backend/app/factors/universe.py` | ST / suspended / <120d IPO / <50M 20d amount / bottom 10% market cap |
| Factor Math & Pipeline | `backend/app/factors/definitions.py`, `pipeline.py` | Factor formulas, median fill -> MAD -> SW industry + log-mcap OLS -> Z-score |
| Single-Factor Analytics | `backend/app/factors/analyzer.py` | Non-overlapping IC/RankIC/ICIR, 5-quantile NAV, Spearman heatmap, 20d decay |
| Purged Walk-Forward ML | `backend/app/ml/splitter.py`, `lgbm_scorer.py`, `model_registry.py` | Embargo >= 5d + label purging, LambdaRank / Huber Regression, TreeSHAP, version rollback |
| Strategy Rules | `backend/app/strategy/regime.py`, `technical.py`, `optimizer.py` | CSI1000+breadth regime, MA20/overheat/MACD divergence, Risk Parity + 12%/25% caps + buffer |
| Exact Backtest Engine | `backend/app/backtest/matching.py`, `engine.py`, `walk_forward_opt.py` | Event-driven A-share matching, T+1/lots/limits/fees/stops/fuse, 2D sensitivity heatmap |
| Paper & Weekly Review | `backend/app/paper/recommender.py`, `service.py`, `reviewer.py` | Top-3 TreeSHAP recommendation cards, 14:40 matching, paper vs theory tracking error |
| Risk & Broker Gateway | `backend/app/broker/risk_middleware.py`, `gateway.py`, `*_adapter.py` | Single/industry caps, drawdown fuse, rate limit, Paper / Assisted / Compliant QMT |

## 2. Adding a New Alpha Factor (Safe Protocol)

1. Register metadata in `FACTOR_DEFINITIONS` (`backend/app/core/constants.py`).
2. Implement strictly causal calculation in `compute_factors()` (`backend/app/factors/definitions.py`):
   - Price/return/MA/volatility MUST use `panel.hfq_close` / `panel.hfq_high` / `panel.hfq_low`.
   - Fundamental fields MUST come from DuckDB `financial_pit` joined strictly on `trade_date > ann_date`.
   - External fragile feeds MUST mark `degraded[name]` and `status[name] = 'DEGRADED_MISSING'` if unavailable — never impute random numbers.
3. Recompute & verify zero look-ahead:
   ```powershell
   backend\.venv\Scripts\python.exe -u scripts\m1_verify_real_data.py --skip-sync
   cd backend && .venv\Scripts\python.exe -m pytest tests/test_data_and_factors.py
   ```
   Confirm `test_no_lookahead_in_factors`, `test_asof_join_strictly_uses_past_announcements`, and `test_neutralized_residual_orthogonal_to_size` all pass.

## 3. Running Walk-Forward Optimization & Model Iteration

1. **Out-of-sample Backtest & Comparison**:
   ```powershell
   backend\.venv\Scripts\python.exe -u scripts\m2_verify_backtest.py
   ```
   Inspect `data/reports/m2_backtest_report.json` for OOS RankIC, Sharpe, Calmar, Max Drawdown & Recovery Days, Risk Parity vs Equal Weight delta, and the 3×3 Walk-Forward sensitivity heatmap.
2. **Weekly Review & Controlled Retrain**:
   - Trigger `POST /api/v1/paper/reviews/generate` (or UI Page 5) to detect trailing 20-day IC reversals and slippage drag.
   - If multiple core factors reverse, trigger `POST /api/v1/models/retrain` to register a new snapshot without overwriting prior versions; compare OOS RankIC/ICIR and roll back immediately via `POST /api/v1/models/{version_id}/activate` if live tracking degrades.

## 4. Non-Negotiable Verification Gate Before Finishing Any Change

Always run:
```powershell
cd backend && .venv\Scripts\python.exe -m pytest
```
Never weaken `MIN_EMBARGO_DAYS = 5`, A-share T+1 / price-limit / lot-size / fee rules, or `RiskMiddleware` pre-trade enforcement.
