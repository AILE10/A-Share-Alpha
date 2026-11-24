"""Explainable real-time stock recommendation engine with TreeSHAP Top-3 factor attribution,
technical resonance pattern tags, target weight, and stop-loss / take-profit levels."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.api.ws_manager import ws_manager
from app.core.calendar import today_cn
from app.core.constants import ALL_FACTOR_NAMES, FACTOR_META_MAP
from app.core.logging import get_logger
from app.db.duckdb_client import DuckDBClient, get_duckdb
from app.db.models import PaperAccount, PaperOrder, PaperPosition, SignalRecommendation
from app.db.sqlite_client import SQLiteDatabase, get_sqlite_db
from app.factors.service import FactorEngine, to_jsonable
from app.ml.lgbm_scorer import LGBMConfig, LGBMScorer, build_ml_dataset
from app.ml.model_registry import ModelRegistry
from app.strategy.dynamic_holding import (
    DynamicHoldingConfig,
    HeldPositionState,
    compute_dynamic_stop_price,
    select_dynamic_portfolio,
)
from app.strategy.optimizer import PortfolioConfig, select_portfolio_names, solve_portfolio_weights
from app.strategy.regime import compute_market_regime, persist_market_regime
from app.strategy.technical import compute_technical_signals

logger = get_logger(__name__)


@dataclass
class RecommendationBundle:
    trade_date: str
    regime: dict
    model_version_id: str
    recommendations: list[dict]
    top_candidates_codes: list[str]


class RecommendationEngine:
    """Produces explainable BUY / SELL / HOLD recommendations using active LightGBM + real prices."""

    def __init__(
        self,
        duckdb_client: DuckDBClient | None = None,
        sqlite_db: SQLiteDatabase | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.db = duckdb_client or get_duckdb()
        self.sqlite = sqlite_db or get_sqlite_db()
        self.registry = registry or ModelRegistry(self.sqlite)

    def _ensure_active_model(self, codes: list[str] | None = None):
        try:
            return self.registry.load_active_booster()
        except FileNotFoundError:
            logger.info("No active model in registry yet; training initial production LightGBM snapshot")
            meta, _ = self.registry.retrain_and_register(
                codes=codes,
                model_type="regression",
                notes="首次初始化训练快照 (Huber 10d · 3-Seed Bagging)",
                duckdb_client=self.db,
            )
            return self.registry.load_active_booster()

    def generate(
        self,
        account_id: str = "default_paper",
        codes: list[str] | None = None,
        live_quotes: pd.DataFrame | None = None,
        portfolio_config: PortfolioConfig | None = None,
        dynamic_config: DynamicHoldingConfig | None = None,
        use_dynamic_holding: bool = True,
        hard_stop_pct: float = 0.08,
        atr_mult: float = 2.5,
        persist: bool = True,
        broadcast: bool = True,
    ) -> RecommendationBundle:
        booster, model_meta = self._ensure_active_model(codes=codes)
        feats = model_meta.get("feature_list") or list(ALL_FACTOR_NAMES)

        f_run = FactorEngine(self.db).build(codes=codes)
        panel = f_run.panel
        tech = compute_technical_signals(panel)
        regime_df = compute_market_regime(panel)
        persist_market_regime(regime_df, self.db)

        latest_dt = panel.dates[-1]
        latest_str = str(latest_dt.date())
        reg_row = regime_df.loc[latest_dt]
        regime_info = {
            "trade_date": latest_str,
            "regime": str(reg_row["regime"]),
            "target_exposure": round(float(reg_row["target_exposure"]), 4),
            "csi1000_close": round(float(reg_row["csi1000_close"]), 2),
            "csi1000_ma20": round(float(reg_row["csi1000_ma20"]), 2) if pd.notna(reg_row["csi1000_ma20"]) else None,
            "csi1000_ma60": round(float(reg_row["csi1000_ma60"]), 2) if pd.notna(reg_row["csi1000_ma60"]) else None,
            "breadth_above_ma20": round(float(reg_row["breadth_above_ma20"]), 4),
        }

        # Score latest cross-section + exact per-stock TreeSHAP contributions
        latest_df = build_ml_dataset(
            f_run.std,
            f_run.universe.eligible,
            fwd_excess=None,
            feature_names=feats,
            dates_filter=pd.DatetimeIndex([latest_dt]),
        )
        if latest_df.empty:
            raise ValueError(f"No eligible stocks with standardized factors on {latest_str}")

        scorer = LGBMScorer(LGBMConfig(mode=model_meta.get("model_type", "lambdarank"), feature_names=feats))
        scores_arr, shap_df = scorer.predict_with_shap(latest_df, booster)
        latest_df = latest_df.copy()
        latest_df["score"] = scores_arr

        score_series = pd.Series(scores_arr, index=latest_df["ts_code"].to_numpy())
        shap_by_code = shap_df.copy()
        shap_by_code.index = latest_df["ts_code"].to_numpy()

        # Current holdings & account equity (for small-capital A-share 100-share lot sizing awareness)
        with self.sqlite.session_scope() as session:
            acct_row = session.get(PaperAccount, account_id)
            acct_equity = float(acct_row.total_equity) if acct_row is not None else 0.0
            pos_rows = session.query(PaperPosition).filter(PaperPosition.account_id == account_id).all()
        held_map = {p.ts_code: p for p in pos_rows if p.total_shares > 0}
        held_set = set(held_map.keys())

        # Optional live quote price overlay (real Top-100 snapshot!)
        quote_map: dict[str, dict] = {}
        if live_quotes is not None and not live_quotes.empty:
            for _, r in live_quotes.iterrows():
                quote_map[str(r["ts_code"])] = r.to_dict()

        def _ref_price(code: str) -> float:
            q = quote_map.get(code)
            if q and pd.notna(q.get("last_price")) and float(q["last_price"]) > 0:
                return float(q["last_price"])
            if code in panel.close.columns and pd.notna(panel.close.loc[latest_dt, code]):
                return float(panel.close.loc[latest_dt, code])
            return held_map[code].current_price if code in held_map else 0.0

        dyn_cfg = dynamic_config or DynamicHoldingConfig()
        stopped_out: set[str] = set()
        stop_reasons: dict[str, str] = {}
        for code, pos in held_map.items():
            px = _ref_price(code)
            atr = float(tech.atr_14_raw.loc[latest_dt, code]) if (code in tech.atr_14_raw.columns and pd.notna(tech.atr_14_raw.loc[latest_dt, code])) else 0.0
            hi_since = max(pos.highest_price_since_entry, px)
            if use_dynamic_holding:
                hard_stop, atr_stop, trail_label = compute_dynamic_stop_price(
                    avg_cost=pos.avg_cost,
                    highest_price=hi_since,
                    atr_raw=atr,
                    hard_stop_pct=hard_stop_pct,
                    atr_stop_mult=atr_mult,
                    dyn_cfg=dyn_cfg,
                )
            else:
                hard_stop = pos.avg_cost * (1.0 - hard_stop_pct)
                atr_stop = hi_since - atr_mult * atr if atr > 0 else 0.0
                trail_label = f"ATR移动止损({atr_mult}xATR)"
            if px > 0 and px <= hard_stop:
                stopped_out.add(code)
                stop_reasons[code] = f"触发个股硬止损(-{hard_stop_pct:.0%})"
            elif px > 0 and atr_stop > 0 and px <= atr_stop:
                stopped_out.add(code)
                stop_reasons[code] = f"触发{trail_label}"

        base_cfg = portfolio_config or PortfolioConfig(buffer_rank=dyn_cfg.buffer_rank if use_dynamic_holding else 25)
        target_exp = float(reg_row["target_exposure"])
        regime_label = str(reg_row["regime"])
        eff_holdings = base_cfg.target_holdings
        eligible_for_sel = f_run.universe.eligible.loc[latest_dt].copy()
        if acct_equity > 0 and target_exp > 1e-5:
            target_budget = acct_equity * target_exp
            min_slots_for_cap = max(1, int(np.ceil(target_exp / max(base_cfg.max_single_weight, 1e-6))))
            # Ensure each target slot has ~10,000 CNY budget in small-capital/defensive modes so 100-share
            # A-share lot rounding does not silently skip normal-priced top-ranked stocks
            if target_budget < base_cfg.target_holdings * 10_000.0:
                eff_holdings = max(
                    min_slots_for_cap,
                    min(base_cfg.target_holdings, max(2, int(round(target_budget / 10_000.0)))),
                )
            # Never force-liquidate newly bought holdings just because small equity drift crossed a rounding boundary
            if held_set and len(held_set) <= base_cfg.target_holdings:
                eff_holdings = max(eff_holdings, len(held_set))
            # Exclude NEW candidates whose 1-lot cost (100 shares) exceeds affordable slot cap (never blocks existing holds)
            slot_budget = min(acct_equity * base_cfg.max_single_weight, target_budget / max(eff_holdings, 1) * 1.15)
            max_lot_px = slot_budget / 100.0
            if max_lot_px > 0:
                px_series = pd.Series({c: _ref_price(c) for c in eligible_for_sel.index})
                affordable = (px_series > 0) & (px_series <= max_lot_px)
                is_held = pd.Series([c in held_set for c in eligible_for_sel.index], index=eligible_for_sel.index)
                filtered = eligible_for_sel & (affordable | is_held)
                if int(filtered.sum()) >= eff_holdings:
                    eligible_for_sel = filtered

        p_cfg = PortfolioConfig(
            top_candidates=base_cfg.top_candidates,
            target_holdings=eff_holdings,
            buffer_rank=base_cfg.buffer_rank,
            max_single_weight=base_cfg.max_single_weight,
            max_industry_weight=base_cfg.max_industry_weight,
            weight_method=base_cfg.weight_method,
            cov_window=base_cfg.cov_window,
        )
        industries = panel.industry_of()
        names = panel.name_of()

        hold_reasons: dict[str, str] = {}
        sell_reasons: dict[str, str] = {}
        buy_reasons: dict[str, str] = {}

        if use_dynamic_holding:
            # Compute live trading-day holding duration & consecutive weak-rank streak for existing holdings
            today_d = today_cn()
            held_states: dict[str, HeldPositionState] = {}
            streak_by_code: dict[str, int] = {c: 0 for c in held_map}
            if held_map:
                trail_dates = panel.dates[-max(1, dyn_cfg.exit_confirm_days):]
                trail_df = build_ml_dataset(
                    f_run.std, f_run.universe.eligible, fwd_excess=None, feature_names=feats, dates_filter=trail_dates
                )
                if not trail_df.empty:
                    trail_df = trail_df.copy()
                    trail_df["score"] = scorer.predict(trail_df, booster)
                    for _, grp in trail_df.groupby("trade_date", sort=True):
                        ordered_codes = grp.sort_values("score", ascending=False)["ts_code"].tolist()
                        rm = {c: i + 1 for i, c in enumerate(ordered_codes)}
                        for c in held_map:
                            if rm.get(c, 9999) > dyn_cfg.buffer_rank:
                                streak_by_code[c] += 1
                            else:
                                streak_by_code[c] = 0

                for code, pos in held_map.items():
                    h_days = 0
                    if pos.entry_date:
                        try:
                            ed = pd.to_datetime(pos.entry_date).date()
                            cnt_d = self.db.query_scalar(
                                "SELECT COUNT(*) FROM trade_calendar WHERE is_open AND trade_date > ? AND trade_date <= ?",
                                [ed, today_d],
                            )
                            h_days = int(cnt_d or 0)
                        except Exception:  # noqa: BLE001
                            h_days = 0
                    px = _ref_price(code)
                    abv = bool(tech.above_ma20.loc[latest_dt, code]) if code in tech.above_ma20.columns else True
                    div = bool(tech.macd_top_div.loc[latest_dt, code]) if code in tech.macd_top_div.columns else False
                    held_states[code] = HeldPositionState(
                        ts_code=code,
                        hold_days=h_days,
                        avg_cost=pos.avg_cost,
                        current_price=px,
                        highest_price=max(pos.highest_price_since_entry, px),
                        above_ma20=abv,
                        macd_top_div=div,
                        weak_rank_streak=streak_by_code.get(code, 0),
                    )

            # Live post-stop cooldown in weak/neutral regimes (prevents catching falling knives)
            allow_new_entries = True
            if regime_label != "OFFENSIVE" and dyn_cfg.stop_cooldown_days > 0:
                with self.sqlite.session_scope() as session:
                    recent_stop = (
                        session.query(PaperOrder)
                        .filter(
                            PaperOrder.account_id == account_id,
                            PaperOrder.side == "SELL",
                            PaperOrder.status == "FILLED",
                            PaperOrder.reason_summary.contains("止损"),
                        )
                        .order_by(PaperOrder.created_at.desc())
                        .first()
                    )
                if recent_stop is not None and recent_stop.created_at is not None:
                    stop_d = recent_stop.created_at.date()
                    days_since_stop = int(
                        self.db.query_scalar(
                            "SELECT COUNT(*) FROM trade_calendar WHERE is_open AND trade_date > ? AND trade_date <= ?",
                            [stop_d, today_d],
                        )
                        or 0
                    )
                    if days_since_stop < dyn_cfg.stop_cooldown_days:
                        allow_new_entries = False

            dyn_res = select_dynamic_portfolio(
                scores=score_series,
                tech_passed=tech.passed.loc[latest_dt],
                eligible=eligible_for_sel,
                held_states=held_states,
                stopped_out=stopped_out,
                port_cfg=p_cfg,
                dyn_cfg=dyn_cfg,
                industries=industries,
                regime=regime_label,
                target_exposure=target_exp,
                allow_new_entries=allow_new_entries,
            )
            selected = dyn_res.selected
            rank_map = dyn_res.rank_map
            hold_reasons = dyn_res.hold_reasons
            sell_reasons = dyn_res.sell_reasons
            buy_reasons = dyn_res.buy_reasons
        else:
            selected, rank_map = select_portfolio_names(
                scores=score_series,
                tech_passed=tech.passed.loc[latest_dt],
                eligible=eligible_for_sel,
                current_holdings=held_set,
                stopped_out=stopped_out,
                config=p_cfg,
                industries=industries,
            )

        target_weights = solve_portfolio_weights(
            selected=selected,
            returns_window=panel.returns,
            industries=industries,
            target_exposure=target_exp,
            config=p_cfg,
        )

        def _top3_factors(code: str) -> list[dict]:
            if code not in shap_by_code.index:
                return []
            s_row = shap_by_code.loc[code]
            ordered = s_row.reindex(s_row.abs().sort_values(ascending=False).index)
            items = []
            for fname in ordered.index[:3]:
                meta = FACTOR_META_MAP.get(fname, {})
                raw_v = (
                    float(f_run.factors.raw[fname].loc[latest_dt, code])
                    if fname in f_run.factors.raw and code in f_run.factors.raw[fname].columns and pd.notna(f_run.factors.raw[fname].loc[latest_dt, code])
                    else None
                )
                std_v = (
                    float(f_run.std[fname].loc[latest_dt, code])
                    if fname in f_run.std and code in f_run.std[fname].columns and pd.notna(f_run.std[fname].loc[latest_dt, code])
                    else None
                )
                items.append({
                    "name": fname,
                    "label": str(meta.get("label", fname)),
                    "dimension": str(meta.get("dimension", "")),
                    "raw_value": round(raw_v, 4) if raw_v is not None else None,
                    "std_value": round(std_v, 4) if std_v is not None else None,
                    "shap": round(float(s_row[fname]), 5),
                })
            return items

        recommendations: list[dict] = []

        # 1) BUY & HOLD targets (if target_exposure == 0 on extreme crash day, still surface Top-5
        #    candidates with 0% weight & full SHAP/technical attribution so investor sees clear watch cards)
        display_codes = selected if selected else list(score_series.sort_values(ascending=False).index[:5])
        for code in display_codes:
            tw = target_weights.get(code, 0.0)
            if tw <= 1e-5 and float(reg_row["target_exposure"]) > 1e-5:
                continue
            px = _ref_price(code)
            atr = (
                float(tech.atr_14_raw.loc[latest_dt, code])
                if (code in tech.atr_14_raw.columns and pd.notna(tech.atr_14_raw.loc[latest_dt, code]))
                else px * 0.03
            )
            stop_loss = round(max(px * (1.0 - hard_stop_pct), px - atr_mult * atr), 2) if px > 0 else 0.0
            take_profit = round(max(px * 1.15, px + 3.5 * atr), 2) if px > 0 else 0.0
            action = "HOLD" if (code in held_set or tw <= 1e-5) else "BUY"
            patterns = tech.describe_stock(latest_dt, code)
            top3 = _top3_factors(code)
            rk = int(rank_map.get(code, 0))
            if action == "HOLD":
                rationale = hold_reasons.get(code, f"模型综合排名 #{rk}（缓冲带保留）")
            else:
                rationale = buy_reasons.get(code, f"模型综合排名 #{rk}，通过技术面共振筛选")
            if patterns:
                rationale += f"，触发形态：{'、'.join(patterns[:2])}"

            recommendations.append({
                "signal_id": f"sig_{uuid.uuid4().hex[:12]}",
                "trade_date": latest_str,
                "ts_code": code,
                "name": str(names.get(code, code)),
                "sw_l1_name": str(industries.get(code, "未分类")),
                "action": action,
                "composite_rank": rk,
                "model_score": round(float(score_series.get(code, 0.0)), 5),
                "suggested_weight": round(tw, 4),
                "ref_price": round(px, 2),
                "stop_loss_price": stop_loss,
                "take_profit_price": take_profit,
                "top_factors": top3,
                "tech_patterns": patterns,
                "rationale": rationale,
                "status": "ACTIVE",
            })

        # 2) SELL recommendations (currently held, dropped from target or stopped out)
        selected_set = set(selected)
        for code, pos in held_map.items():
            if code in selected_set and code not in stopped_out:
                continue
            px = _ref_price(code)
            rk = int(rank_map.get(code, 999))
            patterns = tech.describe_stock(latest_dt, code)
            top3 = _top3_factors(code)
            reason = (
                stop_reasons.get(code, "触发个股止损红线(-8% / 2.5×ATR)")
                if code in stopped_out
                else sell_reasons.get(code, f"模型排名滑落至 #{rk}（跌出前 {p_cfg.buffer_rank} 缓冲带）")
            )
            recommendations.append({
                "signal_id": f"sig_{uuid.uuid4().hex[:12]}",
                "trade_date": latest_str,
                "ts_code": code,
                "name": pos.name or str(names.get(code, code)),
                "sw_l1_name": pos.sw_l1_name or str(industries.get(code, "未分类")),
                "action": "SELL",
                "composite_rank": rk,
                "model_score": round(float(score_series.get(code, 0.0)), 5),
                "suggested_weight": 0.0,
                "ref_price": round(px, 2),
                "stop_loss_price": pos.stop_loss_price,
                "take_profit_price": 0.0,
                "top_factors": top3,
                "tech_patterns": patterns,
                "rationale": reason,
                "status": "ACTIVE",
            })

        # Order: BUY -> SELL -> HOLD sorted by rank
        action_prio = {"BUY": 0, "SELL": 1, "HOLD": 2}
        recommendations.sort(key=lambda r: (action_prio.get(r["action"], 9), r["composite_rank"]))

        if persist:
            with self.sqlite.session_scope() as session:
                session.query(SignalRecommendation).filter(SignalRecommendation.status == "ACTIVE").update(
                    {SignalRecommendation.status: "ARCHIVED"}
                )
                for rec in recommendations:
                    session.add(
                        SignalRecommendation(
                            signal_id=rec["signal_id"],
                            trade_date=rec["trade_date"],
                            ts_code=rec["ts_code"],
                            name=rec["name"],
                            sw_l1_name=rec["sw_l1_name"],
                            action=rec["action"],
                            composite_rank=rec["composite_rank"],
                            model_score=rec["model_score"],
                            suggested_weight=rec["suggested_weight"],
                            ref_price=rec["ref_price"],
                            stop_loss_price=rec["stop_loss_price"],
                            take_profit_price=rec["take_profit_price"],
                            top_factors_json=json.dumps(to_jsonable(rec["top_factors"]), ensure_ascii=False),
                            tech_patterns_json=json.dumps(rec["tech_patterns"], ensure_ascii=False),
                            rationale=rec["rationale"],
                            status="ACTIVE",
                        )
                    )

        top_candidates = list(score_series.sort_values(ascending=False).index[:100])
        if broadcast:
            ws_manager.broadcast_threadsafe(
                "recommendations",
                to_jsonable({
                    "trade_date": latest_str,
                    "regime": regime_info,
                    "model_version_id": model_meta["version_id"],
                    "items": recommendations,
                }),
            )
            ws_manager.broadcast_threadsafe("market_regime", to_jsonable(regime_info))

        return RecommendationBundle(
            trade_date=latest_str,
            regime=regime_info,
            model_version_id=model_meta["version_id"],
            recommendations=recommendations,
            top_candidates_codes=top_candidates,
        )

    def get_latest_persisted(self) -> list[dict]:
        with self.sqlite.session_scope() as session:
            rows = (
                session.query(SignalRecommendation)
                .filter(SignalRecommendation.status == "ACTIVE")
                .order_by(SignalRecommendation.composite_rank.asc())
                .all()
            )
            return [
                {
                    "signal_id": r.signal_id,
                    "trade_date": r.trade_date,
                    "generated_at": r.generated_at.isoformat() if r.generated_at else "",
                    "ts_code": r.ts_code,
                    "name": r.name,
                    "sw_l1_name": r.sw_l1_name,
                    "action": r.action,
                    "composite_rank": r.composite_rank,
                    "model_score": r.model_score,
                    "suggested_weight": r.suggested_weight,
                    "ref_price": r.ref_price,
                    "stop_loss_price": r.stop_loss_price,
                    "take_profit_price": r.take_profit_price,
                    "top_factors": json.loads(r.top_factors_json or "[]"),
                    "tech_patterns": json.loads(r.tech_patterns_json or "[]"),
                    "rationale": r.rationale,
                    "status": r.status,
                }
                for r in rows
            ]
