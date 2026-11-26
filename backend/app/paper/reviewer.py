"""Weekly review & closed-loop iteration analyzer:
compares real paper trading performance vs theoretical backtest, attributes tracking error,
slippage drag & missed fills, detects rolling 20-day factor IC decay, and recommends retrain/downweight."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd

from app.backtest.engine import AShareBacktestEngine, BacktestConfig
from app.core.calendar import today_cn
from app.core.constants import FACTOR_META_MAP
from app.db.duckdb_client import DuckDBClient, get_duckdb
from app.db.models import BacktestRun, PaperNavDaily, PaperOrder, WeeklyReview
from app.db.sqlite_client import SQLiteDatabase, get_sqlite_db
from app.factors.definitions import forward_excess_returns
from app.factors.service import FactorEngine, to_jsonable
from app.ml.lgbm_scorer import LGBMConfig, LGBMScorer
from app.ml.model_registry import ModelRegistry
from app.strategy.regime import compute_market_regime
from app.strategy.technical import compute_technical_signals


class WeeklyReviewEngine:
    def __init__(
        self,
        duckdb_client: DuckDBClient | None = None,
        sqlite_db: SQLiteDatabase | None = None,
    ) -> None:
        self.db = duckdb_client or get_duckdb()
        self.sqlite = sqlite_db or get_sqlite_db()

    def generate_review(
        self,
        account_id: str = "default_paper",
        lookback_days: int = 7,
        progress_cb: Callable[[str, float], None] | None = None,
    ) -> dict:
        def _emit(stage: str, pct: float) -> None:
            if progress_cb is not None:
                progress_cb(stage, pct)

        _emit("正在汇总模拟实盘净值、滑点损耗与涨跌停漏单统计...", 0.12)
        end_d = today_cn()
        start_d = end_d - timedelta(days=lookback_days)

        # 1) Paper account NAV & real fill costs over the review window
        with self.sqlite.session_scope() as session:
            nav_rows = (
                session.query(PaperNavDaily)
                .filter(PaperNavDaily.account_id == account_id)
                .order_by(PaperNavDaily.trade_date.asc())
                .all()
            )
            order_rows = (
                session.query(PaperOrder)
                .filter(PaperOrder.account_id == account_id)
                .order_by(PaperOrder.created_at.desc())
                .limit(200)
                .all()
            )

        if len(nav_rows) >= 2:
            window_nav = [r for r in nav_rows if r.trade_date >= start_d] or nav_rows[-min(5, len(nav_rows)):]
            paper_ret = float(window_nav[-1].total_equity / max(window_nav[0].total_equity, 1.0) - 1.0)
            w_start = window_nav[0].trade_date.isoformat()
            w_end = window_nav[-1].trade_date.isoformat()
            avg_eq = float(np.mean([r.total_equity for r in window_nav]))
        elif len(nav_rows) == 1:
            paper_ret = float(nav_rows[0].cum_return)
            w_start = nav_rows[0].trade_date.isoformat()
            w_end = nav_rows[0].trade_date.isoformat()
            avg_eq = float(nav_rows[0].total_equity)
        else:
            paper_ret = 0.0
            w_start = start_d.isoformat()
            w_end = end_d.isoformat()
            avg_eq = 1_000_000.0

        filled_orders = [o for o in order_rows if o.status == "FILLED"]
        rejected_limit = [o for o in order_rows if o.status == "REJECTED_LIMIT"]
        total_slippage = sum(float(o.slippage_cost or 0.0) for o in filled_orders)
        total_fees = sum(
            float((o.commission or 0.0) + (o.stamp_duty or 0.0) + (o.transfer_fee or 0.0)) for o in filled_orders
        )
        slippage_drag = round(total_slippage / max(avg_eq, 1.0), 6)
        missed_fill_impact = round(len(rejected_limit) * 0.0015, 6)  # ~15 bps opportunity drag per limit-locked miss

        # 2) Theoretical backtest return over the same recent trading window
        # Prefer the latest persisted full-market BacktestRun (or cached AlphaStrategyService) to avoid
        # 200+ second synchronous 6-seed retraining on every weekly review click.
        _emit("正在对齐同期理论回测基准收益...", 0.35)
        theory_bt_ret = 0.0
        used_persisted = False
        try:
            with self.sqlite.session_scope() as session:
                latest_bt = (
                    session.query(BacktestRun)
                    .filter(BacktestRun.status == "DONE")
                    .order_by(BacktestRun.created_at.desc())
                    .first()
                )
                bt_curve = json.loads(latest_bt.equity_curve_json or "[]") if latest_bt is not None else []
            if len(bt_curve) >= 2:
                window_pts = [pt for pt in bt_curve if str(pt.get("date", "")) >= w_start]
                if len(window_pts) < 2:
                    window_pts = bt_curve[-min(6, len(bt_curve)):]
                nav0 = float(window_pts[0].get("nav") or window_pts[0].get("equity") or 1.0)
                nav1 = float(window_pts[-1].get("nav") or window_pts[-1].get("equity") or 1.0)
                if nav0 > 0:
                    theory_bt_ret = float(nav1 / nav0 - 1.0)
                    used_persisted = True
        except Exception:  # noqa: BLE001
            used_persisted = False

        if not used_persisted:
            try:
                from app.strategy.alpha_strategy import AlphaStrategyService

                _emit("未命中已持久化回测，正在通过二级缓存计算近期理论基准...", 0.48)
                svc = AlphaStrategyService(self.db)
                bundle = svc.run_full_pipeline(
                    lgbm_config=LGBMConfig(mode="regression", label_horizon=10, embargo_days=10),
                    bt_config=BacktestConfig(),
                    include_ew_compare=False,
                    include_heatmap=False,
                    use_cache=True,
                )
                if len(bundle.backtest.daily_nav) >= 2:
                    tail = bundle.backtest.daily_nav.tail(min(6, len(bundle.backtest.daily_nav)))
                    theory_bt_ret = float(tail.iloc[-1] / tail.iloc[0] - 1.0)
            except Exception:  # noqa: BLE001
                theory_bt_ret = paper_ret

        tracking_error = round(paper_ret - theory_bt_ret, 6)

        # 3) Rolling 20-day Factor IC Decay Detection from DuckDB factor_ic_daily
        _emit("正在扫描四维 20 因子近 20 日 IC 衰减与反转预警...", 0.72)
        ic_df = self.db.query_df(
            "SELECT trade_date, factor_name, ic, rank_ic FROM factor_ic_daily ORDER BY factor_name, trade_date"
        )
        decayed_factors: list[dict] = []
        if not ic_df.empty:
            for fname, grp in ic_df.groupby("factor_name"):
                valid = grp["rank_ic"].dropna()
                if len(valid) < 10:
                    continue
                hist_mean = float(valid.mean())
                recent_20 = float(valid.tail(20).mean())
                status = "STABLE"
                if np.sign(recent_20) != np.sign(hist_mean) and abs(hist_mean) > 0.01:
                    status = "REVERSED"
                elif abs(recent_20) < 0.5 * abs(hist_mean) and abs(hist_mean) > 0.015:
                    status = "DECAYING"
                if status in ("REVERSED", "DECAYING"):
                    meta = FACTOR_META_MAP.get(str(fname), {})
                    decayed_factors.append({
                        "factor_name": str(fname),
                        "label": str(meta.get("label", fname)),
                        "dimension": str(meta.get("dimension", "")),
                        "hist_rank_ic": round(hist_mean, 4),
                        "recent_20d_rank_ic": round(recent_20, 4),
                        "decay_status": status,
                    })

        # 4) Actionable optimization recommendations
        advice: list[dict] = []
        reversed_list = [d["factor_name"] for d in decayed_factors if d["decay_status"] == "REVERSED"]
        decaying_list = [d["factor_name"] for d in decayed_factors if d["decay_status"] == "DECAYING"]
        if reversed_list:
            advice.append({
                "type": "DOWNWEIGHT_OR_RETRAIN",
                "severity": "HIGH",
                "title": f"检测到 {len(reversed_list)} 个因子近 20 日 IC 反转",
                "detail": f"因子 {', '.join(reversed_list)} 近期 RankIC 符号反转，建议触发一键滚动重训或暂时降权。",
                "suggested_action": "RETRAIN",
            })
        if decaying_list:
            advice.append({
                "type": "MONITOR_DECAY",
                "severity": "MEDIUM",
                "title": f"检测到 {len(decaying_list)} 个因子有效度衰减超 50%",
                "detail": f"因子 {', '.join(decaying_list)} 近 20 日 IC 明显走弱，建议结合下一周表现评估是否调降特征权重。",
                "suggested_action": "DOWNWEIGHT",
            })
        if abs(slippage_drag) > 0.004:
            advice.append({
                "type": "EXECUTION_SLIPPAGE",
                "severity": "MEDIUM",
                "title": "实盘滑点损耗偏高",
                "detail": f"本周累计滑点损耗 {slippage_drag:.2%}，建议提高换手缓冲带（如 Top 25 → Top 30）或改用尾盘分批限价单。",
                "suggested_action": "TUNE_BUFFER",
            })
        if not advice:
            advice.append({
                "type": "HEALTHY",
                "severity": "LOW",
                "title": "因子与实盘跟踪表现健康",
                "detail": f"跟踪误差 {tracking_error:+.2%}，滑点损耗 {slippage_drag:.2%}，累计费用 ¥{total_fees:.2f}，维持当前激活模型运行。",
                "suggested_action": "KEEP",
            })

        active_model = ModelRegistry(self.sqlite).get_active()
        active_vid = active_model["version_id"] if active_model else ""
        review_id = f"rev_{uuid.uuid4().hex[:10]}"

        with self.sqlite.session_scope() as session:
            row = WeeklyReview(
                review_id=review_id,
                week_start=w_start,
                week_end=w_end,
                paper_return=round(paper_ret, 6),
                theory_bt_return=round(theory_bt_ret, 6),
                tracking_error=tracking_error,
                slippage_drag=slippage_drag,
                missed_fill_impact=missed_fill_impact,
                decayed_factors_json=json.dumps(to_jsonable(decayed_factors), ensure_ascii=False),
                optimization_advice_json=json.dumps(to_jsonable(advice), ensure_ascii=False),
                model_version_id=active_vid,
            )
            session.add(row)
            session.flush()
            return self._row_to_dict(row)

    def list_reviews(self, limit: int = 20) -> list[dict]:
        with self.sqlite.session_scope() as session:
            rows = session.query(WeeklyReview).order_by(WeeklyReview.created_at.desc()).limit(limit).all()
            return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: WeeklyReview) -> dict:
        return {
            "review_id": row.review_id,
            "week_start": row.week_start,
            "week_end": row.week_end,
            "created_at": row.created_at.isoformat() if row.created_at else "",
            "paper_return": row.paper_return,
            "theory_bt_return": row.theory_bt_return,
            "tracking_error": row.tracking_error,
            "slippage_drag": row.slippage_drag,
            "missed_fill_impact": row.missed_fill_impact,
            "decayed_factors": json.loads(row.decayed_factors_json or "[]"),
            "optimization_advice": json.loads(row.optimization_advice_json or "[]"),
            "model_version_id": row.model_version_id,
        }
