"""SQLAlchemy ORM models for studio operational state, paper trading, reviews, models and risk."""

from __future__ import annotations

from datetime import datetime, date

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.calendar import now_cn
from app.db.sqlite_client import Base


def _utcnow_cn() -> datetime:
    return now_cn().replace(tzinfo=None)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), default="Default Alpha Backtest")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn)
    status: Mapped[str] = mapped_column(String(24), default="DONE")
    start_date: Mapped[str] = mapped_column(String(16))
    end_date: Mapped[str] = mapped_column(String(16))
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    equity_curve_json: Mapped[str] = mapped_column(Text, default="[]")
    drawdown_curve_json: Mapped[str] = mapped_column(Text, default="[]")
    trades_json: Mapped[str] = mapped_column(Text, default="[]")
    wf_heatmap_json: Mapped[str] = mapped_column(Text, default="{}")
    error_message: Mapped[str] = mapped_column(Text, default="")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn)
    model_type: Mapped[str] = mapped_column(String(32), default="lambdarank")
    train_start: Mapped[str] = mapped_column(String(16), default="")
    train_end: Mapped[str] = mapped_column(String(16), default="")
    embargo_days: Mapped[int] = mapped_column(Integer, default=5)
    artifact_path: Mapped[str] = mapped_column(String(512), default="")
    feature_list_json: Mapped[str] = mapped_column(Text, default="[]")
    oos_ic: Mapped[float] = mapped_column(Float, default=0.0)
    oos_rank_ic: Mapped[float] = mapped_column(Float, default=0.0)
    oos_icir: Mapped[float] = mapped_column(Float, default=0.0)
    feature_importance_json: Mapped[str] = mapped_column(Text, default="{}")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")


class PaperAccount(Base):
    __tablename__ = "paper_accounts"

    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    broker_mode: Mapped[str] = mapped_column(String(24), default="PAPER")
    initial_cash: Mapped[float] = mapped_column(Float, default=1_000_000.0)
    cash: Mapped[float] = mapped_column(Float, default=1_000_000.0)
    total_equity: Mapped[float] = mapped_column(Float, default=1_000_000.0)
    peak_equity: Mapped[float] = mapped_column(Float, default=1_000_000.0)
    current_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    fuse_status: Mapped[str] = mapped_column(String(24), default="NORMAL")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn, onupdate=_utcnow_cn)


class PaperPosition(Base):
    __tablename__ = "paper_positions"
    __table_args__ = (UniqueConstraint("account_id", "ts_code", name="uq_account_position"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    sw_l1_name: Mapped[str] = mapped_column(String(64), default="未分类")
    total_shares: Mapped[int] = mapped_column(Integer, default=0)
    sellable_shares: Mapped[int] = mapped_column(Integer, default=0)
    avg_cost: Mapped[float] = mapped_column(Float, default=0.0)
    entry_date: Mapped[str] = mapped_column(String(16), default="")
    highest_price_since_entry: Mapped[float] = mapped_column(Float, default=0.0)
    current_price: Mapped[float] = mapped_column(Float, default=0.0)
    stop_loss_price: Mapped[float] = mapped_column(Float, default=0.0)
    atr_stop_price: Mapped[float] = mapped_column(Float, default=0.0)
    market_value: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn, onupdate=_utcnow_cn)


class PaperOrder(Base):
    __tablename__ = "paper_orders"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    sw_l1_name: Mapped[str] = mapped_column(String(64), default="未分类")
    side: Mapped[str] = mapped_column(String(8))  # BUY / SELL
    order_type: Mapped[str] = mapped_column(String(16), default="LIMIT")
    request_qty: Mapped[int] = mapped_column(Integer, default=0)
    filled_qty: Mapped[int] = mapped_column(Integer, default=0)
    limit_price: Mapped[float] = mapped_column(Float, default=0.0)
    avg_fill_price: Mapped[float] = mapped_column(Float, default=0.0)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    stamp_duty: Mapped[float] = mapped_column(Float, default=0.0)
    transfer_fee: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_cost: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="SUBMITTED")
    broker_mode: Mapped[str] = mapped_column(String(24), default="PAPER")
    signal_id: Mapped[str] = mapped_column(String(64), default="")
    reject_reason: Mapped[str] = mapped_column(Text, default="")
    reason_summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn, onupdate=_utcnow_cn)


class PaperNavDaily(Base):
    __tablename__ = "paper_nav_daily"
    __table_args__ = (UniqueConstraint("account_id", "trade_date", name="uq_account_nav_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    total_equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    market_value: Mapped[float] = mapped_column(Float)
    daily_return: Mapped[float] = mapped_column(Float, default=0.0)
    cum_return: Mapped[float] = mapped_column(Float, default=0.0)
    csi1000_cum_return: Mapped[float] = mapped_column(Float, default=0.0)
    csi300_cum_return: Mapped[float] = mapped_column(Float, default=0.0)
    excess_return_1000: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    regime: Mapped[str] = mapped_column(String(24), default="NEUTRAL")
    turnover: Mapped[float] = mapped_column(Float, default=0.0)


class SignalRecommendation(Base):
    __tablename__ = "signal_recommendations"

    signal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trade_date: Mapped[str] = mapped_column(String(16), index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn)
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    sw_l1_name: Mapped[str] = mapped_column(String(64), default="未分类")
    action: Mapped[str] = mapped_column(String(16))  # BUY / SELL / HOLD
    composite_rank: Mapped[int] = mapped_column(Integer, default=0)
    model_score: Mapped[float] = mapped_column(Float, default=0.0)
    suggested_weight: Mapped[float] = mapped_column(Float, default=0.0)
    ref_price: Mapped[float] = mapped_column(Float, default=0.0)
    stop_loss_price: Mapped[float] = mapped_column(Float, default=0.0)
    take_profit_price: Mapped[float] = mapped_column(Float, default=0.0)
    top_factors_json: Mapped[str] = mapped_column(Text, default="[]")
    tech_patterns_json: Mapped[str] = mapped_column(Text, default="[]")
    rationale: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="ACTIVE")


class WeeklyReview(Base):
    __tablename__ = "weekly_reviews"

    review_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    week_start: Mapped[str] = mapped_column(String(16))
    week_end: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn)
    paper_return: Mapped[float] = mapped_column(Float, default=0.0)
    theory_bt_return: Mapped[float] = mapped_column(Float, default=0.0)
    tracking_error: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_drag: Mapped[float] = mapped_column(Float, default=0.0)
    missed_fill_impact: Mapped[float] = mapped_column(Float, default=0.0)
    decayed_factors_json: Mapped[str] = mapped_column(Text, default="[]")
    optimization_advice_json: Mapped[str] = mapped_column(Text, default="[]")
    model_version_id: Mapped[str] = mapped_column(String(64), default="")


class RiskConfig(Base):
    __tablename__ = "risk_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    max_single_weight: Mapped[float] = mapped_column(Float, default=0.12)
    max_industry_weight: Mapped[float] = mapped_column(Float, default=0.25)
    stock_hard_stop_pct: Mapped[float] = mapped_column(Float, default=0.08)
    stock_atr_mult: Mapped[float] = mapped_column(Float, default=2.5)
    drawdown_warn_pct: Mapped[float] = mapped_column(Float, default=0.08)
    drawdown_halt_pct: Mapped[float] = mapped_column(Float, default=0.12)
    max_orders_per_min: Mapped[int] = mapped_column(Integer, default=30)
    max_orders_per_day: Mapped[int] = mapped_column(Integer, default=200)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn, onupdate=_utcnow_cn)


class BrokerConfig(Base):
    __tablename__ = "broker_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    active_mode: Mapped[str] = mapped_column(String(24), default="PAPER")
    wecom_webhook_url: Mapped[str] = mapped_column(String(512), default="")
    dingtalk_webhook_url: Mapped[str] = mapped_column(String(512), default="")
    dingtalk_secret: Mapped[str] = mapped_column(String(256), default="")
    qmt_userdata_path: Mapped[str] = mapped_column(String(512), default="")
    qmt_account_id: Mapped[str] = mapped_column(String(128), default="")
    qmt_dry_run: Mapped[bool] = mapped_column(Boolean, default=True)
    tushare_token_configured: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn, onupdate=_utcnow_cn)


class RiskAuditLog(Base):
    __tablename__ = "risk_audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_cn, index=True)
    order_id: Mapped[str] = mapped_column(String(64), default="")
    ts_code: Mapped[str] = mapped_column(String(16), default="")
    side: Mapped[str] = mapped_column(String(8), default="")
    rule_name: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(16))  # PASS / REJECT / CLIPPED
    detail_json: Mapped[str] = mapped_column(Text, default="{}")
