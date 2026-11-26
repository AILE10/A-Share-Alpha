"""Unified pre-trade RiskMiddleware (Guardrail #4).

Every order destined for PaperBrokerAdapter, AssistedSignalAdapter or CompliantQMTAdapter
MUST pass through RiskMiddleware.check_and_route() before reaching any execution adapter.
Checks:
  1. Order frequency rate limiting (sliding 1-minute & daily counters)
  2. Portfolio drawdown circuit breaker (blocks BUY when HALTED, halves single cap on WARNING)
  3. Basic A-share sanity (100-share lots for BUY, T+1 sellable check for SELL, price > 0)
  4. Single-stock position weight cap (default <= 12%)
  5. Shenwan L1 industry concentration cap (default <= 25%)
Every decision (PASS / REJECT / CLIPPED) is persisted to SQLite risk_audit_logs.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

from app.broker.base import BaseBrokerAdapter, OrderReceipt, OrderRequest
from app.core.calendar import today_cn
from app.core.constants import DEFAULT_WARN_EXPOSURE_MULT, LOT_SIZE
from app.core.logging import get_logger
from app.db.models import PaperAccount, PaperPosition, RiskAuditLog, RiskConfig
from app.db.sqlite_client import SQLiteDatabase, get_sqlite_db

logger = get_logger(__name__)


@dataclass
class RiskCheckResult:
    passed: bool
    decision: str         # PASS / REJECT / CLIPPED
    rule_name: str
    reason: str
    detail: dict


class RiskMiddleware:
    """Pre-trade risk enforcement & audit middleware wrapping all broker adapters."""

    def __init__(self, sqlite_db: SQLiteDatabase | None = None) -> None:
        self.sqlite = sqlite_db or get_sqlite_db()
        self._lock = threading.Lock()
        self._minute_window: deque[float] = deque()
        self._day_key: str = ""
        self._day_count: int = 0

    def reset_rate_counters(self) -> None:
        with self._lock:
            self._minute_window.clear()
            self._day_count = 0

    def get_config(self) -> dict:
        with self.sqlite.session_scope() as session:
            cfg = session.get(RiskConfig, 1)
            if cfg is None:
                cfg = RiskConfig(id=1)
                session.add(cfg)
                session.flush()
            return {
                "max_single_weight": cfg.max_single_weight,
                "max_industry_weight": cfg.max_industry_weight,
                "stock_hard_stop_pct": cfg.stock_hard_stop_pct,
                "stock_atr_mult": cfg.stock_atr_mult,
                "drawdown_warn_pct": cfg.drawdown_warn_pct,
                "drawdown_halt_pct": cfg.drawdown_halt_pct,
                "max_orders_per_min": cfg.max_orders_per_min,
                "max_orders_per_day": cfg.max_orders_per_day,
            }

    def update_config(self, updates: dict) -> dict:
        allowed = {
            "max_single_weight", "max_industry_weight", "stock_hard_stop_pct", "stock_atr_mult",
            "drawdown_warn_pct", "drawdown_halt_pct", "max_orders_per_min", "max_orders_per_day",
        }
        with self.sqlite.session_scope() as session:
            cfg = session.get(RiskConfig, 1)
            if cfg is None:
                cfg = RiskConfig(id=1)
                session.add(cfg)
            for k, v in updates.items():
                if k in allowed and v is not None:
                    setattr(cfg, k, v)
            session.flush()
        return self.get_config()

    def _log_audit(self, order_id: str, req: OrderRequest, res: RiskCheckResult) -> None:
        with self.sqlite.session_scope() as session:
            session.add(
                RiskAuditLog(
                    order_id=order_id,
                    ts_code=req.ts_code,
                    side=req.side,
                    rule_name=res.rule_name,
                    decision=res.decision,
                    detail_json=json.dumps({"reason": res.reason, **res.detail}, ensure_ascii=False),
                )
            )

    def _check_rate_limit(self, max_per_min: int, max_per_day: int) -> RiskCheckResult | None:
        now = time.monotonic()
        today_key = today_cn().isoformat()
        with self._lock:
            if self._day_key != today_key:
                self._day_key = today_key
                self._day_count = 0
            while self._minute_window and now - self._minute_window[0] > 60.0:
                self._minute_window.popleft()
            if len(self._minute_window) >= max_per_min:
                return RiskCheckResult(
                    False, "REJECT", "RATE_LIMIT_MINUTE",
                    f"1分钟报撤单频率超限 ({len(self._minute_window)}/{max_per_min})",
                    {"count_1m": len(self._minute_window), "limit_1m": max_per_min},
                )
            if self._day_count >= max_per_day:
                return RiskCheckResult(
                    False, "REJECT", "RATE_LIMIT_DAILY",
                    f"当日报撤单总次数超限 ({self._day_count}/{max_per_day})",
                    {"count_day": self._day_count, "limit_day": max_per_day},
                )
            self._minute_window.append(now)
            self._day_count += 1
        return None

    def validate(self, req: OrderRequest, order_id: str = "") -> RiskCheckResult:
        cfg = self.get_config()

        # 1) Order frequency rate limit
        rate_err = self._check_rate_limit(cfg["max_orders_per_min"], cfg["max_orders_per_day"])
        if rate_err is not None:
            self._log_audit(order_id, req, rate_err)
            return rate_err

        # 2) Basic order validity
        if req.ref_price <= 0 or req.qty <= 0:
            res = RiskCheckResult(
                False, "REJECT", "ORDER_SANITY",
                f"无效委托价格({req.ref_price})或数量({req.qty})",
                {"ref_price": req.ref_price, "qty": req.qty},
            )
            self._log_audit(order_id, req, res)
            return res

        if req.side == "BUY" and (req.qty < LOT_SIZE or req.qty % LOT_SIZE != 0):
            res = RiskCheckResult(
                False, "REJECT", "LOT_SIZE_RULE",
                f"A股买入数量必须为100股整数倍，当前委托 {req.qty} 股",
                {"qty": req.qty, "lot_size": LOT_SIZE},
            )
            self._log_audit(order_id, req, res)
            return res

        # Load current account & positions from SQLite
        with self.sqlite.session_scope() as session:
            acct = session.get(PaperAccount, req.account_id)
            if acct is None:
                res = RiskCheckResult(False, "REJECT", "ACCOUNT_NOT_FOUND", f"账户不存在: {req.account_id}", {})
                self._log_audit(order_id, req, res)
                return res
            positions = session.query(PaperPosition).filter(PaperPosition.account_id == req.account_id).all()
            pos_map = {p.ts_code: p for p in positions}
            total_equity = max(float(acct.total_equity), float(acct.cash), 1.0)
            cur_dd = float(acct.current_drawdown)
            fuse = str(acct.fuse_status)

            # 3) SELL checks (always permitted even under drawdown fuse so exposure can be reduced,
            #    but still enforces T+1 sellable shares when tracked in local positions)
            if req.side == "SELL":
                pos = pos_map.get(req.ts_code)
                if pos is not None and pos.sellable_shares <= 0:
                    res = RiskCheckResult(
                        False, "REJECT", "T_PLUS_1_PRECHECK",
                        f"{req.ts_code} T+1 冻结中（可卖 0 / 总持仓 {pos.total_shares}），禁止当日卖出",
                        {"total_shares": pos.total_shares, "sellable_shares": pos.sellable_shares},
                    )
                    self._log_audit(order_id, req, res)
                    return res
                ok = RiskCheckResult(True, "PASS", "PRE_TRADE_OK", "卖出风控放行", {"side": "SELL"})
                self._log_audit(order_id, req, ok)
                return ok

            # 4) Portfolio drawdown circuit breaker (blocks BUY)
            if fuse == "HALTED" or cur_dd <= -cfg["drawdown_halt_pct"]:
                res = RiskCheckResult(
                    False, "REJECT", "DRAWDOWN_CIRCUIT_BREAKER",
                    f"组合回撤触发熔断 ({cur_dd:.2%} <= -{cfg['drawdown_halt_pct']:.2%})，禁止新增买入",
                    {"current_drawdown": cur_dd, "halt_threshold": -cfg["drawdown_halt_pct"], "fuse_status": fuse},
                )
                self._log_audit(order_id, req, res)
                return res

            order_notional = req.qty * req.ref_price

            # 5) Single-stock weight cap (scaled to 70% under WARNING state, matching DrawdownFuseConfig)
            eff_single_cap = cfg["max_single_weight"] * (DEFAULT_WARN_EXPOSURE_MULT if fuse == "WARNING" else 1.0)
            existing_mv = float(pos_map[req.ts_code].market_value) if req.ts_code in pos_map else 0.0
            proj_single_w = (existing_mv + order_notional) / total_equity
            if proj_single_w > eff_single_cap + 1e-4:
                res = RiskCheckResult(
                    False, "REJECT", "SINGLE_STOCK_CAP",
                    f"{req.ts_code} 买入后单票仓位 {proj_single_w:.2%} 超出上限 {eff_single_cap:.2%}",
                    {"projected_weight": round(proj_single_w, 4), "limit": eff_single_cap, "fuse_status": fuse},
                )
                self._log_audit(order_id, req, res)
                return res

            # 6) Shenwan L1 industry concentration cap
            ind_name = req.sw_l1_name or "未分类"
            existing_ind_mv = sum(
                float(p.market_value) for p in positions if (p.sw_l1_name or "未分类") == ind_name
            )
            proj_ind_w = (existing_ind_mv + order_notional) / total_equity
            if proj_ind_w > cfg["max_industry_weight"] + 1e-4:
                res = RiskCheckResult(
                    False, "REJECT", "INDUSTRY_CONCENTRATION_CAP",
                    f"行业[{ind_name}]买入后集中度 {proj_ind_w:.2%} 超出上限 {cfg['max_industry_weight']:.2%}",
                    {"industry": ind_name, "projected_weight": round(proj_ind_w, 4), "limit": cfg["max_industry_weight"]},
                )
                self._log_audit(order_id, req, res)
                return res

        ok = RiskCheckResult(
            True, "PASS", "PRE_TRADE_OK", "前置风控全部通过",
            {
                "projected_single_weight": round(proj_single_w, 4),
                "projected_industry_weight": round(proj_ind_w, 4),
                "current_drawdown": round(cur_dd, 4),
            },
        )
        self._log_audit(order_id, req, ok)
        return ok

    def execute(self, adapter: BaseBrokerAdapter, req: OrderRequest) -> OrderReceipt:
        """Validate through RiskMiddleware, then delegate to the active BrokerAdapter."""
        pre_id = f"ord_{uuid.uuid4().hex[:12]}"
        check = self.validate(req, order_id=pre_id)
        if not check.passed:
            logger.warning("RiskMiddleware REJECTED %s %s qty=%d: %s", req.side, req.ts_code, req.qty, check.reason)
            return OrderReceipt(
                order_id=pre_id,
                ts_code=req.ts_code,
                side=req.side,
                broker_mode=adapter.mode,
                status="REJECTED_RISK",
                requested_qty=req.qty,
                reject_reason=f"[{check.rule_name}] {check.reason}",
                extra={"risk_detail": check.detail},
            )
        return adapter.submit_order(req)

    def get_audit_logs(self, limit: int = 100) -> list[dict]:
        with self.sqlite.session_scope() as session:
            rows = session.query(RiskAuditLog).order_by(RiskAuditLog.timestamp.desc()).limit(limit).all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat(),
                    "order_id": r.order_id,
                    "ts_code": r.ts_code,
                    "side": r.side,
                    "rule_name": r.rule_name,
                    "decision": r.decision,
                    "detail": json.loads(r.detail_json or "{}"),
                }
                for r in rows
            ]
