"""Unified BaseBrokerAdapter interface & Order/Account DTOs (Adapter Pattern, Guardrail #4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

BrokerMode = Literal["PAPER", "ASSISTED", "QMT"]
OrderSide = Literal["BUY", "SELL"]


@dataclass
class OrderRequest:
    account_id: str
    ts_code: str
    name: str
    sw_l1_name: str
    side: OrderSide
    qty: int
    ref_price: float
    pre_close: float = 0.0
    ask1: float = 0.0
    bid1: float = 0.0
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    volume: float = 100_000.0
    is_st: bool = False
    is_suspended: bool = False
    limit_up: float = 0.0
    limit_down: float = 0.0
    signal_id: str = ""
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    suggested_weight: float = 0.0
    reason_summary: str = ""
    top_factors: list[dict] = field(default_factory=list)
    tech_patterns: list[str] = field(default_factory=list)


@dataclass
class OrderReceipt:
    order_id: str
    ts_code: str
    side: OrderSide
    broker_mode: BrokerMode
    status: str  # FILLED / PENDING_CONFIRM / SUBMITTED / REJECTED_RISK / REJECTED_LIMIT / CANCELLED
    requested_qty: int
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0
    slippage_cost: float = 0.0
    reject_reason: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class BaseBrokerAdapter(ABC):
    """Abstract compliant broker adapter. Never write reverse-engineering or GUI-click bots."""

    mode: BrokerMode = "PAPER"

    @abstractmethod
    def submit_order(self, req: OrderRequest) -> OrderReceipt:
        """Execute or stage an order AFTER RiskMiddleware has approved it."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending order."""

    @abstractmethod
    def get_account_state(self, account_id: str = "default_paper") -> dict:
        """Return current cash, equity, drawdown, fuse status and positions."""

    @abstractmethod
    def health_check(self) -> dict:
        """Verify adapter connectivity / compliance status."""
