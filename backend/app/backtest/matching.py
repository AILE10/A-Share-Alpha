"""Single source of truth for A-share exchange matching rules (Guardrail #2).

Enforces:
  * T+1 (shares bought on day t cannot be sold until day t+1)
  * 100-share lot size for buys (odd-lot permitted ONLY for full clearance on sells)
  * Board & ST price limits (MAIN 10%, MAIN ST 5%, ChiNext/STAR 20%, BSE 30%)
  * Limit-up lock blocks buys; limit-down lock blocks sells; suspended blocks both
  * Exact fees: commission max(amount * 0.00015, 5.0), sell stamp duty 0.05%, transfer fee 0.001%
  * Adverse slippage bounded inside [limit_down, limit_up]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.constants import (
    COMMISSION_RATE,
    DEFAULT_SLIPPAGE_BPS,
    LOT_SIZE,
    MIN_COMMISSION,
    STAMP_DUTY_RATE,
    TRANSFER_FEE_RATE,
    calc_limit_prices,
    round_price,
)

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class BarQuote:
    ts_code: str
    open: float
    high: float
    low: float
    close: float
    pre_close: float
    volume: float
    is_st: bool = False
    is_suspended: bool = False
    limit_up: float | None = None
    limit_down: float | None = None
    ask1: float | None = None
    bid1: float | None = None

    def limits(self) -> tuple[float, float]:
        if self.limit_up and self.limit_down and self.limit_up > 0 and self.limit_down > 0:
            return (float(self.limit_up), float(self.limit_down))
        return calc_limit_prices(self.pre_close, self.ts_code, is_st=self.is_st)


@dataclass(frozen=True)
class FeeBreakdown:
    commission: float
    stamp_duty: float
    transfer_fee: float
    slippage_cost: float

    @property
    def total_fees(self) -> float:
        return self.commission + self.stamp_duty + self.transfer_fee


@dataclass(frozen=True)
class FillResult:
    filled: bool
    ts_code: str
    side: Side
    requested_qty: int
    filled_qty: int
    ref_price: float
    exec_price: float
    gross_amount: float
    fees: FeeBreakdown
    net_cash_delta: float   # negative for BUY (cash outflow), positive for SELL (cash inflow)
    reject_code: str = ""
    reject_reason: str = ""


def compute_fees(side: Side, gross_amount: float, slippage_cost: float = 0.0) -> FeeBreakdown:
    """Exact A-share transaction cost breakdown."""
    if gross_amount <= 0:
        return FeeBreakdown(0.0, 0.0, 0.0, 0.0)
    commission = max(round(gross_amount * COMMISSION_RATE, 2), MIN_COMMISSION)
    stamp_duty = round(gross_amount * STAMP_DUTY_RATE, 2) if side == "SELL" else 0.0
    transfer_fee = round(gross_amount * TRANSFER_FEE_RATE, 2)
    return FeeBreakdown(
        commission=commission,
        stamp_duty=stamp_duty,
        transfer_fee=transfer_fee,
        slippage_cost=round(abs(slippage_cost), 2),
    )


def round_buy_lots(qty: int) -> int:
    """Round buy share quantity DOWN to integer multiples of 100 shares."""
    return max(0, (int(qty) // LOT_SIZE) * LOT_SIZE)


def is_limit_up_locked(quote: BarQuote, exec_ref: float, mode: Literal["open", "close", "intraday"] = "open") -> bool:
    """Detect whether buy orders are blocked by limit-up sealing."""
    up, _ = quote.limits()
    if up <= 0:
        return False
    eps = 0.005
    # One-word limit-up board (high == low == limit_up)
    if quote.high > 0 and quote.low > 0 and abs(quote.high - quote.low) < eps and quote.high >= up - eps:
        return True
    if mode == "open" and quote.open >= up - eps:
        return True
    if mode == "close" and quote.close >= up - eps:
        return True
    if mode == "intraday":
        # Sealed at limit-up if ask1 is missing/zero or >= limit_up while price is at limit_up
        if quote.close >= up - eps and (quote.ask1 is None or quote.ask1 <= 0 or quote.ask1 >= up - eps):
            return True
    return exec_ref >= up - eps


def is_limit_down_locked(quote: BarQuote, exec_ref: float, mode: Literal["open", "close", "intraday"] = "open") -> bool:
    """Detect whether sell orders are blocked by limit-down sealing."""
    _, down = quote.limits()
    if down <= 0:
        return False
    eps = 0.005
    if quote.high > 0 and quote.low > 0 and abs(quote.high - quote.low) < eps and quote.low <= down + eps:
        return True
    if mode == "open" and quote.open <= down + eps:
        return True
    if mode == "close" and quote.close <= down + eps:
        return True
    if mode == "intraday":
        if quote.close <= down + eps and (quote.bid1 is None or quote.bid1 <= 0 or quote.bid1 <= down + eps):
            return True
    return exec_ref <= down + eps


def match_order(
    quote: BarQuote,
    side: Side,
    requested_qty: int,
    available_cash: float = float("inf"),
    sellable_shares: int = 0,
    total_shares: int = 0,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    price_mode: Literal["open", "close", "intraday"] = "open",
) -> FillResult:
    """Simulate a single A-share order fill enforcing T+1, lots, limits, suspension & exact fees."""
    zero_fees = FeeBreakdown(0.0, 0.0, 0.0, 0.0)

    if quote.is_suspended or quote.volume <= 0 or (quote.close <= 0 and quote.open <= 0):
        return FillResult(
            False, quote.ts_code, side, requested_qty, 0, 0.0, 0.0, 0.0, zero_fees, 0.0,
            reject_code="SUSPENDED", reject_reason=f"{quote.ts_code} 停牌或无成交，无法撮合",
        )

    if price_mode == "intraday":
        if side == "BUY":
            ref_price = quote.ask1 if (quote.ask1 and quote.ask1 > 0) else quote.close
        else:
            ref_price = quote.bid1 if (quote.bid1 and quote.bid1 > 0) else quote.close
    elif price_mode == "close":
        ref_price = quote.close
    else:
        ref_price = quote.open if quote.open > 0 else quote.close

    if ref_price <= 0:
        return FillResult(
            False, quote.ts_code, side, requested_qty, 0, 0.0, 0.0, 0.0, zero_fees, 0.0,
            reject_code="INVALID_PRICE", reject_reason=f"{quote.ts_code} 无有效行情价格",
        )

    up, down = quote.limits()

    if side == "BUY":
        if is_limit_up_locked(quote, ref_price, mode=price_mode):
            return FillResult(
                False, quote.ts_code, side, requested_qty, 0, ref_price, 0.0, 0.0, zero_fees, 0.0,
                reject_code="LIMIT_UP_LOCKED",
                reject_reason=f"{quote.ts_code} 涨停封板 ({up:.2f})，禁止买入",
            )
        slip_price = round_price(ref_price * (1.0 + slippage_bps / 10_000.0))
        exec_price = min(slip_price, up) if up > 0 else slip_price
        qty = round_buy_lots(requested_qty)
        if qty < LOT_SIZE:
            return FillResult(
                False, quote.ts_code, side, requested_qty, 0, ref_price, exec_price, 0.0, zero_fees, 0.0,
                reject_code="BELOW_LOT_SIZE",
                reject_reason=f"买入数量 {requested_qty} 不足一手 (100 股)",
            )
        # Scale down by 100-share lots if cash (including fees) is insufficient
        while qty >= LOT_SIZE:
            gross = round(qty * exec_price, 2)
            fees = compute_fees("BUY", gross, slippage_cost=(exec_price - ref_price) * qty)
            total_out = gross + fees.total_fees
            if total_out <= available_cash + 1e-6:
                return FillResult(
                    True, quote.ts_code, side, requested_qty, qty, ref_price, exec_price,
                    gross, fees, -round(total_out, 2),
                )
            qty -= LOT_SIZE
        return FillResult(
            False, quote.ts_code, side, requested_qty, 0, ref_price, exec_price, 0.0, zero_fees, 0.0,
            reject_code="INSUFFICIENT_CASH",
            reject_reason=f"可用资金 {available_cash:.2f} 不足买入至少 100 股 ({exec_price:.2f})",
        )

    # SELL branch --------------------------------------------------------------
    if sellable_shares <= 0:
        return FillResult(
            False, quote.ts_code, side, requested_qty, 0, ref_price, 0.0, 0.0, zero_fees, 0.0,
            reject_code="T_PLUS_1_LOCKED",
            reject_reason=f"{quote.ts_code} T+1 冻结中（可卖 0 股 / 总持仓 {total_shares} 股），当日不可卖出",
        )
    if is_limit_down_locked(quote, ref_price, mode=price_mode):
        return FillResult(
            False, quote.ts_code, side, requested_qty, 0, ref_price, 0.0, 0.0, zero_fees, 0.0,
            reject_code="LIMIT_DOWN_LOCKED",
            reject_reason=f"{quote.ts_code} 跌停封板 ({down:.2f})，无法卖出，顺延至次日重试",
        )

    slip_price = round_price(ref_price * (1.0 - slippage_bps / 10_000.0))
    exec_price = max(slip_price, down) if down > 0 else slip_price

    # Allow selling full sellable position (including odd-lots after splits), otherwise 100-share multiples
    qty = min(int(requested_qty), int(sellable_shares))
    if qty < sellable_shares and qty >= LOT_SIZE:
        qty = (qty // LOT_SIZE) * LOT_SIZE
    if qty <= 0:
        return FillResult(
            False, quote.ts_code, side, requested_qty, 0, ref_price, exec_price, 0.0, zero_fees, 0.0,
            reject_code="ZERO_QTY", reject_reason="卖出数量为 0",
        )

    gross = round(qty * exec_price, 2)
    fees = compute_fees("SELL", gross, slippage_cost=(ref_price - exec_price) * qty)
    net_in = round(gross - fees.total_fees, 2)
    return FillResult(
        True, quote.ts_code, side, requested_qty, qty, ref_price, exec_price,
        gross, fees, net_in,
    )
