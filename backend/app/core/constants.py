"""A-Share market rules, trading fees, board classification and factor metadata."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

BoardType = Literal["MAIN", "CHINEXT", "STAR", "BSE"]

# A-Share real trading constraints & fees (post Aug-2023 stamp duty reform)
LOT_SIZE: int = 100
COMMISSION_RATE: float = 0.00015  # 万1.5
MIN_COMMISSION: float = 5.0       # 最低 5 元
STAMP_DUTY_RATE: float = 0.0005   # 卖出 0.05%
TRANSFER_FEE_RATE: float = 0.00001  # 过户费 0.001%（双向）
DEFAULT_SLIPPAGE_BPS: float = 10.0  # 默认滑点 10 bps (0.1%)

# Benchmark indices
BENCHMARK_CSI1000 = "000852.SH"
BENCHMARK_CSI300 = "000300.SH"
BENCHMARK_SHCOMP = "000001.SH"

BENCHMARK_INDEX_META = {
    BENCHMARK_CSI1000: {"name": "中证1000", "ak_symbol": "sh000852", "bs_code": "sh.000852"},
    BENCHMARK_CSI300: {"name": "沪深300", "ak_symbol": "sh000300", "bs_code": "sh.000300"},
    BENCHMARK_SHCOMP: {"name": "上证指数", "ak_symbol": "sh000001", "bs_code": "sh.000001"},
}

# Strategy defaults
DEFAULT_REBALANCE_DAYS = 5
MIN_EMBARGO_DAYS = 5
DEFAULT_MIN_LIST_DAYS = 120
DEFAULT_MIN_AVG_AMOUNT_20D = 50_000_000.0  # 5000 万元
DEFAULT_MICRO_CAP_QUANTILE = 0.10          # 剔除总市值后 10%
DEFAULT_MIN_PRICE = 2.5                    # 防 1 元面值退市硬底线（剔除 < 2.5 元低价仙股）
DEFAULT_BULL_MOM_H120_WEIGHT = 0.10        # OFFENSIVE 强牛期温和动量倾斜：距120日新高权重（非强牛=0）
DEFAULT_BULL_MOM_RET20_WEIGHT = 0.08       # OFFENSIVE 强牛期温和动量倾斜：20日动量权重（非强牛=0）

# 4-Dimension 20 Core Factors Metadata
FACTOR_DEFINITIONS: list[dict[str, str | int]] = [
    # 1. 量价与动量反转
    {"name": "ret_20d", "label": "20日收益率", "dimension": "量价与动量反转", "direction": 1},
    {"name": "ret_60d", "label": "60日收益率", "dimension": "量价与动量反转", "direction": 1},
    {"name": "rev_5d", "label": "5日短期反转", "dimension": "量价与动量反转", "direction": 1},
    {"name": "pv_corr_20d", "label": "20日量价相关性", "dimension": "量价与动量反转", "direction": -1},
    {"name": "intraday_pos", "label": "日内收盘位置", "dimension": "量价与动量反转", "direction": -1},
    {"name": "smart_intra_minus_on_20d", "label": "20日日内-隔夜收益差", "dimension": "量价与动量反转", "direction": -1},
    # 2. 流动性与资金流
    {"name": "turnover_ratio_20_60", "label": "20/60日异常换手", "dimension": "流动性与资金流", "direction": -1},
    {"name": "turn_cv_20d", "label": "20日换手变异系数", "dimension": "流动性与资金流", "direction": -1},
    {"name": "turn_ratio_5_20", "label": "5/20日对数换手比", "dimension": "流动性与资金流", "direction": -1},
    {"name": "amihud_20d", "label": "Amihud非流动性", "dimension": "流动性与资金流", "direction": 1},
    {"name": "vol_ratio_5d", "label": "5日量比", "dimension": "流动性与资金流", "direction": 1},
    {"name": "main_net_inflow_ratio", "label": "主力净流入占比", "dimension": "流动性与资金流", "direction": 1},
    # 3. 波动率与筹码结构
    {"name": "idio_vol_20d", "label": "20日特质波动率", "dimension": "波动率与筹码结构", "direction": -1},
    {"name": "downside_vol_ratio_20d", "label": "下行波动率占比", "dimension": "波动率与筹码结构", "direction": -1},
    {"name": "price_pct_60d", "label": "60日价格分位数", "dimension": "波动率与筹码结构", "direction": -1},
    {"name": "high_120d_prox", "label": "距120日新高距离", "dimension": "波动率与筹码结构", "direction": 1},
    # 4. 基本面质量与排雷
    {"name": "ep_ttm", "label": "EP (1/PE_TTM)", "dimension": "基本面质量与排雷", "direction": 1},
    {"name": "bp", "label": "BP (1/PB)", "dimension": "基本面质量与排雷", "direction": 1},
    {"name": "roe_ttm", "label": "ROE_TTM (PIT)", "dimension": "基本面质量与排雷", "direction": 1},
    {"name": "np_yoy_q", "label": "单季归母净利同比 (PIT)", "dimension": "基本面质量与排雷", "direction": 1},
]

ALL_FACTOR_NAMES: list[str] = [str(f["name"]) for f in FACTOR_DEFINITIONS]
FACTOR_META_MAP: dict[str, dict[str, str | int]] = {str(f["name"]): f for f in FACTOR_DEFINITIONS}


def normalize_ts_code(raw_code: str) -> str:
    """Normalize stock code (e.g. '600519', 'sh.600519', 'SH600519', '600519.SH') -> '600519.SH'."""
    code = raw_code.strip().upper()
    if "." in code:
        parts = code.split(".")
        if parts[0] in {"SH", "SZ", "BJ"}:
            return f"{parts[1].zfill(6)}.{parts[0]}"
        if parts[1] in {"SH", "SZ", "BJ"}:
            return f"{parts[0].zfill(6)}.{parts[1]}"
    for prefix in ("SH", "SZ", "BJ"):
        if code.startswith(prefix) and len(code) == 8:
            return f"{code[2:]}.{prefix}"
    digits = "".join(ch for ch in code if ch.isdigit()).zfill(6)
    if digits.startswith(("6", "9")):
        if digits.startswith("92"):
            return f"{digits}.BJ"
        return f"{digits}.SH"
    if digits.startswith(("0", "2", "3")):
        return f"{digits}.SZ"
    if digits.startswith(("4", "8")):
        return f"{digits}.BJ"
    return f"{digits}.SH"


def ts_code_to_symbol(ts_code: str) -> str:
    return normalize_ts_code(ts_code).split(".")[0]


def ts_code_to_baostock(ts_code: str) -> str:
    norm = normalize_ts_code(ts_code)
    symbol, exch = norm.split(".")
    return f"{exch.lower()}.{symbol}"


def ts_code_to_ak_prefix(ts_code: str) -> str:
    norm = normalize_ts_code(ts_code)
    symbol, exch = norm.split(".")
    return f"{exch.lower()}{symbol}"


def detect_board(ts_code: str) -> BoardType:
    norm = normalize_ts_code(ts_code)
    symbol, exch = norm.split(".")
    if exch == "BJ" or symbol.startswith(("4", "8", "92")):
        return "BSE"
    if symbol.startswith(("688", "689")):
        return "STAR"
    if symbol.startswith(("300", "301")):
        return "CHINEXT"
    return "MAIN"


def get_price_limit_pct(ts_code: str, is_st: bool = False) -> float:
    """Return A-share daily price limit ratio according to board & ST status."""
    board = detect_board(ts_code)
    if board == "BSE":
        return 0.30
    if board in ("CHINEXT", "STAR"):
        return 0.20
    return 0.05 if is_st else 0.10


def round_price(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def calc_limit_prices(pre_close: float, ts_code: str, is_st: bool = False) -> tuple[float, float]:
    """Calculate exact exchange rounded limit-up and limit-down prices."""
    if pre_close is None or pre_close <= 0:
        return (0.0, 0.0)
    pct = Decimal(str(get_price_limit_pct(ts_code, is_st=is_st)))
    base = Decimal(str(round(float(pre_close), 4)))
    cent = Decimal("0.01")
    up = (base * (Decimal(1) + pct)).quantize(cent, rounding=ROUND_HALF_UP)
    down = (base * (Decimal(1) - pct)).quantize(cent, rounding=ROUND_HALF_UP)
    return (float(up), float(down))
