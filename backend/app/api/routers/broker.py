"""Broker Gateway (Paper / Assisted / Compliant QMT) & Pre-Trade RiskMiddleware endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.ws_manager import ws_manager
from app.broker.base import OrderRequest
from app.broker.gateway import BrokerGateway
from app.core.constants import normalize_ts_code
from app.db.duckdb_client import get_duckdb
from app.db.models import PaperAccount
from app.db.sqlite_client import get_sqlite_db
from app.factors.service import to_jsonable

router = APIRouter(tags=["Broker & Risk Control"])


class BrokerConfigUpdate(BaseModel):
    active_mode: str | None = Field(default=None, pattern="^(PAPER|ASSISTED|QMT)$")
    wecom_webhook_url: str | None = None
    dingtalk_webhook_url: str | None = None
    dingtalk_secret: str | None = None
    qmt_userdata_path: str | None = None
    qmt_account_id: str | None = None
    qmt_dry_run: bool | None = None


class RiskConfigUpdate(BaseModel):
    max_single_weight: float | None = Field(default=None, ge=0.02, le=0.50)
    max_industry_weight: float | None = Field(default=None, ge=0.05, le=0.80)
    stock_hard_stop_pct: float | None = Field(default=None, ge=0.02, le=0.30)
    stock_atr_mult: float | None = Field(default=None, ge=0.5, le=6.0)
    drawdown_warn_pct: float | None = Field(default=None, ge=0.02, le=0.30)
    drawdown_halt_pct: float | None = Field(default=None, ge=0.05, le=0.40)
    max_orders_per_min: int | None = Field(default=None, ge=1, le=200)
    max_orders_per_day: int | None = Field(default=None, ge=1, le=2000)


class ConfirmFillRequest(BaseModel):
    filled_qty: int = Field(gt=0)
    avg_fill_price: float = Field(gt=0)
    commission: float | None = None
    stamp_duty: float | None = None
    transfer_fee: float | None = None


class ManualOrderRequest(BaseModel):
    ts_code: str
    side: str = Field(pattern="^(BUY|SELL)$")
    qty: int = Field(gt=0)
    ref_price: float | None = None
    reason_summary: str = "人工通过工作台下单"


def _gateway() -> BrokerGateway:
    return BrokerGateway(
        get_sqlite_db(),
        on_stage_order=lambda payload: ws_manager.broadcast_threadsafe("assisted_order", payload),
    )


@router.get("/broker/config")
def get_broker_config() -> dict:
    return to_jsonable(_gateway().get_config())


@router.put("/broker/config")
def update_broker_config(req: BrokerConfigUpdate) -> dict:
    return to_jsonable(_gateway().update_config(req.model_dump(exclude_none=True)))


@router.post("/broker/webhook/test")
def test_broker_webhooks() -> dict:
    return to_jsonable({"results": _gateway().test_webhooks()})


@router.get("/broker/assisted/pending")
def list_assisted_pending() -> list[dict]:
    return to_jsonable(_gateway().assisted.list_pending())


@router.post("/broker/assisted/{order_id}/confirm-fill")
def confirm_assisted_fill(order_id: str, req: ConfirmFillRequest) -> dict:
    gw = _gateway()
    try:
        res = gw.assisted.confirm_manual_fill(
            order_id=order_id,
            filled_qty=req.filled_qty,
            avg_fill_price=req.avg_fill_price,
            commission=req.commission,
            stamp_duty=req.stamp_duty,
            transfer_fee=req.transfer_fee,
        )
        ws_manager.broadcast_threadsafe("paper_nav", gw.paper.get_account_state().get("account", {}))
        return to_jsonable(res)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/broker/assisted/{order_id}/cancel")
def cancel_assisted_order(order_id: str) -> dict:
    return to_jsonable(_gateway().assisted.cancel_order(order_id))


@router.post("/broker/manual-order")
def submit_manual_order(req: ManualOrderRequest) -> dict:
    """Route a manual workbench order through RiskMiddleware -> active BrokerAdapter."""
    code = normalize_ts_code(req.ts_code)
    db = get_duckdb()
    basic = db.query_df("SELECT name, sw_l1_name FROM stock_basic WHERE ts_code = ?", [code])
    name = str(basic.iloc[0]["name"]) if not basic.empty else code
    ind = str(basic.iloc[0]["sw_l1_name"]) if not basic.empty else "未分类"
    last_bar = db.query_df(
        "SELECT open, high, low, close, pre_close, volume, is_st, is_suspended, limit_up, limit_down "
        "FROM daily_bars WHERE ts_code = ? ORDER BY trade_date DESC LIMIT 1",
        [code],
    )
    if req.ref_price is not None and req.ref_price > 0:
        px = float(req.ref_price)
        pre_c = float(last_bar.iloc[0]["pre_close"]) if not last_bar.empty else px
    elif not last_bar.empty:
        px = float(last_bar.iloc[0]["close"])
        pre_c = float(last_bar.iloc[0]["pre_close"])
    else:
        raise HTTPException(status_code=400, detail=f"找不到 {code} 行情价格，请指定 ref_price")

    order_req = OrderRequest(
        account_id="default_paper",
        ts_code=code,
        name=name,
        sw_l1_name=ind,
        side=req.side,  # type: ignore[arg-type]
        qty=req.qty,
        ref_price=px,
        pre_close=pre_c,
        ask1=px,
        bid1=px,
        reason_summary=req.reason_summary,
    )
    receipt = _gateway().submit_order(order_req)
    return to_jsonable(receipt.to_dict())


@router.get("/risk/config")
def get_risk_config() -> dict:
    return to_jsonable(_gateway().risk.get_config())


@router.put("/risk/config")
def update_risk_config(req: RiskConfigUpdate) -> dict:
    return to_jsonable(_gateway().risk.update_config(req.model_dump(exclude_none=True)))


@router.post("/risk/fuse-reset")
def reset_drawdown_fuse() -> dict:
    sqlite = get_sqlite_db()
    with sqlite.session_scope() as session:
        acct = session.get(PaperAccount, "default_paper")
        if acct is None:
            raise HTTPException(status_code=404, detail="default_paper account not found")
        acct.fuse_status = "NORMAL"
        acct.peak_equity = max(acct.peak_equity, acct.total_equity)
        acct.current_drawdown = 0.0
        session.flush()
        return {"ok": True, "fuse_status": acct.fuse_status, "total_equity": acct.total_equity}


@router.get("/risk/audit-logs")
def get_risk_audit_logs(limit: int = 100) -> list[dict]:
    return to_jsonable(_gateway().risk.get_audit_logs(limit=limit))
