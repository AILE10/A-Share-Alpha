"""WeCom (企业微信) and DingTalk (钉钉) Markdown webhook notifiers for Assisted Signal orders."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
import urllib.parse

import httpx

from app.broker.base import OrderRequest
from app.core.calendar import now_cn
from app.core.logging import get_logger

logger = get_logger(__name__)


def format_trade_ticket_markdown(req: OrderRequest, order_id: str) -> str:
    action_cn = "买入 (BUY)" if req.side == "BUY" else "卖出 (SELL)"
    factors_lines = "\n".join(
        f"- {f.get('label', f.get('name'))}: `{f.get('value', 0):.3f}` (SHAP `{f.get('shap', 0):+.4f}`)"
        for f in (req.top_factors[:3] if req.top_factors else [])
    ) or "- 定期调仓 / 风控止损触发"
    tech_str = "、".join(req.tech_patterns) if req.tech_patterns else "无特殊形态"
    stop_str = f"{req.stop_loss_price:.2f}" if req.stop_loss_price > 0 else "-"
    tp_str = f"{req.take_profit_price:.2f}" if req.take_profit_price > 0 else "-"
    return (
        f"### A-Share Alpha Studio 辅助交易单\n"
        f"- **单号**: `{order_id}`\n"
        f"- **方向**: **{action_cn}**\n"
        f"- **标的**: `{req.ts_code}` {req.name} ({req.sw_l1_name})\n"
        f"- **建议数量**: `{req.qty}` 股 (目标仓位 `{req.suggested_weight:.1%}`)\n"
        f"- **参考价**: `¥{req.ref_price:.2f}` | **止损参考**: `¥{stop_str}` | **止盈参考**: `¥{tp_str}`\n"
        f"- **技术共振**: {tech_str}\n"
        f"- **Top 因子归因**:\n{factors_lines}\n"
        f"- **触发原因**: {req.reason_summary or '模型调仓'}\n"
        f"- **生成时间**: {now_cn().strftime('%Y-%m-%d %H:%M:%S')} (请在券商终端人工核对后下单并回写成交)"
    )


def _sign_dingtalk(url: str, secret: str) -> str:
    if not secret:
        return url
    ts = str(round(time.time() * 1000))
    string_to_sign = f"{ts}\n{secret}"
    hmac_code = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}timestamp={ts}&sign={sign}"


def send_wecom_markdown(webhook_url: str, content: str, timeout: float = 8.0) -> dict:
    if not webhook_url:
        return {"channel": "wecom", "sent": False, "reason": "未配置企业微信 Webhook"}
    try:
        resp = httpx.post(webhook_url, json={"msgtype": "markdown", "markdown": {"content": content}}, timeout=timeout)
        ok = resp.status_code == 200 and resp.json().get("errcode", 0) == 0
        return {"channel": "wecom", "sent": ok, "status_code": resp.status_code, "body": resp.text[:200]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("WeCom webhook send failed: %s", exc)
        return {"channel": "wecom", "sent": False, "error": str(exc)}


def send_dingtalk_markdown(webhook_url: str, title: str, content: str, secret: str = "", timeout: float = 8.0) -> dict:
    if not webhook_url:
        return {"channel": "dingtalk", "sent": False, "reason": "未配置钉钉 Webhook"}
    try:
        signed = _sign_dingtalk(webhook_url, secret)
        resp = httpx.post(
            signed,
            json={"msgtype": "markdown", "markdown": {"title": title, "text": content}},
            timeout=timeout,
        )
        ok = resp.status_code == 200 and resp.json().get("errcode", 0) == 0
        return {"channel": "dingtalk", "sent": ok, "status_code": resp.status_code, "body": resp.text[:200]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("DingTalk webhook send failed: %s", exc)
        return {"channel": "dingtalk", "sent": False, "error": str(exc)}
