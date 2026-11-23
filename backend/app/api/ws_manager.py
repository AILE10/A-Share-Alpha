"""FastAPI WebSocket connection hub for real-time quant stream pushing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket

from app.core.calendar import now_cn
from app.core.logging import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """Thread-safe async WebSocket broadcaster supporting sync background worker triggers."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        logger.info("WebSocket connected (active=%d)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)

    async def broadcast(self, channel: str, payload: Any) -> None:
        if not self._connections:
            return
        envelope = json.dumps(
            {
                "channel": channel,
                "timestamp": now_cn().isoformat(),
                "payload": payload,
            },
            ensure_ascii=False,
            default=str,
        )
        stale: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(envelope)
            except Exception:  # noqa: BLE001
                stale.append(ws)
        for ws in stale:
            self._connections.discard(ws)

    def broadcast_threadsafe(self, channel: str, payload: Any) -> None:
        """Emit an event from APScheduler or worker threads onto FastAPI's event loop."""
        if self._loop is None or not self._loop.is_running() or not self._connections:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(channel, payload), self._loop)
        except Exception as exc:  # noqa: BLE001
            logger.debug("WS threadsafe broadcast skipped: %s", exc)


ws_manager = WebSocketManager()
