"""Unified thread-safe background action task manager with live WebSocket progress streaming
and polling recovery for all slow workbench operations (signals refresh, 14:40 rebalance,
intraday refresh, realtime regime, factor recompute, weekly review, model retrain, data sync)."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from app.api.ws_manager import ws_manager
from app.core.calendar import now_cn
from app.core.logging import get_logger
from app.factors.service import to_jsonable

logger = get_logger(__name__)

ProgressEmitter = Callable[[str, float], None]


@dataclass
class ActionTask:
    task_id: str
    kind: str
    title: str
    status: str = "RUNNING"  # RUNNING | DONE | FAILED
    stage: str = "任务已提交，正在初始化..."
    pct: float = 0.02
    started_mono: float = field(default_factory=time.monotonic)
    created_at: str = field(default_factory=lambda: now_cn().isoformat())
    finished_at: str | None = None
    elapsed_sec: float = 0.0
    result: Any = None
    error: str = ""

    def snapshot(self, include_result: bool = True) -> dict:
        elapsed = self.elapsed_sec if self.status != "RUNNING" else round(time.monotonic() - self.started_mono, 2)
        payload = {
            "task_id": self.task_id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "stage": self.stage,
            "pct": round(max(0.0, min(1.0, float(self.pct))), 4),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "elapsed_sec": elapsed,
            "error": self.error,
        }
        if include_result and self.status == "DONE":
            payload["result"] = self.result
        return to_jsonable(payload)


class BackgroundTaskManager:
    """Manages non-blocking workbench tasks with deduplication, WS push & polling recovery."""

    def __init__(self, max_history: int = 40) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, ActionTask] = {}
        self._max_history = max_history

    def _prune_locked(self) -> None:
        if len(self._tasks) <= self._max_history:
            return
        finished = [
            (tid, t.started_mono)
            for tid, t in self._tasks.items()
            if t.status != "RUNNING"
        ]
        finished.sort(key=lambda x: x[1])
        excess = len(self._tasks) - self._max_history
        for tid, _ in finished[:excess]:
            self._tasks.pop(tid, None)

    def _broadcast(self, task: ActionTask, include_result: bool = False) -> None:
        ws_manager.broadcast_threadsafe("action_task", task.snapshot(include_result=include_result))

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.snapshot(include_result=True) if task is not None else None

    def list_tasks(self, kind: str | None = None, active_only: bool = False) -> list[dict]:
        with self._lock:
            items = list(self._tasks.values())
        if kind:
            items = [t for t in items if t.kind == kind]
        if active_only:
            items = [t for t in items if t.status == "RUNNING"]
        items.sort(key=lambda t: t.started_mono, reverse=True)
        return [t.snapshot(include_result=False) for t in items[:20]]

    def submit(
        self,
        kind: str,
        title: str,
        fn: Callable[[ProgressEmitter], Any],
        dedupe_running: bool = True,
    ) -> dict:
        """Start `fn(emit)` in a daemon thread. If an identical `kind` is already RUNNING
        and `dedupe_running=True`, attach to and return the active task cleanly."""
        with self._lock:
            if dedupe_running:
                for existing in self._tasks.values():
                    if existing.kind == kind and existing.status == "RUNNING":
                        return existing.snapshot(include_result=False)
            task_id = f"tsk_{kind}_{uuid.uuid4().hex[:8]}"
            task = ActionTask(task_id=task_id, kind=kind, title=title)
            self._tasks[task_id] = task
            self._prune_locked()
            init_snap = task.snapshot(include_result=False)

        self._broadcast(task, include_result=False)

        def _emit(stage: str, pct: float) -> None:
            with self._lock:
                task.stage = stage
                task.pct = max(0.0, min(0.99, float(pct)))
                task.elapsed_sec = round(time.monotonic() - task.started_mono, 2)
            self._broadcast(task, include_result=False)

        def _runner() -> None:
            try:
                res = fn(_emit)
                with self._lock:
                    task.status = "DONE"
                    task.stage = "执行完成"
                    task.pct = 1.0
                    task.elapsed_sec = round(time.monotonic() - task.started_mono, 2)
                    task.finished_at = now_cn().isoformat()
                    task.result = to_jsonable(res)
                self._broadcast(task, include_result=False)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Background action task %s (%s) failed", task.task_id, kind)
                with self._lock:
                    task.status = "FAILED"
                    task.stage = f"执行失败: {exc}"
                    task.error = str(exc)
                    task.elapsed_sec = round(time.monotonic() - task.started_mono, 2)
                    task.finished_at = now_cn().isoformat()
                self._broadcast(task, include_result=False)

        threading.Thread(target=_runner, daemon=True, name=f"action_{kind}").start()
        return init_snap


action_task_manager = BackgroundTaskManager()
