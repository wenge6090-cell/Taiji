"""
GoalCoroutineManager — asyncio-based worker pool for goal-driven loops.

Each worker atomically claims a goal entry task from the shared pending
queue, then pushes that goal through the full sixiang loop until completion.
Prior to execution, the worker checks the goal's ``meta.json`` control
plane fields (paused, warnings, anqu request) and reacts accordingly.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Awaitable, Callable

from loguru import logger

from vingobot.core.goal_meta import read_goal_meta, update_goal_meta
from vingobot.core.pending_queue import PendingQueue


class WorkerPool:
    """Pool of asyncio workers that drive goals through sixiang loops concurrently.

    Each worker:
    1. Atomically claims the next pending task for a goal.
    2. Marks the goal as active (non-locking, for monitoring).
    3. Calls ``execute_complete_sixiang_loop(goal_id, description, signal)``.
    4. Finalises (persists trajectory, cleans up .processing file).
    5. Returns to the queue.
    """

    def __init__(
        self,
        *,
        max_workers: int = 3,
        poll_interval_ms: int = 500,
        run_task_fn: Callable[[str, str, asyncio.Task | None], Awaitable[object]],
    ) -> None:
        """
        Args:
            max_workers: Maximum concurrent goal-driving workers.
            poll_interval_ms: Idle sleep between queue polls (ms).
            run_task_fn: ``async fn(goal_id, description, signal) -> GoalResult``.
        """
        self._max_workers = max(max_workers, 1)
        self._poll_interval = max(poll_interval_ms, 100) / 1000.0
        self._run_task_fn = run_task_fn
        self._running = False
        self._workers: list[asyncio.Task[None]] = []
        self._active_goals: dict[str, int] = {}  # goal_id → worker_id
        self._stopped = asyncio.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def active_count(self) -> int:
        return len([w for w in self._workers if not w.done()])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch all workers and begin processing pending tasks."""
        if self._running:
            return
        self._running = True
        self._stopped.clear()
        logger.info("[协程池] 启动 {} 个 worker", self._max_workers)
        for i in range(self._max_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(task)

    async def stop(self) -> None:
        """Gracefully shut down all workers, cancelling remaining tasks."""
        if not self._running:
            return
        self._running = False
        for t in self._workers:
            if not t.done():
                t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._active_goals.clear()
        self._stopped.set()
        logger.info("[协程池] 已停止")

    async def wait_stopped(self) -> None:
        """Block until ``stop()`` is called."""
        await self._stopped.wait()

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self, worker_id: int) -> None:
        queue = PendingQueue()

        while self._running:
            claimed = queue.try_consume_next(
                exclude_prefixes=["cognition-evolution", "dmn"],
            )
            if claimed is None:
                try:
                    await asyncio.sleep(self._poll_interval)
                except asyncio.CancelledError:
                    break
                continue

            task, file_path = claimed

            # ── Control plane: check meta.json before consuming ──────
            meta = read_goal_meta(task.goal_id)
            if meta is None:
                logger.warning(
                    "[协程{}] 目标 {} meta.json 不存在，丢弃任务", worker_id, task.goal_id
                )
                queue.delete_task_file(file_path)
                continue

            if meta.status == "paused":
                logger.info("[协程{}] 目标 {} 已暂停，任务放回队列", worker_id, task.goal_id)
                queue.enqueue(task)
                queue.delete_task_file(file_path)
                continue

            if meta.warnings:
                logger.warning(
                    "[协程{}] 目标 {} 有异常警告: {}，丢弃任务等待人工处理",
                    worker_id,
                    task.goal_id,
                    meta.warnings,
                )
                queue.delete_task_file(file_path)
                continue

            if meta.last_anqu_at == "":
                now = datetime.now(timezone.utc).isoformat()
                update_goal_meta(task.goal_id, last_anqu_at=now)
                logger.info(
                    "[协程{}] 目标 {} 暗驱信号已确认 (last_anqu_at={})",
                    worker_id,
                    task.goal_id,
                    now,
                )

            # ── Execute ──────────────────────────────────────────────
            # Mark goal as active for monitoring
            self._active_goals[task.goal_id] = worker_id
            try:
                await update_goal_meta(task.goal_id, last_active="")
            except Exception:
                pass

            try:
                await self._run_task_fn(task.goal_id, task.description, asyncio.current_task())
            except asyncio.CancelledError:
                logger.info("[协程{}] 任务被中断 (goal={})", worker_id, task.goal_id)
            except Exception:
                logger.exception(
                    "[协程{}] 任务失败 (goal={}, desc={})",
                    worker_id,
                    task.goal_id,
                    task.description[:80],
                )
            finally:
                # Cleanup
                queue.delete_task_file(file_path)
                self._active_goals.pop(task.goal_id, None)

        logger.info("[协程{}] worker 已退出", worker_id)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def is_goal_active(self, goal_id: str) -> bool:
        return goal_id in self._active_goals

    def get_active_goals(self) -> dict[str, int]:
        return dict(self._active_goals)
