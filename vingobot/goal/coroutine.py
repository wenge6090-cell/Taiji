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

from vingobot.core.goal_meta import read_goal_meta, scan_active_goals, update_goal_meta
from vingobot.core.pending_queue import PendingQueue, PendingTask


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
        on_task_complete: Callable[[str, bool, str], Awaitable[None]] | None = None,
    ) -> None:
        """
        Args:
            max_workers: Maximum concurrent goal-driving workers.
            poll_interval_ms: Idle sleep between queue polls (ms).
            run_task_fn: ``async fn(goal_id, description, signal) -> GoalResult``.
            on_task_complete: Optional callback ``async fn(goal_id, success, summary)``
                invoked after each sixiang loop completes.
        """
        self._max_workers = max(max_workers, 1)
        self._poll_interval = max(poll_interval_ms, 100) / 1000.0
        self._run_task_fn = run_task_fn
        self._on_task_complete = on_task_complete
        self._running = False
        self._workers: list[asyncio.Task[None]] = []
        self._active_goals: dict[str, int] = {}  # goal_id → worker_id
        self._stopped = asyncio.Event()
        self._scanner_task: asyncio.Task[None] | None = None
        self._scanner_interval = 30  # seconds between self-driven scans

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

        # ── Recover orphaned .processing files from previous crashes ──
        try:
            recovered = PendingQueue.cleanup_orphan_tasks(timeout_ms=0)  # 0 = recover all immediately
            if recovered > 0:
                logger.info("[协程池] 启动时恢复 {} 个孤儿任务", recovered)
        except Exception:
            logger.warning("[协程池] 孤儿任务清理失败，继续启动")

        # ── Launch self-driven scanner ────────────────────────────
        self._scanner_task = asyncio.create_task(self._self_driven_scanner())

        logger.info("[协程池] 启动 {} 个 worker", self._max_workers)
        for i in range(self._max_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(task)

    async def stop(self) -> None:
        """Gracefully shut down all workers, cancelling remaining tasks."""
        if not self._running:
            return
        self._running = False
        # Cancel scanner first
        if self._scanner_task and not self._scanner_task.done():
            self._scanner_task.cancel()
        for t in self._workers:
            if not t.done():
                t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        if self._scanner_task:
            try:
                await self._scanner_task
            except asyncio.CancelledError:
                pass
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
                result = await self._run_task_fn(task.goal_id, task.description, asyncio.current_task())
                # ── TPN→DMN feedback ──────────────────────
                if self._on_task_complete is not None:
                    success = (
                        getattr(result, "status", "") == "completed"
                        if result is not None else False
                    )
                    try:
                        await self._on_task_complete(
                            task.goal_id, success, task.description[:120],
                        )
                    except Exception:
                        logger.debug("[协程{}] on_task_complete 回调异常", worker_id)
            except asyncio.CancelledError:
                logger.info("[协程{}] 任务被中断 (goal={})", worker_id, task.goal_id)
                if self._on_task_complete is not None:
                    try:
                        await self._on_task_complete(
                            task.goal_id, False, f"cancelled: {task.description[:100]}",
                        )
                    except Exception:
                        pass
            except Exception:
                logger.exception(
                    "[协程{}] 任务失败 (goal={}, desc={})",
                    worker_id,
                    task.goal_id,
                    task.description[:80],
                )
                if self._on_task_complete is not None:
                    try:
                        await self._on_task_complete(
                            task.goal_id, False, f"error: {task.description[:100]}",
                        )
                    except Exception:
                        pass
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

    # ------------------------------------------------------------------
    # Self-driven scanner
    # ------------------------------------------------------------------

    async def _self_driven_scanner(self) -> None:
        """Periodically scan self-driven goals and enqueue tasks.

        Wakes every ``_scanner_interval`` seconds, iterates all active
        goals with ``self_driven.enabled=True``, and enqueues a task when
        the elapsed time since ``last_active`` exceeds the configured
        interval.
        """
        while self._running:
            try:
                await self._scan_and_enqueue_self_driven()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[自驱动扫描] 扫描周期异常")

            try:
                await asyncio.sleep(self._scanner_interval)
            except asyncio.CancelledError:
                break

        logger.info("[自驱动扫描] 已退出")

    async def _scan_and_enqueue_self_driven(self) -> None:
        """One scan cycle: check all self-driven goals."""
        queue = PendingQueue()
        now = datetime.now(timezone.utc)

        try:
            active_goals = scan_active_goals()
        except Exception:
            logger.warning("[自驱动扫描] 获取活跃目标失败")
            return

        for meta in active_goals:
            sd = meta.self_driven
            if not sd.enabled:
                continue

            # Skip if already being processed
            if meta.id in self._active_goals:
                continue

            # Skip if there's already a pending task for this goal
            if any(t.goal_id == meta.id for t in queue.scan_pending()):
                continue

            # Check elapsed time
            if meta.last_active:
                try:
                    last = datetime.fromisoformat(meta.last_active)
                    elapsed_min = (now - last).total_seconds() / 60.0
                except (ValueError, TypeError):
                    elapsed_min = float("inf")
            else:
                elapsed_min = float("inf")  # never active → should trigger

            if elapsed_min < sd.interval_minutes:
                continue

            # ── Enqueue self-driven task ─────────────────────────
            task = PendingTask(
                goal_id=meta.id,
                description=f"自驱动触发：继续推进目标 '{meta.name or meta.id}'",
                source="self_driven",
            )
            fp = queue.enqueue(task)
            logger.info(
                "[自驱动扫描] 入队目标 '{}' (上次活跃: {:.0f}分钟前, 间隔: {}分钟) → {}",
                meta.id,
                elapsed_min,
                sd.interval_minutes,
                fp,
            )
            # Update last_active so we don't re-enqueue on next scan
            try:
                update_goal_meta(meta.id, last_active=now.isoformat())
            except Exception:
                pass
