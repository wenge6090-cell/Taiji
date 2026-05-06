"""
默认目标（对话模式适配）— Dialogue mode wrapped as default goal.

vingobot 原有的对话行为保持不变，但被封装为"default"目标下的六爻循环。
用户在对话模式发送的消息会自动创建 default 目标的入口任务，然后由
六爻循环引擎驱动执行。

关键差异：对话模式走完整六爻循环（明觉→编织器→阳→阴→执行器→暗驱），
但限制 max_rounds=10、max_goal_tasks=1，实现快速单任务响应。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.core.pending_queue import PendingQueue, PendingTask
from vingobot.core.workspace import ensure_goal_dir
from vingobot.goal.types import GoalResult

# ---------------------------------------------------------------------------
# Default goal creation
# ---------------------------------------------------------------------------


def ensure_default_goal(workspace_root: str | Path | None = None) -> str:
    """Create the 'default' goal if it doesn't exist.

    Returns the goal_id ('default').
    """
    goal_id = "default"
    if workspace_root:
        from vingobot.core.workspace import init_workspace

        init_workspace(workspace_root)

    ensure_goal_dir(
        goal_id,
        priority=1,
        name="默认对话",
        description="用户对话与系统自我演化",
    )
    return goal_id


# ---------------------------------------------------------------------------
# Dialogue mode — push message as pending task
# ---------------------------------------------------------------------------


def push_dialogue_task(message: str) -> str | None:
    """Push a user message as a pending task under the 'default' goal.

    This is the bridge between the traditional chat interface and the
    sixiang goal-driven loop.  The message becomes a task description
    that will be picked up by a worker in the coroutine pool.

    Returns the task file path, or None on failure.
    """
    goal_id = ensure_default_goal()

    queue = PendingQueue()
    task = PendingTask(
        goal_id=goal_id,
        description=message,
        source="dialogue",
    )

    try:
        file_path = queue.enqueue(task)
        logger.info("[对话→目标] 已加入队列: goal=default, desc={}", message[:60])
        return file_path
    except Exception:
        logger.exception("[对话→目标] 入队失败")
        return None


# ---------------------------------------------------------------------------
# Simplified sixiang loop for dialogue mode
# ---------------------------------------------------------------------------


async def run_dialogue_goal(
    message: str,
    *,
    provider: Any = None,
    tool_registry: Any = None,
    workspace_root: str | Path | None = None,
    signal: asyncio.Task | None = None,
) -> GoalResult:
    """Run a single-turn dialogue through the sixiang loop (simplified).

    In dialogue mode, each user message is one task.  There is no
    multi-round iteration — it's a single Yang→Executor→response cycle.
    """
    goal_id = ensure_default_goal(workspace_root)

    # Set provider for sixiang modules
    if provider is not None:
        _inject_provider(provider)

    # Full sixiang loop
    from vingobot.goal.sixiang_loop import execute_complete_sixiang_loop

    result = await execute_complete_sixiang_loop(
        goal_id=goal_id,
        initial_description=message,
        signal=signal,
        max_rounds=10,  # Dialogue mode: fewer rounds
        max_goal_tasks=1,  # Single task only
        max_rework_rounds=1,
    )

    return result


# ---------------------------------------------------------------------------
# Batch goal creation from CLI
# ---------------------------------------------------------------------------


async def run_goal(
    goal_id: str,
    description: str,
    *,
    provider: Any = None,
    workspace_root: str | Path | None = None,
    signal: asyncio.Task | None = None,
) -> GoalResult:
    """Run a single goal through the complete sixiang loop.

    This is the entry point for::

        vingobot sixiang run "write a hello world program"
    """
    ensure_goal_dir(
        goal_id,
        description=description,
    )

    if provider is not None:
        _inject_provider(provider)

    from vingobot.goal.sixiang_loop import execute_complete_sixiang_loop

    return await execute_complete_sixiang_loop(
        goal_id=goal_id,
        initial_description=description,
        signal=signal,
        max_rounds=30,
        max_goal_tasks=100,
        max_rework_rounds=3,
    )


# ---------------------------------------------------------------------------
# Worker pool lifecycle helpers
# ---------------------------------------------------------------------------


async def start_sixiang_workers(
    *,
    max_workers: int = 3,
    provider: Any = None,
    signal: asyncio.Task | None = None,
) -> Any:
    """Start the sixiang coroutine worker pool.

    Blocks until ``pool.stop()`` is called — callers that need non-blocking
    behaviour (e.g. programmatic API) should wrap this in
    ``asyncio.create_task()``.

    Returns the ``WorkerPool`` instance (after stop).
    """
    from vingobot.goal.coroutine import WorkerPool
    from vingobot.goal.sixiang_loop import execute_complete_sixiang_loop

    if provider is not None:
        _inject_provider(provider)

    pool = WorkerPool(
        max_workers=max_workers,
        run_task_fn=execute_complete_sixiang_loop,
    )

    await pool.start()
    await pool.wait_stopped()
    return pool


# ---------------------------------------------------------------------------
# Provider injection helper
# ---------------------------------------------------------------------------


def _inject_provider(provider: Any) -> None:
    """Inject the provider into all sixiang modules.

    Each module gets its own ``set_provider()`` call so the
    provider is available for lazy ``_get_provider()`` consumers.
    """
    _ALL_MODULES = ["mingjue", "weaver", "yang", "yin", "anqu"]
    for mod_name in _ALL_MODULES:
        try:
            mod = __import__(f"vingobot.goal.{mod_name}", fromlist=["set_provider"])
            if hasattr(mod, "set_provider"):
                mod.set_provider(provider)
        except Exception:
            pass


def _inject_sixiang_providers(config: Any) -> None:
    """Inject per-agent providers into all sixiang modules from config.

    Creates an independent LLM provider for each sixiang agent that has
    a per-agent override in the config, and injects it into the
    corresponding module.  Agents without per-agent config will lazily
    load their own provider when first used.
    """
    _SIXIANG_AGENTS = ["mingjue", "weaver", "yang", "yin", "anqu"]
    from vingobot.providers.factory import build_sixiang_provider_snapshot

    for agent_name in _SIXIANG_AGENTS:
        try:
            snapshot = build_sixiang_provider_snapshot(config, agent_name)
            module = __import__(
                f"vingobot.goal.{agent_name}",
                fromlist=["set_provider"],
            )
            if hasattr(module, "set_provider"):
                module.set_provider(snapshot.provider)
        except Exception:
            from loguru import logger as _lg

            _lg.debug("[六爻] 注入 provider 失败: {}", agent_name)
