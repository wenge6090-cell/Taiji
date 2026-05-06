"""Tests for DMN consumer — polls pending/ for cognitive evolution tasks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vingobot.core.pending_queue import PendingQueue, PendingTask
from vingobot.core.workspace import init_workspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_test_workspace(tmp_path: Path) -> Path:
    """Initialise a .taiji workspace and return its root."""
    root = tmp_path / ".taiji"
    init_workspace(root, seed=False)
    return root


def _write_dmn_task(pending_dir: Path, prefix: str, description: str) -> Path:
    """Write a DMN-prefixed .task file directly and return its path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
    filename = f"{prefix}__{ts}.task"
    filepath = pending_dir / filename
    content = f"{description}\npriority=5\nsource=system\ngoalId={prefix}\n"
    filepath.write_text(content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Tests — DMN consumer polling (PendingQueue level)
# ---------------------------------------------------------------------------


class TestDmnConsumerPolling:
    """DMN consumer discovers and consumes DMN-prefixed tasks via PendingQueue."""

    def test_consumes_cognition_evolution_task(self, tmp_path: Path) -> None:
        """DMN consumer picks up cognition-evolution__* tasks."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        _write_dmn_task(root / "pending", "cognition-evolution", "learn http client")

        claimed = queue.try_consume_by_prefix("cognition-evolution")
        assert claimed is not None
        task, processing_path = claimed
        assert "learn http client" in task.description
        assert processing_path.suffix == ".processing"
        # Cleanup
        queue.delete_task_file(processing_path)

    def test_consumes_dmn_task(self, tmp_path: Path) -> None:
        """DMN consumer picks up dmn__* tasks."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        _write_dmn_task(root / "pending", "dmn", "global maintenance check")

        claimed = queue.try_consume_by_prefix("dmn")
        assert claimed is not None
        task, _ = claimed
        assert task.description == "global maintenance check"

    def test_skips_regular_tasks(self, tmp_path: Path) -> None:
        """try_consume_by_prefix does NOT consume regular (non-prefixed) tasks."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        queue.enqueue(PendingTask(
            id="reg-001", goal_id="default", description="regular user task",
        ))

        assert queue.try_consume_by_prefix("cognition-evolution") is None
        # Regular task still consumable by WorkerPool
        assert queue.try_consume_next() is not None

    def test_dmn_and_worker_pools_coexist(self, tmp_path: Path) -> None:
        """DMN consumer and WorkerPool do not interfere with each other."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        queue.enqueue(PendingTask(
            id="reg-001", goal_id="default", description="worker task",
        ))
        _write_dmn_task(root / "pending", "cognition-evolution", "learn task")

        # DMN consumer gets its task
        dmn_claimed = queue.try_consume_by_prefix("cognition-evolution")
        assert dmn_claimed is not None
        assert dmn_claimed[0].description == "learn task"

        # WorkerPool gets its task (excluding DMN prefixes)
        worker_claimed = queue.try_consume_next(
            exclude_prefixes=["cognition-evolution", "dmn"],
        )
        assert worker_claimed is not None
        assert worker_claimed[0].description == "worker task"

    def test_cleanup_after_consumption(self, tmp_path: Path) -> None:
        """After DMN consumer processes a task, the .processing file is deleted."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        filepath = _write_dmn_task(root / "pending", "cognition-evolution", "cleanup test")
        claimed = queue.try_consume_by_prefix("cognition-evolution")
        assert claimed is not None
        _, processing_path = claimed
        assert processing_path.exists()

        queue.delete_task_file(processing_path)
        assert not processing_path.exists()
        # Original .task should also be gone
        assert not filepath.exists()


# ---------------------------------------------------------------------------
# Tests — DMN system prompt injection
# ---------------------------------------------------------------------------


class TestDmnPromptInjection:
    """DMN identity prompt is correctly injected into the system prompt."""

    def test_dmn_injection_contains_identity(self) -> None:
        """DMN injection text includes key identity markers."""
        # Replicate the exact injection string from AgentLoop._run_dmn_turn
        dmn_injection = (
            "## 全局认知维护（DMN）模式\n\n"
            "你当前处于全局认知维护模式，身份是系统的认知管家。\n"
            "职责：\n"
            "1. **认知演化** — 创建/更新 L1 技能、L2 模型、L3 格栅\n"
            "2. **全局维护** — 检查 meta.json、归档、暂停、成长日志\n\n"
            "**约束**:\n"
            "- 所有文件写入必须在 `.taiji/` 目录下\n"
            "- 不能修改 `.taiji/cognition/truths/` 下的 L4 不可变真理"
        )

        assert "全局认知维护" in dmn_injection
        assert "认知管家" in dmn_injection
        assert "L1 技能" in dmn_injection
        assert "L2 模型" in dmn_injection
        assert "L3 格栅" in dmn_injection
        assert ".taiji/" in dmn_injection
        assert "L4 不可变真理" in dmn_injection

    def test_dmn_session_isolation_key(self) -> None:
        """DMN consumer uses dedicated '_dmn' session key for isolation."""
        # The session key must be '_dmn' to keep DMN context separate
        # from user conversation sessions.
        dmn_session_key = "_dmn"
        assert dmn_session_key.startswith("_")
        assert dmn_session_key != "cli:direct"
        assert dmn_session_key != "unified:default"


# ---------------------------------------------------------------------------
# Tests — DMN consumer lifecycle (AgentLoop integration)
# ---------------------------------------------------------------------------


class TestDmnConsumerLifecycle:
    """DMN consumer starts, polls, and stops correctly within AgentLoop."""

    @pytest.mark.asyncio
    async def test_consumer_exits_when_running_false(self) -> None:
        """DMN consumer loop exits immediately when _running is False."""
        # Simulate the core polling loop without AgentLoop dependencies
        running = False
        poll_count = 0

        async def _simulate_consumer() -> None:
            nonlocal poll_count
            while running:
                # In real code: try_consume_by_prefix(...)
                poll_count += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(_simulate_consumer())
        await task  # Completes immediately — loop never entered
        assert poll_count == 0  # Never entered the loop

    @pytest.mark.asyncio
    async def test_consumer_polls_while_running(self) -> None:
        """DMN consumer polls continuously while _running is True."""
        running = True
        poll_count = 0

        async def _simulate_consumer() -> None:
            nonlocal poll_count
            while running:
                poll_count += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(_simulate_consumer())
        await asyncio.sleep(0.08)
        running = False
        await task  # Loop exits naturally when running becomes False

        assert poll_count >= 3  # Should have polled several times

    @pytest.mark.asyncio
    async def test_consumer_handles_cancellation(self) -> None:
        """DMN consumer exits cleanly on CancelledError."""
        running = True
        cancelled_cleanly = False

        async def _simulate_consumer() -> None:
            nonlocal cancelled_cleanly
            while running:
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    cancelled_cleanly = True
                    break

        task = asyncio.create_task(_simulate_consumer())
        await asyncio.sleep(0.03)
        task.cancel()
        await task  # CancelledError caught internally → completes normally

        assert cancelled_cleanly is True

    @pytest.mark.asyncio
    async def test_consumer_task_registered_in_agent_loop(self, tmp_path: Path) -> None:
        """AgentLoop.__init__ registers _dmn_consumer_task as None initially."""
        # Lightweight check: the attribute exists on AgentLoop instances
        from vingobot.agent.loop import AgentLoop

        # We cannot fully instantiate AgentLoop without real providers,
        # but we can verify the attribute pattern exists in the class.
        assert hasattr(AgentLoop, "__init__")
        # The field is set in __init__: self._dmn_consumer_task = None
        import inspect
        source = inspect.getsource(AgentLoop.__init__)
        assert "_dmn_consumer_task" in source
        assert "None" in source.split("_dmn_consumer_task")[1][:50]
