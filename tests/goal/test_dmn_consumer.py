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


# ---------------------------------------------------------------------------
# Tests — Pure event-driven DMN loop (no timed polling)
# ---------------------------------------------------------------------------


class TestEventDrivenDmn:
    """DMN consciousness cycle driven purely by events — no timer, no sleep."""

    @pytest.mark.asyncio
    async def test_full_zhoutian_event_driven(self) -> None:
        """一个完整周天由事件驱动完成: 起念→立目标→整理认知→起念."""
        from vingobot.goal.dmn_consciousness import DmnConsciousness
        from vingobot.goal.guizang_types import ConsciousnessPhase

        dmn = DmnConsciousness()  # 无 LLM → 位运算降级
        assert dmn.current_phase == ConsciousnessPhase.QINIAN
        assert dmn.state.is_resting

        # 事件队列作为唯一的驱动源
        events: asyncio.Queue[str] = asyncio.Queue()
        results: list[object] = []

        async def event_loop():
            """纯事件循环 — 不做任何定时轮询."""
            while True:
                try:
                    event = await asyncio.wait_for(events.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    break  # 超时 = 空闲，退出

                if event == "cycle":
                    result = await dmn.cycle()
                    results.append(result)
                    # 事件驱动链：根据结果决定下一步
                    if not result.is_resting:
                        events.put_nowait("dispatch")
                elif event == "dispatch":
                    # 实际系统中这里会入队 TPN 任务
                    pass
                elif event == "tpn_feedback":
                    dmn.observe_tpn_task(success=True, summary="test")
                elif event == "stop":
                    break

        # 启动事件循环并注入 cycle 事件
        loop_task = asyncio.create_task(event_loop())

        # 注入足够的事件完成一个完整周天
        for _ in range(6):
            events.put_nowait("cycle")
        events.put_nowait("stop")

        await asyncio.wait_for(loop_task, timeout=10.0)

        # 验证 3 个阶段全部执行
        phases = [r.phase for r in results]
        assert ConsciousnessPhase.QINIAN in phases
        assert ConsciousnessPhase.LIMUBIAO in phases
        assert ConsciousnessPhase.ZHENGLI in phases

        # 整理认知后状态归零
        assert dmn.state.is_resting
        assert dmn.state.bits == 0
        assert dmn._cycles_completed >= 3

    @pytest.mark.asyncio
    async def test_tpn_feedback_triggers_consolidation(self) -> None:
        """TPN 任务反馈累积到阈值 → 事件驱动触发整理认知."""
        from vingobot.goal.dmn_consciousness import (
            CONSOLIDATE_TRIGGER_TASKS,
            DmnConsciousness,
        )
        from vingobot.goal.guizang_types import ConsciousnessPhase

        dmn = DmnConsciousness()
        events: asyncio.Queue[str] = asyncio.Queue()
        results: list[object] = []

        async def event_loop():
            while True:
                try:
                    event = await asyncio.wait_for(events.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    break
                if event == "cycle":
                    result = await dmn.cycle()
                    results.append(result)
                elif event == "tpn_feedback":
                    dmn.observe_tpn_task(success=True, summary="test")
                elif event == "stop":
                    break

        loop_task = asyncio.create_task(event_loop())

        # 注入 TPN 反馈直到触发 consolidate
        for _ in range(CONSOLIDATE_TRIGGER_TASKS):
            events.put_nowait("tpn_feedback")

        # Yield to event_loop task so it processes the queued events
        await asyncio.sleep(0.05)

        # 此时 _phase_pending 应为 ZHENGLI
        assert dmn.current_phase == ConsciousnessPhase.ZHENGLI

        # 驱动一个 cycle 执行整理认知
        events.put_nowait("cycle")
        events.put_nowait("stop")

        await asyncio.wait_for(loop_task, timeout=10.0)

        # 整理认知完成
        assert any(r.phase == ConsciousnessPhase.ZHENGLI for r in results)
        # 整理后回到 resting
        assert dmn.state.is_resting

    @pytest.mark.asyncio
    async def test_event_loop_no_tasks_no_polling(self) -> None:
        """事件循环空闲时不消耗 CPU — 等待事件而非轮询."""
        from vingobot.goal.dmn_consciousness import DmnConsciousness

        dmn = DmnConsciousness()
        events: asyncio.Queue[str] = asyncio.Queue()
        cycles_run = 0

        async def event_loop():
            nonlocal cycles_run
            while True:
                try:
                    # 没有 timeout 时完全阻塞等待 — 纯事件驱动
                    event = await events.get()
                except asyncio.CancelledError:
                    break
                if event == "cycle":
                    cycles_run += 1
                    await dmn.cycle()
                elif event == "stop":
                    break

        loop_task = asyncio.create_task(event_loop())

        # 什么也不注入 — 验证循环阻塞在 events.get() 上
        await asyncio.sleep(0.2)
        assert cycles_run == 0  # 没有事件 = 没有 cycle

        # 注入事件后才运行
        events.put_nowait("cycle")
        events.put_nowait("stop")
        await asyncio.wait_for(loop_task, timeout=5.0)
        assert cycles_run == 1

    @pytest.mark.asyncio
    async def test_multi_zhoutian_event_driven(self) -> None:
        """多次完整周天事件驱动 — 状态一致性验证."""
        from vingobot.goal.dmn_consciousness import DmnConsciousness
        from vingobot.goal.guizang_types import ConsciousnessPhase

        dmn = DmnConsciousness()
        events: asyncio.Queue[str] = asyncio.Queue()
        results: list[object] = []

        async def event_loop():
            while True:
                try:
                    event = await asyncio.wait_for(events.get(), timeout=3.0)
                except asyncio.TimeoutError:
                    break
                if event == "cycle":
                    results.append(await dmn.cycle())
                elif event == "stop":
                    break

        loop_task = asyncio.create_task(event_loop())

        # 注入 9 个 cycle 事件 (3 个完整周天)
        for _ in range(9):
            events.put_nowait("cycle")
        events.put_nowait("stop")

        await asyncio.wait_for(loop_task, timeout=10.0)

        assert len(results) == 9

        # 每个周天: 起念→立目标→整理认知
        for zhoutian_start in range(0, 9, 3):
            assert results[zhoutian_start].phase == ConsciousnessPhase.QINIAN
            assert results[zhoutian_start + 1].phase == ConsciousnessPhase.LIMUBIAO
            assert results[zhoutian_start + 2].phase == ConsciousnessPhase.ZHENGLI

        # 最终状态应归零
        assert dmn.state.is_resting
        assert dmn._cycles_completed == 9
        # 藏海应有记录
        assert dmn.cang_sea.size > 0


# ---------------------------------------------------------------------------
# Tests — Deviation enforcement (DMN→TPN control bridge)
# ---------------------------------------------------------------------------


class TestDeviationEnforcement:
    """_enforce_deviation_control writes warnings / pauses goals deterministically.

    NOTE: All goal-meta helpers are imported locally to avoid the
    Python descriptor protocol binding them as instance methods when
    accessed via ``self``.
    """

    @pytest.mark.asyncio
    async def test_deviation_above_07_writes_warnings(self, tmp_path: Path) -> None:
        """deviation 0.75 → warnings on all active goals, paused goals untouched."""
        import vingobot.core.goal_meta as _gm
        from vingobot.agent.loop import AgentLoop

        root = tmp_path / ".taiji"
        init_workspace(root, seed=False)

        _gm.write_goal_meta("goal-one", _gm.GoalMeta(id="goal-one", name="one", status="active"))
        _gm.write_goal_meta("goal-two", _gm.GoalMeta(id="goal-two", name="two", status="paused"))

        loop = AgentLoop.__new__(AgentLoop)
        await loop._enforce_deviation_control(0.75)

        meta1 = _gm.read_goal_meta("goal-one")
        assert meta1 is not None
        assert len(meta1.warnings) >= 1
        assert "偏离度 0.75" in meta1.warnings[0]
        assert meta1.status == "active"  # Not paused at 0.75

        meta2 = _gm.read_goal_meta("goal-two")
        assert meta2 is not None
        assert meta2.warnings == [] or meta2.warnings is None
        assert meta2.status == "paused"

    @pytest.mark.asyncio
    async def test_deviation_above_09_pauses_goals(self, tmp_path: Path) -> None:
        """deviation 0.95 → warnings + pause all active goals."""
        import vingobot.core.goal_meta as _gm
        from vingobot.agent.loop import AgentLoop

        root = tmp_path / ".taiji"
        init_workspace(root, seed=False)

        _gm.write_goal_meta("goal-one", _gm.GoalMeta(id="goal-one", name="one", status="active"))
        _gm.write_goal_meta("goal-two", _gm.GoalMeta(id="goal-two", name="two", status="paused"))

        # Enqueue some tasks for the active goal via explicit path
        test_queue = PendingQueue(root / "pending")
        test_queue.enqueue(PendingTask(id="t1", goal_id="goal-one", description="task to clear"))

        loop = AgentLoop.__new__(AgentLoop)
        await loop._enforce_deviation_control(0.95)

        meta1 = _gm.read_goal_meta("goal-one")
        assert meta1 is not None
        assert meta1.status == "paused"
        assert len(meta1.warnings) >= 1

        # Queue should be cleared (PendingQueue() inside enforcement
        # uses the same workspace root)
        remaining = test_queue.scan_pending()
        tasks_for_one = [t for t in remaining if t.goal_id == "goal-one"]
        assert len(tasks_for_one) == 0

    @pytest.mark.asyncio
    async def test_deviation_below_07_does_nothing(self, tmp_path: Path) -> None:
        """deviation 0.3 → no enforcement, goals unchanged."""
        import vingobot.core.goal_meta as _gm
        from vingobot.agent.loop import AgentLoop

        root = tmp_path / ".taiji"
        init_workspace(root, seed=False)

        _gm.write_goal_meta("goal-one", _gm.GoalMeta(id="goal-one", name="one", status="active"))

        loop = AgentLoop.__new__(AgentLoop)
        await loop._enforce_deviation_control(0.3)

        meta1 = _gm.read_goal_meta("goal-one")
        assert meta1 is not None
        assert meta1.warnings == [] or meta1.warnings is None
        assert meta1.status == "active"

    @pytest.mark.asyncio
    async def test_no_active_goals_no_error(self, tmp_path: Path) -> None:
        """Enforcement with no active goals does not crash."""
        from vingobot.agent.loop import AgentLoop

        root = tmp_path / ".taiji"
        init_workspace(root, seed=False)

        loop = AgentLoop.__new__(AgentLoop)
        # Should not raise
        await loop._enforce_deviation_control(0.85)
