"""Tests for DMN tool — status, start, stop, and trigger actions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from vingobot.agent.tools.dmn import DmnTool
from vingobot.core.workspace import init_workspace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_test_workspace(tmp_path: Path) -> Path:
    """Initialise a .taiji workspace and return its root."""
    root = tmp_path / ".taiji"
    init_workspace(root, seed=False)
    return root


def _make_loop(*, dmn_running: bool = False, loop_running: bool = True) -> MagicMock:
    """Build a MagicMock AgentLoop with DMN consumer state."""
    loop = MagicMock()
    loop._running = loop_running
    dmn_task = MagicMock()
    if dmn_running:
        dmn_task.done.return_value = False
    else:
        dmn_task.done.return_value = True
    loop._dmn_consumer_task = dmn_task

    # _run_dmn_consumer must be an async callable for asyncio.create_task
    async def _async_noop() -> None:
        pass

    loop._run_dmn_consumer = _async_noop
    return loop


# ---------------------------------------------------------------------------
# Tests — status action
# ---------------------------------------------------------------------------


class TestDmnStatus:
    """DMN status dashboard tests."""

    def test_status_stopped(self, tmp_path: Path) -> None:
        """Status shows stopped when DMN consumer is not running."""
        _init_test_workspace(tmp_path)
        loop = _make_loop(dmn_running=False)
        tool = DmnTool(loop)

        result = tool._do_status()

        assert "DMN 认知网络状态" in result
        assert "已停止" in result

    def test_status_running(self, tmp_path: Path) -> None:
        """Status shows running when DMN consumer is active."""
        _init_test_workspace(tmp_path)
        loop = _make_loop(dmn_running=True)
        tool = DmnTool(loop)

        result = tool._do_status()

        assert "运行中" in result

    def test_status_pending_tasks(self, tmp_path: Path) -> None:
        """Status counts pending DMN tasks."""
        root = _init_test_workspace(tmp_path)
        pending_dir = root / "pending"

        # Add a cognition-evolution task
        (pending_dir / "cognition-evolution__05__check__20260101.task").write_text(
            "check skills\npriority=5\nsource=user\ngoalId=cognition-evolution\n",
            encoding="utf-8",
        )

        # Add a dmn task
        (pending_dir / "dmn__review__20260101.task").write_text(
            "review models\npriority=3\nsource=system\ngoalId=cognition-evolution\n",
            encoding="utf-8",
        )

        # Add a regular task (should NOT be counted)
        (pending_dir / "some-goal__task1.task").write_text(
            "regular task\ngoalId=some-goal\n",
            encoding="utf-8",
        )

        loop = _make_loop(dmn_running=True)
        tool = DmnTool(loop)

        result = tool._do_status()

        assert "待处理: 2 个" in result

    def test_status_cognition_health(self, tmp_path: Path) -> None:
        """Status shows cognition library counts."""
        root = _init_test_workspace(tmp_path)

        # Create assets
        skills_dir = root / "cognition" / "skills" / "test-skill"
        skills_dir.mkdir(parents=True, exist_ok=True)

        models_dir = root / "cognition" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "test.md").write_text("# model", encoding="utf-8")

        loop = _make_loop(dmn_running=False)
        tool = DmnTool(loop)

        result = tool._do_status()

        assert "L1 技能 (skills):" in result
        assert "L2 思维模型 (models):" in result
        assert "L3 认知格栅 (grids):" in result

    def test_status_no_pending_dir(self, tmp_path: Path) -> None:
        """Status handles missing pending directory gracefully."""
        root = _init_test_workspace(tmp_path)
        # Remove pending dir
        import shutil

        shutil.rmtree(root / "pending", ignore_errors=True)

        loop = _make_loop(dmn_running=False)
        tool = DmnTool(loop)

        result = tool._do_status()

        # Should not crash
        assert "DMN 认知网络状态" in result

    def test_status_current_task(self, tmp_path: Path) -> None:
        """Status shows currently-processing DMN task."""
        root = _init_test_workspace(tmp_path)

        # Create a processing file simulating an in-progress DMN task
        pending_dir = root / "pending"
        (pending_dir / "cognition-evolution__05__check_skills__20260101.processing").write_text(
            "检查技能覆盖情况\npriority=5\nsource=user\ngoalId=cognition-evolution\n",
            encoding="utf-8",
        )

        loop = _make_loop(dmn_running=True)
        tool = DmnTool(loop)

        result = tool._do_status()

        assert "正在处理" in result
        assert "cognition-evolution" in result
        assert "检查技能覆盖情况" in result


# ---------------------------------------------------------------------------
# Tests — start / stop
# ---------------------------------------------------------------------------


class TestDmnStartStop:
    """DMN start/stop tests."""

    def test_start_when_stopped(self) -> None:
        """Start creates a new DMN consumer task when stopped."""
        loop = _make_loop(dmn_running=False)
        tool = DmnTool(loop)

        async def _run() -> str:
            return tool._do_start()

        result = asyncio.run(_run())

        assert "已启动" in result

    def test_start_when_already_running(self) -> None:
        """Start returns running message when already running."""
        loop = _make_loop(dmn_running=True)
        tool = DmnTool(loop)

        result = tool._do_start()

        assert "已在运行中" in result

    def test_start_when_loop_stopped(self) -> None:
        """Start fails when AgentLoop is not running."""
        loop = _make_loop(dmn_running=False, loop_running=False)
        tool = DmnTool(loop)

        result = tool._do_start()

        assert "AgentLoop 未运行" in result

    def test_stop_when_running(self) -> None:
        """Stop cancels the DMN consumer task."""
        loop = _make_loop(dmn_running=True)
        tool = DmnTool(loop)

        result = tool._do_stop()

        assert "已发出停止信号" in result
        loop._dmn_consumer_task.cancel.assert_called_once()

    def test_stop_when_already_stopped(self) -> None:
        """Stop returns not-running message when already stopped."""
        loop = _make_loop(dmn_running=False)
        tool = DmnTool(loop)

        result = tool._do_stop()

        assert "未在运行" in result


# ---------------------------------------------------------------------------
# Tests — trigger action
# ---------------------------------------------------------------------------


class TestDmnTrigger:
    """DMN trigger (cognition-evolution enqueue) tests."""

    def test_trigger_creates_task_file(self, tmp_path: Path) -> None:
        """Trigger writes a cognition-evolution prefixed .task file."""
        root = _init_test_workspace(tmp_path)
        loop = _make_loop(dmn_running=True)
        tool = DmnTool(loop)

        result = tool._do_trigger({"description": "检查技能覆盖情况"})

        assert "已入队认知演化任务" in result
        pending_dir = root / "pending"
        dmn_files = list(pending_dir.glob("cognition-evolution__*.task"))
        assert len(dmn_files) >= 1
        content = dmn_files[0].read_text(encoding="utf-8")
        assert "goalId=cognition-evolution" in content

    def test_trigger_default_description(self, tmp_path: Path) -> None:
        """Trigger uses default description when none provided."""
        _init_test_workspace(tmp_path)
        loop = _make_loop(dmn_running=True)
        tool = DmnTool(loop)

        result = tool._do_trigger({})

        assert "已入队认知演化任务" in result

    def test_trigger_warns_when_dmn_stopped(self, tmp_path: Path) -> None:
        """Trigger warns when DMN consumer is not running."""
        _init_test_workspace(tmp_path)
        loop = _make_loop(dmn_running=False)
        tool = DmnTool(loop)

        result = tool._do_trigger({"description": "test"})

        assert "DMN 消费者未运行" in result


# ---------------------------------------------------------------------------
# Tests — execute dispatch
# ---------------------------------------------------------------------------


class TestDmnExecute:
    """DMN execute() dispatch tests."""

    def test_execute_status(self, tmp_path: Path) -> None:
        """Execute dispatches 'status' action correctly."""
        _init_test_workspace(tmp_path)
        loop = _make_loop(dmn_running=False)
        tool = DmnTool(loop)

        async def _run() -> str:
            return await tool.execute(action="status")

        result = asyncio.run(_run())

        assert "DMN 认知网络状态" in result

    def test_execute_start(self) -> None:
        """Execute dispatches 'start' action correctly."""
        loop = _make_loop(dmn_running=False)
        tool = DmnTool(loop)

        async def _run() -> str:
            return await tool.execute(action="start")

        result = asyncio.run(_run())

        assert "已启动" in result

    def test_execute_stop(self) -> None:
        """Execute dispatches 'stop' action correctly."""
        loop = _make_loop(dmn_running=True)
        tool = DmnTool(loop)

        async def _run() -> str:
            return await tool.execute(action="stop")

        result = asyncio.run(_run())

        assert "已发出停止信号" in result

    def test_execute_trigger(self, tmp_path: Path) -> None:
        """Execute dispatches 'trigger' action and writes task file."""
        root = _init_test_workspace(tmp_path)
        loop = _make_loop(dmn_running=True)
        tool = DmnTool(loop)

        async def _run() -> str:
            return await tool.execute(action="trigger", description="测试触发")

        result = asyncio.run(_run())

        assert "已入队认知演化任务" in result
        pending_dir = root / "pending"
        dmn_files = list(pending_dir.glob("cognition-evolution__*.task"))
        assert len(dmn_files) >= 1

    def test_execute_unknown_action(self, tmp_path: Path) -> None:
        """Execute returns error for unknown action."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = DmnTool(loop)

        async def _run() -> str:
            return await tool.execute(action="fly-to-moon")

        result = asyncio.run(_run())

        assert "未知 DMN 动作" in result
