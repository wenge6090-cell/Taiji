"""Tests for TPN tool — status and trigger actions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from vingobot.agent.tools.tpn import TpnTool
from vingobot.core.goal_meta import GoalMeta, SelfDrivenConfig, write_goal_meta
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


def _make_loop(*, tpn_bot=None, dmn_running: bool = False) -> MagicMock:
    """Build a MagicMock AgentLoop with just enough attributes for TpnTool."""
    loop = MagicMock()
    loop._tpn_bot = tpn_bot
    dmn_task = MagicMock()
    if dmn_running:
        dmn_task.done.return_value = False
    else:
        dmn_task.done.return_value = True
    loop._dmn_consumer_task = dmn_task
    return loop


def _make_bot(*, sixiang_running: bool = False) -> MagicMock:
    """Build a MagicMock vingobot with sixiang pool state."""
    bot = MagicMock()
    bot.sixiang_running = sixiang_running
    pool = MagicMock()
    pool.active_count = 0
    bot._sixiang_pool = pool
    bot.start_sixiang = MagicMock()
    bot.stop_sixiang = MagicMock()
    return bot


# ---------------------------------------------------------------------------
# Tests — status action
# ---------------------------------------------------------------------------


class TestTpnStatus:
    """TPN status dashboard tests."""

    def test_status_pool_stopped(self, tmp_path: Path) -> None:
        """Status shows 'stopped' when sixiang pool is not running."""
        _init_test_workspace(tmp_path)
        bot = _make_bot(sixiang_running=False)
        loop = _make_loop(tpn_bot=bot, dmn_running=False)
        tool = TpnTool(loop)

        result = tool._do_status()

        assert "TPN 状态面板" in result
        assert "已停止" in result
        assert "DMN 消费者" in result

    def test_status_pool_running(self, tmp_path: Path) -> None:
        """Status shows 'running' when pool is active."""
        _init_test_workspace(tmp_path)
        bot = _make_bot(sixiang_running=True)
        loop = _make_loop(tpn_bot=bot, dmn_running=False)
        tool = TpnTool(loop)

        result = tool._do_status()

        assert "运行中" in result

    def test_status_no_tpn_bot(self, tmp_path: Path) -> None:
        """Status gracefully handles missing _tpn_bot reference."""
        _init_test_workspace(tmp_path)
        loop = _make_loop(tpn_bot=None, dmn_running=False)
        tool = TpnTool(loop)

        result = tool._do_status()

        assert "未初始化" in result or "_tpn_bot" in result

    def test_status_pending_queue_counts(self, tmp_path: Path) -> None:
        """Status counts regular and DMN tasks in the pending queue."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        # Add regular tasks
        queue.enqueue(PendingTask(id="", goal_id="goal-a", description="task a"))
        queue.enqueue(PendingTask(id="", goal_id="goal-b", description="task b"))

        # Add DMN task (cognition-evolution prefix)
        dmn_file = root / "pending" / "cognition-evolution__05__learn-skill__test.task"
        dmn_file.write_text(
            "learn skill\npriority=5\nsource=system\ngoalId=cognition-evolution\n",
            encoding="utf-8",
        )

        bot = _make_bot(sixiang_running=False)
        loop = _make_loop(tpn_bot=bot)
        tool = TpnTool(loop)

        result = tool._do_status()

        assert "总任务数: 3" in result
        assert "常规任务: 2" in result
        assert "认知演化任务 (DMN): 1" in result

    def test_status_cognition_health(self, tmp_path: Path) -> None:
        """Status shows cognition library health counts."""
        root = _init_test_workspace(tmp_path)

        # Create a test skill
        skills_dir = root / "cognition" / "skills" / "test-skill"
        skills_dir.mkdir(parents=True, exist_ok=True)

        # Create a test model
        models_dir = root / "cognition" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "test-model.md").write_text("# test model", encoding="utf-8")

        bot = _make_bot(sixiang_running=False)
        loop = _make_loop(tpn_bot=bot)
        tool = TpnTool(loop)

        result = tool._do_status()

        assert "认知库健康度" in result
        assert "L1 技能 (skills):" in result
        assert "L2 思维模型 (models):" in result

    def test_status_goal_summary(self, tmp_path: Path) -> None:
        """Status shows goal summaries from meta.json files."""
        root = _init_test_workspace(tmp_path)

        goals_dir = root / "goals"
        (goals_dir / "active-goal").mkdir(parents=True, exist_ok=True)
        write_goal_meta(
            "active-goal",
            GoalMeta(
                id="active-goal",
                name="活跃目标",
                status="active",
                priority=8,
                self_driven=SelfDrivenConfig(enabled=True, interval_minutes=15),
            ),
        )

        (goals_dir / "paused-goal").mkdir(parents=True, exist_ok=True)
        write_goal_meta(
            "paused-goal",
            GoalMeta(
                id="paused-goal",
                name="暂停目标",
                status="paused",
                priority=3,
                warnings=["low balance"],
            ),
        )

        bot = _make_bot(sixiang_running=False)
        loop = _make_loop(tpn_bot=bot)
        tool = TpnTool(loop)

        result = tool._do_status()

        assert "目标概览" in result
        assert "活跃目标" in result
        assert "暂停目标" in result
        assert "自驱:关" in result or "自驱" in result


# ---------------------------------------------------------------------------
# Tests — list action
# ---------------------------------------------------------------------------


class TestTpnList:
    """TPN list-goals tests."""

    def test_list_empty(self, tmp_path: Path) -> None:
        """List returns '(无目标)' when no goals exist."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_list()

        assert "无目标" in result

    def test_list_with_goals(self, tmp_path: Path) -> None:
        """List shows all goals with status, priority, and self-driven flag."""
        root = _init_test_workspace(tmp_path)

        goals_dir = root / "goals"
        (goals_dir / "g1").mkdir(parents=True, exist_ok=True)
        write_goal_meta(
            "g1",
            GoalMeta(
                id="g1",
                name="目标一",
                status="active",
                priority=7,
                description="第一个目标",
            ),
        )

        (goals_dir / "g2").mkdir(parents=True, exist_ok=True)
        write_goal_meta(
            "g2",
            GoalMeta(
                id="g2",
                name="目标二",
                status="completed",
                priority=5,
                self_driven=SelfDrivenConfig(enabled=True, interval_minutes=30),
            ),
        )

        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_list()

        assert "目标一" in result
        assert "目标二" in result
        assert "active" in result
        assert "completed" in result


# ---------------------------------------------------------------------------
# Tests — trigger action
# ---------------------------------------------------------------------------


class TestTpnTrigger:
    """TPN trigger (manual Anqu evaluation) tests."""

    def test_trigger_enqueues_task(self, tmp_path: Path) -> None:
        """Trigger enqueues a task to the pending queue for the given goal."""
        root = _init_test_workspace(tmp_path)

        # Create a goal first
        goals_dir = root / "goals" / "test-goal"
        goals_dir.mkdir(parents=True, exist_ok=True)
        write_goal_meta(
            "test-goal",
            GoalMeta(id="test-goal", name="测试目标", status="active", priority=5),
        )

        bot = _make_bot(sixiang_running=False)
        loop = _make_loop(tpn_bot=bot)
        tool = TpnTool(loop)

        result = tool._do_trigger({"goal_id": "test-goal"})

        assert "已向目标" in result
        assert "test-goal" in result
        # Check task was actually written
        queue = PendingQueue(root / "pending")
        tasks = queue.scan_pending()
        assert len(tasks) >= 1
        assert tasks[0].goal_id == "test-goal"

    def test_trigger_with_custom_description(self, tmp_path: Path) -> None:
        """Trigger accepts a custom description."""
        root = _init_test_workspace(tmp_path)

        goals_dir = root / "goals" / "custom-goal"
        goals_dir.mkdir(parents=True, exist_ok=True)
        write_goal_meta(
            "custom-goal",
            GoalMeta(id="custom-goal", status="active"),
        )

        bot = _make_bot()
        loop = _make_loop(tpn_bot=bot)
        tool = TpnTool(loop)

        result = tool._do_trigger(
            {
                "goal_id": "custom-goal",
                "description": "请评估数据库迁移进度",
            }
        )

        assert "数据库迁移" in result
        queue = PendingQueue(root / "pending")
        tasks = queue.scan_pending()
        assert any("数据库迁移" in t.description for t in tasks)

    def test_trigger_nonexistent_goal(self, tmp_path: Path) -> None:
        """Trigger returns error for non-existent goal."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_trigger({"goal_id": "no-such-goal"})

        assert "不存在" in result

    def test_trigger_no_goal_id(self, tmp_path: Path) -> None:
        """Trigger returns error when goal_id is missing."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_trigger({})

        assert "必须提供 goal_id" in result


# ---------------------------------------------------------------------------
# Tests — create action
# ---------------------------------------------------------------------------


class TestTpnCreate:
    """TPN create-goal tests."""

    def test_create_minimal(self, tmp_path: Path) -> None:
        """Create a goal with minimal required fields."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_create(
            {
                "goal_id": "minimal-goal",
                "description": "最小目标",
            }
        )

        assert "已创建" in result
        assert "minimal-goal" in result

        # Verify on disk
        from vingobot.core.goal_meta import read_goal_meta

        meta = read_goal_meta("minimal-goal")
        assert meta is not None
        assert meta.id == "minimal-goal"
        assert meta.description == "最小目标"
        assert meta.status == "active"
        assert meta.priority == 5

    def test_create_with_priority(self, tmp_path: Path) -> None:
        """Create a goal with custom priority."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_create(
            {
                "goal_id": "hi-pri-goal",
                "description": "高优先级",
                "priority": 9,
            }
        )

        assert "已创建" in result
        from vingobot.core.goal_meta import read_goal_meta

        meta = read_goal_meta("hi-pri-goal")
        assert meta.priority == 9

    def test_create_with_self_driven(self, tmp_path: Path) -> None:
        """Create a goal with self-driven wake enabled."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_create(
            {
                "goal_id": "auto-goal",
                "description": "自动目标",
                "self_driven_enabled": True,
                "self_driven_interval_minutes": 15,
            }
        )

        assert "已创建" in result
        assert "自驱: 启用" in result
        from vingobot.core.goal_meta import read_goal_meta

        meta = read_goal_meta("auto-goal")
        assert meta.self_driven.enabled is True
        assert meta.self_driven.interval_minutes == 15

    def test_create_invalid_goal_id(self, tmp_path: Path) -> None:
        """Create rejects invalid goal_id."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_create({"goal_id": "", "description": "bad"})
        assert "必须提供 goal_id" in result

        result = tool._do_create({"goal_id": "a" * 65, "description": "bad"})
        assert "最长" in result

    def test_create_clamps_priority(self, tmp_path: Path) -> None:
        """Create clamps priority to 1-10 range."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        # Priority 0 → clamp to 1
        tool._do_create({"goal_id": "clamp-low", "description": "x", "priority": 0})
        from vingobot.core.goal_meta import read_goal_meta

        assert read_goal_meta("clamp-low").priority == 1  # type: ignore[union-attr]

        # Priority 99 → clamp to 10
        tool._do_create({"goal_id": "clamp-high", "description": "x", "priority": 99})
        assert read_goal_meta("clamp-high").priority == 10  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tests — update action
# ---------------------------------------------------------------------------


class TestTpnUpdate:
    """TPN update-goal tests."""

    def test_update_status(self, tmp_path: Path) -> None:
        """Update changes goal status."""
        root = _init_test_workspace(tmp_path)
        (root / "goals" / "up-goal").mkdir(parents=True, exist_ok=True)
        write_goal_meta("up-goal", GoalMeta(id="up-goal", status="active"))

        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_update({"goal_id": "up-goal", "status": "paused"})

        assert "已更新" in result
        assert "paused" in result

        from vingobot.core.goal_meta import read_goal_meta

        meta = read_goal_meta("up-goal")
        assert meta.status == "paused"  # type: ignore[union-attr]

    def test_update_priority(self, tmp_path: Path) -> None:
        """Update changes goal priority."""
        root = _init_test_workspace(tmp_path)
        (root / "goals" / "pri-goal").mkdir(parents=True, exist_ok=True)
        write_goal_meta("pri-goal", GoalMeta(id="pri-goal", priority=3))

        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_update({"goal_id": "pri-goal", "priority": 8})

        assert "已更新" in result
        from vingobot.core.goal_meta import read_goal_meta

        assert read_goal_meta("pri-goal").priority == 8  # type: ignore[union-attr]

    def test_update_nonexistent(self, tmp_path: Path) -> None:
        """Update returns error for non-existent goal."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_update({"goal_id": "no-such", "status": "paused"})

        assert "不存在" in result

    def test_update_no_changes(self, tmp_path: Path) -> None:
        """Update with no actual changes returns appropriate message."""
        root = _init_test_workspace(tmp_path)
        (root / "goals" / "nc-goal").mkdir(parents=True, exist_ok=True)
        write_goal_meta("nc-goal", GoalMeta(id="nc-goal", status="active"))

        loop = _make_loop()
        tool = TpnTool(loop)

        # Update with same status → no change
        result = tool._do_update({"goal_id": "nc-goal", "status": "active"})

        assert "没有需要更新的字段" in result


# ---------------------------------------------------------------------------
# Tests — delete action
# ---------------------------------------------------------------------------


class TestTpnDelete:
    """TPN delete-goal tests."""

    def test_delete_existing_goal(self, tmp_path: Path) -> None:
        """Delete removes a goal directory."""
        root = _init_test_workspace(tmp_path)

        goal_dir = root / "goals" / "del-goal"
        goal_dir.mkdir(parents=True, exist_ok=True)
        write_goal_meta("del-goal", GoalMeta(id="del-goal", name="待删除"))

        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_delete({"goal_id": "del-goal"})

        assert "已删除" in result
        assert not goal_dir.exists()

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        """Delete returns error for non-existent goal."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = tool._do_delete({"goal_id": "no-such"})

        assert "不存在" in result


# ---------------------------------------------------------------------------
# Tests — execute dispatch
# ---------------------------------------------------------------------------


class TestTpnExecute:
    """TPN execute() dispatch tests."""

    def test_execute_status(self, tmp_path: Path) -> None:
        """Execute dispatches 'status' action correctly."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = asyncio.run(tool.execute(action="status"))

        assert "TPN 状态面板" in result

    def test_execute_list(self, tmp_path: Path) -> None:
        """Execute dispatches 'list' action correctly."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = asyncio.run(tool.execute(action="list"))

        assert "无目标" in result

    def test_execute_unknown_action(self, tmp_path: Path) -> None:
        """Execute returns error for unknown action."""
        _init_test_workspace(tmp_path)
        loop = _make_loop()
        tool = TpnTool(loop)

        result = asyncio.run(tool.execute(action="fly-to-moon"))

        assert "未知 TPN 动作" in result

    def test_execute_trigger(self, tmp_path: Path) -> None:
        """Execute dispatches 'trigger' action and enqueues task."""
        root = _init_test_workspace(tmp_path)

        (root / "goals" / "exec-goal").mkdir(parents=True, exist_ok=True)
        write_goal_meta("exec-goal", GoalMeta(id="exec-goal", status="active"))

        loop = _make_loop()
        tool = TpnTool(loop)

        result = asyncio.run(
            tool.execute(action="trigger", goal_id="exec-goal")
        )

        assert "已向目标" in result
        queue = PendingQueue(root / "pending")
        tasks = queue.scan_pending()
        assert any(t.goal_id == "exec-goal" for t in tasks)
