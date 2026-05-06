"""Tests for the file-system based pending task queue."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from vingobot.core.pending_queue import PendingQueue, PendingTask
from vingobot.core.workspace import init_workspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_test_workspace(tmp_path: Path) -> Path:
    """Initialise a .taiji workspace inside *tmp_path* and return its root."""
    root = tmp_path / ".taiji"
    init_workspace(root, seed=False)
    return root


def _create_stale_task_dir(
    goals_dir: Path, goal_id: str, task_id: str, age_minutes: int = 60
) -> Path:
    """Create a task directory with a manifest in 'pending' status that is
    *age_minutes* old."""
    task_dir = goals_dir / goal_id / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    created = datetime.now(timezone.utc).timestamp() - age_minutes * 60
    created_iso = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()

    manifest = {
        "task_id": task_id,
        "goal_id": goal_id,
        "description": "stale task",
        "status": "pending",
        "created_at": created_iso,
        "updated_at": created_iso,
        "round_count": 0,
        "max_rounds": 30,
        "priority": 5,
        "source": "user",
    }
    (task_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return task_dir


# ---------------------------------------------------------------------------
# Tests for PendingQueue
# ---------------------------------------------------------------------------


class TestPendingQueue:
    """Basic queue operations."""

    def test_enqueue_and_scan(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        task = PendingTask(
            id="test-001",
            goal_id="default",
            description="测试任务",
            priority=5,
            source="user",
        )
        filename = queue.enqueue(task)
        assert filename.endswith(".task")

        tasks = queue.scan_pending()
        assert len(tasks) == 1
        assert tasks[0].description == "测试任务"
        assert tasks[0].goal_id == "default"

    def test_try_consume_next(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        task = PendingTask(id="consume-001", goal_id="default", description="消费测试")
        queue.enqueue(task)

        claimed = queue.try_consume_next()
        assert claimed is not None
        consumed, processing_path = claimed
        assert consumed.description == "消费测试"
        assert processing_path.suffix == ".processing"

        # Queue should be empty now
        assert queue.scan_pending() == []

    def test_try_consume_next_empty(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")
        assert queue.try_consume_next() is None

    def test_has_duplicate(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        queue.enqueue(PendingTask(id="dup-001", goal_id="default", description="编写登录功能"))
        queue.enqueue(PendingTask(id="dup-002", goal_id="other", description="不同目标的相同描述"))

        # Same description → duplicate
        assert queue.has_duplicate("编写登录功能") is True
        assert queue.has_duplicate("编写登录功能", goal_id="default") is True

        # Same description, wrong goal → not duplicate
        assert queue.has_duplicate("编写登录功能", goal_id="nonexistent") is False

        # Different description → not duplicate
        assert queue.has_duplicate("完全不同的任务") is False

    def test_delete_task_file(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        task = PendingTask(id="clean-001", goal_id="default", description="清理测试")
        filename = queue.enqueue(task)

        # Delete the .task file
        task_path = root / "pending" / filename
        assert task_path.exists()
        queue.delete_task_file(task_path)
        assert not task_path.exists()

    def test_list_tasks(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        queue.enqueue(PendingTask(id="list-001", goal_id="a", description="task a"))
        queue.enqueue(PendingTask(id="list-002", goal_id="b", description="task b"))

        listed = queue.list_tasks()
        assert len(listed) == 2

    def test_length_property(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        assert queue.length == 0
        queue.enqueue(PendingTask(id="len-001", goal_id="default", description="任务"))
        assert queue.length == 1


# ---------------------------------------------------------------------------
# Tests for prefix-based consumption
# ---------------------------------------------------------------------------


class TestPrefixConsume:
    """try_consume_by_prefix() and try_consume_next(exclude_prefixes=...)."""

    def test_consume_by_prefix_matches_only(self, tmp_path: Path) -> None:
        """try_consume_by_prefix only consumes files starting with the prefix."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        # Write a regular task (no prefix)
        queue.enqueue(PendingTask(id="reg-001", goal_id="default", description="regular task"))

        # Write DMN-prefixed tasks directly (mimicking _process_evolution_actions)
        dmn_file1 = root / "pending" / "cognition-evolution__05__learn-skill__test.task"
        dmn_file1.write_text("learn skill task\npriority=5\nsource=system\ngoalId=cognition-evolution\n")
        dmn_file2 = root / "pending" / "dmn__test-maintenance.task"
        dmn_file2.write_text("dmn maintenance task\npriority=5\nsource=system\ngoalId=dmn\n")

        # Consume by prefix — should get the cognition-evolution task
        claimed = queue.try_consume_by_prefix("cognition-evolution")
        assert claimed is not None
        task, _ = claimed
        assert task.description == "learn skill task"
        assert task.goal_id == "cognition-evolution"

        # Regular task still available via try_consume_next
        remaining = queue.scan_pending()
        assert len(remaining) == 2  # regular + dmn (cognition-evolution was consumed)

    def test_consume_by_prefix_no_match(self, tmp_path: Path) -> None:
        """try_consume_by_prefix returns None when no matching prefix."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        queue.enqueue(PendingTask(id="reg-001", goal_id="default", description="regular task"))

        claimed = queue.try_consume_by_prefix("cognition-evolution")
        assert claimed is None  # No matching prefix

    def test_exclude_prefixes_skips_dmn_tasks(self, tmp_path: Path) -> None:
        """try_consume_next(exclude_prefixes=[...]) skips DMN-prefixed files."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        queue.enqueue(PendingTask(id="reg-001", goal_id="default", description="regular task"))
        (root / "pending" / "cognition-evolution__test.task").write_text(
            "dmn task\npriority=5\nsource=system\ngoalId=cognition-evolution\n"
        )

        claimed = queue.try_consume_next(exclude_prefixes=["cognition-evolution", "dmn"])
        assert claimed is not None
        task, _ = claimed
        assert task.goal_id == "default"  # Got the regular task, not DMN

        # DMN task still in queue
        assert queue.try_consume_by_prefix("cognition-evolution") is not None

    def test_exclude_prefixes_empty_equivalent_to_none(self, tmp_path: Path) -> None:
        """Empty exclude_prefixes list is equivalent to no exclusion."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        queue.enqueue(PendingTask(id="reg-001", goal_id="default", description="regular task"))

        claimed_with_empty = queue.try_consume_next(exclude_prefixes=[])
        assert claimed_with_empty is not None

        # Re-create queue and test without parameter
        queue2 = PendingQueue(root / "pending")
        queue2.enqueue(PendingTask(id="reg-002", goal_id="default", description="second task"))
        claimed_without = queue2.try_consume_next()
        assert claimed_without is not None

    def test_multiple_prefix_exclusion(self, tmp_path: Path) -> None:
        """Multiple prefixes in exclude_prefixes all get skipped."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        queue.enqueue(PendingTask(id="reg-001", goal_id="default", description="regular task"))
        (root / "pending" / "cognition-evolution__a.task").write_text(
            "cog task\npriority=5\nsource=system\ngoalId=cognition-evolution\n"
        )
        (root / "pending" / "dmn__b.task").write_text(
            "dmn task\npriority=5\nsource=system\ngoalId=dmn\n"
        )

        claimed = queue.try_consume_next(
            exclude_prefixes=["cognition-evolution", "dmn"]
        )
        assert claimed is not None
        task, _ = claimed
        assert task.goal_id == "default"  # Only the regular task consumed

    def test_consume_by_prefix_empty_queue(self, tmp_path: Path) -> None:
        """try_consume_by_prefix returns None on empty queue."""
        root = _init_test_workspace(tmp_path)
        queue = PendingQueue(root / "pending")

        assert queue.try_consume_by_prefix("cognition-evolution") is None


# ---------------------------------------------------------------------------
# Tests for orphan cleanup
# ---------------------------------------------------------------------------


class TestCleanupOrphanTasks:
    """cleanup_orphan_tasks() — P0 critical feature."""

    def test_archives_stale_pending_task(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        goals_dir = root / "goals"
        (goals_dir / "test-goal" / "tasks").mkdir(parents=True)

        _create_stale_task_dir(goals_dir, "test-goal", "task-001", age_minutes=60)

        cleaned = PendingQueue.cleanup_orphan_tasks(timeout_ms=30 * 60 * 1000)
        assert cleaned == 1

        # Verify status changed to 'archived'
        manifest_path = goals_dir / "test-goal" / "tasks" / "task-001" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "archived"

    def test_ignores_recent_pending_task(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        goals_dir = root / "goals"
        (goals_dir / "test-goal" / "tasks").mkdir(parents=True)

        _create_stale_task_dir(goals_dir, "test-goal", "task-001", age_minutes=5)

        cleaned = PendingQueue.cleanup_orphan_tasks(timeout_ms=30 * 60 * 1000)
        assert cleaned == 0  # Only 5 min old → not stale

    def test_ignores_non_pending_tasks(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        goals_dir = root / "goals"
        (goals_dir / "test-goal" / "tasks").mkdir(parents=True)

        task_dir = _create_stale_task_dir(goals_dir, "test-goal", "task-001", age_minutes=60)

        # Manually set status to 'completed'
        manifest_path = task_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        cleaned = PendingQueue.cleanup_orphan_tasks(timeout_ms=30 * 60 * 1000)
        assert cleaned == 0  # completed → not orphan

    def test_empty_goals_dir(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        # No goals exist
        cleaned = PendingQueue.cleanup_orphan_tasks()
        assert cleaned == 0

    def test_no_tasks_dir(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        goals_dir = root / "goals"
        (goals_dir / "empty-goal").mkdir(parents=True)
        # Goal has no tasks/ dir
        cleaned = PendingQueue.cleanup_orphan_tasks()
        assert cleaned == 0

    def test_multiple_goals_mixed(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        goals_dir = root / "goals"

        # Goal A: 2 stale pending tasks
        (goals_dir / "goal-a" / "tasks").mkdir(parents=True)
        _create_stale_task_dir(goals_dir, "goal-a", "stale-1", age_minutes=60)
        _create_stale_task_dir(goals_dir, "goal-a", "stale-2", age_minutes=120)

        # Goal B: 1 recent task (should NOT be archived)
        _create_stale_task_dir(goals_dir, "goal-b", "recent-1", age_minutes=5)

        cleaned = PendingQueue.cleanup_orphan_tasks(timeout_ms=30 * 60 * 1000)
        assert cleaned == 2

        # Verify which ones were archived
        def _get_status(goal: str, task: str) -> str:
            p = goals_dir / goal / "tasks" / task / "manifest.json"
            return json.loads(p.read_text(encoding="utf-8"))["status"]

        assert _get_status("goal-a", "stale-1") == "archived"
        assert _get_status("goal-a", "stale-2") == "archived"
        assert _get_status("goal-b", "recent-1") == "pending"  # unchanged

    def test_task_with_invalid_created_at(self, tmp_path: Path) -> None:
        root = _init_test_workspace(tmp_path)
        goals_dir = root / "goals"
        (goals_dir / "test-goal" / "tasks").mkdir(parents=True)

        task_dir = _create_stale_task_dir(goals_dir, "test-goal", "task-001", age_minutes=60)
        # Corrupt created_at
        manifest_path = task_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["created_at"] = "not-a-date"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        cleaned = PendingQueue.cleanup_orphan_tasks(timeout_ms=30 * 60 * 1000)
        assert cleaned == 0  # Invalid date → cannot determine age → skip
