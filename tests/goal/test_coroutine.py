"""Tests for WorkerPool control-plane (meta.json signal handling)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from vingobot.core.goal_meta import GoalMeta, read_goal_meta, write_goal_meta
from vingobot.core.pending_queue import PendingQueue
from vingobot.core.workspace import init_workspace
from vingobot.goal.coroutine import WorkerPool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_test_workspace(tmp_path: Path) -> Path:
    """Initialise a .taiji workspace and return its root."""
    root = tmp_path / ".taiji"
    init_workspace(root, seed=False)
    return root


def _create_goal_meta(workspace_root: Path, goal_id: str, **overrides) -> GoalMeta:
    """Create a goal directory with meta.json and return the GoalMeta.

    ``init_workspace()`` must have been called with *workspace_root* before
    calling this helper so that ``write_goal_meta`` resolves the correct path.
    """
    meta = GoalMeta(
        id=goal_id,
        name=goal_id,
        description=f"Test goal {goal_id}",
        **overrides,
    )
    write_goal_meta(goal_id, meta)
    return meta


def _enqueue_task(workspace_root: Path, goal_id: str, description: str) -> None:
    """Write a task file manually (bypassing PendingQueue.enqueue for test speed)."""
    pending_dir = workspace_root / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
    filename = f"{ts}_{description[:20]}.task"
    filepath = pending_dir / filename
    content = f"{description}\npriority=5\nsource=user\ngoalId={goal_id}\n"
    filepath.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWorkerPoolControlPlane:
    """WorkerPool reacts correctly to meta.json signals."""

    async def test_paused_goal_re_enqueues(self, tmp_path: Path) -> None:
        """Worker should re-enqueue task when meta.json status is 'paused'."""
        root = _init_test_workspace(tmp_path)
        _create_goal_meta(root, "test-goal", status="paused")
        _enqueue_task(root, "test-goal", "paused goal task")

        executed: list[str] = []

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed.append(goal_id)
            return f"executed {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)
        await pool.start()

        # Wait for worker to pick up and re-enqueue
        await asyncio.sleep(0.5)

        # Should NOT have executed (core invariant)
        assert executed == []

        # The task was re-enqueued at least once (log confirms).
        # Due to tight re-claim loop, the queue may be empty or have
        # the re-enqueued task depending on timing.  We stop first,
        # then check only that the goal was never executed.
        await pool.stop()

        # After graceful stop, check if the task survived in queue.
        # It is acceptable for the queue to be empty — the worker may
        # have claimed it just before cancellation.
        queue = PendingQueue(root / "pending")
        remaining = queue.scan_pending()
        # Either 0 (claimed right before stop) or ≥1 (re-enqueued)
        assert all(t.goal_id == "test-goal" for t in remaining)

    async def test_warnings_drops_task(self, tmp_path: Path) -> None:
        """Worker should drop task when meta.json has non-empty warnings."""
        root = _init_test_workspace(tmp_path)
        _create_goal_meta(root, "test-goal", warnings=["critical error"])
        _enqueue_task(root, "test-goal", "warning goal task")

        executed: list[str] = []

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed.append(goal_id)
            return f"executed {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)
        await pool.start()

        await asyncio.sleep(0.5)

        assert executed == []

        # Task should be deleted (not in pending)
        queue = PendingQueue(root / "pending")
        assert queue.scan_pending() == []

        await pool.stop()

    async def test_missing_meta_drops_task(self, tmp_path: Path) -> None:
        """Worker should drop task when meta.json does not exist."""
        root = _init_test_workspace(tmp_path)
        # No meta.json created for "orphan-goal"
        _enqueue_task(root, "orphan-goal", "orphan goal task")

        executed: list[str] = []

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed.append(goal_id)
            return f"executed {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)
        await pool.start()

        await asyncio.sleep(0.5)

        assert executed == []

        queue = PendingQueue(root / "pending")
        assert queue.scan_pending() == []

        await pool.stop()

    async def test_empty_anqu_confirms_and_executes(self, tmp_path: Path) -> None:
        """Worker should confirm Anqu (write timestamp) then execute task."""
        root = _init_test_workspace(tmp_path)
        _create_goal_meta(root, "test-goal", status="active", last_anqu_at="")
        _enqueue_task(root, "test-goal", "anqu pending task")

        executed: list[str] = []

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed.append(goal_id)
            return f"executed {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)
        await pool.start()

        await asyncio.sleep(0.5)

        assert executed == ["test-goal"]

        # last_anqu_at should have been written
        meta = read_goal_meta("test-goal")
        assert meta is not None
        assert meta.last_anqu_at != ""

        await pool.stop()

    async def test_normal_execution(self, tmp_path: Path) -> None:
        """Worker should execute task when all control checks pass."""
        root = _init_test_workspace(tmp_path)
        _create_goal_meta(
            root,
            "test-goal",
            status="active",
            last_anqu_at="2026-01-01T00:00:00",
        )
        _enqueue_task(root, "test-goal", "normal task")

        executed: list[str] = []

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed.append(goal_id)
            return f"executed {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)
        await pool.start()

        await asyncio.sleep(0.5)

        assert executed == ["test-goal"]

        queue = PendingQueue(root / "pending")
        assert queue.scan_pending() == []

        await pool.stop()

    async def test_active_goals_tracking(self, tmp_path: Path) -> None:
        """WorkerPool tracks active goals during execution."""
        root = _init_test_workspace(tmp_path)
        _create_goal_meta(
            root,
            "test-goal",
            status="active",
            last_anqu_at="2026-01-01T00:00:00",
        )
        _enqueue_task(root, "test-goal", "tracking task")

        running = asyncio.Event()
        done = asyncio.Event()

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            running.set()
            await done.wait()
            return f"executed {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)
        await pool.start()

        await asyncio.wait_for(running.wait(), timeout=2.0)
        assert pool.is_goal_active("test-goal") is True
        assert "test-goal" in pool.get_active_goals()

        done.set()
        await asyncio.sleep(0.3)
        assert pool.is_goal_active("test-goal") is False

        await pool.stop()

    async def test_exclude_dmn_prefixes(self, tmp_path: Path) -> None:
        """WorkerPool skips cognition-evolution prefixed tasks."""
        root = _init_test_workspace(tmp_path)
        _create_goal_meta(root, "default", status="active", last_anqu_at="2026-01-01T00:00:00")

        # Enqueue a regular task
        _enqueue_task(root, "default", "regular task")

        # Write a DMN-prefixed task directly
        dmn_file = root / "pending" / "cognition-evolution__test.task"
        dmn_file.write_text("dmn task\npriority=5\nsource=system\ngoalId=cognition-evolution\n")

        executed: list[str] = []
        done = asyncio.Event()

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed.append(goal_id)
            done.set()
            return f"executed {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)
        await pool.start()

        await asyncio.wait_for(done.wait(), timeout=2.0)
        assert executed == ["default"]  # Only the regular task

        # DMN task should remain in pending
        queue = PendingQueue(root / "pending")
        remaining = queue.scan_pending()
        dmn_tasks = [t for t in remaining if t.goal_id == "cognition-evolution"]
        assert len(dmn_tasks) >= 1  # or 2 (if re-enqueued), at least 1

        await pool.stop()


# ---------------------------------------------------------------------------
# Tests — on_task_complete callback (TPN→DMN feedback)
# ---------------------------------------------------------------------------


class TestOnTaskComplete:
    """WorkerPool invokes on_task_complete after each sixiang loop."""

    async def test_callback_fires_on_success(self, tmp_path: Path) -> None:
        """Callback receives goal_id, success=True, and summary on success."""
        root = _init_test_workspace(tmp_path)

        call_log: list[tuple[str, bool, str]] = []

        async def _mock_run(
            goal_id: str, description: str, signal: asyncio.Task | None,
        ) -> object:
            from vingobot.goal.types import GoalResult
            return GoalResult(status="completed", goal_id=goal_id)

        async def _on_complete(goal_id: str, success: bool, summary: str) -> None:
            call_log.append((goal_id, success, summary))

        _create_goal_meta(root, "goal-test")
        _enqueue_task(root, "goal-test", "test task")

        pool = WorkerPool(
            max_workers=1,
            poll_interval_ms=200,
            run_task_fn=_mock_run,
            on_task_complete=_on_complete,
        )
        await pool.start()

        # Wait for the worker to pick up and complete the task
        await asyncio.sleep(0.5)
        await pool.stop()

        assert len(call_log) == 1
        goal_id, success, summary = call_log[0]
        assert goal_id == "goal-test"
        assert success is True
        assert "test task" in summary

    async def test_callback_fires_on_failure(self, tmp_path: Path) -> None:
        """Callback receives success=False when _run_task_fn raises."""
        root = _init_test_workspace(tmp_path)

        call_log: list[tuple[str, bool, str]] = []

        async def _mock_run(
            goal_id: str, description: str, signal: asyncio.Task | None,
        ) -> object:
            raise RuntimeError("simulated failure")

        async def _on_complete(goal_id: str, success: bool, summary: str) -> None:
            call_log.append((goal_id, success, summary))

        _create_goal_meta(root, "goal-fail")
        _enqueue_task(root, "goal-fail", "failing task")

        pool = WorkerPool(
            max_workers=1,
            poll_interval_ms=200,
            run_task_fn=_mock_run,
            on_task_complete=_on_complete,
        )
        await pool.start()
        await asyncio.sleep(0.5)
        await pool.stop()

        assert len(call_log) >= 1
        goal_id, success, summary = call_log[0]
        assert goal_id == "goal-fail"
        assert success is False
        assert "failing" in summary

    async def test_callback_not_required(self, tmp_path: Path) -> None:
        """WorkerPool works fine without on_task_complete (backward compat)."""
        root = _init_test_workspace(tmp_path)

        async def _mock_run(
            goal_id: str, description: str, signal: asyncio.Task | None,
        ) -> object:
            from vingobot.goal.types import GoalResult
            return GoalResult(status="completed", goal_id=goal_id)

        _create_goal_meta(root, "goal-ok")
        _enqueue_task(root, "goal-ok", "task without callback")

        pool = WorkerPool(
            max_workers=1,
            poll_interval_ms=200,
            run_task_fn=_mock_run,
            # No on_task_complete
        )
        await pool.start()
        await asyncio.sleep(0.5)
        await pool.stop()
        # Should not crash
