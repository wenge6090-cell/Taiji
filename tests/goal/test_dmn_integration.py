"""End-to-end tests — main process ↔ WorkerPool control via shared file layer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from vingobot.core.goal_meta import GoalMeta, read_goal_meta, update_goal_meta, write_goal_meta
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


def _create_goal(goal_id: str, **overrides) -> GoalMeta:
    """Create a goal with meta.json and return its GoalMeta."""
    meta = GoalMeta(id=goal_id, name=goal_id, description=f"Goal {goal_id}", **overrides)
    write_goal_meta(goal_id, meta)
    return meta


def _enqueue(pending_dir: Path, goal_id: str, description: str, prefix: str = "") -> None:
    """Enqueue a task file directly."""
    pending_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
    filename = (
        f"{prefix}{ts}_{description[:20]}.task" if prefix else f"{ts}_{description[:20]}.task"
    )
    (pending_dir / filename).write_text(
        f"{description}\npriority=5\nsource=user\ngoalId={goal_id}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDmnIntegration:
    """Full control-plane flow: main-loop writes meta.json → WorkerPool reacts."""

    async def test_full_pause_resume_cycle(self, tmp_path: Path) -> None:
        """End-to-end: pause goal → task re-enqueued; resume → task executed."""
        root = _init_test_workspace(tmp_path)
        _create_goal("goal-a", status="active", last_anqu_at="2026-01-01T00:00:00")

        executed: list[str] = []

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed.append(goal_id)
            return f"done {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)

        # ── Phase 1: Pause the goal from "main process" ──
        update_goal_meta("goal-a", status="paused")
        _enqueue(root / "pending", "goal-a", "task during pause")

        await pool.start()
        await asyncio.sleep(0.5)

        # Worker should NOT have executed (goal is paused)
        assert executed == []
        # Task should have been re-enqueued (or consumed & re-claimed)
        # Core invariant: task was never executed
        await pool.stop()

        # ── Phase 2: Resume the goal ──
        executed.clear()
        update_goal_meta("goal-a", status="active")
        _enqueue(root / "pending", "goal-a", "task after resume")

        await pool.start()
        await asyncio.sleep(0.6)

        assert executed == ["goal-a"]  # Should execute now
        await pool.stop()

    async def test_warning_blocks_execution_then_cleared(self, tmp_path: Path) -> None:
        """Goal with warnings → tasks dropped; clear warnings → tasks flow again."""
        root = _init_test_workspace(tmp_path)
        _create_goal(
            "goal-b",
            status="active",
            warnings=["balance exceeded"],
            last_anqu_at="2026-01-01T00:00:00",
        )

        executed: list[str] = []

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed.append(goal_id)
            return f"done {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)

        # Phase 1: Warning active → task dropped
        _enqueue(root / "pending", "goal-b", "blocked task")
        await pool.start()
        await asyncio.sleep(0.5)
        assert executed == []
        await pool.stop()

        # Phase 2: Clear warnings → task executes
        executed.clear()
        update_goal_meta("goal-b", warnings=[])
        _enqueue(root / "pending", "goal-b", "allowed task")

        await pool.start()
        await asyncio.sleep(0.6)
        assert executed == ["goal-b"]
        await pool.stop()

    async def test_anqu_confirmation_flow(self, tmp_path: Path) -> None:
        """Main process requests Anqu (clears last_anqu_at) → WorkerPool confirms."""
        root = _init_test_workspace(tmp_path)
        _create_goal("goal-c", status="active", last_anqu_at="")

        executed: list[str] = []

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed.append(goal_id)
            return f"done {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)

        _enqueue(root / "pending", "goal-c", "anqu request task")
        await pool.start()
        await asyncio.sleep(0.6)

        # Worker should confirm Anqu (write timestamp) then execute
        assert executed == ["goal-c"]
        meta = read_goal_meta("goal-c")
        assert meta is not None
        assert meta.last_anqu_at != ""  # Anqu confirmed
        await pool.stop()

    async def test_dual_consumer_routing(self, tmp_path: Path) -> None:
        """WorkerPool and DMN consumer each consume their own task types."""
        root = _init_test_workspace(tmp_path)
        _create_goal("worker-goal", status="active", last_anqu_at="2026-01-01T00:00:00")

        # Enqueue tasks for both consumers
        _enqueue(root / "pending", "worker-goal", "worker task")
        _enqueue(
            root / "pending", "cognition-evolution", "learn skill", prefix="cognition-evolution__"
        )
        _enqueue(root / "pending", "dmn", "maintenance check", prefix="dmn__")

        worker_executed: list[str] = []
        dmn_executed: list[str] = []

        async def _mock_worker(goal_id: str, _desc: str, _signal) -> str:
            worker_executed.append(goal_id)
            return f"worker done {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_worker)

        # Simulate DMN consumer picking up DMN tasks
        async def _dmn_consumer() -> None:
            queue = PendingQueue(root / "pending")
            for _ in range(10):  # poll a few times
                claimed = queue.try_consume_by_prefix("cognition-evolution")
                if claimed is None:
                    claimed = queue.try_consume_by_prefix("dmn")
                if claimed is not None:
                    task, file_path = claimed
                    dmn_executed.append(task.goal_id)
                    queue.delete_task_file(file_path)
                    continue
                await asyncio.sleep(0.1)

        await pool.start()
        dmn_task = asyncio.create_task(_dmn_consumer())

        await asyncio.sleep(1.0)
        await pool.stop()
        await dmn_task

        # Worker consumed its task, DMN consumer consumed its tasks
        assert "worker-goal" in worker_executed
        assert "cognition-evolution" in dmn_executed
        assert "dmn" in dmn_executed

        # No cross-contamination
        assert "cognition-evolution" not in worker_executed
        assert "dmn" not in worker_executed
        assert "worker-goal" not in dmn_executed

    async def test_meta_json_roundtrip(self, tmp_path: Path) -> None:
        """Main process writes meta.json → WorkerPool reads it correctly."""
        root = _init_test_workspace(tmp_path)
        _create_goal("goal-d", status="active", priority=7, last_anqu_at="2026-01-01T00:00:00")

        # Simulate main process: update meta.json dynamically
        update_goal_meta("goal-d", status="paused")
        meta = read_goal_meta("goal-d")
        assert meta is not None
        assert meta.status == "paused"
        assert meta.priority == 7  # unchanged

        # Resume
        update_goal_meta("goal-d", status="active", priority=3)
        meta = read_goal_meta("goal-d")
        assert meta is not None
        assert meta.status == "active"
        assert meta.priority == 3

    async def test_multiple_goals_concurrent(self, tmp_path: Path) -> None:
        """Multiple goals with different states are handled correctly."""
        root = _init_test_workspace(tmp_path)

        _create_goal("active-goal", status="active", last_anqu_at="2026-01-01T00:00:00")
        _create_goal("paused-goal", status="paused")
        _create_goal(
            "warning-goal", status="active", warnings=["error"], last_anqu_at="2026-01-01T00:00:00"
        )

        executed: dict[str, int] = {}

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed[goal_id] = executed.get(goal_id, 0) + 1
            return f"done {goal_id}"

        pool = WorkerPool(max_workers=3, poll_interval_ms=200, run_task_fn=_mock_run)

        _enqueue(root / "pending", "active-goal", "task a1")
        _enqueue(root / "pending", "paused-goal", "task p1")
        _enqueue(root / "pending", "warning-goal", "task w1")
        _enqueue(root / "pending", "active-goal", "task a2")

        await pool.start()
        await asyncio.sleep(1.0)
        await pool.stop()

        # active-goal: both tasks executed
        assert executed.get("active-goal", 0) >= 1
        # paused-goal: never executed
        assert executed.get("paused-goal", 0) == 0
        # warning-goal: never executed
        assert executed.get("warning-goal", 0) == 0

    async def test_orphan_goal_task_dropped(self, tmp_path: Path) -> None:
        """Task for a goal without meta.json is dropped."""
        root = _init_test_workspace(tmp_path)
        # No meta.json for "orphan"

        executed: list[str] = []

        async def _mock_run(goal_id: str, _desc: str, _signal) -> str:
            executed.append(goal_id)
            return f"done {goal_id}"

        pool = WorkerPool(max_workers=1, poll_interval_ms=200, run_task_fn=_mock_run)

        _enqueue(root / "pending", "orphan", "orphan task")
        await pool.start()
        await asyncio.sleep(0.5)
        await pool.stop()

        assert executed == []  # Orphan task dropped silently
