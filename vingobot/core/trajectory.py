"""
Goal trajectory — cross-task progress tracking.

Each goal keeps a ``trajectory.json`` file that records completed / failed
tasks, key outputs, blockers, and an auto-generated progress snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vingobot.core.workspace import get_goal_dir


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryEntry:
    """One task's outcome recorded in the goal trajectory."""

    task_id: str = ""
    timestamp: str = ""
    status: str = "completed"  # completed | failed
    summary: str = ""
    key_outputs: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    round_count: int = 0


@dataclass
class GoalTrajectory:
    """Full trajectory file content for a goal."""

    recent: list[TrajectoryEntry] = field(default_factory=list)
    history_summary: str = ""
    progress_snapshot: str = ""


# ---------------------------------------------------------------------------
# File path
# ---------------------------------------------------------------------------

def _trajectory_path(goal_id: str) -> Path:
    return get_goal_dir(goal_id) / "trajectory.json"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def ensure_trajectory_file(goal_id: str) -> None:
    """Create an empty ``trajectory.json`` if one does not exist."""
    tp = _trajectory_path(goal_id)
    if tp.is_file():
        return
    tp.parent.mkdir(parents=True, exist_ok=True)
    trajectory = GoalTrajectory()
    tp.write_text(
        json.dumps(_trajectory_to_dict(trajectory), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_trajectory(goal_id: str) -> GoalTrajectory:
    """Read the full trajectory; returns empty on missing/corrupt file."""
    tp = _trajectory_path(goal_id)
    if not tp.is_file():
        return GoalTrajectory()
    try:
        data = json.loads(tp.read_text(encoding="utf-8"))
        return _dict_to_trajectory(data)
    except (json.JSONDecodeError, OSError):
        return GoalTrajectory()


def read_trajectory_snapshot(goal_id: str) -> str:
    """Return the progress snapshot string (convenience shim)."""
    return read_trajectory(goal_id).progress_snapshot


def append_trajectory_entry(goal_id: str, entry: TrajectoryEntry) -> None:
    """Append a new entry to the goal trajectory, capping recent at 5."""
    ensure_trajectory_file(goal_id)
    traj = read_trajectory(goal_id)

    # Cap recent entries
    traj.recent.append(entry)
    if len(traj.recent) > 5:
        overflow = traj.recent[:-5]
        # Summarise overflow into history_summary
        overflow_text = "; ".join(
            f"{e.task_id}({e.status})" for e in overflow
        )
        traj.history_summary = (
            f"{traj.history_summary}; {overflow_text}".strip("; ")
        )
        traj.recent = traj.recent[-5:]

    # Auto-generate progress snapshot
    completed = sum(1 for e in traj.recent if e.status == "completed")
    traj.progress_snapshot = (
        f"最近 {len(traj.recent)} 个任务中 {completed} 个已完成"
    )

    _write_trajectory(goal_id, traj)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _trajectory_to_dict(traj: GoalTrajectory) -> dict[str, Any]:
    return {
        "recent": [asdict(e) for e in traj.recent],
        "history_summary": traj.history_summary,
        "progress_snapshot": traj.progress_snapshot,
    }


def _dict_to_trajectory(data: dict[str, Any]) -> GoalTrajectory:
    recent_raw = data.get("recent") or []
    return GoalTrajectory(
        recent=[
            TrajectoryEntry(
                task_id=e.get("task_id", ""),
                timestamp=e.get("timestamp", ""),
                status=e.get("status", "completed"),
                summary=e.get("summary", ""),
                key_outputs=list(e.get("key_outputs", [])),
                blockers=list(e.get("blockers", [])),
                suggestions=list(e.get("suggestions", [])),
                round_count=int(e.get("round_count", 0)),
            )
            for e in recent_raw
        ],
        history_summary=str(data.get("history_summary", "")),
        progress_snapshot=str(data.get("progress_snapshot", "")),
    )


def _write_trajectory(goal_id: str, traj: GoalTrajectory) -> None:
    tp = _trajectory_path(goal_id)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(
        json.dumps(_trajectory_to_dict(traj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
