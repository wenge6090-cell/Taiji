"""Goal progress tracking — master progress file.

Each goal keeps a ``trajectory.json`` that is **overwritten** after each
Anqu decision with the current aggregated goal progress.  This gives
Mingjue, Anqu, and Weaver a single-file, one-read summary of "where
are we now?" without having to comb through multiple memory entries.

Data flow::
    Anqu decision → ``update_goal_progress()`` → trajectory.json
    GoalContext   → ``read_progress_snapshot()`` ← trajectory.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vingobot.core.workspace import get_goal_dir


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TaskProgressEntry:
    """Brief snapshot of one completed task in the progress file."""

    task_id: str = ""
    status: str = ""  # completed | failed | auto_terminated
    summary: str = ""
    round_count: int = 0


@dataclass
class GoalProgress:
    """Master progress state — overwritten after each Anqu decision.

    This is the single source of truth for "how far along is this goal?"
    """

    goal_id: str = ""
    status: str = "active"
    total_tasks: int = 0
    goal_progress_pct: int | None = None
    current_assessment: str = ""
    remaining_work: str = ""
    recent_tasks: list[TaskProgressEntry] = field(default_factory=list)
    last_updated: str = ""


# ---------------------------------------------------------------------------
# File path
# ---------------------------------------------------------------------------


def _progress_path(goal_id: str) -> Path:
    """Return the path to the master progress file (trajectory.json)."""
    return get_goal_dir(goal_id) / "trajectory.json"


# ---------------------------------------------------------------------------
# Read / Write
# ---------------------------------------------------------------------------


def read_goal_progress(goal_id: str) -> GoalProgress:
    """Read the master progress file; returns an empty ``GoalProgress`` on
    missing or corrupt data."""
    tp = _progress_path(goal_id)
    if not tp.is_file():
        return GoalProgress()
    try:
        data = json.loads(tp.read_text(encoding="utf-8"))
        recent_raw = data.get("recent_tasks") or []
        return GoalProgress(
            goal_id=data.get("goal_id", ""),
            status=data.get("status", "active"),
            total_tasks=int(data.get("total_tasks", 0)),
            goal_progress_pct=data.get("goal_progress_pct"),
            current_assessment=str(data.get("current_assessment", "")),
            remaining_work=str(data.get("remaining_work", "")),
            recent_tasks=[
                TaskProgressEntry(
                    task_id=e.get("task_id", ""),
                    status=e.get("status", ""),
                    summary=e.get("summary", ""),
                    round_count=int(e.get("round_count", 0)),
                )
                for e in recent_raw
            ],
            last_updated=str(data.get("last_updated", "")),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return GoalProgress()


def update_goal_progress(
    goal_id: str,
    *,
    task_id: str = "",
    task_status: str = "",
    task_summary: str = "",
    round_count: int = 0,
    goal_progress_pct: int | None = None,
    current_assessment: str = "",
    remaining_work: str = "",
    total_tasks: int = 0,
    goal_status: str = "active",
) -> None:
    """Merge new progress data into the master file and write back.

    Called after each Anqu decision.  ``recent_tasks`` keeps a rolling
    window of the last 10 task entries.
    """
    progress = read_goal_progress(goal_id)

    progress.goal_id = goal_id
    progress.status = goal_status
    if total_tasks > 0:
        progress.total_tasks = total_tasks
    else:
        progress.total_tasks += 1
    if goal_progress_pct is not None:
        progress.goal_progress_pct = goal_progress_pct
    if current_assessment:
        progress.current_assessment = current_assessment[:400]
    if remaining_work:
        progress.remaining_work = remaining_work[:300]

    # Rolling window of recent tasks
    if task_id:
        progress.recent_tasks.append(
            TaskProgressEntry(
                task_id=task_id,
                status=task_status,
                summary=(task_summary or "")[:300],
                round_count=round_count,
            )
        )
        if len(progress.recent_tasks) > 10:
            progress.recent_tasks = progress.recent_tasks[-10:]

    progress.last_updated = datetime.now(timezone.utc).isoformat()

    tp = _progress_path(goal_id)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(
        json.dumps(_progress_to_dict(progress), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Summary for LLM consumption
# ---------------------------------------------------------------------------


def read_progress_snapshot(goal_id: str) -> str:
    """Return a concise, readable one-line progress summary.

    This is what gets injected into Mingjue / Anqu / Weaver
    system prompts as ``trajectory_snapshot``.
    """
    progress = read_goal_progress(goal_id)
    parts: list[str] = []
    if progress.goal_progress_pct is not None:
        parts.append(f"总进度: {progress.goal_progress_pct}%")
    completed = sum(1 for t in progress.recent_tasks if t.status == "completed")
    if progress.recent_tasks:
        parts.append(f"已完成 {completed}/{len(progress.recent_tasks)} 个近期任务")
    if progress.current_assessment:
        parts.append(f"最新评估: {progress.current_assessment[:200]}")
    if progress.status != "active":
        parts.append(f"状态: {progress.status}")
    return " | ".join(parts) if parts else ""


# ── Backward-compat alias ─────────────────────────────────────────
read_trajectory_snapshot = read_progress_snapshot


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _progress_to_dict(progress: GoalProgress) -> dict[str, Any]:
    return {
        "goal_id": progress.goal_id,
        "status": progress.status,
        "total_tasks": progress.total_tasks,
        "goal_progress_pct": progress.goal_progress_pct,
        "current_assessment": progress.current_assessment,
        "remaining_work": progress.remaining_work,
        "recent_tasks": [
            {
                "task_id": e.task_id,
                "status": e.status,
                "summary": e.summary,
                "round_count": e.round_count,
            }
            for e in progress.recent_tasks
        ],
        "last_updated": progress.last_updated,
    }
