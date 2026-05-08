"""
Goal context builder — provides a snapshot of a goal's state for Mingjue.

Reads goal metadata, blueprint summary, memory files, trajectory snapshot,
and recent task statuses from the filesystem, returning a structured
``GoalContext`` that Mingjue can use to translate a fuzzy task description
into a concrete, executable action plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vingobot.core.goal_meta import read_goal_meta, GoalMeta
from vingobot.core.manifest import read_manifest
from vingobot.core.trajectory import read_progress_snapshot
from vingobot.core.workspace import get_goal_dir


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GoalContextMeta:
    status: str = "active"
    priority: int = 5
    self_driven: bool = False


@dataclass
class RecentTaskStatus:
    task_id: str
    status: str
    summary_snippet: str


@dataclass
class GoalContext:
    """Snapshot of a goal's current state, built from file-system data."""

    goal_id: str
    meta: GoalContextMeta = field(default_factory=GoalContextMeta)
    blueprint_summary: str = ""
    memory_summary: str = ""
    trajectory_snapshot: str = ""
    recent_task_statuses: list[RecentTaskStatus] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_goal_context(goal_id: str) -> GoalContext | None:
    """Build a ``GoalContext`` from the on-disk goal directory.

    Returns ``None`` when the goal does not exist or has no valid metadata.
    """
    goal_dir = get_goal_dir(goal_id)
    if not goal_dir.is_dir():
        return None

    meta = read_goal_meta(goal_id)
    if meta is None:
        return None

    return GoalContext(
        goal_id=goal_id,
        meta=GoalContextMeta(
            status=meta.status,
            priority=meta.priority,
            self_driven=meta.self_driven.enabled,
        ),
        blueprint_summary=_read_blueprint_summary(meta),
        memory_summary=_read_memory_summary(goal_dir),
        trajectory_snapshot=read_progress_snapshot(goal_id),
        recent_task_statuses=_read_recent_task_statuses(goal_dir),
    )


def refresh_goal_context(goal_id: str) -> GoalContext | None:
    """Re-read the goal context (convenience alias, identical to ``load_goal_context``)."""
    return load_goal_context(goal_id)


# ---------------------------------------------------------------------------
# Internal readers
# ---------------------------------------------------------------------------

def _read_blueprint_summary(meta: GoalMeta) -> str:
    """Return the first 1000 characters of the goal's blueprint."""
    return (meta.blueprint or "")[:1000]


def _read_memory_summary(goal_dir: Path) -> str:
    """Read the last 5 goal-memory entries (both .json and .md).

    For .json entries the key fields are formatted as readable text.
    For .md entries the first 3 non-header lines are used.
    """
    memory_dir = goal_dir / "memory"
    if not memory_dir.is_dir():
        return ""

    try:
        files = sorted(
            p for p in memory_dir.iterdir()
            if p.is_file() and p.suffix in (".json", ".md")
        )[-5:]
    except OSError:
        return ""

    parts: list[str] = []
    for f in files:
        try:
            if f.suffix == ".json":
                snippet = _format_json_memory(f)
            else:
                content = f.read_text(encoding="utf-8")
                lines = [
                    l.strip() for l in content.splitlines()
                    if l.strip() and not l.startswith("#")
                ]
                snippet = " | ".join(lines[:3])[:200]
            if snippet:
                parts.append(f"[{f.name}] {snippet}")
        except OSError:
            continue
    return "\n".join(parts)


def _format_json_memory(path: Path) -> str:
    """Format a JSON goal-memory entry as a short readable snippet."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entry_type = data.get("type", "?")
    if entry_type == "mingjue":
        summary = data.get("summary", "")[:120]
        trigram = data.get("trigram", "?")
        source = data.get("source_type", "?")
        return f"[明觉] {summary} (卦:{trigram}, 来源:{source})"
    elif entry_type == "anqu":
        action = data.get("action", "?")
        pct = data.get("goal_progress_pct")
        pct_str = f"进度:{pct}%" if pct is not None else ""
        accomplished = data.get("what_was_accomplished", "")[:120]
        parts_t = [f"[暗驱] {accomplished}"]
        if pct_str:
            parts_t.append(pct_str)
        parts_t.append(f"决策:{action}")
        return " | ".join(parts_t)
    # Fallback: just show top-level fields
    fields = " | ".join(f"{k}:{str(v)[:60]}" for k, v in data.items() if isinstance(v, str))
    return fields[:200]


def _read_recent_task_statuses(goal_dir: Path) -> list[RecentTaskStatus]:
    tasks_dir = goal_dir / "tasks"
    if not tasks_dir.is_dir():
        return []

    try:
        task_dirs = sorted(
            d for d in tasks_dir.iterdir()
            if d.is_dir()
        )[-2:]
    except OSError:
        return []

    result: list[RecentTaskStatus] = []
    for td in task_dirs:
        mf = read_manifest(str(td))
        if mf is None:
            continue
        snippet = ""
        summary_path = td / "outputs" / "99-summary.md"
        if summary_path.is_file():
            try:
                text = summary_path.read_text(encoding="utf-8")
                first = next(
                    (l.strip() for l in text.splitlines() if l.strip()), ""
                )
                snippet = first[:200]
            except OSError:
                pass
        result.append(RecentTaskStatus(
            task_id=td.name,
            status=mf.status,
            summary_snippet=snippet,
        ))
    return result
