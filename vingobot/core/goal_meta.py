"""
Goal metadata — per-goal ``meta.json`` read / write / scan operations.

Each goal directory contains a ``meta.json`` file that tracks the goal's
status, priority, creation timestamp, last-active timestamp,
and self-driven kick configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vingobot.core.workspace import get_goal_dir, get_workspace_paths


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SelfDrivenConfig:
    enabled: bool = False
    interval_minutes: int = 30


@dataclass
class GoalMeta:
    """Persistent metadata for a single goal."""

    id: str
    name: str = ""
    description: str = ""
    status: str = "active"  # active | paused | completed | archived
    priority: int = 5  # 1-10, higher = more important
    created_at: str = ""
    last_active: str = ""
    self_driven: SelfDrivenConfig = field(default_factory=SelfDrivenConfig)

    # ── Control plane fields (written by main-loop, read by WorkerPool) ──
    last_anqu_at: str = ""           # empty = WorkerPool should run Anqu
    warnings: list[str] = field(default_factory=list)  # non-empty = skip, wait for manual
    rounds_completed: int = 0
    auto_managed: bool = True        # False = main-loop scanner skips this goal

    # ── Per-goal anti-pattern table (目标格栅) ──
    known_traps: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalMeta:
        sd = data.get("self_driven") or {}
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            status=str(data.get("status", "active")),
            priority=int(data.get("priority", 5)),
            created_at=str(data.get("created_at", "")),
            last_active=str(data.get("last_active", "")),
            self_driven=SelfDrivenConfig(
                enabled=bool(sd.get("enabled", False)),
                interval_minutes=int(sd.get("interval_minutes", 30)),
            ),
            # Control plane fields (backward compatible defaults)
            last_anqu_at=str(data.get("last_anqu_at", "")),
            warnings=list(data.get("warnings") or []),
            rounds_completed=int(data.get("rounds_completed", 0)),
            auto_managed=bool(data.get("auto_managed", True)),
            # Per-goal anti-pattern table
            known_traps=list(data.get("known_traps") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "self_driven": {
                "enabled": self.self_driven.enabled,
                "interval_minutes": self.self_driven.interval_minutes,
            },
            "last_anqu_at": self.last_anqu_at,
            "warnings": self.warnings,
            "rounds_completed": self.rounds_completed,
            "auto_managed": self.auto_managed,
            "known_traps": self.known_traps,
        }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _meta_path(goal_id: str) -> Path:
    return get_goal_dir(goal_id) / "meta.json"


def read_goal_meta(goal_id: str) -> GoalMeta | None:
    """Read ``meta.json`` for *goal_id*, returning ``None`` if not found."""
    mp = _meta_path(goal_id)
    if not mp.is_file():
        return None
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
        return GoalMeta.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def write_goal_meta(goal_id: str, meta: GoalMeta) -> None:
    """Write (overwrite) the ``meta.json`` for *goal_id*."""
    mp = _meta_path(goal_id)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def update_goal_meta(goal_id: str, **overrides: Any) -> GoalMeta | None:
    """Patch fields of the goal meta and persist. Returns the updated meta."""
    meta = read_goal_meta(goal_id)
    if meta is None:
        return None

    for key, value in overrides.items():
        if hasattr(meta, key):
            setattr(meta, key, value)

    meta.last_active = datetime.now(timezone.utc).isoformat()
    write_goal_meta(goal_id, meta)
    return meta


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------

def _all_goal_ids() -> list[str]:
    goals_root = get_workspace_paths().goals
    if not goals_root.is_dir():
        return []
    return sorted(
        d.name for d in goals_root.iterdir() if d.is_dir()
    )


def scan_active_goals() -> list[GoalMeta]:
    """Return all goals whose status is ``"active"``, ordered by priority desc."""
    metas = [read_goal_meta(gid) for gid in _all_goal_ids()]
    active = [m for m in metas if m is not None and m.status == "active"]
    return sorted(active, key=lambda m: -m.priority)


def get_all_goals() -> list[GoalMeta]:
    """Return every goal with a valid ``meta.json``."""
    metas = [read_goal_meta(gid) for gid in _all_goal_ids()]
    return [m for m in metas if m is not None]
