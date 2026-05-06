"""
Task manifest — per-task ``manifest.json`` read / write operations.

Each task directory contains a ``manifest.json`` that records the task's
identity, status, creation time, and round count for the sixiang loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TaskManifest:
    """Lightweight manifest stored alongside every task's working directory."""

    task_id: str = ""
    goal_id: str = ""
    description: str = ""
    status: str = "pending"  # pending | running | completed | failed | archived
    created_at: str = ""
    updated_at: str = ""
    round_count: int = 0
    max_rounds: int = 30
    priority: int = 5
    source: str = "user"  # user | self_driven | system

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskManifest:
        return cls(
            task_id=str(data.get("task_id", "")),
            goal_id=str(data.get("goal_id", "")),
            description=str(data.get("description", "")),
            status=str(data.get("status", "pending")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            round_count=int(data.get("round_count", 0)),
            max_rounds=int(data.get("max_rounds", 30)),
            priority=int(data.get("priority", 5)),
            source=str(data.get("source", "user")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "round_count": self.round_count,
            "max_rounds": self.max_rounds,
            "priority": self.priority,
            "source": self.source,
        }


def create_manifest(
    task_dir: str | Path,
    task_id: str,
    goal_id: str,
    description: str = "",
    *,
    priority: int = 5,
    source: str = "user",
    max_rounds: int = 30,
) -> TaskManifest:
    """Create and persist a fresh ``TaskManifest`` in *task_dir*."""
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    mf = TaskManifest(
        task_id=task_id,
        goal_id=goal_id,
        description=description,
        status="pending",
        created_at=now,
        updated_at=now,
        round_count=0,
        max_rounds=max_rounds,
        priority=priority,
        source=source,
    )
    _write(task_dir, mf)
    return mf


def read_manifest(task_dir: str | Path) -> TaskManifest | None:
    """Read an existing ``manifest.json``; return ``None`` if missing."""
    path = Path(task_dir) / "manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TaskManifest.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def update_manifest_status(
    task_dir: str | Path,
    status: str,
    *,
    round_count: int | None = None,
) -> TaskManifest | None:
    """Patch the manifest status (and optionally round_count) and persist."""
    mf = read_manifest(task_dir)
    if mf is None:
        return None
    mf.status = status
    mf.updated_at = datetime.now(timezone.utc).isoformat()
    if round_count is not None:
        mf.round_count = round_count
    _write(Path(task_dir), mf)
    return mf


def _write(task_dir: Path, mf: TaskManifest) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "manifest.json").write_text(
        json.dumps(mf.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
