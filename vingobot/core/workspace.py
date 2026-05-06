"""
Workspace manager for the sixiang (六爻) goal-driven architecture.

Manages the .taiji directory structure under the vingobot workspace:
    workspace/.taiji/
        pending/          # Task queue files (*.task)
        cognition/        # Cognitive layer
            skills/       # L1 — Skill definitions
            models/       # L2 — Experience models
            grids/        # L3 — Cognitive grids
        goals/            # Goal directories
            {goal_id}/
                meta.json
                blueprint.md
                memory/   # Goal-level memory files
                tasks/    # Task subdirectories
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from typing import Any


@dataclass
class WorkspacePaths:
    """Resolved paths for the sixiang workspace layout."""

    root: Path
    pending: Path
    cognition: Path
    skills: Path
    models: Path
    grids: Path
    truths: Path  # L4 immutable truths
    goals: Path


_workspace_paths: WorkspacePaths | None = None


TRUTHS_DIRNAME = "truths"  # subdirectory name under cognition/


def init_workspace(root: str | Path | None = None, *, seed: bool = True) -> WorkspacePaths:
    """Initialise the sixiang workspace directory tree.

    Creates the full ``.taiji/`` layout and optionally seeds the cognition
    layers (L1–L4) with default content so the agent can use them immediately.

    Args:
        root: Custom root directory. Defaults to ``<vingobot-workspace>/.taiji``.
        seed: If True, populate L1–L4 cognition layers with defaults.

    Returns:
        A ``WorkspacePaths`` instance pointing to the resolved directories.
    """
    global _workspace_paths

    if root is None:
        from vingobot.config.paths import get_workspace_path
        ws = get_workspace_path()
        root = ws / ".taiji"

    root = Path(root).expanduser().resolve()

    wp = WorkspacePaths(
        root=root,
        pending=root / "pending",
        cognition=root / "cognition",
        skills=root / "cognition" / "skills",
        models=root / "cognition" / "models",
        grids=root / "cognition" / "grids",
        truths=root / "cognition" / TRUTHS_DIRNAME,
        goals=root / "goals",
    )

    dirs = [wp.pending, wp.cognition, wp.skills, wp.models, wp.grids, wp.truths, wp.goals]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    _workspace_paths = wp

    # ── Seed cognition layers if requested ─────────────────────────────
    if seed:
        _seed_l1_skills(wp.skills)
        _seed_l2_models(wp.models)
        _seed_l3_grids(wp.grids)
        _seed_l4_truths(wp.truths)

    return wp


def get_workspace_paths() -> WorkspacePaths:
    """Return the cached workspace paths; initialise lazily if needed."""
    global _workspace_paths
    if _workspace_paths is None:
        return init_workspace()
    return _workspace_paths


# ---------------------------------------------------------------------------
# Cognition seeding helpers
# ---------------------------------------------------------------------------


def _seed_from_templates(templates_subdir: str, target_dir: Path) -> int:
    """Read all files from a bundled templates sub-directory and write them
    idempotently (never overwrite existing files) to *target_dir*.

    Returns the number of files written.
    """
    from importlib.resources import files as pkg_files

    src_dir = pkg_files("vingobot") / "templates" / templates_subdir
    if not src_dir.is_dir():
        return 0

    count = 0
    for f in sorted(src_dir.iterdir()):
        if f.name.startswith(".") or f.name == "__pycache__":
            continue
        target = target_dir / f.name
        if target.exists():
            continue
        content = f.read_bytes()
        target.write_bytes(content)
        count += 1
    return count


def _seed_l1_skills(skills_dir: Path) -> None:
    """Seed L1 skills from bundled templates (if any)."""
    _seed_from_templates("cognition/skills", skills_dir)


def _seed_l2_models(models_dir: Path) -> None:
    """Seed L2 experience models from bundled templates (if any)."""
    _seed_from_templates("cognition/models", models_dir)


def _seed_l3_grids(grids_dir: Path) -> None:
    """Seed L3 cognitive grids from bundled templates — includes MD thinking
    frameworks, legacy taiji JSON grids, and standard GridFile-format JSONs."""
    count = _seed_from_templates("cognition/grids", grids_dir)
    if count:
        logger.info("[L3] 播种了 {} 个认知格栅文件", count)


# ---------------------------------------------------------------------------
# Goal directory helpers
# ---------------------------------------------------------------------------


def get_goal_dir(goal_id: str) -> Path:
    """Resolve the directory for a single goal."""
    return get_workspace_paths().goals / goal_id


def get_task_dir(goal_id: str, task_id: str) -> Path:
    """Resolve the directory for a single task."""
    return get_goal_dir(goal_id) / "tasks" / task_id


def create_task_folder(goal_id: str, task_id: str) -> Path:
    """Create the on-disk directory tree for a task."""
    task_dir = get_task_dir(goal_id, task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "outputs").mkdir(exist_ok=True)
    return task_dir


# ── Characters that cause path nesting on POSIX and Windows ──
_UNSAFE_PATH_CHARS = frozenset("/\\")


def _validate_goal_id(goal_id: str) -> None:
    """Raise ``ValueError`` if *goal_id* contains path-separator characters."""
    if not goal_id:
        raise ValueError("goal_id 不能为空")
    if any(c in _UNSAFE_PATH_CHARS for c in goal_id):
        raise ValueError(
            f"goal_id 包含不安全的路径字符 (/ 或 \\): {goal_id!r}"
        )


def ensure_goal_dir(
    goal_id: str,
    *,
    priority: int = 5,
    name: str | None = None,
    description: str | None = None,
    blueprint: str = "",
) -> Path:
    """Create a goal directory if it does not already exist, populating
    ``meta.json`` (with embedded blueprint), ``memory/`` and ``tasks/``.

    Returns the resolved goal directory path.
    """
    _validate_goal_id(goal_id)
    goal_dir = get_goal_dir(goal_id)
    if goal_dir.exists():
        return goal_dir

    goal_dir.mkdir(parents=True, exist_ok=True)
    (goal_dir / "memory").mkdir(exist_ok=True)
    (goal_dir / "tasks").mkdir(exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    meta: dict[str, Any] = {
        "id": goal_id,
        "name": name or (goal_id if goal_id != "default" else "太极系统"),
        "description": (
            description
            or (goal_id if goal_id != "default" else "默认目标：用户对话与系统自我演化")
        ),
        "status": "active",
        "priority": priority if goal_id != "default" else 1,
        "created_at": now,
        "last_active": now,
        "self_driven": {"enabled": False, "interval_minutes": 30},
        "blueprint": blueprint,
    }
    (goal_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return goal_dir


# ---------------------------------------------------------------------------
# L4 Truth seeding
# ---------------------------------------------------------------------------


def _seed_l4_truths(truths_dir: Path) -> None:
    """Seed the L4 truths directory with default immutable truth files.

    Copies bundled truth definitions from ``vingobot/templates/cognition/truths/``
    into the workspace cognition layer.  Files are only created if they do not
    already exist (existing truths are never overwritten).
    """
    import json
    from importlib.resources import files as pkg_files

    bundled_truths = ["truth_identity.json", "truth_safety.json"]
    for filename in bundled_truths:
        target = truths_dir / filename
        if target.exists():
            continue  # never overwrite existing truths

        try:
            src = pkg_files("vingobot") / "templates" / "cognition" / "truths" / filename
            if src.is_file():
                data = json.loads(src.read_text(encoding="utf-8"))
                data["workspace_seeded_at"] = datetime.now(timezone.utc).isoformat()
                target.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("[L4] 种子真理文件: {}", filename)
        except Exception as exc:
            logger.warning("[L4] 创建真理文件失败 {}: {}", filename, exc)
