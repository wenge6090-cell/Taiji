"""
File-system-based pending task queue.

Queue items are serialised as flat ``.task`` files under ``pending/``.
Atomic consumption is achieved via ``os.replace()`` rename to
``.processing``, making the queue safe for multiple concurrent workers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from vingobot.core.workspace import get_workspace_paths


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

TaskSource = Literal["user", "self_driven", "vassal_report", "system"]


@dataclass
class PendingTask:
    """A single pending task entry from the file-system queue."""

    goal_id: str
    description: str
    id: str = ""
    priority: int = 5
    source: TaskSource = "user"
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    filename_prefix: str = ""

    def to_file_content(self) -> str:
        """Serialise the task to the flat ``.task`` file format."""
        lines = [self.description]
        lines.append(f"priority={self.priority}")
        lines.append(f"source={self.source}")
        lines.append(f"goalId={self.goal_id}")
        if self.metadata:
            lines.append(f"metadata={json.dumps(self.metadata, ensure_ascii=False)}")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_file(cls, filepath: Path, filename: str) -> PendingTask | None:
        """Parse a ``.task`` (or ``.processing``) file back to a ``PendingTask``."""
        try:
            content = filepath.read_text(encoding="utf-8")
        except OSError:
            return None

        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if not lines:
            return None

        description = lines[0]
        meta: dict[str, str] = {}
        for line in lines[1:]:
            m = re.match(r"^(\w+)=(.+)$", line)
            if m:
                meta[m.group(1)] = m.group(2)

        ts_match = re.match(r"^(\d{8}T\d{6})", filename)
        created_at = ts_match.group(1) if ts_match else ""

        parsed_meta: dict[str, Any] = {}
        if "metadata" in meta:
            try:
                parsed_meta = json.loads(meta["metadata"])
            except json.JSONDecodeError:
                pass

        return cls(
            id=filename.replace(".task", "").replace(".processing", ""),
            goal_id=meta.get("goalId", "default"),
            description=description,
            priority=int(meta.get("priority", 5)),
            source=meta.get("source", "user"),  # type: ignore[arg-type]
            created_at=created_at,
            metadata=parsed_meta,
        )


# ---------------------------------------------------------------------------
# PendingQueue
# ---------------------------------------------------------------------------

class PendingQueue:
    """File-system queue backed by ``pending/`` directory.

    Supports atomic consumption via ``.task`` → ``.processing`` rename,
    which is safe for multiple concurrent workers on the same filesystem.
    """

    def __init__(self, pending_dir: str | Path | None = None) -> None:
        if pending_dir is None:
            pending_dir = get_workspace_paths().pending
        self.pending_dir = Path(pending_dir)
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(
        self,
        task: PendingTask,
    ) -> str:
        """Write a new task to a ``.task`` file. Returns the filename stem.

        Uses atomic write (temp file + ``os.replace``) to prevent
        partial writes and accidental overwrites of concurrently-created
        files.
        """
        prefix = task.filename_prefix or ""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_desc = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", "-", task.description[:30])
        filename = f"{prefix}{ts}_{safe_desc}.task"
        filepath = self.pending_dir / filename

        try:
            # Write to temp file first, then atomically rename
            tmp_path = filepath.with_suffix(".tmp")
            tmp_path.write_text(task.to_file_content(), encoding="utf-8")
            os.replace(tmp_path, filepath)
            return filename
        except OSError:
            # Clean up temp file if rename failed
            tmp_path.unlink(missing_ok=True)
            # Append a microsecond discriminator
            import time

            time.sleep(0.002)
            ts2 = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            filename2 = f"{prefix}{ts2}_{safe_desc}.task"
            filepath2 = self.pending_dir / filename2
            tmp_path2 = filepath2.with_suffix(".tmp")
            tmp_path2.write_text(task.to_file_content(), encoding="utf-8")
            os.replace(tmp_path2, filepath2)
            return filename2

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def _scan_task_files(self, suffix: str = ".task") -> list[Path]:
        """Return ``.task`` or ``.processing`` files sorted by name."""
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            p for p in self.pending_dir.iterdir()
            if p.is_file() and p.suffix == suffix
        )
        return files

    def scan_pending(self) -> list[PendingTask]:
        """Return all enqueued (non-processed) tasks."""
        result: list[PendingTask] = []
        for fp in self._scan_task_files(".task"):
            task = PendingTask.from_file(fp, fp.name)
            if task is not None:
                result.append(task)
        return result

    # ------------------------------------------------------------------
    # Atomic consume
    # ------------------------------------------------------------------

    def try_consume_next(
        self,
        *,
        exclude_prefixes: list[str] | None = None,
    ) -> tuple[PendingTask, Path] | None:
        """Atomically claim the next task by renaming ``.task`` → ``.processing``.

        Returns ``(task, processing_path)`` on success, or ``None`` if the
        queue is empty or all tasks are currently being processed by other
        workers.

        Args:
            exclude_prefixes: Optional list of filename prefixes to skip.
                Tasks whose filename starts with any of these prefixes are
                left for other consumers (e.g. DMN).
        """
        for fp in self._scan_task_files(".task"):
            if exclude_prefixes:
                if any(fp.name.startswith(p) for p in exclude_prefixes):
                    continue
            processing_path = fp.with_suffix(".processing")
            try:
                fp.rename(processing_path)
            except OSError:
                # Another worker already claimed it
                continue
            task = PendingTask.from_file(processing_path, processing_path.name)
            if task is not None:
                return task, processing_path
            # Malformed file — clean up
            processing_path.unlink(missing_ok=True)
        return None

    def has_pending_by_prefix(self, prefix: str) -> bool:
        """Check whether any ``.task`` file starts with *prefix*.

        Lightweight check — does NOT claim or rename any file.
        Useful for idle-detection logic before attempting a consume.
        """
        for fp in self._scan_task_files(".task"):
            if fp.name.startswith(prefix):
                return True
        return False

    def try_consume_by_prefix(self, prefix: str) -> tuple[PendingTask, Path] | None:
        """Atomically claim a task whose filename starts with *prefix*.

        Only scans ``.task`` files matching ``{prefix}*``.  Other consumers
        (that use ``try_consume_next`` without prefix filtering) are not
        affected.

        Returns ``(task, processing_path)`` on success, or ``None``.
        """
        for fp in self._scan_task_files(".task"):
            if not fp.name.startswith(prefix):
                continue
            processing_path = fp.with_suffix(".processing")
            try:
                fp.rename(processing_path)
            except OSError:
                continue
            task = PendingTask.from_file(processing_path, processing_path.name)
            if task is not None:
                return task, processing_path
            processing_path.unlink(missing_ok=True)
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_task_file(self, path: str | Path) -> None:
        """Remove a ``.processing`` or ``.task`` file after completion."""
        path = Path(path)
        # Also clean up the original .task if it exists (belt-and-suspenders)
        task_path = path.with_suffix(".task")
        task_path.unlink(missing_ok=True)
        path.unlink(missing_ok=True)

    def delete_tasks_for_goal(self, goal_id: str) -> int:
        """Delete all ``.task`` and ``.processing`` files for *goal_id*.

        Returns the number of files deleted.
        """
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        deleted = 0
        for sfx in (".task", ".processing"):
            for fp in list(self.pending_dir.glob(f"*{sfx}")):
                try:
                    task = PendingTask.from_file(fp, fp.name)
                except Exception:
                    logger.debug("[队列] 解析任务失败: {}", fp)
                    continue
                if task is not None and task.goal_id == goal_id:
                    try:
                        fp.unlink()
                        deleted += 1
                    except OSError:
                        logger.debug("[队列] 删除任务失败: {}", fp)
        return deleted

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    def has_duplicate(
        self,
        description: str,
        goal_id: str | None = None,
        threshold: float = 0.25,
    ) -> bool:
        """Check for a semantically-similar pending task using
        character-level 2-gram Jaccard similarity.
        """
        new_grams = self._char_ngrams(description, 2)
        if not new_grams:
            return False

        for task in self.scan_pending():
            if goal_id is not None and task.goal_id != goal_id:
                continue
            existing = self._char_ngrams(task.description, 2)
            if not existing:
                continue
            intersection = new_grams & existing
            union = new_grams | existing
            if union and len(intersection) / len(union) >= threshold:
                return True
        return False

    @staticmethod
    def _char_ngrams(text: str, n: int) -> set[str]:
        normalised = re.sub(r"[^\w\u4e00-\u9fa5]", "", text.lower())
        padded = "#" * (n - 1) + normalised + "$" * (n - 1)
        return {padded[i:i + n] for i in range(len(padded) - n + 1)}

    # ------------------------------------------------------------------
    # Orphan cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def cleanup_orphan_tasks(
        timeout_ms: int = 30 * 60 * 1000,
    ) -> int:
        """Scan all goal task directories and archive tasks stuck in
        ``pending`` status longer than *timeout_ms*.

        Also recovers orphaned ``.processing`` files in the pending
        directory — these are tasks that were claimed by a worker that
        crashed before completing or cleaning up.

        A task is an "orphan" when its ``manifest.json`` was created but
        the task was never picked up by a worker (e.g. the process crashed
        before the pending file was written, or the worker failed before
        updating the status).  These accumulate on disk and should be
        cleaned up periodically.

        Returns the number of tasks archived.
        """
        from vingobot.core.manifest import read_manifest, update_manifest_status
        from vingobot.core.workspace import get_workspace_paths

        goals_dir = get_workspace_paths().goals
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        cleaned = 0

        # ── Phase 1: Clean orphan .processing files in pending/ ──────
        pending_dir = get_workspace_paths().pending
        if pending_dir.is_dir():
            for fp in sorted(pending_dir.iterdir()):
                if not fp.is_file() or fp.suffix != ".processing":
                    continue
                try:
                    mtime_ms = int(fp.stat().st_mtime * 1000)
                except OSError:
                    continue
                if now_ms - mtime_ms >= timeout_ms:
                    # Atomically rename back to .task for retry
                    task_path = fp.with_suffix(".task")
                    try:
                        fp.rename(task_path)
                        logger.info("[清理] 恢复孤儿处理文件: {}", task_path.name)
                        cleaned += 1
                    except OSError:
                        fp.unlink(missing_ok=True)
                        cleaned += 1

        # ── Phase 2: Clean orphan manifests ──────────────────────────
        if not goals_dir.is_dir():
            if cleaned > 0:
                logger.info("[清理] 共清理 {} 个孤儿任务", cleaned)
            return cleaned

        for goal_dir in sorted(goals_dir.iterdir()):
            if not goal_dir.is_dir():
                continue
            tasks_dir = goal_dir / "tasks"
            if not tasks_dir.is_dir():
                continue

            for task_dir in sorted(tasks_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                manifest = read_manifest(task_dir)
                if manifest is None:
                    continue
                if manifest.status != "pending":
                    continue

                # Parse creation time
                try:
                    created = datetime.fromisoformat(manifest.created_at)
                    created_ms = int(created.timestamp() * 1000)
                except (ValueError, TypeError):
                    continue

                if now_ms - created_ms >= timeout_ms:
                    update_manifest_status(task_dir, "archived")
                    logger.info(
                        "[清理] 孤儿任务已归档: {} (目标: {}, 创建于 {})",
                        manifest.task_id,
                        manifest.goal_id,
                        manifest.created_at,
                    )
                    cleaned += 1

        if cleaned > 0:
            logger.info("[清理] 共清理 {} 个孤儿任务", cleaned)
        return cleaned

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        return len(self.scan_pending())

    def list_tasks(self) -> list[PendingTask]:
        """Return all pending tasks (alias for ``scan_pending``)."""
        return self.scan_pending()
