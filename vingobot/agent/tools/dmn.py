"""
DMN (Default Mode Network) — cognitive evolution manager.

Lets the agent inspect and control the DMN consumer: the background
loop that polls for ``cognition-evolution__*`` and ``dmn__*`` tasks
and processes them via LLM → cognition_evolver.

The DMN consumer is an independent async loop in AgentLoop (NOT part
of the sixiang WorkerPool).  It runs alongside the main message loop,
consuming cognitive-evolution tasks and maintaining the cognition
library (skills / models / grids).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from vingobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from vingobot.agent.loop import AgentLoop


# ---------------------------------------------------------------------------
# Action names
# ---------------------------------------------------------------------------

_ACTION_STATUS = "status"
_ACTION_START = "start"
_ACTION_STOP = "stop"
_ACTION_TRIGGER = "trigger"

_VALID_ACTIONS = frozenset({_ACTION_STATUS, _ACTION_START, _ACTION_STOP, _ACTION_TRIGGER})


class DmnTool(Tool):
    """Control the DMN (Default Mode Network) cognitive-evolution consumer.

    The DMN consumer is a background loop that polls for
    ``cognition-evolution__*`` and ``dmn__*`` tasks, processes each
    via LLM, and creates/updates cognition assets (L1 skills, L2 models,
    L3 grids).  It runs independently of the sixiang goal-execution pool.

    Key use case: when a DMN task blocks dialogue mode for too long,
    use **stop** to cancel the current DMN processing and regain control.

    Actions:

    * **status** — DMN dashboard: consumer running/stopped, cognition
      library health, pending evolution task count.

    * **start** — Start the DMN consumer if it is not running.

    * **stop** — Stop (cancel) the DMN consumer.  The currently-processing
      task is cancelled; the task file is cleaned up so it is not
      re-consumed on restart.

    * **trigger** — Manually enqueue a cognition-evolution task.
      The DMN consumer (if running) picks it up and the LLM evaluates
      whether to create or update skills, models, or grids.
    """

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop

    # ------------------------------------------------------------------
    # Tool metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "dmn"

    @property
    def description(self) -> str:
        return (
            "Control the DMN (Default Mode Network) cognitive-evolution consumer. "
            "Use **status** to inspect DMN state and cognition library health. "
            "Use **start**/**stop** to control the background evolution loop. "
            "Use **trigger** to manually enqueue a cognition-evolution task "
            "(e.g. 'check whether we need a new HTTP skill'). "
            "The DMN consumer runs independently of the sixiang goal-execution pool."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "DMN action to perform.",
                    "enum": sorted(_VALID_ACTIONS),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "For **trigger**: natural-language description of the "
                        "cognitive-evolution task (e.g. 'check HTTP client coverage')."
                    ),
                },
            },
            "required": ["action"],
        }

    # ------------------------------------------------------------------
    # Execute dispatch
    # ------------------------------------------------------------------

    async def execute(self, action: str, **kwargs: Any) -> str:
        if action not in _VALID_ACTIONS:
            return f"未知 DMN 动作: {action}。有效动作: {', '.join(sorted(_VALID_ACTIONS))}"

        try:
            if action == _ACTION_STATUS:
                return self._do_status()
            if action == _ACTION_START:
                return self._do_start()
            if action == _ACTION_STOP:
                return self._do_stop()
            if action == _ACTION_TRIGGER:
                return self._do_trigger(kwargs)
        except Exception as exc:
            logger.exception("[DMN工具] 执行 '{}' 失败", action)
            return f"DMN 错误 ({action}): {exc}"

        return f"未实现的 DMN 动作: {action}"

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def _do_status(self) -> str:
        lines: list[str] = ["# 🧠 DMN 认知网络状态\n"]

        # ── Consumer state ─────────────────────────────────────
        lines.append("## DMN 消费者")
        if self._dmn_running():
            lines.append("- 状态: 🟢 运行中")
        else:
            lines.append("- 状态: ⚪ 已停止")

        # ── Pending evolution tasks ────────────────────────────
        lines.append("")
        lines.append("## 待处理认知演化任务")
        self._append_evolution_tasks(lines)

        # ── Currently-processing task ───────────────────────────
        lines.append("")
        lines.append("## 正在处理")
        self._append_current_task(lines)

        # ── Cognition library health ────────────────────────────
        lines.append("")
        lines.append("## 认知库健康度")
        self._append_cognition_health(lines)

        # ── Recent cognition changes ────────────────────────────
        lines.append("")
        lines.append("## 最近认知资产变动")
        self._append_recent_changes(lines)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # start / stop
    # ------------------------------------------------------------------

    def _do_start(self) -> str:
        if self._dmn_running():
            return "DMN 消费者已在运行中。"

        if not self._loop._running:
            return "错误: AgentLoop 未运行，无法启动 DMN 消费者。"

        try:
            self._loop._dmn_consumer_task = asyncio.create_task(
                self._loop._run_dmn_consumer(),
                name="dmn-consumer",
            )
            logger.info("[DMN工具] DMN 消费者已手动启动")
            return "✅ DMN 消费者已启动，开始消费认知演化任务。"
        except Exception as exc:
            return f"启动 DMN 消费者失败: {exc}"

    def _do_stop(self) -> str:
        if not self._dmn_running():
            return "DMN 消费者未在运行。"

        task = self._loop._dmn_consumer_task
        try:
            task.cancel()  # type: ignore[union-attr]
            logger.info("[DMN工具] DMN 消费者已手动停止")
            return (
                "⏹ DMN 消费者已发出停止信号。\n"
                "   当前正在处理的认知演化任务将被取消，"
                "任务文件已清理不会重复消费。\n"
                "   用 `dmn start` 重新启动。"
            )
        except Exception as exc:
            return f"停止 DMN 消费者失败: {exc}"

    # ------------------------------------------------------------------
    # trigger
    # ------------------------------------------------------------------

    def _do_trigger(self, kwargs: dict[str, Any]) -> str:
        try:
            from vingobot.core.workspace import get_workspace_paths

            description = str(kwargs.get("description", "")).strip()
            if not description:
                description = "手动触发的认知演化任务"

            wp = get_workspace_paths()
            pending_dir = wp.root / "pending"

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_desc = "".join(c if c.isalnum() or c in "-_" else "_" for c in description[:40])
            safe_prefix = f"cognition-evolution__05__{safe_desc}"
            filename = f"{safe_prefix}__{ts}.task"

            filepath = pending_dir / filename
            content = f"{description}\npriority=5\nsource=user\ngoalId=cognition-evolution\n"
            filepath.write_text(content, encoding="utf-8")

            dmn_note = (
                "DMN 消费者将自动拾取此任务。"
                if self._dmn_running()
                else "⚠ DMN 消费者未运行，任务将在启动后被消费。"
            )
            return (
                f"🧠 已入队认知演化任务。\n"
                f"   文件: {filename}\n"
                f"   描述: {description[:120]}\n"
                f"   {dmn_note}"
            )

        except Exception as exc:
            return f"入队认知演化任务失败: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dmn_running(self) -> bool:
        task = self._loop._dmn_consumer_task
        return task is not None and not task.done()

    # ------------------------------------------------------------------
    # Status helpers (static — no loop state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _append_evolution_tasks(lines: list[str]) -> None:
        """Count pending cognition-evolution and dmn tasks."""
        try:
            from vingobot.core.workspace import get_workspace_paths

            wp = get_workspace_paths()
            pending_dir = wp.root / "pending"
            if not pending_dir.is_dir():
                lines.append("- (pending 目录不存在)")
                return

            task_files = sorted(pending_dir.glob("*.task"))
            evo_tasks = [
                f
                for f in task_files
                if f.name.startswith("cognition-evolution__") or f.name.startswith("dmn__")
            ]
            if not evo_tasks:
                lines.append("- 无待处理任务")
                return

            lines.append(f"- 待处理: {len(evo_tasks)} 个")
            for tf in evo_tasks[:10]:
                content = tf.read_text(encoding="utf-8")
                first_line = content.split("\n")[0][:80]
                lines.append(f"  - `{tf.name}`: {first_line}")
            if len(evo_tasks) > 10:
                lines.append(f"  - ... 还有 {len(evo_tasks) - 10} 个")
        except Exception as exc:
            lines.append(f"- 读取失败: {exc}")

    @staticmethod
    def _append_current_task(lines: list[str]) -> None:
        """Show the currently-processing DMN task (if any)."""
        try:
            from vingobot.core.workspace import get_workspace_paths

            wp = get_workspace_paths()
            pending_dir = wp.root / "pending"
            if not pending_dir.is_dir():
                lines.append("- (pending 目录不存在)")
                return

            # Look for .processing files with DMN prefixes
            processing = sorted(
                f
                for f in pending_dir.glob("*.processing")
                if f.name.startswith("cognition-evolution__") or f.name.startswith("dmn__")
            )
            if not processing:
                lines.append("- 无正在处理的任务")
                return

            for pf in processing[:3]:
                content = pf.read_text(encoding="utf-8")
                first_line = content.split("\n")[0][:100]
                name = pf.name.removesuffix(".processing")
                lines.append(f"- 🔄 `{name}`: {first_line}")
            if len(processing) > 3:
                lines.append(f"- ... 还有 {len(processing) - 3} 个在处理中")
        except Exception as exc:
            lines.append(f"- 读取失败: {exc}")

    @staticmethod
    def _append_cognition_health(lines: list[str]) -> None:
        try:
            from vingobot.core.workspace import get_workspace_paths

            wp = get_workspace_paths()
            skills_dir = wp.skills
            models_dir = wp.models
            grids_dir = wp.grids

            skill_count = len(list(skills_dir.iterdir())) if skills_dir.is_dir() else 0
            model_count = len(list(models_dir.iterdir())) if models_dir.is_dir() else 0
            grid_count = len(list(grids_dir.iterdir())) if grids_dir.is_dir() else 0
            total = skill_count + model_count + grid_count

            lines.append(f"- L1 技能 (skills): {skill_count} 个")
            lines.append(f"- L2 思维模型 (models): {model_count} 个")
            lines.append(f"- L3 认知格栅 (grids): {grid_count} 个")
            lines.append(f"- 认知资产总计: {total} 个")
        except Exception as exc:
            lines.append(f"- 读取失败: {exc}")

    @staticmethod
    def _append_recent_changes(lines: list[str]) -> None:
        """Show recently modified cognition assets (last 24h, max 10)."""
        try:
            from vingobot.core.workspace import get_workspace_paths

            wp = get_workspace_paths()
            skills_dir = wp.skills
            models_dir = wp.models
            grids_dir = wp.grids

            now = datetime.now(timezone.utc).timestamp()
            cutoff = now - 24 * 3600  # 24 hours

            recent: list[tuple[float, str, str]] = []  # (mtime, type, name)

            for label, base in [
                ("skill", skills_dir),
                ("model", models_dir),
                ("grid", grids_dir),
            ]:
                if not base.is_dir():
                    continue
                for child in base.iterdir():
                    if child.is_dir():
                        for f in child.rglob("*"):
                            if f.is_file() and f.suffix in (".md", ".yaml", ".json"):
                                mtime = f.stat().st_mtime
                                if mtime > cutoff:
                                    recent.append((mtime, label, f.relative_to(base).as_posix()))
                    elif child.is_file():
                        mtime = child.stat().st_mtime
                        if mtime > cutoff:
                            recent.append((mtime, label, child.name))

            if not recent:
                lines.append("- 过去 24 小时内无认知资产变动")
                return

            recent.sort(key=lambda x: x[0], reverse=True)
            for mtime, label, name in recent[:10]:
                ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%m-%d %H:%M")
                lines.append(f"- `[{label}]` {name} ({ts})")

            if len(recent) > 10:
                lines.append(f"- ... 还有 {len(recent) - 10} 个")
        except Exception as exc:
            lines.append(f"- 读取失败: {exc}")
