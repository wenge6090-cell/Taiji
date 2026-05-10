"""
TPN (Task Positive Network) — sixiang control & goal management tool.

Lets the agent inspect and control the sixiang (六爻) goal-driven loop:
start/stop the worker pool, view pending-queue and cognition-library
status, manage goals, and manually trigger Anqu evaluations or
cognitive-evolution tasks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from vingobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from vingobot.agent.loop import AgentLoop


# ---------------------------------------------------------------------------
# Action names
# ---------------------------------------------------------------------------

_ACTION_START = "start"
_ACTION_STOP = "stop"
_ACTION_STATUS = "status"
_ACTION_LIST = "list"
_ACTION_CREATE = "create"
_ACTION_UPDATE = "update"
_ACTION_DELETE = "delete"
_ACTION_TRIGGER = "trigger"

_VALID_ACTIONS = frozenset(
    {
        _ACTION_START,
        _ACTION_STOP,
        _ACTION_STATUS,
        _ACTION_LIST,
        _ACTION_CREATE,
        _ACTION_UPDATE,
        _ACTION_DELETE,
        _ACTION_TRIGGER,
    }
)


def _is_safe_goal_id(goal_id: str) -> bool:
    """Allow Unicode letters (incl.中文), digits and ``-`` ``_`` only."""
    if not goal_id:
        return False
    for ch in goal_id:
        if ch.isalnum() or ch in ("-", "_"):
            continue
        return False
    return True


class TpnTool(Tool):
    """Control the sixiang (六爻) goal-driven loop and manage goals.

    Actions:

    * **start** — Start the sixiang auto-loop (background worker pool).
      Workers pick up pending tasks and drive goals through
      Mingjue → Task → Anqu cycles.

    * **stop** — Stop the sixiang auto-loop gracefully.  Currently
      running tasks finish; no new tasks are picked up.

    * **status** — Show the full TPN dashboard: pool state, pending
      queue depth, cognition library health (skills / models / grids),
      and a summary of all goals with their current status.

    * **list** — List all goals with id, name, status, priority, and
      self-driven configuration.

    * **create** — Create a new goal.  Required: ``goal_id`` and
      ``description``.  Optional: ``priority`` (1-10, default 5),
      ``self_driven_enabled``, ``self_driven_interval_minutes``.

    * **update** — Update a goal's metadata.  You can change status
      ("active" / "paused" / "completed" / "archived"), priority,
      description, or self-driven settings.  Only the fields you
      provide are changed; others stay the same.

    * **delete** — Delete a goal and all its tasks.  This is
      irreversible — warn the user first.

    * **trigger** — Manually trigger one round of sixiang processing
      for a specific goal.  This enqueues a task that goes through
      Mingjue → Task → Anqu, producing a fresh Anqu decision and
      any cognitive evolution actions.  Use this to "kick" a goal.

      For cognitive-evolution tasks, use the **dmn** tool instead.
    """

    _MAX_GOAL_ID_LEN = 64

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop

    # ------------------------------------------------------------------
    # Tool metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "tpn"

    @property
    def description(self) -> str:
        return (
            "Control the sixiang (六爻) goal-driven loop and manage goals.\n"
            "Actions: start, stop, status, list, create, update, delete, "
            "trigger.\n"
            "\n"
            "- start / stop: Control the sixiang background worker pool.\n"
            "- status: Full TPN dashboard — pool state, pending queue, "
            "cognition health, goal summary.\n"
            "- list: List all goals with status.\n"
            "- create: Create a new goal (goal_id, description required).\n"
            "- update: Change goal status/priority/description/self_driven.\n"
            "- delete: Irreversibly delete a goal — warn the user first.\n"
            "- trigger: Kick a goal through one Mingjue→Task→Anqu cycle.\n"
            "For cognitive-evolution tasks, use the **dmn** tool."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(_VALID_ACTIONS),
                    "description": "Which TPN action to perform.",
                },
                "goal_id": {
                    "type": "string",
                    "description": (
                        "Goal ID for create/update/delete/trigger. "
                        "Use snake_case, e.g. 'refactor-db', 'write-tests'."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Goal description (for create/update/trigger).",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Goal priority 1-10, higher = more important (default 5).",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "paused", "completed", "archived"],
                    "description": "Goal status to set (for update).",
                },
                "self_driven_enabled": {
                    "type": "boolean",
                    "description": "Enable/disable self-driven wake for this goal.",
                },
                "self_driven_interval_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1440,
                    "description": "Self-driven wake interval in minutes.",
                },
            },
            "required": ["action"],
        }

    # ------------------------------------------------------------------
    # Execute dispatch
    # ------------------------------------------------------------------

    async def execute(self, action: str, **kwargs: Any) -> str:
        if action not in _VALID_ACTIONS:
            return f"未知 TPN 动作: {action}。有效动作: {', '.join(sorted(_VALID_ACTIONS))}"

        try:
            if action == _ACTION_START:
                return await self._do_start()
            if action == _ACTION_STOP:
                return await self._do_stop()
            if action == _ACTION_STATUS:
                return self._do_status()
            if action == _ACTION_LIST:
                return self._do_list()
            if action == _ACTION_CREATE:
                return self._do_create(kwargs)
            if action == _ACTION_UPDATE:
                return self._do_update(kwargs)
            if action == _ACTION_DELETE:
                return self._do_delete(kwargs)
            if action == _ACTION_TRIGGER:
                return self._do_trigger(kwargs)
        except Exception as exc:
            logger.exception("[TPN] 执行 '{}' 失败", action)
            return f"TPN 错误 ({action}): {exc}"

        return f"未实现的 TPN 动作: {action}"

    # ------------------------------------------------------------------
    # start / stop
    # ------------------------------------------------------------------

    async def _do_start(self) -> str:
        bot = getattr(self._loop, "_tpn_bot", None)
        if bot is None:
            return "错误: 无法访问六爻池控制器（_tpn_bot 未设置）"

        if bot.sixiang_running:
            return "六爻协程池已经在运行中。"

        try:
            await bot.start_sixiang()
            return "✅ 六爻协程池已启动。Worker 开始消费 pending 队列中的任务。"
        except Exception as exc:
            return f"启动六爻协程池失败: {exc}"

    async def _do_stop(self) -> str:
        bot = getattr(self._loop, "_tpn_bot", None)
        if bot is None:
            return "错误: 无法访问六爻池控制器（_tpn_bot 未设置）"

        if not bot.sixiang_running:
            return "六爻协程池未在运行。"

        try:
            await bot.stop_sixiang()
            return "⏹ 六爻协程池已停止。当前运行中的任务会完成，不再拾取新任务。"
        except Exception as exc:
            return f"停止六爻协程池失败: {exc}"

    # ------------------------------------------------------------------
    # status — full TPN dashboard
    # ------------------------------------------------------------------

    def _do_status(self) -> str:
        lines: list[str] = ["# TPN 状态面板\n"]

        # ── Pool state ──────────────────────────────────────────
        bot = getattr(self._loop, "_tpn_bot", None)
        if bot is not None and bot.sixiang_running:
            pool = bot._sixiang_pool
            active = pool.active_count if pool else "?"
            lines.append("## 六爻协程池")
            lines.append(f"- 状态: 🟢 运行中 (active workers: {active})")
        elif bot is not None:
            lines.append("## 六爻协程池")
            lines.append("- 状态: ⚪ 已停止")
        else:
            lines.append("## 六爻协程池")
            lines.append("- 状态: ⚫ 未初始化（_tpn_bot 不可用）")

        # ── DMN consumer ───────────────────────────────────────
        dmn_task = self._loop._dmn_consumer_task
        dmn_running = dmn_task is not None and not dmn_task.done()
        lines.append("## DMN 消费者")
        lines.append(f"- 状态: {'🟢 运行中' if dmn_running else '⚪ 已停止'}")

        # ── Pending queue ──────────────────────────────────────
        try:
            from vingobot.core.pending_queue import PendingQueue

            queue = PendingQueue()
            all_tasks = queue.scan_pending()
            regular = [t for t in all_tasks if not t.goal_id.startswith("cognition-evolution")]
            dmn_tasks = [t for t in all_tasks if t.goal_id.startswith("cognition-evolution")]
            lines.append("## Pending 队列")
            lines.append(f"- 总任务数: {len(all_tasks)}")
            lines.append(f"  - 常规任务: {len(regular)}")
            lines.append(f"  - 认知演化任务 (DMN): {len(dmn_tasks)}")
            if all_tasks:
                lines.append("")
                lines.append("### 队列中的任务:")
                for t in all_tasks[:20]:
                    desc = t.description[:60].replace("\n", " ")
                    lines.append(f"- `{t.goal_id}` — {desc}")
                if len(all_tasks) > 20:
                    lines.append(f"- ... 还有 {len(all_tasks) - 20} 个任务")
        except Exception as exc:
            lines.append("## Pending 队列")
            lines.append(f"- 读取失败: {exc}")

        # ── Cognition health ───────────────────────────────────
        lines.append("")
        lines.append("## 认知库健康度")
        self._append_cognition_health(lines)

        # ── Goal summary ───────────────────────────────────────
        lines.append("")
        lines.append("## 目标概览")
        self._append_goal_summary(lines)

        return "\n".join(lines)

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
            grid_count = len(list(grids_dir.glob("*.json"))) if grids_dir.is_dir() else 0

            lines.append(f"- L1 技能 (skills): {skill_count} 个")
            lines.append(f"- L2 思维模型 (models): {model_count} 个")
            lines.append(f"- L3 认知格栅 (grids): {grid_count} 个")
            lines.append(f"- 认知资产总计: {skill_count + model_count + grid_count} 个")

            # Grid details
            if grid_count > 0:
                try:
                    from vingobot.goal.cognition_tools import list_all_grids

                    grids = list_all_grids(str(grids_dir))
                    for g in grids[:10]:
                        proficiency_bar = "█" * int(g["proficiency"] * 10) + "░" * (
                            10 - int(g["proficiency"] * 10)
                        )
                        lines.append(
                            f"  - {g['domain']} "
                            f"[{proficiency_bar}] {g['proficiency']:.0%} "
                            f"skills:{g['skill_count']} models:{g['model_count']}"
                        )
                    if len(grids) > 10:
                        lines.append(f"  - ... 还有 {len(grids) - 10} 个格栅")
                except Exception:
                    pass

        except Exception as exc:
            lines.append(f"- 读取失败: {exc}")

    @staticmethod
    def _append_goal_summary(lines: list[str]) -> None:
        try:
            from vingobot.core.goal_meta import get_all_goals

            goals = get_all_goals()
            if not goals:
                lines.append("(无目标)")
                return

            status_icon = {"active": "●", "paused": "◐", "completed": "○", "archived": "✕"}
            for g in goals:
                gid = g.id
                name = g.name or gid
                status = g.status
                priority = g.priority
                sd_enabled = g.self_driven.enabled
                icon = status_icon.get(status, "?")

                parts = [f"{icon} **{name}** (`{gid}`): {status}, 优先级:{priority}"]
                if sd_enabled:
                    parts.append(f", ⏰自驱:{g.self_driven.interval_minutes}min")
                else:
                    parts.append(", 自驱:关")

                if g.warnings:
                    parts.append(f", ⚠{len(g.warnings)}个警告")

                if g.rounds_completed:
                    parts.append(f", {g.rounds_completed}轮")

                lines.append("- " + "".join(parts))

        except Exception as exc:
            lines.append(f"- 读取失败: {exc}")

    # ------------------------------------------------------------------
    # list — all goals
    # ------------------------------------------------------------------

    def _do_list(self) -> str:
        try:
            from vingobot.core.goal_meta import get_all_goals

            goals = get_all_goals()
            if not goals:
                return "(无目标)"

            lines: list[str] = ["# 所有目标\n"]
            status_icon = {"active": "🔄", "paused": "⏸", "completed": "✅", "archived": "📦"}
            for g in goals:
                gid = g.id
                name = g.name or gid
                status = g.status
                priority = g.priority
                desc = (g.description or "")[:80]
                icon = status_icon.get(status, "❓")
                sd_str = "⏰" if g.self_driven.enabled else "  "

                lines.append(f"{icon} {sd_str} **{name}** | `{gid}` | {status} | P{priority}")
                if desc:
                    lines.append(f"   {desc}")

            return "\n".join(lines)

        except Exception as exc:
            return f"列出目标失败: {exc}"

    # ------------------------------------------------------------------
    # create — new goal
    # ------------------------------------------------------------------

    def _do_create(self, kwargs: dict[str, Any]) -> str:
        goal_id = str(kwargs.get("goal_id", "")).strip()
        if not goal_id:
            return "错误: 必须提供 goal_id"
        if len(goal_id) > self._MAX_GOAL_ID_LEN:
            return f"错误: goal_id 最长 {self._MAX_GOAL_ID_LEN} 字符"
        if "/" in goal_id or "\\" in goal_id:
            return "错误: goal_id 不能包含路径分隔符 (/ 或 \\)"
        if not _is_safe_goal_id(goal_id):
            return "错误: goal_id 只能包含字母、数字、中文、- 和 _"

        description = str(kwargs.get("description", "")).strip()
        blueprint = str(kwargs.get("blueprint", "")).strip()
        priority = int(kwargs.get("priority", 5))
        priority = max(1, min(10, priority))

        sd_enabled = bool(kwargs.get("self_driven_enabled", False))
        sd_interval = int(kwargs.get("self_driven_interval_minutes", 30))
        sd_interval = max(1, min(1440, sd_interval))

        try:
            from vingobot.core.goal_meta import GoalMeta, SelfDrivenConfig, write_goal_meta
            from vingobot.core.workspace import ensure_goal_dir

            ensure_goal_dir(
                goal_id,
                priority=priority,
                description=description or None,
                blueprint=blueprint,
            )

            meta = GoalMeta(
                id=goal_id,
                name=goal_id,
                description=description,
                status="active",
                priority=priority,
                created_at=datetime.now(timezone.utc).isoformat(),
                self_driven=SelfDrivenConfig(
                    enabled=sd_enabled,
                    interval_minutes=sd_interval,
                ),
            )
            write_goal_meta(goal_id, meta)

            parts = [f"✅ 目标 `{goal_id}` 已创建 (优先级: {priority})"]
            if description:
                parts.append(f"   描述: {description}")
            if sd_enabled:
                parts.append(f"   自驱: 启用 (每 {sd_interval} 分钟)")
            return "\n".join(parts)

        except Exception as exc:
            return f"创建目标失败: {exc}"

    # ------------------------------------------------------------------
    # update — modify goal metadata
    # ------------------------------------------------------------------

    def _do_update(self, kwargs: dict[str, Any]) -> str:
        goal_id = str(kwargs.get("goal_id", "")).strip()
        if not goal_id:
            return "错误: 必须提供 goal_id"

        try:
            from vingobot.core.goal_meta import SelfDrivenConfig, read_goal_meta, update_goal_meta

            meta = read_goal_meta(goal_id)
            if meta is None:
                return f"错误: 目标 `{goal_id}` 不存在"

            updates: dict[str, Any] = {}
            changes: list[str] = []

            if "status" in kwargs and kwargs["status"] is not None:
                new_status = str(kwargs["status"])
                if new_status in ("active", "paused", "completed", "archived"):
                    if new_status != meta.status:
                        updates["status"] = new_status
                        changes.append(f"状态: {meta.status} → {new_status}")

            if "priority" in kwargs and kwargs["priority"] is not None:
                new_pri = max(1, min(10, int(kwargs["priority"])))
                if new_pri != meta.priority:
                    updates["priority"] = new_pri
                    changes.append(f"优先级: {meta.priority} → {new_pri}")

            if "description" in kwargs and kwargs["description"] is not None:
                new_desc = str(kwargs["description"]).strip()
                if new_desc != meta.description:
                    # description is not in update_goal_meta, write it separately
                    meta.description = new_desc
                    from vingobot.core.goal_meta import write_goal_meta

                    write_goal_meta(goal_id, meta)
                    changes.append("描述: 已更新")

            # Self-driven config
            sd_updates: dict[str, Any] = {}
            if "self_driven_enabled" in kwargs and kwargs["self_driven_enabled"] is not None:
                sd_updates["enabled"] = bool(kwargs["self_driven_enabled"])
            if (
                "self_driven_interval_minutes" in kwargs
                and kwargs["self_driven_interval_minutes"] is not None
            ):
                sd_updates["interval_minutes"] = max(
                    1, min(1440, int(kwargs["self_driven_interval_minutes"]))
                )

            if sd_updates:
                new_sd = SelfDrivenConfig(
                    enabled=sd_updates.get("enabled", meta.self_driven.enabled),
                    interval_minutes=sd_updates.get(
                        "interval_minutes", meta.self_driven.interval_minutes
                    ),
                )
                meta.self_driven = new_sd
                from vingobot.core.goal_meta import write_goal_meta

                write_goal_meta(goal_id, meta)
                changes.append(
                    f"自驱: enabled={new_sd.enabled}, interval={new_sd.interval_minutes}min"
                )

            if updates:
                update_goal_meta(goal_id, **updates)

            if not changes:
                return f"目标 `{goal_id}` 没有需要更新的字段。"

            return f"✅ 目标 `{goal_id}` 已更新:\n" + "\n".join(f"  - {c}" for c in changes)

        except Exception as exc:
            return f"更新目标失败: {exc}"

    # ------------------------------------------------------------------
    # delete — remove a goal
    # ------------------------------------------------------------------

    def _do_delete(self, kwargs: dict[str, Any]) -> str:
        goal_id = str(kwargs.get("goal_id", "")).strip()
        if not goal_id:
            return "错误: 必须提供 goal_id"

        try:
            import shutil

            from vingobot.core.goal_meta import read_goal_meta
            from vingobot.core.workspace import get_goal_dir

            meta = read_goal_meta(goal_id)
            if meta is None:
                return f"错误: 目标 `{goal_id}` 不存在"

            goal_dir = get_goal_dir(goal_id)
            if not goal_dir.exists():
                return f"错误: 目标目录 `{goal_id}` 不存在"

            shutil.rmtree(goal_dir)
            logger.warning("[TPN] 已删除目标: {} ({})", goal_id, meta.name or goal_id)
            return f"🗑 目标 `{goal_id}` 已删除（{meta.name or goal_id}）。此操作不可撤销。"

        except Exception as exc:
            return f"删除目标失败: {exc}"

    # ------------------------------------------------------------------
    # trigger — enqueue a task for sixiang evaluation
    # ------------------------------------------------------------------

    def _do_trigger(self, kwargs: dict[str, Any]) -> str:
        goal_id = str(kwargs.get("goal_id", "")).strip()
        if not goal_id:
            return "错误: 必须提供 goal_id（要对哪个目标触发评估？）"

        description = str(kwargs.get("description", "")).strip()
        if not description:
            description = (
                f"评估目标 '{goal_id}' 的当前进展，检查完成标准是否已满足，"
                f"并做出下一步决策（继续/完成/失败/回炉）。"
            )

        try:
            from vingobot.core.goal_meta import read_goal_meta
            from vingobot.core.pending_queue import PendingQueue, PendingTask

            meta = read_goal_meta(goal_id)
            if meta is None:
                return f"错误: 目标 `{goal_id}` 不存在。请先用 create 创建。"

            queue = PendingQueue()
            task = PendingTask(
                id="",
                goal_id=goal_id,
                description=description,
                priority=meta.priority,
                source="user",
                metadata={
                    "tpn_trigger": True,
                    "triggered_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            filename = queue.enqueue(task)

            return (
                f"🎯 已向目标 `{goal_id}` 的 pending 队列发送触发任务。\n"
                f"   文件: {filename}\n"
                f"   描述: {description[:120]}\n"
                f"   {'' if self._pool_running() else '⚠ 六爻协程池未运行，任务将在池启动后被消费。'}"
            )

        except Exception as exc:
            return f"触发任务失败: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pool_running(self) -> bool:
        bot = getattr(self._loop, "_tpn_bot", None)
        return bot is not None and bot.sixiang_running
