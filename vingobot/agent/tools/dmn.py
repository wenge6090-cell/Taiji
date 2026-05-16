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
_ACTION_ANALYZE = "analyze"

_VALID_ACTIONS = frozenset({
    _ACTION_STATUS, _ACTION_START, _ACTION_STOP, _ACTION_TRIGGER, _ACTION_ANALYZE,
})


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

    * **analyze** — Cross-task execution analytics. Scans all goal/task
      directories and aggregates round execution facts into a structured
      ``ExecutionInsight`` report covering gua efficiency, tool failure
      patterns, yin approval trends, and stuck-loop detection.
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
            "Use **analyze** to inspect cross-task execution patterns "
            "(gua efficiency, tool failures, yin approval patterns, stuck loops). "
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
            if action == _ACTION_ANALYZE:
                return await self._do_analyze()
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
    # analyze — cross-task execution analytics
    # ------------------------------------------------------------------

    async def _do_analyze(self) -> str:
        """Scan all goals/tasks, aggregate RoundExecutionFact data, and
        produce an ``ExecutionInsight`` report."""
        import json
        from datetime import datetime, timezone

        from vingobot.goal.types import (
            CognitiveStat,
            ExecutionInsight,
            StuckLoopRecord,
            ToolStatItem,
            YinDecisionStat,
        )

        from vingobot.core.goal_meta import read_goal_meta
        from vingobot.core.manifest import read_manifest
        from vingobot.core.workspace import get_workspace_paths

        wp = get_workspace_paths()
        goals_dir = wp.goals

        insight = ExecutionInsight(
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )

        if not goals_dir.is_dir():
            return "未找到 goal 目录，无法分析。"

        # ── 读操作工具集（用于自读循环检测） ──────────────────────────
        _READ_ONLY_TOOLS = frozenset({
            "read_file", "search_codebase", "search_file",
            "grep_code", "list_dir", "search_symbol",
            "search_web", "fetch_content",
        })

        # ── 缓存一轮读过的 round 数据：{round_path: parsed_dict} ────
        _round_cache: dict[str, dict | None] = {}

        for goal_dir in sorted(goals_dir.iterdir()):
            if not goal_dir.is_dir():
                continue
            goal_id = goal_dir.name
            meta = read_goal_meta(goal_id)
            if meta is None:
                continue
            insight.total_goals += 1

            tasks_dir = goal_dir / "tasks"
            if not tasks_dir.is_dir():
                continue

            for task_dir in sorted(tasks_dir.iterdir()):
                if not task_dir.is_dir():
                    continue

                manifest = read_manifest(task_dir)
                if manifest is None:
                    continue

                insight.total_tasks += 1
                status = manifest.status
                insight.task_status_breakdown[status] = (
                    insight.task_status_breakdown.get(status, 0) + 1
                )

                # ── Read 06-execution-facts.json ───────────────────────
                facts_path = task_dir / "06-execution-facts.json"
                if not facts_path.is_file():
                    continue

                try:
                    facts_data = json.loads(facts_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue

                if not isinstance(facts_data, list):
                    continue

                outputs_dir = task_dir / "outputs"

                # ── Aggregate per-round facts ──────────────────────────
                for round_data in facts_data:
                    if not isinstance(round_data, dict):
                        continue

                    insight.total_rounds += 1

                    current_gua = str(round_data.get("current_gua", ""))
                    sixiang = str(round_data.get("sixiang", ""))
                    yao = round_data.get("yao", 0)
                    exec_status = str(round_data.get("execution_status", "skipped"))
                    tool_call_count = int(round_data.get("tool_call_count", 0))
                    yin_decision = str(round_data.get("yin_decision", "skipped"))
                    yin_reason = str(round_data.get("yin_reason", ""))

                    is_success = exec_status == "success"
                    is_failure = exec_status in ("failure", "exec_failed")

                    # gua
                    if current_gua:
                        if current_gua not in insight.gua_stats:
                            insight.gua_stats[current_gua] = CognitiveStat()
                        gs = insight.gua_stats[current_gua]
                        gs.count += 1
                        gs.total_rounds += 1
                        gs.total_tool_calls += tool_call_count
                        if is_success:
                            gs.success_count += 1
                        if is_failure:
                            gs.failure_count += 1

                    # sixiang
                    if sixiang:
                        if sixiang not in insight.sixiang_stats:
                            insight.sixiang_stats[sixiang] = CognitiveStat()
                        ss = insight.sixiang_stats[sixiang]
                        ss.count += 1
                        ss.total_rounds += 1
                        ss.total_tool_calls += tool_call_count
                        if is_success:
                            ss.success_count += 1
                        if is_failure:
                            ss.failure_count += 1

                    # yao
                    yao_key = str(yao)
                    if yao_key and yao_key != "0":
                        if yao_key not in insight.yao_stats:
                            insight.yao_stats[yao_key] = CognitiveStat()
                        ys = insight.yao_stats[yao_key]
                        ys.count += 1
                        ys.total_rounds += 1
                        ys.total_tool_calls += tool_call_count
                        if is_success:
                            ys.success_count += 1
                        if is_failure:
                            ys.failure_count += 1

                    # yin
                    y = insight.yin_stats
                    if yin_decision != "skipped":
                        y.total += 1
                        if yin_decision == "approved":
                            y.approved += 1
                        elif yin_decision == "rejected":
                            y.rejected += 1
                        elif yin_decision == "modified":
                            y.modified += 1

                        if yin_decision == "rejected" and yin_reason:
                            reason_snippet = yin_reason[:60]
                            found = False
                            for i, (r, _) in enumerate(y.top_rejection_reasons):
                                if r == reason_snippet:
                                    cnt = y.top_rejection_reasons[i][1]
                                    y.top_rejection_reasons[i] = (r, cnt + 1)
                                    found = True
                                    break
                            if not found:
                                y.top_rejection_reasons.append((reason_snippet, 1))

                # ── Tool-level analysis + stuck-loop detection ───────
                # (一次遍历所有 round 文件，避免重复读盘)
                if outputs_dir.is_dir():
                    consecutive_reads = 0
                    stuck_rounds: list[int] = []

                    round_files = sorted(
                        outputs_dir.glob("*-round.json"),
                        key=lambda x: int(x.name.split("-")[0])
                        if x.name.split("-")[0].isdigit()
                        else 0,
                    )
                    for rf in round_files:
                        # Use cache to avoid re-reading
                        cache_key = str(rf)
                        if cache_key not in _round_cache:
                            try:
                                _round_cache[cache_key] = json.loads(
                                    rf.read_text(encoding="utf-8")
                                )
                            except (json.JSONDecodeError, OSError):
                                _round_cache[cache_key] = None
                        rdata = _round_cache[cache_key]
                        if not isinstance(rdata, dict):
                            continue

                        tool_calls = rdata.get("tool_calls") or rdata.get(
                            "yang_tool_calls", []
                        )
                        if not isinstance(tool_calls, list):
                            tool_calls = []
                        results = rdata.get("execution_results", [])
                        if not isinstance(results, list):
                            results = []

                        # ── Tool-level stats ────────────────────────
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            tname = tc.get("name") or tc.get(
                                "function", {}
                            ).get("name", "")
                            if not tname:
                                continue
                            if tname not in insight.tool_stats:
                                insight.tool_stats[tname] = ToolStatItem()
                            insight.tool_stats[tname].call_count += 1

                        for res in results:
                            if not isinstance(res, dict):
                                continue
                            call_info = res.get("call", {})
                            tname = (
                                call_info.get("name", "")
                                if isinstance(call_info, dict)
                                else ""
                            )
                            status_val = str(res.get("status", ""))
                            error = str(res.get("error", ""))

                            if not tname:
                                continue

                            if tname not in insight.tool_stats:
                                insight.tool_stats[tname] = ToolStatItem()
                                insight.tool_stats[tname].call_count += 1

                            ts = insight.tool_stats[tname]
                            if status_val in ("error", "failure"):
                                ts.failure_count += 1
                                if len(error) > 3:
                                    err_snip = error[:80]
                                    new_errors = []
                                    found = False
                                    for err, cnt in ts.top_errors:
                                        if err == err_snip:
                                            new_errors.append(
                                                (err, cnt + 1)
                                            )
                                            found = True
                                        else:
                                            new_errors.append((err, cnt))
                                    if not found:
                                        new_errors.append((err_snip, 1))
                                    new_errors.sort(key=lambda x: -x[1])
                                    ts.top_errors = new_errors[:5]
                            if status_val == "exec_failed":
                                ts.exec_failed_count += 1

                        # ── Stuck-loop detection ─────────────────────
                        if tool_calls:
                            tnames = set()
                            for tc in tool_calls:
                                if isinstance(tc, dict):
                                    tn = tc.get("name") or tc.get(
                                        "function", {}
                                    ).get("name", "")
                                    if tn:
                                        tnames.add(tn)

                            if tnames and tnames.issubset(_READ_ONLY_TOOLS):
                                # round number from filename
                                round_num = int(rf.name.split("-")[0])
                                consecutive_reads += 1
                                stuck_rounds.append(round_num)
                            else:
                                consecutive_reads = 0
                                stuck_rounds = []

                    if consecutive_reads >= 3:
                        rng = f"R{stuck_rounds[0]}-R{stuck_rounds[-1]}"
                        insight.stuck_loops.append(
                            StuckLoopRecord(
                                goal_id=goal_id,
                                task_id=task_dir.name,
                                round_range=rng,
                                consecutive_read_rounds=consecutive_reads,
                                detection_reason=(
                                    f"连续{consecutive_reads}轮仅读操作"
                                ),
                            )
                        )
                        insight.total_stuck_loops += 1

        # Sort rejection reasons
        insight.yin_stats.top_rejection_reasons.sort(key=lambda x: -x[1])

        return insight.to_text()

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
