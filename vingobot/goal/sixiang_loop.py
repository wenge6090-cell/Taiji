"""
Complete sixiang (六爻) loop — goal outer-loop driver.

This is the top-level entry point that drives a single goal through the full
sixiang cycle:

    Mingjue → Task Inner Loop → Anqu → (repeat or terminate)

The inner loop (Weaver → Yang → Yin → Executor) is delegated to
``task_inner_loop.py``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from loguru import logger

from vingobot.core.goal_context import load_goal_context, refresh_goal_context
from vingobot.core.goal_meta import update_goal_meta
from vingobot.core.manifest import create_manifest, update_manifest_status
from vingobot.core.trajectory import TrajectoryEntry, append_trajectory_entry
from vingobot.core.workspace import create_task_folder, ensure_goal_dir
from vingobot.goal.anqu import run_anqu
from vingobot.goal.grid_types import CognitionEvolutionAction
from vingobot.goal.mingjue import run_mingjue
from vingobot.goal.task_inner_loop import execute_task_inner_loop
from vingobot.goal.types import (
    AnquAction,
    AnquDecision,
    GoalResult,
    MingjueOutput,
    MingjueSource,
)

# Defaults
_DEFAULT_MAX_ROUNDS = 30
_DEFAULT_MAX_REWORKS = 3
_DEFAULT_MAX_GOAL_TASKS = 100


async def execute_complete_sixiang_loop(
    goal_id: str,
    initial_description: str,
    signal: asyncio.Task | None = None,
    *,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    max_rework_rounds: int = _DEFAULT_MAX_REWORKS,
    max_goal_tasks: int = _DEFAULT_MAX_GOAL_TASKS,
) -> GoalResult:
    """Run the full sixiang outer-loop for one goal.

    Args:
        goal_id: Target goal identifier.
        initial_description: First task description from the pending queue.
        signal: Optional asyncio Task for cancellation support.
        max_rounds: Per-task iteration cap.
        max_rework_rounds: Extra rounds granted on rework.
        max_goal_tasks: Maximum sequential tasks before forced termination.
    """

    # Ensure goal directory exists
    ensure_goal_dir(goal_id)

    goal_context = load_goal_context(goal_id)
    if goal_context is None:
        return GoalResult(status="failed", goal_id=goal_id, reason="Goal not found")

    task_description = initial_description
    mode: str = "initial"  # initial | continuation | rework
    previous_task_summary = ""
    continuation_context = ""
    rework_instruction = ""
    previous_mingjue: MingjueOutput | None = None
    task_count = 0
    total_rounds = 0
    consecutive_empty_fallbacks = 0  # track consecutive empty next_task_description

    while task_count < max_goal_tasks:
        if signal is not None and signal.cancelled():
            return GoalResult(status="aborted", goal_id=goal_id, reason="Cancelled by signal")

        task_count += 1

        # ── 1. 初爻·明觉 ──────────────────────────────────────
        task_id = f"task-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{task_count:03d}"
        task_dir = create_task_folder(goal_id, task_id)
        create_manifest(
            task_dir,
            task_id=task_id,
            goal_id=goal_id,
            description=task_description,
        )

        source: MingjueSource
        if mode == "initial":
            source = MingjueSource(type="initial_goal", description=task_description)
        elif mode == "continuation":
            source = MingjueSource(
                type="anqu_continuation",
                description=task_description,
                previous_task_summary=previous_task_summary,
                continuation_context=continuation_context,
            )
        else:
            source = MingjueSource(
                type="rework",
                description=task_description,
                rework_instruction=rework_instruction,
                previous_output=previous_mingjue,
            )

        mingjue_output = await run_mingjue(goal_context, source, signal=signal)
        # 明觉落地工作环境：注入实际任务目录
        mingjue_output.context.task_dir = str(task_dir)
        previous_mingjue = mingjue_output

        # ── 2. 任务内循环 ──────────────────────────────────────
        inner_result = await execute_task_inner_loop(
            task_dir=task_dir,
            mingjue_output=mingjue_output,
            goal_context=goal_context,
            signal=signal,
            max_rounds=max_rounds + (max_rework_rounds if mode == "rework" else 0),
        )
        total_rounds += inner_result.rounds_executed

        # Update manifest
        update_manifest_status(
            task_dir,
            "completed" if inner_result.task_completed else "failed",
            round_count=inner_result.rounds_executed,
        )

        # ── 3. 上爻·暗驱 ──────────────────────────────────────
        anqu_decision = await run_anqu(
            goal_context=goal_context,
            current_task_facts=inner_result.facts,
            final_content=inner_result.final_content,
            task_dir=str(task_dir),
            signal=signal,
            cognitive_usage=inner_result.cognitive_usage,
        )

        # Persist trajectory entry
        _persist_trajectory(
            goal_id,
            task_id,
            anqu_decision,
            rounds=inner_result.rounds_executed,
        )

        # Process cognitive evolution actions (enqueue as learning tasks)
        _process_evolution_actions(anqu_decision.evolution_actions, task_id)

        # ── 4. Route based on Anqu decision ────────────────────
        action: AnquAction = anqu_decision.action

        if action == "goal_completed":
            update_goal_meta(goal_id, status="completed")
            return GoalResult(
                status="completed", goal_id=goal_id, reason=anqu_decision.task_summary
            )

        if action == "goal_failed":
            update_goal_meta(goal_id, status="failed")
            return GoalResult(
                status="failed",
                goal_id=goal_id,
                reason=anqu_decision.failure_reason or "Anqu declared failure",
            )

        if action == "goal_next_task":
            if not anqu_decision.next_task_description:
                logger.warning(
                    "[六爻] 暗驱返回 goal_next_task 但未提供 next_task_description，"
                    "回退使用旧任务描述 (连续 {} 次)",
                    consecutive_empty_fallbacks + 1,
                )
                consecutive_empty_fallbacks += 1
                if consecutive_empty_fallbacks >= 2:
                    update_goal_meta(goal_id, status="failed")
                    return GoalResult(
                        status="failed",
                        goal_id=goal_id,
                        reason="暗驱连续两次返回 goal_next_task 但未提供新任务描述，终止以防止死循环",
                    )
            else:
                consecutive_empty_fallbacks = 0  # reset on valid description
            task_description = anqu_decision.next_task_description or task_description
            mode = "continuation"
            previous_task_summary = anqu_decision.task_summary or ""
            continuation_context = anqu_decision.continuation_context or ""
            goal_context = refresh_goal_context(goal_id)
            continue

        if action in ("continue_task", "verify_task", "learn_task"):
            rework_instruction = anqu_decision.rework_instruction or "请重新审题并继续执行"
            mode = "rework"
            continue

        logger.warning("[六爻] 未知暗驱动作: {}", action)
        break

    # ── Max goal tasks reached ──────────────────────────────────
    return GoalResult(
        status="completed",
        goal_id=goal_id,
        reason=f"达到最大目标任务数 ({max_goal_tasks})",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persist_trajectory(
    goal_id: str,
    task_id: str,
    decision: AnquDecision,
    *,
    rounds: int = 0,
) -> None:
    """Append a trajectory entry for a completed task."""
    status = "completed" if decision.action in ("goal_completed", "goal_next_task") else "failed"
    entry = TrajectoryEntry(
        task_id=task_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status=status,
        summary=decision.task_summary or decision.next_task_description or "",
        round_count=rounds,
    )
    try:
        append_trajectory_entry(goal_id, entry)
    except Exception:
        logger.exception("Failed to persist trajectory for goal {}", goal_id)


def _process_evolution_actions(
    actions: list[CognitionEvolutionAction],
    source_task_id: str,
) -> None:
    """Enqueue cognitive evolution tasks under the special ``cognition-evolution`` goal.

    Each ``CognitionEvolutionAction`` is written as a task file to the
    ``.taiji/pending/`` directory, prefixed with priority for ordering.
    The WorkerPool will pick them up and process them asynchronously.
    """
    if not actions:
        return

    try:
        from vingobot.core.workspace import get_workspace_paths

        wp = get_workspace_paths()
        pending_dir = wp.pending
        pending_dir.mkdir(parents=True, exist_ok=True)

        for action in actions:
            # Build a task description from the evolution action
            task_desc = _build_evolution_task_description(action)

            # Priority-based filename for ordering
            priority_prefix = f"{10 - action.priority:02d}"  # higher priority = earlier
            filename = (
                f"cognition-evolution__{priority_prefix}__"
                f"{action.action}__{action.target_name}__"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.task"
            )

            task_file = pending_dir / filename
            task_file.write_text(task_desc, encoding="utf-8")

            logger.info(
                "[六爻] 入队认知演化任务: {} {} (优先级 {})",
                action.action,
                action.target_name,
                action.priority,
            )

    except Exception as exc:
        logger.warning("[六爻] 处理认知演化动作失败: {}", exc)


def _build_evolution_task_description(action: CognitionEvolutionAction) -> str:
    """Build a human-readable task description from an evolution action."""
    lines = [
        f"# 认知演化任务: {action.action}",
        f"目标: {action.target_name}",
        f"描述: {action.description}",
        f"来源任务: {action.source_task_id}",
        f"来源目标: {action.source_goal_id}",
        f"优先级: {action.priority}/10",
        "",
    ]

    if action.context:
        lines.append("## 上下文")
        import json

        lines.append(json.dumps(action.context, ensure_ascii=False, indent=2))
        lines.append("")

    # Action-specific instructions
    if action.action == "learn_skill":
        lines.extend(
            [
                "## 执行说明",
                "1. 分析缺失的技能/工具需求",
                "2. 搜索现有技能库，确认是否已存在类似技能",
                "3. 如不存在，创建新的L1技能定义 (SKILL.md + 实现)",
                "4. 更新相关L3格栅的 skills 列表",
                "5. 更新认知导航索引",
            ]
        )
    elif action.action == "precipitate_skill":
        lines.extend(
            [
                "## 执行说明",
                "1. 回顾源任务的成功SOP",
                "2. 提取可复用的步骤和方法",
                "3. 创建L1技能定义 (SKILL.md)",
                "4. 注册到相关L3格栅的 skills 列表",
                "5. 更新认知导航索引",
            ]
        )
    elif action.action == "precipitate_model":
        lines.extend(
            [
                "## 执行说明",
                "1. 回顾源任务中产生的方法论或思维模式",
                "2. 抽象为通用的L2经验模型",
                "3. 创建模型文件 (.md) 到 models 目录",
                "4. 关联到相关L3格栅的 models 列表",
                "5. 更新认知导航索引",
            ]
        )
    elif action.action == "create_grid":
        lines.extend(
            [
                "## 执行说明",
                "1. 确定新认知领域的范围和边界",
                "2. 识别相关的L1技能和L2模型",
                "3. 创建标准化的L3格栅JSON文件",
                "4. 更新认知导航索引",
            ]
        )

    return "\n".join(lines)
