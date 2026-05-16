"""
Complete sixiang (六爻) loop — goal outer-loop driver.

This is the top-level entry point that drives a single goal through the full
sixiang cycle:

    Mingjue → Task Inner Loop → Anqu → (repeat or terminate)

The inner loop (Weaver → Yang → Yin) is delegated to
``task_inner_loop.py``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from vingobot.core.goal_context import GoalContext, load_goal_context, refresh_goal_context
from vingobot.core.goal_meta import read_goal_meta, update_goal_meta
from vingobot.core.manifest import create_manifest, update_manifest_status
from vingobot.core.trajectory import update_goal_progress
from vingobot.core.workspace import create_task_folder, ensure_goal_dir, get_goal_dir
from vingobot.goal.anqu import run_anqu
from vingobot.goal.grid_types import CognitionEvolutionAction
from vingobot.goal.mingjue import run_mingjue
from vingobot.goal.sibian import run_sibian
from vingobot.goal.task_inner_loop import InnerLoopResult, execute_task_inner_loop
from vingobot.goal.types import (
    AnquAction,
    AnquDecision,
    GoalResult,
    MingjueOutput,
    MingjueSource,
    SibianDecision,
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
    consecutive_continuations = 0  # track consecutive goal_next_task decisions
    goal_progress_history: list[int] = []  # track goal_progress_pct across tasks
    suggested_trigram = ""  # Anqu's suggested gua for next task
    anqu_task_summary = ""  # Anqu's summary of previous task (for 思变)
    anqu_reason = ""  # Anqu's routing reason (for 思变)
    anqu_goal_progress_pct: int | None = None  # Anqu's progress assessment (for 思变)
    stagnation_attempts = 0  # blueprint review attempts after stagnation detection

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
                suggested_trigram=suggested_trigram,
                anqu_task_summary=anqu_task_summary,
                anqu_reason=anqu_reason,
                anqu_goal_progress_pct=anqu_goal_progress_pct,
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

        # 保存明觉的进度基线评估
        mingjue_progress = mingjue_output.goal_progress_pct or 0

        # ── 明觉记忆持久化 ──────────────────────────────────────
        _persist_memory(goal_id, task_id, "mingjue", {
            "summary": mingjue_output.summary,
            "concrete_goal": (mingjue_output.concrete_goal or "")[:300],
            "trigram": mingjue_output.trigram,
            "trigram_reason": mingjue_output.trigram_reason,
            "source_type": source.type,
            "goal_progress_pct": mingjue_progress,
        })

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
            total_tasks=task_count,
            mingjue_progress_pct=mingjue_progress,
        )

        # ── 暗驱记忆持久化 ──────────────────────────────────────
        _persist_memory(goal_id, task_id, "anqu", {
            "action": anqu_decision.action,
            "what_was_accomplished": (anqu_decision.task_summary or "")[:300],
            "next_task_description": (anqu_decision.next_task_description or "")[:200],
            "goal_progress_pct": anqu_decision.goal_progress_pct,
            "reason": (anqu_decision.continuation_context or anqu_decision.rework_instruction or "")[:300],
            "suggested_trigram": anqu_decision.suggested_trigram or "",
        })

        # ── 暗驱已知陷阱提案 → 写入待确认文件 ──────────────────
        if anqu_decision.known_traps_proposal:
            _write_pending_traps(goal_id, task_id, anqu_decision.known_traps_proposal)

        # ── 3.5 learn_task → DMN 认知演化 ──────────────────
        if anqu_decision.action == "learn_task":
            logger.info("[六爻] learn_task → 路由到 DMN 认知演化")
            # Create a default precipitate skill action if Anqu didn't provide one
            evolution = list(anqu_decision.evolution_actions) if anqu_decision.evolution_actions else []
            if not evolution:
                from vingobot.goal.grid_types import CognitionEvolutionAction
                evolution = [
                    CognitionEvolutionAction(
                        action="precipitate_skill",
                        target_name=f"task_experience_{task_id}",
                        description=f"沉淀任务 {task_id} 的执行经验为可复用技能",
                        priority=5,
                    )
                ]
            _process_evolution_actions(evolution, task_id, inner_result=inner_result, task_dir=str(task_dir))
            # Treat as task completed, route to goal_next_task
            anqu_decision = AnquDecision(
                action="goal_next_task",
                task_summary=anqu_decision.task_summary,
                next_task_description=anqu_decision.next_task_description or task_description,
                goal_progress_pct=anqu_decision.goal_progress_pct,
                suggested_trigram=anqu_decision.suggested_trigram,
                continuation_context=anqu_decision.continuation_context,
            )

        # ── 3.6 回炉子循环 ────────────────────────────────────
        # 暗驱要求继续/验证 → 同目录重启内循环，不创建新任务文件夹
        _MAX_REWORK_ATTEMPTS = 5
        rework_attempt = 0

        while anqu_decision.action in ("continue_task", "verify_task"):
            rework_attempt += 1
            if rework_attempt > _MAX_REWORK_ATTEMPTS:
                logger.warning(
                    "[六爻] 回炉达上限 ({} 次)，强制终止任务 {}",
                    _MAX_REWORK_ATTEMPTS, task_id,
                )
                anqu_decision = AnquDecision(
                    action="goal_failed",
                    task_summary=anqu_decision.task_summary,
                    failure_reason=f"回炉 {_MAX_REWORK_ATTEMPTS} 次后仍无法完成，强制终止",
                )
                break

            # ── 写入暗驱指令文件 ────────────────────────────
            _write_anqu_instruction(
                task_dir, anqu_decision, rework_attempt, inner_result=inner_result,
            )

            # ── 更新 manifest 状态 ───────────────────────────
            rework_status_map = {
                "continue_task": "in_progress",
                "verify_task": "verifying",
            }
            update_manifest_status(
                task_dir,
                status=rework_status_map[anqu_decision.action],
            )

            logger.info(
                "[六爻] 回炉 #{}/{}: {} → {} (指令已写入 05-anqu-instruction.md)",
                rework_attempt, _MAX_REWORK_ATTEMPTS,
                task_id, anqu_decision.action,
            )

            # ── 同目录重启内循环 ─────────────────────────────
            inner_result = await execute_task_inner_loop(
                task_dir=task_dir,
                mingjue_output=mingjue_output,
                goal_context=goal_context,
                signal=signal,
                max_rounds=max_rounds + max_rework_rounds,
                rework_attempt=rework_attempt,
                rework_action=anqu_decision.action,
            )
            total_rounds += inner_result.rounds_executed

            # ── 更新 manifest ─────────────────────────────────
            update_manifest_status(
                task_dir,
                "completed" if inner_result.task_completed else "failed",
                round_count=inner_result.rounds_executed,
            )

            # ── 再次运行暗驱评估 ─────────────────────────────
            anqu_decision = await run_anqu(
                goal_context=goal_context,
                current_task_facts=inner_result.facts,
                final_content=inner_result.final_content,
                task_dir=str(task_dir),
                signal=signal,
                cognitive_usage=inner_result.cognitive_usage,
                total_tasks=task_count,
                mingjue_progress_pct=mingjue_progress,
            )

            # ── 回炉记忆持久化 ───────────────────────────────
            _persist_memory(goal_id, task_id, "anqu_rework", {
                "action": anqu_decision.action,
                "rework_attempt": rework_attempt,
                "what_was_accomplished": (anqu_decision.task_summary or "")[:300],
                "goal_progress_pct": anqu_decision.goal_progress_pct,
                "reason": (anqu_decision.continuation_context or anqu_decision.rework_instruction or "")[:300],
            })

            # ── 处理认知演化动作 ─────────────────────────────
            _process_evolution_actions(
                anqu_decision.evolution_actions,
                task_id,
                inner_result=inner_result,
                task_dir=str(task_dir),
            )

        # ── 交付物自动聚拢 ──────────────────────────────────────
        _link_task_deliverables(str(task_dir), goal_id)

        # ── 总进度写入 ──────────────────────────────────────────
        _task_status = "completed" if inner_result.task_completed else "failed"
        if inner_result is not None and inner_result.final_content and "自读循环" in inner_result.final_content:
            _task_status = "auto_terminated"
        update_goal_progress(
            goal_id,
            task_id=task_id,
            task_status=_task_status,
            task_summary=(
                anqu_decision.task_summary
                or anqu_decision.next_task_description
                or ""
            )[:300],
            round_count=inner_result.rounds_executed,
            goal_progress_pct=(
                anqu_decision.goal_progress_pct
                if anqu_decision.goal_progress_pct is not None
                else mingjue_progress
            ),
            current_assessment=(
                anqu_decision.task_summary
                or anqu_decision.continuation_context
                or ""
            ),
            remaining_work=anqu_decision.next_task_description or "",
            total_tasks=task_count,
            goal_status=(
                "completed" if anqu_decision.action == "goal_completed"
                else "failed" if anqu_decision.action == "goal_failed"
                else "active"
            ),
        )

        # ── 任务完成计数 ──────────────────────────────────────
        try:
            meta = read_goal_meta(goal_id)
            if meta is not None:
                update_goal_meta(goal_id, rounds_completed=meta.rounds_completed + 1)
        except Exception:
            pass

        # Process cognitive evolution actions (enqueue as learning tasks)
        _process_evolution_actions(
            anqu_decision.evolution_actions,
            task_id,
            inner_result=inner_result,
            task_dir=str(task_dir),
        )

        # ── 4. Route based on Anqu decision ────────────────────
        action: AnquAction = anqu_decision.action

        if action == "goal_completed":
            update_goal_meta(goal_id, status="completed")
            return GoalResult(
                status="completed", goal_id=goal_id, reason=anqu_decision.task_summary
            )

        if action == "goal_failed":
            # ── 思变救援机会: 暗驱打算放弃,但在执行前给思变一次介入机会 ──
            # 当 needs_sibian=True (连续失败/任务数高/已知陷阱),思变可提供替代策略
            if anqu_decision.needs_sibian:
                sibian_decision = await run_sibian(
                    goal_context=goal_context,
                    anqu_decision=anqu_decision,
                    total_tasks=task_count,
                    goal_progress_history=goal_progress_history,
                    signal=signal,
                )
                _save_sibian_decision(goal_id, task_id, sibian_decision)

                # 思变给出了救援策略(非 abort 非 continue)→覆盖失败决策,继续推进
                if sibian_decision.action not in ("abort", "continue"):
                    task_description = anqu_decision.next_task_description or task_description
                    mode = "continuation"
                    previous_task_summary = anqu_decision.task_summary or ""
                    continuation_context = anqu_decision.continuation_context or ""
                    suggested_trigram = anqu_decision.suggested_trigram or ""

                    # ── 注入思变策略到明觉上下文 ─────────────────
                    strategy_text = sibian_decision.strategy or sibian_decision.reason or ""
                    if strategy_text:
                        action_labels = {
                            "push_through": "🚀 强攻突破",
                            "navigate_around": "🧭 绕行换路",
                            "wait_gather": "⏳ 主动等待",
                            "decompose": "📐 拆解降维",
                            "escalate": "🆘 升级求助",
                            "continue": "➡️ 照常推进",
                        }
                        label = action_labels.get(sibian_decision.action, "思变决策(目标失败救援)")
                        continuation_context += (
                            f"\n\n## {label}\n"
                            f"思变节点连山分析（目标失败救援）：\n"
                            f"- 六气: {sibian_decision.liuq}\n"
                            f"- 六甲: {sibian_decision.liujia}\n"
                            f"- 对峙: {sibian_decision.duizhi}\n"
                            f"- 策略: {strategy_text}\n"
                            f"- 时机: {sibian_decision.timing}\n"
                        )

                    # ── 卦象建议 ─────────────────────────────
                    if sibian_decision.trigram_hint:
                        suggested_trigram = sibian_decision.trigram_hint

                    # ── 蓝图微调（navigate_around / decompose 时可选） ──
                    if sibian_decision.blueprint_revision and sibian_decision.action in ("navigate_around", "decompose"):
                        _apply_blueprint_revision(goal_id, sibian_decision.blueprint_revision)
                        continuation_context += (
                            f"\n\n## 🔄 蓝图微调\n"
                            f"思变节点对蓝图的任务拆解/顺序进行了调整：\n"
                            f"{sibian_decision.blueprint_revision[:500]}\n"
                        )

                    logger.info(
                        "[六爻] 思变救援: 暗驱 goal_failed → 思变 {} → 覆盖为续行",
                        sibian_decision.action,
                    )
                    goal_context = refresh_goal_context(goal_id)
                    continue

                # 思变也同意放弃 → 执行暗驱的失败决策
                logger.info(
                    "[六爻] 思变确认放弃: 暗驱 goal_failed → 思变 {} → 执行终止",
                    sibian_decision.action,
                )

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

            # ── 链深度追踪 ──────────────────────────────────
            consecutive_continuations += 1

            # ── 滞涨检测: 两阶段 ─────────────────────────
            pct = anqu_decision.goal_progress_pct if anqu_decision.goal_progress_pct is not None else mingjue_progress
            if pct is not None:
                goal_progress_history.append(pct)
                _stagnation_warning = ""  # captured here, injected after variable assignments
                if len(goal_progress_history) >= 3:
                    last3 = goal_progress_history[-3:]
                    if max(last3) - min(last3) <= 5:
                        stagnation_attempts += 1
                        if stagnation_attempts >= 2:
                            # Stage 2: DMN review already attempted, still stuck → terminate
                            update_goal_meta(goal_id, status="completed")
                            return GoalResult(
                                status="completed",
                                goal_id=goal_id,
                                reason=(
                                    f"连续 {len(goal_progress_history)} 个任务目标进度停滞 ({last3})，"
                                    f"已触发 {stagnation_attempts - 1} 次蓝图重审，仍无进展，自动终止"
                                ),
                            )
                        # Stage 1: trigger DMN blueprint review, reset, continue
                        logger.warning(
                            "[六爻] 滞涨检测 (第 {} 次): 连续 {} 个任务进度停滞 {} → 触发 DMN 蓝图重审",
                            stagnation_attempts, len(goal_progress_history), last3,
                        )
                        _trigger_blueprint_review(
                            goal_id=goal_id,
                            task_id=task_id,
                            stagnation_history=goal_progress_history,
                        )
                        # Capture warning text — inject AFTER variable assignments below
                        _stagnation_warning = (
                            f"## ⚠️ 滞涨警告（第 {stagnation_attempts} 次）\n"
                            f"连续 {len(goal_progress_history)} 个任务的目标进度在 {last3} 之间停滞。"
                            f"已触发 DMN 蓝图重审任务。请考虑：\n"
                            f"- 当前任务拆解方式是否合理？\n"
                            f"- 是否需要更换策略方向？\n"
                            f"- 目标判定标准是否需要调整？\n"
                        )
                        goal_progress_history = []  # reset for fresh measurement after review

            task_description = anqu_decision.next_task_description or task_description
            mode = "continuation"
            previous_task_summary = anqu_decision.task_summary or ""
            continuation_context = anqu_decision.continuation_context or ""
            # ── 滞涨警告注入（在变量赋值之后，避免被覆盖）───
            if _stagnation_warning:
                continuation_context = _stagnation_warning + "\n" + continuation_context
            anqu_task_summary = anqu_decision.task_summary or ""
            anqu_reason = anqu_decision.continuation_context or anqu_decision.rework_instruction or ""
            anqu_goal_progress_pct = anqu_decision.goal_progress_pct
            # ── 注入第一步具体行动 ────────────────────────────
            if anqu_decision.next_task_concrete_action:
                continuation_context += (
                    f"\n\n**下一任务第一步具体行动**: {anqu_decision.next_task_concrete_action}"
                )
            # ── 传递卦象建议 ────────────────────────────────
            suggested_trigram = anqu_decision.suggested_trigram or ""

            # ── 5. 五爻·思变 (连山策略引擎) ─────────────────
            # 暗驱触发：仅当 Anqu 检测到跨任务停滞/失败模式时运行
            if anqu_decision.needs_sibian:
                sibian_decision = await run_sibian(
                    goal_context=goal_context,
                    anqu_decision=anqu_decision,
                    total_tasks=task_count,
                    goal_progress_history=goal_progress_history,
                    signal=signal,
                )
                _save_sibian_decision(goal_id, task_id, sibian_decision)

                if sibian_decision.action == "abort":
                    update_goal_meta(goal_id, status="failed")
                    return GoalResult(
                        status="failed",
                        goal_id=goal_id,
                        reason=f"思变节点终止: {sibian_decision.reason}",
                    )

                # ── 注入思变策略到明觉上下文 ─────────────────
                strategy_text = sibian_decision.strategy or sibian_decision.reason or ""
                if strategy_text:
                    action_labels = {
                        "push_through": "🚀 强攻突破",
                        "navigate_around": "🧭 绕行换路",
                        "wait_gather": "⏳ 主动等待",
                        "decompose": "📐 拆解降维",
                        "escalate": "🆘 升级求助",
                        "continue": "➡️ 照常推进",
                    }
                    label = action_labels.get(sibian_decision.action, "思变决策")
                    continuation_context += (
                        f"\n\n## {label}\n"
                        f"思变节点连山分析：\n"
                        f"- 六气: {sibian_decision.liuq}\n"
                        f"- 六甲: {sibian_decision.liujia}\n"
                        f"- 对峙: {sibian_decision.duizhi}\n"
                        f"- 策略: {strategy_text}\n"
                        f"- 时机: {sibian_decision.timing}\n"
                    )

                # ── 卦象建议 ─────────────────────────────
                if sibian_decision.trigram_hint:
                    suggested_trigram = sibian_decision.trigram_hint

                # ── 蓝图微调（navigate_around / decompose 时可选） ──
                if sibian_decision.blueprint_revision and sibian_decision.action in ("navigate_around", "decompose"):
                    _apply_blueprint_revision(goal_id, sibian_decision.blueprint_revision)
                    continuation_context += (
                        f"\n\n## 🔄 蓝图微调\n"
                        f"思变节点对蓝图的任务拆解/顺序进行了调整：\n"
                        f"{sibian_decision.blueprint_revision[:500]}\n"
                    )

                # ── 上下文刷新 ──────────────────────────
                if sibian_decision.timing == "after_refresh":
                    logger.info("[六爻] 思变建议 after_refresh，刷新目标上下文")
                goal_context = refresh_goal_context(goal_id)
            else:
                goal_context = refresh_goal_context(goal_id)

            continue

        # ── If still in rework after max attempts → was caught above as goal_failed
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


def _build_rework_diagnostics(facts: list) -> str:
    """Build a diagnostic summary from execution facts for rework guidance.

    Analyses what went wrong in the failed task and provides specific,
    actionable advice for the rework attempt.
    """
    from vingobot.goal.types import RoundExecutionFact

    if not facts:
        return ""

    round_count = len(facts)
    successes = sum(1 for f in facts if getattr(f, "execution_status", "") == "success")
    failures = sum(
        1 for f in facts
        if getattr(f, "execution_status", "") in ("failure", "partial_failure", "exec_failed")
    )
    exec_failures = sum(
        1 for f in facts
        if getattr(f, "execution_status", "") == "exec_failed"
    )
    read_only_rounds = sum(
        1 for f in facts
        if getattr(f, "tool_call_count", 0) > 0
        and getattr(f, "execution_status", "") in ("success", "skipped")
    )

    lines: list[str] = ["## 上一轮执行诊断"]

    # ── Exec failure degrade ladder (must precede general patterns) ──
    if exec_failures >= 2:
        lines.append(f"检测到连续 {exec_failures} 次 exec 失败（超时或非零退出码）。")
        lines.append("建议回炉策略：")
        lines.append("1. **禁止再次执行脚本**——降级为手工产出")
        lines.append("2. 用 write_file 写入分析报告/简报/文档作为降级交付物")
        lines.append("3. 如需修复脚本，用 edit_file 修改后用 task_complete 提交")
        lines.append("4. 禁止用 read_file 读取自己的执行事实文件来\"理解问题\"")
    elif exec_failures == 1:
        lines.append(f"检测到 1 次 exec 失败。")
        lines.append("建议回炉策略：")
        lines.append("1. 若脚本自身有问题，用 edit_file 修复后重试一次")
        lines.append("2. 若修复无效，降级为 write_file 手工产出")
        lines.append("3. **不要**反复读取执行事实文件来\"理解错误\"——直接行动")

    # Pattern detection
    if round_count >= 12 and failures > 0:
        lines.append(f"检测到自读循环模式：{round_count}轮中只读{read_only_rounds}轮，{failures}轮失败。")
        lines.append("建议回炉策略：")
        lines.append("1. 第一轮直接调用 write_file 产出文件，不要先读")
        lines.append("2. 仅读取绝对必要的 1-2 个文件后立即产出")
        lines.append("3. 产出后立即调用 task_complete 结束")
    elif failures >= 3:
        lines.append(f"检测到高失败率：{failures}/{round_count} 轮执行失败。")
        lines.append("建议回炉策略：")
        lines.append("1. 减少工具调用数量，每次只调用 1 个 write_file")
        lines.append("2. 先产出最小可行版本，不要追求完美")
    elif read_only_rounds >= round_count * 0.8:
        lines.append(f"检测到过度读取：{read_only_rounds}/{round_count} 轮为纯读取操作。")
        lines.append("建议回炉策略：首轮直接产出文件，跳过探索性读取。")
    elif round_count >= 20:
        lines.append(f"检测到轮次过多（{round_count}轮）。")
        lines.append("建议回炉策略：采用最小可行策略，2-3 轮内产出并结束。")
    else:
        lines.append(f"执行概况：{round_count}轮，{successes}成功/{failures}失败。")
        lines.append("建议回炉策略：聚焦核心产出，减少探索操作。")

    return "\n".join(lines)


def _write_anqu_instruction(
    task_dir: Path,
    anqu_decision: AnquDecision,
    rework_attempt: int,
    *,
    inner_result: InnerLoopResult | None = None,
) -> None:
    """Write 05-anqu-instruction.md into the task directory for Yang to read on restart.

    Yang reads this file on the first round of a rework cycle to understand
    the specific goal (continue / verify / learn) without needing Mingjue.
    """
    from datetime import datetime, timezone

    task_dir = Path(task_dir)
    action = anqu_decision.action
    instruction_text = anqu_decision.rework_instruction or anqu_decision.continuation_context or ""

    action_label = {
        "continue_task": "继续执行",
        "verify_task": "验证产出",
        "learn_task": "沉淀经验",
    }.get(action, action)

    lines = [
        f"# 暗驱回炉指令",
        f"",
        f"- **生成时间**: {datetime.now(timezone.utc).isoformat()}",
        f"- **回炉次数**: 第 {rework_attempt} 次",
        f"- **回炉类型**: {action}（{action_label}）",
        f"",
        f"## 暗驱评估",
        f"",
        f"{anqu_decision.task_summary or '(无)'}",
        f"",
        f"## 具体指令",
        f"",
        f"{instruction_text or '请继续推进任务。'}",
        f"",
        f"## 执行要求",
        f"",
    ]

    if action == "verify_task":
        lines.extend([
            f"- 你需要**验证上一轮产出物的正确性**，而不是创建新文件。",
            f"- 优先使用 exec 运行验证脚本（如 ffprobe、pytest、python 脚本检查）。",
            f"- exec 运行失败是预期行为——它暴露了问题，不要因为 exec 返回非零就放弃。",
            f"- 如果 exec 验证成功，用 task_complete 上报验证结果。",
            f"- 如果 exec 验证失败，用 edit_file 修复产出物后再次 exec 验证。",
            f"- **不要**写新文件来替代旧产出——编辑修复已有的文件。",
        ])
    elif action == "continue_task":
        # Inject diagnostics from previous execution
        if inner_result is not None:
            diagnostics = _build_rework_diagnostics(inner_result.facts)
            if diagnostics:
                lines.append(diagnostics)
                lines.append("")
        lines.extend([
            f"- 上一轮任务未完成，请在同一目录下继续工作。",
            f"- 首先检查 outputs/ 目录中已有的产出物。",
            f"- 优先用 edit_file 修复已有文件，而不是创建新文件。",
            f"- 如果确实需要新文件，确保是推进任务的必要产出。",
        ])
    elif action == "learn_task":
        lines.extend([
            f"- 上一轮任务已完成，现在需要提炼经验为可复用的认知资产。",
            f"- 回顾 outputs/ 目录中的产出物和执行路径。",
            f"- 用 write_file 创建 L1 技能 (SKILL.md) 或 L2 模型文件。",
            f"- 产出物写入目标 deliverable/ 目录或认知库目录。",
        ])

    try:
        (task_dir / "05-anqu-instruction.md").write_text(
            "\n".join(lines), encoding="utf-8",
        )
        logger.info("[六爻] 写入暗驱指令: {}", task_dir / "05-anqu-instruction.md")
    except OSError as exc:
        logger.error("[六爻] 写入暗驱指令失败: {}", exc)


def _persist_memory(
    goal_id: str,
    task_id: str,
    entry_type: str,
    data: dict,
) -> None:
    """Write a goal-level memory entry (Mingjue or Anqu decision).

    Stored as ``memory/{entry_type}-{task_id}.json`` under the goal directory.
    Consumed by ``_read_memory_summary`` in the next task's context.
    """
    from vingobot.core.workspace import get_goal_dir

    memory_dir = get_goal_dir(goal_id) / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "type": entry_type,
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }

    path = memory_dir / f"{task_id}__{entry_type}.json"
    try:
        path.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("[六爻] 写入{}记忆: {}", entry_type, path)
    except OSError:
        logger.warning("[六爻] 写入{}记忆失败: {}", entry_type, path)


def _apply_blueprint_revision(goal_id: str, revision_content: str) -> None:
    """Apply a blueprint revision from Sibian to the goal's blueprint file.

    Writes to ``<goal_dir>/blueprint.md``.  Also creates a revision history entry
    at ``<goal_dir>/blueprint-revisions/`` for audit.
    """
    from vingobot.core.workspace import get_goal_dir

    goal_dir = get_goal_dir(goal_id)
    bp_file = goal_dir / "blueprint.md"
    try:
        bp_file.write_text(revision_content, encoding="utf-8")
        logger.info("[六爻] 思变修订蓝图: {}", bp_file)

        # ── 记录修订历史 ───────────────────────────────────
        rev_dir = goal_dir / "blueprint-revisions"
        rev_dir.mkdir(parents=True, exist_ok=True)
        rev_file = rev_dir / f"sibian-revision-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.md"
        rev_file.write_text(revision_content, encoding="utf-8")
    except OSError as exc:
        logger.error("[六爻] 写入蓝图修订失败: {}", exc)


def _save_sibian_decision(goal_id: str, task_id: str, decision: SibianDecision) -> None:
    """Persist Sibian's Lianshan decision to ``<goal_dir>/sibian/`` for audit / review.

    Saves the full SibianDecision (action, liuq, liujia, sanyuan, duizhi,
    strategy, trigram_hint, timing, reason, blueprint_revision) as JSON.
    """
    from vingobot.core.workspace import get_goal_dir

    goal_dir = get_goal_dir(goal_id)
    sibian_dir = goal_dir / "sibian"
    sibian_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "task_id": task_id,
        "action": decision.action,
        "reason": decision.reason,
        "strategy": decision.strategy,
        "liuq": decision.liuq,
        "liujia": decision.liujia,
        "sanyuan": decision.sanyuan,
        "duizhi": decision.duizhi,
        "trigram_hint": decision.trigram_hint,
        "timing": decision.timing,
        "blueprint_revision": decision.blueprint_revision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path = sibian_dir / f"sibian-decision-{task_id}.json"
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("[六爻] 保存思变决策: {}", path)
    except OSError:
        logger.warning("[六爻] 保存思变决策失败: {}", path)


def _process_evolution_actions(
    actions: list[CognitionEvolutionAction],
    source_task_id: str,
    *,
    inner_result: InnerLoopResult | None = None,
    task_dir: str = "",
) -> None:
    """Enqueue cognitive evolution tasks under the special ``cognition-evolution`` goal.

    Each ``CognitionEvolutionAction`` is written as a task file to the
    ``.taiji/pending/`` directory, prefixed with priority for ordering.
    The DMN consumer will pick them up and process them asynchronously.

    When ``inner_result`` and ``task_dir`` are provided, the task description
    is enriched with execution facts so the DMN LLM can analyze real task
    data when creating cognitive assets.
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
            task_desc = _build_evolution_task_description(
                action,
                inner_result=inner_result,
                task_dir=task_dir,
            )

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


def _trigger_blueprint_review(
    goal_id: str,
    task_id: str,
    stagnation_history: list[int],
) -> None:
    """Enqueue a DMN blueprint review task when stagnation is detected.

    Creates a high-priority ``review_blueprint`` cognition evolution task
    that asks DMN to re-examine the blueprint, trajectory, and execution
    history to decide whether the goal needs re-scoping, re-decomposition,
    or should be declared blocked.
    """
    try:
        from vingobot.core.workspace import get_workspace_paths
        from vingobot.goal.grid_types import CognitionEvolutionAction

        wp = get_workspace_paths()
        pending_dir = wp.pending
        pending_dir.mkdir(parents=True, exist_ok=True)

        action = CognitionEvolutionAction(
            action="review_blueprint",
            target_name=f"goal_{goal_id}_blueprint",
            description=(
                f"目标 {goal_id} 连续 {len(stagnation_history)} 个任务进度停滞在 "
                f"{stagnation_history}，需要重审蓝图。请分析："
                f"1) 目标拆解是否合理 2) 当前策略是否有效 "
                f"3) 是否需要修改完成判定标准 4) 是否应该标记为 blocked"
            ),
            source_task_id=task_id,
            source_goal_id=goal_id,
            priority=9,  # highest priority — blocked goal needs immediate attention
            context={"stagnation_history": stagnation_history},
        )

        filename = (
            f"cognition-evolution__09__"
            f"review_blueprint__goal_{goal_id}_blueprint__"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.task"
        )

        task_file = pending_dir / filename
        task_desc = _build_evolution_task_description(action)
        task_file.write_text(task_desc, encoding="utf-8")

        logger.info(
            "[六爻] 入队蓝图重审任务: goal={} stagnation_rounds={} history={}",
            goal_id,
            len(stagnation_history),
            stagnation_history,
        )

    except Exception as exc:
        logger.warning("[六爻] 创建蓝图重审任务失败: {}", exc)


def _resolve_written_path(path: str, task_dir: str) -> str:
    """Resolve a write_file path to an absolute path for cross-task use.

    Yang may write files with relative paths (e.g. "outputs/report.md").
    When these paths are injected into the next task's system prompt,
    they must be absolute so that read_file resolves correctly regardless
    of the current working directory.
    """
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    resolved = (Path(task_dir) / p).resolve()
    if resolved.is_file():
        return str(resolved)
    # Fallback: file may have been written but not yet flushed, return raw
    return str(resolved)


def _scan_task_outputs(task_dir: str) -> list[str]:
    """Scan a completed task's output files for written deliverables.

    Reads all round JSON files in ``outputs/`` and extracts file paths
    from successful ``write_file`` calls, deduplicated and sorted.
    """
    if not task_dir:
        return []
    out_dir = Path(task_dir) / "outputs"
    if not out_dir.is_dir():
        return []

    written: set[str] = set()
    for rf in sorted(out_dir.glob("*-round.json")):
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        # Scan approved_calls + results (exec rounds)
        approved = data.get("approved_calls") or []
        results = data.get("results") or []
        for call, result in zip(approved, results):
            if call.get("name") == "write_file" and result.get("status") == "success":
                path = _resolve_written_path(call.get("arguments", {}).get("path", ""), task_dir)
                if path:
                    written.add(path)

        # Also scan raw tool_calls (pre-approval, for task_complete rounds)
        for tc in data.get("tool_calls") or []:
            if tc.get("name") == "write_file":
                path = _resolve_written_path(tc.get("arguments", {}).get("path", ""), task_dir)
                if path:
                    written.add(path)

    return sorted(written)


def _link_task_deliverables(task_dir: str, goal_id: str) -> int:
    """Scan task outputs and copy written files to goal deliverables.

    Automatically harvests files written during task execution and copies
    them to ``<goal_dir>/deliverables/`` for cross-task reuse.  Only
    copies files that are not already in the deliverables directory.

    Returns:
        Number of files linked.
    """
    written = _scan_task_outputs(task_dir)
    if not written:
        return 0

    goal_dir = get_goal_dir(goal_id)
    dlv_dir = goal_dir / "deliverables"
    dlv_dir.mkdir(exist_ok=True)

    # Collect existing deliverable names for dedup
    existing: set[str] = set()
    if dlv_dir.is_dir():
        for f in dlv_dir.iterdir():
            if f.is_file():
                existing.add(f.name)

    linked = 0
    for src_path_str in written:
        src = Path(src_path_str)
        if not src.is_file():
            continue

        # Skip files already in deliverables/
        src_name = src.name
        if src_name in existing:
            continue

        # Skip files that were written directly to deliverables/ (already there)
        try:
            if dlv_dir.resolve() in src.resolve().parents:
                continue
        except Exception:
            pass

        dst = dlv_dir / src_name
        try:
            import shutil
            shutil.copy2(src, dst)
            linked += 1
            existing.add(src_name)
            logger.info("[六爻] 交付物归档: {} → {}", src.name, dst)
        except Exception as exc:
            logger.warning("[六爻] 交付物归档失败 {}: {}", src.name, exc)

    if linked:
        logger.info("[六爻] 已归档 {} 个交付物到 {}", linked, dlv_dir)
    return linked


def _build_evolution_task_description(
    action: CognitionEvolutionAction,
    *,
    inner_result: InnerLoopResult | None = None,
    task_dir: str = "",
) -> str:
    """Build a human-readable task description from an evolution action.

    When ``inner_result`` and ``task_dir`` are provided, the description is
    enriched with execution facts so the DMN LLM can:
    - Analyze what the task actually accomplished
    - Read round-by-round execution data from the task directory
    - Extract patterns for skills, insight for models, and structure for grids
    """
    lines = [
        f"# 认知演化任务: {action.action}",
        f"目标: {action.target_name}",
        f"描述: {action.description}",
        f"来源任务: {action.source_task_id}",
        f"来源目标: {action.source_goal_id}",
        f"优先级: {action.priority}/10",
        "",
    ]

    # ── Enrich with execution data ──────────────────────────────
    if task_dir:
        lines.append(f"## 源任务执行数据")
        lines.append(f"")
        lines.append(f"源任务目录: {task_dir}")
        lines.append(f"执行事实文件: {task_dir}/06-execution-facts.json")
        lines.append(f"输出文件目录: {task_dir}/outputs/")
        lines.append(f"清单文件: {task_dir}/manifest.json")
        lines.append(f"")

        if inner_result is not None:
            facts = inner_result.facts
            rounds = inner_result.rounds_executed
            tool_calls = sum(f.tool_call_count for f in facts)
            successes = sum(1 for f in facts if f.execution_status == "success")
            failures = sum(1 for f in facts if f.execution_status in ("failure", "partial_failure", "exec_failed"))

            lines.append(f"- 总轮次: {rounds}")
            lines.append(f"- 工具调用: {tool_calls} 次")
            lines.append(f"- 成功轮次: {successes}")
            lines.append(f"- 失败轮次: {failures}")
            lines.append(f"")

            # Cognitive usage
            cu = inner_result.cognitive_usage
            if cu is not None:
                if cu.grids_loaded:
                    lines.append(f"- 已加载的认知格栅: {', '.join(cu.grids_loaded)}")
                if cu.skills_used:
                    used_set = [s for s in cu.skills_used if s != "__searched__"]
                    if used_set:
                        lines.append(f"- 已使用的技能: {', '.join(used_set)}")
                if cu.models_loaded:
                    lines.append(f"- 已加载的思维模型: {', '.join(cu.models_loaded)}")
                if cu.tools_failed:
                    lines.append(f"- 调用失败的工具: {', '.join(cu.tools_failed)}")
                lines.append(f"")

            # Round-by-round summary
            if facts:
                lines.append(f"## 执行轮次摘要")
                lines.append(f"")
                for f in facts:
                    status_icon = {"success": "✅", "failure": "❌", "partial_failure": "⚠️", "exec_failed": "🕐", "skipped": "⏭"}.get(f.execution_status, "?")
                    intent = f.yang_intent_summary[:100]
                    lines.append(f"- 第{f.round}轮 {status_icon} {f.execution_status}: {intent}")
                    if f.tool_call_count > 0:
                        lines.append(f"  工具 {f.tool_call_count} 个 | 审批: {f.yin_decision}")
                    if f.execution_result_summary:
                        lines.append(f"  结果: {f.execution_result_summary[:200]}")
                lines.append(f"")
    else:
        lines.append(f"## 说明")
        lines.append(f"")
        lines.append(f"无源任务执行数据（可能是手动触发的演化任务）。")
        lines.append(f"")

    # Extra context from Anqu
    if action.context:
        lines.append("## 上下文")
        import json

        lines.append(json.dumps(action.context, ensure_ascii=False, indent=2))
        lines.append("")
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
    elif action.action == "research":
        lines.extend(
            [
                "## 执行说明",
                "1. 分析目标任务中提出的具体问题或失败模式",
                "2. 使用 web_search 搜索最佳实践、替代方案或修复方法",
                "3. 验证搜索到的方案在目标平台（如Windows）的兼容性",
                "4. 如果找到有效方案，创建或更新对应的L1技能",
                "5. 更新认知导航索引",
            ]
        )
    elif action.action == "investigate":
        lines.extend(
            [
                "## 执行说明",
                "1. 读取源任务目录的执行事实文件（06-execution-facts.json）",
                "2. 分析失败轮次的执行结果和工具调用记录",
                "3. 识别失败根因模式（工具配置、路径、权限、平台兼容性等）",
                "4. 根据分析结论创建认知资产（L2模型用于沉淀根因模式，或L3格栅更新）",
                "5. 更新认知导航索引",
            ]
        )
    elif action.action == "review_blueprint":
        lines.extend(
            [
                "## 执行说明",
                "1. 读取目标蓝图文件 (blueprint.md) 和目标轨迹 (trajectory.md)",
                "2. 分析所有已完成任务的执行结果和进度停滞原因",
                "3. 判断：目标定义是否有问题？拆解是否合理？当前策略是否有效？",
                "4. 如果蓝图需要调整 — 用 edit_file 修改 blueprint.md（调整范围、重新拆解、修改判定标准）",
                "5. 如果目标已经实际完成但判定标准过于严格 — 更新完成判定标准",
                "6. 如果目标确实无法达成 — 在 blueprint.md 中标注为 'blocked' 并给出阻塞原因",
                "7. 产出重审报告到目标目录: deliverable/blueprint-review.md",
            ]
        )

    return "\n".join(lines)


def _write_pending_traps(
    goal_id: str,
    task_id: str,
    proposals: list[dict],
) -> None:
    """Write Anqu's known_traps proposals to ``pending_traps.json`` for user confirmation.

    Each proposal is enriched with source metadata (task_id, timestamp) and
    a ``confirmed`` flag (always false when first written).

    The file is stored in the goal directory and can be listed/read by the
    main loop to present proposals to the user for confirmation.
    """
    from vingobot.core.workspace import get_goal_dir

    goal_dir = get_goal_dir(goal_id)
    path = goal_dir / "pending_traps.json"

    # Load existing pending traps if any
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []

    # De-duplicate: skip proposals with same name already in pending
    existing_names = {e.get("name", "") for e in existing}
    added_count = 0
    for proposal in proposals:
        name = proposal.get("name", "")
        if name and name in existing_names:
            logger.debug("[六爻] 跳过重复陷阱提案: {}", name)
            continue
        entry = dict(proposal)
        entry.setdefault("proposed_by", "anqu")
        entry["proposed_at"] = datetime.now(timezone.utc).isoformat()
        entry["source_task_id"] = task_id
        entry["confirmed"] = False
        existing.append(entry)
        existing_names.add(name)
        added_count += 1

    if added_count == 0:
        logger.debug("[六爻] 所有陷阱提案已存在，跳过写入")
        return

    try:
        path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("[六爻] 写入 {} 个待确认陷阱提案到 {}", added_count, path)
    except OSError:
        logger.warning("[六爻] 写入待确认陷阱提案失败: {}", path)
