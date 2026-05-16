"""
上爻·暗驱 — Goal-level decision maker.

Anqu is the highest authority in the sixiang loop.  It receives the
complete context of the just-completed task loop and must answer two
questions definitively:

1. What did the current task accomplish?
2. Has the OVERALL GOAL been achieved?

Based on the answers, Anqu routes the outer loop:
- ``goal_next_task`` — Goal not done, generate the next task.
- ``goal_completed`` — All completion criteria satisfied.
- ``goal_failed`` — Goal cannot be achieved.
- ``continue_task`` / ``verify_task`` / ``learn_task`` — Current task
  needs rework.

**Cognitive evolution** — Anqu is also the natural place to evaluate
whether the cognition layer (L1/L2/L3) needs updating.  Three triggers
baked into Anqu's evaluation:

A. **tools_failed** → ``learn_skill`` — task failed because a skill or
   tool was missing.
B. **successful SOP** → ``precipitate_skill`` — the task's approach is
   reusable and should be captured as an L1 skill.
C. **valuable methodology** → ``precipitate_model`` — the task produced
   insights worth preserving as an L2 experience model.

Anqu outputs 0-N ``CognitionEvolutionAction`` items alongside its
routing decision.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.core.goal_context import GoalContext
from vingobot.core.workspace import get_workspace_paths
from vingobot.goal.grid_types import CognitionEvolutionAction, CognitionUsage
from vingobot.goal.types import AnquAction, AnquDecision, RoundExecutionFact
from vingobot.goal.yin import ABANDONMENT_KEYWORDS, BLUEPRINT_TARGET_PATTERNS


# ---------------------------------------------------------------------------
# Blueprint completion verification — deterministic comparison helper
# ---------------------------------------------------------------------------

# NOTE: BLUEPRINT_TARGET_PATTERNS and ABANDONMENT_KEYWORDS are imported
# from vingobot.goal.yin (single source of truth).


def _verify_blueprint_completion(
    blueprint: str,
    recent_task_statuses: list,
    memory: str,
    trajectory: str,
    total_tasks: int,
) -> str:
    """Build a deterministic blueprint-vs-reality comparison for Anqu.

    Parses the blueprint for quantitative completion targets and compares
    against what the task chain has actually produced.  Also checks for
    self-abandonment contamination in task status summaries.

    Returns a Markdown-formatted comparison section to inject into Anqu's
    system prompt.  Returns empty string if no quantifiable targets found.
    """
    if not blueprint or blueprint == "(无蓝图)":
        return ""

    # ── Extract numeric targets from blueprint ───────────────
    targets: list[tuple[str, int]] = []
    for pattern in BLUEPRINT_TARGET_PATTERNS:
        for m in pattern.finditer(blueprint):
            try:
                count = int(m.group(1))
                if count >= 1:
                    context = blueprint[max(0, m.start() - 30):m.end() + 30]
                    targets.append((context.strip(), count))
            except (ValueError, IndexError):
                continue

    # Deduplicate
    unique_targets: dict[str, int] = {}
    for ctx, count in targets:
        key = ctx[:50]
        if key not in unique_targets or count > unique_targets[key]:
            unique_targets[key] = count

    # ── Scan recent task statuses for abandonment ────────────
    abandonment_signals: list[str] = []
    for task in (recent_task_statuses or []):
        task_id = getattr(task, "task_id", "?")
        status = getattr(task, "status", "?")
        snippet = (getattr(task, "summary_snippet", "") or "").lower()
        for kw in ABANDONMENT_KEYWORDS:
            if kw.lower() in snippet:
                abandonment_signals.append(
                    f"任务 {task_id} ({status}): ...{kw}..."
                )
                break

    # ── Check for user confirmation ─────────────────────────
    combined_text = (memory + " " + trajectory).lower()
    user_confirmed = any(kw in combined_text for kw in (
        "ask_user", "用户确认", "user confirmed",
    ))

    # ── Build comparison section ────────────────────────────
    lines: list[str] = ["## 蓝图完成验证（阴·主动仲裁提供的对照数据）\n"]

    if unique_targets:
        lines.append("### 从蓝图解析出的量化目标")
        for ctx, count in list(unique_targets.items())[:5]:
            lines.append(f"- 预期 **{count}** 个: `{ctx}...`")
        lines.append("")
        lines.append(
            f"### 实际执行情况\n"
            f"- 已完成任务总数: **{total_tasks}**\n"
            f"- 各任务产出摘要请见上方「近期已完成任务」\n"
        )
    else:
        lines.append("（未能从蓝图中解析出量化目标，请基于蓝图整体内容判断）\n")

    if abandonment_signals:
        lines.append("### ⚠️ 检测到自我放弃污染")
        lines.append("以下任务产出中包含放弃/跳过/不可行等语言：")
        for sig in abandonment_signals[:5]:
            lines.append(f"- {sig}")
        lines.append("")
        if user_confirmed:
            lines.append("✓ 目标记忆/轨迹中检测到用户确认记录 — 放弃可能是用户认可的")
        else:
            lines.append("**✗ 未检测到用户确认记录！** 这些放弃声明可能未经用户认可。")
            lines.append("在判定 goal_completed 之前，请仔细核实这些放弃是否合理。")
            lines.append("")

    lines.append(
        "### 判定提醒\n"
        "1. 如果蓝图有明确的量化目标但实际任务产出远低于目标 → 不应判定为 goal_completed\n"
        "2. 如果存在自我放弃信号且无用户确认 → 应继续推进（goal_next_task），不应关闭目标\n"
        "3. 如果用户确实认可了范围的缩小 → 应记录确认证据后再判定 goal_completed\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 强制验证回炉 — 首次 task_complete 检测 + verify instruction
# ---------------------------------------------------------------------------


def _needs_verification(task_dir: str) -> bool:
    """Check if the just-completed task needs a mandatory verification round.

    Every task's first ``task_complete`` triggers a ``verify_task`` rework.
    The marker file ``.anqu_verified`` is created when the first
    ``verify_task`` is issued, so subsequent Anqu calls on the same task
    (after the verification round completes) know verification is done.
    """
    if not task_dir:
        return False
    return not (Path(task_dir) / ".anqu_verified").exists()


def _mark_verification_pending(task_dir: str) -> None:
    """Create the verification marker so re-entry skips to evaluation."""
    try:
        Path(task_dir).mkdir(parents=True, exist_ok=True)
        (Path(task_dir) / ".anqu_verified").write_text("")
        logger.debug("[暗驱] 已写入验证标记: {}", task_dir)
    except OSError:
        logger.warning("[暗驱] 无法写入验证标记: {}", task_dir)


def _build_verify_instruction(
    goal_context: GoalContext,
) -> str:
    """Build the verification instruction for the rework round.

    Instructs Yang to validate every output file in the task's ``outputs/``
    directory — checking integrity via appropriate commands — and produce
    a structured quality report before calling ``task_complete``.
    """
    bp = (goal_context.blueprint_summary or "")[:1000]

    return (
        "## 📋 强制验证指令\n\n"
        "这是任务首次完成后的**强制性验证回炉**。你必须验证上一轮产出物的质量和完整性，"
        "而不是创建新的内容文件。\n\n"
        "### 验证步骤（必须按顺序执行）\n\n"
        "1. **列出产出** — 用 `list_directory` 查看 task_dir/outputs/ 目录，确认有哪些产出文件\n"
        "2. **完整性检查** — 对每个产出文件执行适当的验证命令：\n"
        "   - 视频文件 (.mp4/.mov/.avi/.webm): `ffprobe -v error <file>` — 确认文件完整\n"
        "   - 视频分辨率与音轨: `ffprobe -v quiet -print_format json -show_streams <file>` — 确认有视频流+音频流\n"
        "   - Python 脚本 (.py): `python -c 'compile(open(\"<file>\").read(), \"<file>\", \"exec\")'` — 确认语法正确\n"
        "   - Shell 脚本 (.sh): `bash -n <file>` — 确认语法正确\n"
        "   - JSON 文件 (.json): `python -c 'import json; json.load(open(\"<file>\"))'` — 确认格式有效\n"
        "3. **内容抽查** — 对文本类产出文件，用 `read_file` 读取内容片段，确认非空、内容合理\n"
        "4. **写入验证报告** — 用 `write_file` 将以下内容写入 `outputs/quality-report.md`：\n"
        "   - 被验证的产出文件列表（文件路径 + 大小）\n"
        "   - 每个文件的完整性/语法检查结果（通过/失败）\n"
        "   - 截图/抽样的观察结果\n"
        "   - 总体结论：**全部通过** 或 **部分失败需修复**\n\n"
        "### 蓝图质量参考\n\n"
        f"```\n{bp}\n```\n\n"
        "### 结束条件\n\n"
        "- 所有验证通过 → 调用 `task_complete` 结束任务\n"
        "- 发现文件损坏或格式问题 → 先用 `edit_file` 修复产出，"
        "修复后再次验证，确认通过后调用 `task_complete`\n"
        "- **不要创建新的内容文件** — 这是验证轮次，不是生产轮次\n"
        "- **不要把 exec 验证失败当作任务失败** — exec 返回非零 = 发现问题，"
        "这是预期行为，修复后继续即可\n"
    )


async def run_anqu(
    goal_context: GoalContext,
    current_task_facts: list[RoundExecutionFact],
    final_content: str | None,
    *,
    task_dir: str = "",
    signal: asyncio.Task | None = None,
    cognitive_usage: CognitionUsage | None = None,
    total_tasks: int = 0,
    mingjue_progress_pct: int | None = None,
) -> AnquDecision:
    """Evaluate the current task result and decide the goal's next step.

    Args:
        goal_context: Full goal context snapshot.
        current_task_facts: Round facts from the just-completed task.
        final_content: Yang's final content (if task_complete was called).
        task_dir: The just-completed task's working directory (for output inspection).
        signal: Optional cancellation token.
        cognitive_usage: Cognitive assets used during this task (for evolution).
        total_tasks: Exact task count from the outer loop (used for fallback stop conditions).
        mingjue_progress_pct: Mingjue's baseline progress assessment (0-100) for the
            goal.  Anqu can confirm or adjust it based on actual task results.

    Returns:
        AnquDecision routing the outer loop, optionally with evolution_actions.
    """

    # ── Fast-path: task didn't complete ──────────────────────────
    if final_content is None:
        # Task ran out of rounds → rework with concrete diagnostics
        round_count = len(current_task_facts)
        successes = sum(1 for f in current_task_facts if f.execution_status == "success")
        failures = sum(1 for f in current_task_facts if f.execution_status in ("failure", "partial_failure", "exec_failed"))
        read_only_rounds = sum(
            1 for f in current_task_facts
            if f.tool_call_count > 0 and f.execution_status == "success"
            and f.had_action_request
        )
        # Detect self-read loop pattern
        is_read_loop = round_count >= 12 and failures > 0
        if is_read_loop:
            instruction = (
                f"上一任务在 {round_count} 轮后因自读循环自动终止。"
                "请在下一轮中：1) 首轮直接调用 write_file 写入成果文件，避免纯读取；"
                "2) 使用 list_directory 快速定位所需文件而非逐文件读取；"
                "3) 收集 2-3 个关键文件后就着手产出，不必读完所有文件。"
            )
        else:
            instruction = (
                f"上一任务在 {round_count} 轮后未完成（{successes} 成功/{failures} 失败）。"
                "请在下一轮聚焦最小可行产出，减少探索性读取，优先使用 write_file 交付成果。"
            )
        return AnquDecision(
            action="continue_task",
            task_summary=f"任务在第 {round_count} 轮后仍未完成",
            rework_instruction=instruction,
        )

    # ── 强制验证回炉：任何任务的第一次 task_complete → verify_task ──
    if _needs_verification(task_dir):
        _mark_verification_pending(task_dir)
        logger.info("[暗驱] 首次完成 → 强制验证回炉 (task_dir={})", task_dir)
        return AnquDecision(
            action="verify_task",
            task_summary="任务首次完成，正在执行强制性质量验证",
            rework_instruction=_build_verify_instruction(goal_context),
        )

    # ── Task completed (already verified) → evaluate goal progress ──
    return await _evaluate_goal_progress(
        goal_context,
        current_task_facts,
        final_content,
        cognitive_usage,
        task_dir=task_dir,
        signal=signal,
        total_tasks=total_tasks,
        mingjue_progress_pct=mingjue_progress_pct,
    )


# ---------------------------------------------------------------------------
# Goal progress evaluation
# ---------------------------------------------------------------------------


async def _evaluate_goal_progress(
    goal_context: GoalContext,
    facts: list[RoundExecutionFact],
    final_content: str,
    cognitive_usage: CognitionUsage | None = None,
    task_dir: str = "",
    signal: asyncio.Task | None = None,
    total_tasks: int = 0,
    mingjue_progress_pct: int | None = None,
) -> AnquDecision:
    """Run a lightweight explore-verify loop to decide goal routing.

    Anqu can now inspect task output files, verify completion quality,
    read the blueprint, and gather evidence BEFORE making a goal-level
    decision.  The loop uses only read-only tools and terminates when
    Anqu calls ``task_complete`` with a JSON decision.
    """
    wp = get_workspace_paths()
    goal_dir_path = str(wp.goals / goal_context.goal_id)

    # Build context for Anqu
    blueprint = (
        goal_context.blueprint_summary[:3000] if goal_context.blueprint_summary else "(无蓝图)"
    )
    memory = goal_context.memory_summary[:2000] if goal_context.memory_summary else "(无记忆)"
    trajectory = (
        goal_context.trajectory_snapshot[:2000] if goal_context.trajectory_snapshot else "(新目标)"
    )
    recent = goal_context.recent_task_statuses

    recent_tasks_text = (
        "\n".join(f"- {t.task_id}: {t.status} | {t.summary_snippet[:300]}" for t in (recent or []))
        or "(无近期任务)"
    )

    # Round summary
    total_rounds = len(facts)
    successes = sum(1 for f in facts if f.execution_status == "success")
    failures = sum(1 for f in facts if f.execution_status in ("failure", "partial_failure", "exec_failed"))
    tool_calls_total = sum(f.tool_call_count for f in facts)

    facts_summary = (
        f"任务共 {total_rounds} 轮 | "
        f"成功 {successes} 轮, 失败 {failures} 轮 | "
        f"总工具调用: {tool_calls_total} 次"
    )

    # Build cognitive usage context for evolution evaluation
    # Also detect which grid-recommended skills were NOT used
    grid_available_skills = _get_grid_available_skills(
        cognitive_usage.grids_loaded if cognitive_usage else []
    )
    cognitive_context = _build_cognitive_context(
        cognitive_usage, grid_available_skills,
    )

    # 列出目标目录 + 任务输出目录文件清单（相对于目标目录的路径，方便 read_file）
    goal_file_listing = _list_dir_files(goal_dir_path, "目标目录")
    task_file_listing = _list_dir_files(task_dir, "任务输出目录", base_dir=goal_dir_path) if task_dir else ""

    # 构建明觉进度提示
    progress_text = (
        f"明觉认为当前目标进度为 **{mingjue_progress_pct}%**。"
        if mingjue_progress_pct is not None and mingjue_progress_pct > 0
        else "明觉尚未做进度评估。"
    )

    # ── 蓝图完成验证（阴·主动仲裁对照数据）─────────────────
    blueprint_verification = ""
    if goal_context.blueprint_summary and goal_context.blueprint_summary != "(无蓝图)":
        blueprint_verification = _verify_blueprint_completion(
            blueprint=goal_context.blueprint_summary,
            recent_task_statuses=recent,
            memory=memory,
            trajectory=trajectory,
            total_tasks=total_tasks,
        )

    system_prompt = f"""你是上爻·暗驱，目标的最高决策者。

你的唯一使命：推动目标走向完成。

你可以使用以下验证能力在决策前收集证据：
- list_directory / read_file — 查看目标目录、任务输出、蓝图、记忆、认知库
- query_capabilities — 了解执行环境能力

认知库路径：
- .vingobot/.taiji/cognition/skills/ — L1 技能定义
- .vingobot/.taiji/cognition/models/ — L2 经验模型
- .vingobot/.taiji/cognition/grids/ — L3 认知格栅
- .vingobot/.taiji/cognition/truths/ — L4 不可变真理

**不要在证据不足时猜测。先验证，再决策。**

## 当前目标上下文
- 蓝图: {blueprint}
- 记忆: {memory}
- 轨迹: {trajectory}
- {progress_text}
- 近期已完成任务:
{recent_tasks_text}

{goal_file_listing}

{task_file_listing}

## 当前任务结果

{facts_summary}

Yang 的最终输出:
{final_content[:2000]}

{cognitive_context}

{blueprint_verification}

## 决策指南
- 所有完成标准都已满足 → goal_completed
- 目标明显无法达成 → goal_failed
- 任务结果不完整或质量不高 → continue_task / verify_task / learn_task
- 目标还有差距 → goal_next_task 并给出清晰可执行的下一步
- **next_task_concrete_action（重要）**：必须给出下一任务的第一步具体行动，格式如 "write_file outputs/05-xxx.py 产出..."。**必须包含文件路径和工具名**。
- **技能利用率判定（关键）**：如果上方出现"技能利用率警告"，说明 Worker 有可用技能但未调用。此时任务失败**更可能是技能未使用**而非目标不可行——请判定为 `goal_next_task`（而非 `goal_failed`），并在 next_task_description 中明确指出"必须优先调用已注入的技能工具"。

## 认知演化指南 (evolution 数组)

你的演化指令会被交给 **DMN（Default Mode Network）** — DMN 会通过任务目录直接获取执行数据，
自主完成分析、搜索和认知资产创建。你只需准确判断"需要做什么"：

- **learn_skill** (priority 6-8) — 任务因缺少必要技能或工具而失败
- **research** (priority 5-8) — 工具/库用法有疑问、现有方案在特定平台出错
  例如："exec 工具在 Windows 引号转义有问题，研究跨平台最佳实践"
- **investigate** (priority 3-6) — 失败原因不明确，需深入分析执行数据
  例如："第 5-8 轮反复失败原因不明，请深入分析 execution-facts"
- **precipitate_skill** (priority 4-6) — 形成了可复用的工作模式/SOP
- **precipitate_model** (priority 3-5) — 产出了有价值的方法论或思维模型
  提示：执行过程本身就是最佳学习材料，成功模式和失败教训都值得沉淀
- **create_grid** (priority 2-4) — 需要新的认知领域格栅
  提示：对于有 2 个以上已完成任务的目标，考虑创建领域格栅
- 如无需演化 → 空数组 []

**关键原则**: 让每一次任务执行都沉淀为可复用的认知资产。
DMN 会通过任务目录直接获取执行数据自主分析。请大胆决策。

## 已知陷阱检测 (known_traps_proposal)

你是任务执行的直接观察者——你看到了执行事实（facts_summary），你知道哪些操作模式导致了失败。

如果本次任务中出现了**可复现的失败模式**（特定工具或操作路径反复失败），请将其记录为 known_trap 提案，供下轮明觉读取后提前规避：

- **只在明确观察到失败模式时提案**——不要猜测，必须有执行事实支撑
- 每个陷阱格式：name (snake_case)、description (失败模式描述)、trigger (触发条件)、response (应对策略)
- 例如：exec 安装操作被 L4 安全策略拒绝 → {{"name": "exec-install-blocked", "description": "exec 工具执行安装命令被安全策略拒绝", "trigger": "任务尝试用 exec 安装系统级软件包", "response": "改用容器内预装工具，或在容器初始化阶段提供依赖"}}
- 例如：工具反复调用返回空结果 → {{"name": "empty-search-loop", "description": "搜索/查询工具反复返回空结果但继续调用", "trigger": "连续多轮工具调用无有效返回", "response": "最多尝试 2 次不同来源，然后切换为直接产出策略"}}
- 如果任务中没有新的失败模式 → 空数组 []

## 输出格式
调查完成后，调用 task_complete，summary 字段输出以下 JSON：

{{
  "what_was_accomplished": "本任务实际完成了什么",
  "goal_progress_pct": 0-100,
  "decision": "goal_next_task | goal_completed | goal_failed | continue_task | verify_task | learn_task",
  "next_task_description": "如果 goal_next_task，给出下一个具体任务描述",
  "next_task_concrete_action": "如果 goal_next_task，给出第一步具体行动（含文件路径和工具名）",
  "suggested_trigram": "如果 goal_next_task，卦象: qian(乾)|kun(坤)|zhen(震)|xun(巽)|kan(坎)|li(离)|gen(艮)|dui(兑)，避免连续重复",
  "reason": "决策理由",
  "completion_note": "如果 goal_completed，总结目标成就",
  "evolution": [
    {{
      "action": "learn_skill | precipitate_skill | precipitate_model | create_grid | research | investigate",
      "target_name": "技能/模型/格栅名称（snake_case）",
      "description": "要创建或更新的内容描述",
      "priority": 1-10
    }}
  ],
  "known_traps_proposal": [
    {{
      "name": "陷阱名称（snake_case）",
      "description": "失败模式描述",
      "trigger": "触发条件",
      "response": "应对策略"
    }}
  ]
}}

**goal_progress_pct**: 明觉已评估基线进度，你只需基于任务实际产出**确认或微调**（±5-15%）。无需从零评估。

直接输出 JSON，不要包裹在 markdown 代码块中。
"""

    try:
        from vingobot.goal.lightweight_loop import run_anqu_loop

        cognition_dirs = [
            str(wp.skills),
            str(wp.models),
            str(wp.grids),
        ]

        # Use precise task count from outer loop, fallback to goal_context estimate
        effective_total = total_tasks if total_tasks > 0 else (len(goal_context.recent_task_statuses or []) + 1)

        anqu_provider = _get_provider()

        result = await run_anqu_loop(
            task_dir=goal_dir_path,
            system_prompt=system_prompt,
            goal_dir=goal_dir_path,
            cognition_dirs=cognition_dirs,
            signal=signal,
            provider=anqu_provider,
        )

        if result.task_completed and result.final_content:
            parsed = _parse_anqu_json(result.final_content)
        else:
            logger.warning("[暗驱] 轻量循环未完成，使用回退")
            return _fallback_anqu(final_content, facts, cognitive_usage, total_tasks=effective_total, mingjue_progress_pct=mingjue_progress_pct, task_dir=task_dir)

    except Exception:
        logger.exception("[暗驱] 轻量验证循环失败")
        return _fallback_anqu(final_content, facts, cognitive_usage, total_tasks=effective_total, mingjue_progress_pct=mingjue_progress_pct, task_dir=task_dir)

    decision_raw = parsed.get("decision", "goal_next_task")
    action = _normalize_action(decision_raw)

    # Parse evolution actions
    evolution = _parse_evolution_actions(
        parsed.get("evolution", []),
        source_task_id="",
        source_goal_id=getattr(goal_context, "goal_id", ""),
    )

    # ── Parse known traps proposal ──────────────────────────────
    known_traps_proposal = _parse_known_traps_proposal(
        parsed.get("known_traps_proposal", []),
    )

    # Supplement with computed actions if LLM returned empty evolution
    if not evolution:
        evolution = _compute_evolution_actions(
            final_content, facts, cognitive_usage, task_dir,
            total_tasks=effective_total,
        )

    return AnquDecision(
        action=action,
        next_task_description=parsed.get("next_task_description", ""),
        next_task_concrete_action=parsed.get("next_task_concrete_action", ""),
        suggested_trigram=parsed.get("suggested_trigram", ""),
        goal_progress_pct=_parse_pct(parsed.get("goal_progress_pct")),
        task_summary=parsed.get("what_was_accomplished", final_content[:800]),
        continuation_context=parsed.get("reason", ""),
        rework_instruction=parsed.get("reason", ""),
        failure_reason=parsed.get("reason", "") if action == "goal_failed" else "",
        evolution_actions=evolution,
        known_traps_proposal=known_traps_proposal,
        needs_sibian=_compute_needs_sibian(
            goal_context=goal_context,
            known_traps_proposal=known_traps_proposal,
            facts=facts,
            total_tasks=effective_total,
        ),
    )


def _list_dir_files(dir_path: str, label: str = "", base_dir: str = "") -> str:
    """列出目录下的文件清单，显示相对于 base_dir 的路径。

    当提供 base_dir 时，文件路径以相对 base_dir 的形式显示，
    方便暗驱在只读循环中用 read_file 直接读取。
    """
    try:
        d = Path(dir_path)
        if not d.is_dir():
            return ""
        base = Path(base_dir) if base_dir else None
        lines = [f"## {label}文件清单"]
        for f in sorted(d.iterdir()):
            display = f.name
            if base:
                try:
                    display = str(f.relative_to(base))
                except ValueError:
                    pass
            if f.is_file():
                lines.append(f"- 📄 {display}")
            elif f.is_dir():
                lines.append(f"- 📁 {display}/")
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception:
        return ""


def _get_grid_available_skills(grid_names: list[str]) -> dict[str, list[str]]:
    """Read loaded grid JSON files and extract their skill references.

    Returns a dict mapping grid_name → list of skill names that the
    grid recommends.  An empty dict if grids don't exist or can't be read.
    """
    if not grid_names:
        return {}

    wp = get_workspace_paths()
    result: dict[str, list[str]] = {}

    for name in grid_names:
        grid_path = wp.grids / f"{name}.json"
        try:
            if not grid_path.is_file():
                continue
            data = json.loads(grid_path.read_text(encoding="utf-8"))
            skills = data.get("skills", [])
            skill_names: list[str] = []
            for s in skills:
                if isinstance(s, str):
                    skill_names.append(s)
                elif isinstance(s, dict):
                    skill_names.append(s.get("name", ""))
            result[name] = [n for n in skill_names if n]
        except (OSError, json.JSONDecodeError, ValueError):
            logger.debug("[暗驱] 无法读取格栅文件: {}", grid_path)

    return result


def _build_cognitive_context(
    cognitive_usage: CognitionUsage | None,
    grid_available_skills: dict[str, list[str]] | None = None,
) -> str:
    """Build the cognitive usage context block for Anqu's prompt.

    When *grid_available_skills* is provided, this function also
    computes which grid-recommended skills were **not** used and
    emits a prominent warning — so Anqu can distinguish "task truly
    infeasible" from "available skills simply weren't called".
    """
    if cognitive_usage is None:
        return ""

    parts: list[str] = ["## 认知使用情况"]

    if cognitive_usage.grids_loaded:
        parts.append(f"- 加载的认知格栅: {', '.join(cognitive_usage.grids_loaded)}")
    if cognitive_usage.skills_used:
        parts.append(f"- 使用的技能: {', '.join(cognitive_usage.skills_used)}")
    if cognitive_usage.models_loaded:
        parts.append(f"- 加载的思维模型: {', '.join(cognitive_usage.models_loaded)}")
    if cognitive_usage.tools_failed:
        parts.append(f"- 调用失败的工具: {', '.join(cognitive_usage.tools_failed)}")
    parts.append(f"- 总工具调用次数: {cognitive_usage.tool_calls_total}")

    # ── Skill utilization gap detection ──────────────────────────
    if grid_available_skills:
        used = set(cognitive_usage.skills_used)
        # Filter out internal markers like "__read__"
        used_clean = {s for s in used if not s.startswith("__")}

        all_available: set[str] = set()
        for skills in grid_available_skills.values():
            all_available.update(skills)

        unused = all_available - used_clean

        if unused:
            parts.append("")
            parts.append(
                "### ⚠️ 技能利用率警告"
            )
            parts.append(
                f"当前格栅推荐了以下技能，但 Worker 在本任务中**未调用**: "
                f"{', '.join(sorted(unused))}"
            )
            for grid_name, skills in grid_available_skills.items():
                grid_unused = [s for s in skills if s in unused]
                if grid_unused:
                    parts.append(
                        f"  - 格栅 `{grid_name}` 推荐但未使用: "
                        f"{', '.join(grid_unused)}"
                    )
            parts.append(
                "**重要**: 任务失败可能是技能未调用导致，"
                "不一定是目标不可行。请勿仅因本次未产出即判定 `goal_failed`。"
            )

    return "\n".join(parts)


def _parse_evolution_actions(
    raw_evolution: list[dict[str, Any]],
    source_task_id: str,
    source_goal_id: str,
) -> list[CognitionEvolutionAction]:
    """Parse evolution actions from Anqu's JSON output."""
    if not raw_evolution:
        return []

    valid_actions = {"learn_skill", "precipitate_skill", "precipitate_model", "create_grid", "research", "investigate", "review_blueprint"}
    actions: list[CognitionEvolutionAction] = []

    for item in raw_evolution:
        action_type = item.get("action", "")
        if action_type not in valid_actions:
            logger.warning("[暗驱] 忽略未知演化动作: {}", action_type)
            continue

        actions.append(
            CognitionEvolutionAction(
                action=action_type,  # type: ignore[arg-type]
                target_name=item.get("target_name", ""),
                description=item.get("description", ""),
                source_task_id=source_task_id,
                source_goal_id=source_goal_id,
                priority=int(item.get("priority", 5)),
            )
        )

    return actions


def _parse_known_traps_proposal(
    raw_proposals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse known_traps_proposal from Anqu's JSON output.

    Validates that each proposal has at least name + description.
    Skips incomplete or empty entries.
    """
    if not raw_proposals:
        return []

    valid: list[dict[str, Any]] = []
    for item in raw_proposals:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        description = (item.get("description") or "").strip()
        if not name or not description:
            logger.warning("[暗驱] 跳过不完整的 known_trap 提案: {}", item)
            continue
        valid.append({
            "name": name,
            "description": description,
            "trigger": (item.get("trigger") or "").strip(),
            "response": (item.get("response") or "").strip(),
        })

    return valid


def _compute_evolution_actions(
    final_content: str,
    facts: list[RoundExecutionFact],
    cognitive_usage: CognitionUsage | None = None,
    task_dir: str = "",
    total_tasks: int = 0,
) -> list[CognitionEvolutionAction]:
    """Compute evolution actions from execution facts.

    Used as a supplement when the LLM path returns empty evolution.
    Actions are ordered by priority; only the highest-priority action triggers.
    """
    summary = (final_content or "")[:500]
    round_count = len(facts)
    successes = sum(1 for f in facts if f.execution_status == "success")
    failures = sum(1 for f in facts if f.execution_status in ("failure", "partial_failure", "exec_failed"))
    task_label = Path(task_dir).name if task_dir else "unknown"

    # Tool failures → learn_skill (highest priority — broken tool blocks progress)
    if cognitive_usage and cognitive_usage.tools_failed:
        return [
            CognitionEvolutionAction(
                action="learn_skill",
                target_name=f"fix_{t.replace('-', '_')}",
                description=f"修复/学习工具 '{t}' 的正确用法",
                priority=6,
            )
            for t in cognitive_usage.tools_failed[:2]
        ]

    # High failure rate or many rounds with failures → investigate
    if failures > 0 and (failures / max(round_count, 1) >= 0.3 or (round_count >= 10 and failures > 0)):
        fail_rate = failures / max(round_count, 1)
        priority = 6 if fail_rate >= 0.5 else (5 if fail_rate >= 0.3 else 4)
        return [
            CognitionEvolutionAction(
                action="investigate",
                target_name=f"task_failure_{task_label}",
                description=(
                    f"该任务完成了 {round_count} 轮（{successes} 成功/{failures} 失败），"
                    f"失败率 {fail_rate:.0%}，请分析失败根因并提出改进建议。"
                    f"执行摘要: {summary[:200]}"
                ),
                priority=priority,
            )
        ]

    # Pure successful task with clean execution → precipitate_model
    if successes >= 3 and failures == 0 and task_dir:
        outputs_dir = Path(task_dir) / "outputs"
        if outputs_dir.is_dir() and any(outputs_dir.iterdir()):
            return [
                CognitionEvolutionAction(
                    action="precipitate_model",
                    target_name=f"task_method_{task_label}",
                    description=(
                        f"该任务纯顺利完成了 {round_count} 轮（{successes} 成功），"
                        f"执行模式干净，值得抽象为通用的L2经验模型。"
                        f"任务摘要: {summary[:300]}"
                    ),
                    priority=4,
                )
            ]

    # Successful task with outputs → precipitate_skill
    if successes >= 2 and task_dir:
        outputs_dir = Path(task_dir) / "outputs"
        if outputs_dir.is_dir() and any(outputs_dir.iterdir()):
            return [
                CognitionEvolutionAction(
                    action="precipitate_skill",
                    target_name=f"task_experience_{task_label}",
                    description=(
                        f"该任务完成了 {round_count} 轮（{successes} 成功/{failures} 失败），"
                        f"产出了具体交付物。请分析其执行流程和SOP，沉淀为可复用的L1技能。"
                        f"任务摘要: {summary[:300]}"
                    ),
                    priority=5,
                )
            ]

    # Multiple tasks completed → create_grid (cross-task domain organization)
    if total_tasks >= 3:
        return [
            CognitionEvolutionAction(
                action="create_grid",
                target_name="goal_achievement_grid",
                description=(
                    f"该目标已完成了 {total_tasks} 个任务。"
                    f"请分析所有任务的执行经验（产物、SOP、模式），"
                    f"创建或更新L3认知格栅以整合该领域的认知资产。"
                ),
                priority=3,
            )
        ]

    return []


def _compute_known_traps_proposal(
    facts: list[RoundExecutionFact],
    cognitive_usage: CognitionUsage | None = None,
    round_count: int = 0,
) -> list[dict[str, Any]]:
    """Compute known_traps proposals from execution facts.

    Used in the fallback path when Anqu LLM is unavailable.
    Detects clear failure patterns that should be avoided in future tasks.
    """
    proposals: list[dict[str, Any]] = []
    failures = sum(1 for f in facts if f.execution_status in ("failure", "partial_failure", "exec_failed"))

    # Tool failures (specific known-bad operations)
    if cognitive_usage and cognitive_usage.tools_failed:
        for tool in cognitive_usage.tools_failed[:2]:
            proposals.append({
                "name": f"{tool.replace('-', '_')}_fails",
                "description": f"工具 '{tool}' 在任务中调用失败",
                "trigger": f"任务尝试使用 {tool} 工具",
                "response": f"优先查找替代方案：检查认知库中是否有 {tool} 的使用经验，或咨询用户替代路径",
            })

    # High failure rate pattern → diagnosis addiction risk
    if failures > 0 and round_count >= 10 and failures / max(round_count, 1) >= 0.3:
        proposals.append({
            "name": "diagnosis-addiction",
            "description":
                f"任务 {round_count} 轮中失败 {failures} 次（失败率 {failures/max(round_count,1):.0%}），"
                "可能是反复尝试同一种失败方案而非切换策略",
            "trigger": "连续多轮执行失败且没有产出新的交付物",
            "response": "遇到失败后最多重试一种替代方案，若仍失败则直接切换为最小可行产出模式（write_file 写简化版）",
        })

    # Self-read loop pattern
    read_only = sum(
        1 for f in facts
        if f.tool_call_count > 0 and not f.had_action_request
    )
    if read_only >= 8 and round_count >= 12:
        proposals.append({
            "name": "read-only-loop",
            "description":
                f"任务有 {read_only} 轮为纯读取无写操作（总计 {round_count} 轮），"
                "可能陷入了只读不产的循环",
            "trigger": "连续多轮仅调用 read_file / list_directory / search 等读取工具",
            "response": "每轮必须产生可交付输出（write_file / edit_file），至少每 3 轮一次写入产出",
        })

    return proposals


def _compute_needs_sibian(
    goal_context: GoalContext | None,
    known_traps_proposal: list[dict[str, Any]],
    facts: list[RoundExecutionFact],
    total_tasks: int,
) -> bool:
    """Determine if SiBian (blueprint review) should be triggered.

    SiBian is the symmetric counterpart of Weaver at the goal level:
    - Weaver revises cognitive posture within a task (triggered by Yin)
    - SiBian revises the blueprint across tasks (triggered by Anqu)

    Both use the same 思变↔织 cross-cycle symmetry principle.

    Triggers when:
    - Known traps detected (failure patterns need blueprint adjustment)
    - Cross-task: 2+ consecutive task failures
    - Current task: all-round failure (0 successes in >=5 rounds)
    - High task count: blueprint may need refresh after many tasks
    """
    # ── Known traps triggered — failure pattern needs blueprint review ──
    if known_traps_proposal:
        return True

    # ── Cross-task: 2+ consecutive task failures ──
    if goal_context is not None:
        recent = goal_context.recent_task_statuses
        if recent and len(recent) >= 2:
            last2 = recent[-2:]
            if all(getattr(t, "status", "") in ("failed", "auto_terminated") for t in last2):
                return True

    # ── Current task: all-round failure (0 successes in >=5 rounds) ──
    total_rounds = len(facts)
    if total_rounds >= 5:
        successes = sum(1 for f in facts if f.execution_status == "success")
        if successes == 0:
            return True

    # ── High task count: blueprint likely needs refresh ──
    if total_tasks >= 5:
        return True

    return False


def _fallback_anqu(
    final_content: str,
    facts: list[RoundExecutionFact],
    cognitive_usage: CognitionUsage | None = None,
    total_tasks: int = 0,
    mingjue_progress_pct: int | None = None,
    task_dir: str = "",
) -> AnquDecision:
    """LLM-less fallback — use execution statistics for guidance."""
    summary = (final_content or "")[:500]
    round_count = len(facts)
    successes = sum(1 for f in facts if f.execution_status == "success")
    failures = sum(1 for f in facts if f.execution_status in ("failure", "partial_failure", "exec_failed"))

    # ── Stop condition: too many tasks with poor performance ──
    if total_tasks >= 6 and failures > successes:
        traps = _compute_known_traps_proposal(facts, cognitive_usage, round_count)
        return AnquDecision(
            action="goal_completed",
            task_summary=f"目标已运行 {total_tasks} 个任务，最近任务成功率持续偏低（{successes} 成功/{failures} 失败），建议停止。最后产出: {summary}",
            known_traps_proposal=traps,
        )

    # Build evidence-based continuation context
    context = f"上一任务完成 {round_count} 轮（{successes} 成功/{failures} 失败）。"
    if failures > 0:
        context += f" 有 {failures} 轮执行失败，建议下一轮减少纯读取操作，优先产出具体交付物。"
    else:
        context += " 请基于已收集的信息继续推进，撰写具体交付文件。"
    if round_count >= 12:
        context += " 注意：上一任务轮次较多，可能存在自读循环，下一轮请首轮直接写入交付物。"

    next_desc = "请基于已完成的成果继续推进目标"
    concrete_action = ""
    suggested_trigram = ""
    if failures > successes:
        next_desc += "。注意：上次失败率较高，请聚焦最小可行产出"
        concrete_action = "用 write_file 产出最简可行交付物到 outputs/ 目录，首轮不要探索"
        suggested_trigram = "kan"  # 攻坚 — high failure rate
    elif round_count >= 12:
        concrete_action = "用 write_file 产出交付物到 outputs/ 目录为第一步，首轮不要探索"
        suggested_trigram = "li"  # 整理 — many rounds, need consolidation
    else:
        concrete_action = "先读取 outputs/ 目录中已有产出了解进展，再用 write_file 产出下一步交付物"
        suggested_trigram = "kun"  # 默认执行

    evolution = _compute_evolution_actions(
        final_content, facts, cognitive_usage, task_dir,
        total_tasks=total_tasks,
    )

    traps = _compute_known_traps_proposal(facts, cognitive_usage, round_count)

    return AnquDecision(
        action="goal_next_task",
        task_summary=summary,
        next_task_description=next_desc,
        next_task_concrete_action=concrete_action,
        suggested_trigram=suggested_trigram,
        goal_progress_pct=mingjue_progress_pct,
        continuation_context=context,
        evolution_actions=evolution,
        known_traps_proposal=traps,
        needs_sibian=_compute_needs_sibian(
            goal_context=None,  # fallback path has no goal_context
            known_traps_proposal=traps,
            facts=facts,
            total_tasks=total_tasks,
        ),
    )


def _parse_anqu_json(text: str) -> dict[str, Any]:
    """Extract JSON from Anqu LLM output."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    for fence in ("```json", "```"):
        if fence in text:
            start = text.index(fence) + len(fence)
            end = text.rfind("```")
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, TypeError):
                    pass

    return {"decision": "goal_next_task"}


def _parse_pct(raw: Any) -> int | None:
    """Parse goal_progress_pct from Anqu's JSON, returning None if absent/invalid."""
    if raw is None:
        return None
    try:
        pct = int(raw)
        return max(0, min(100, pct))
    except (ValueError, TypeError):
        return None


def _normalize_action(raw: str) -> AnquAction:
    """Normalize the Anqu decision string to a valid action."""
    valid: set[AnquAction] = {
        "goal_next_task",
        "goal_completed",
        "goal_failed",
        "continue_task",
        "verify_task",
        "learn_task",
    }
    raw = raw.strip().lower()
    if raw in valid:
        return raw  # type: ignore[return-value]
    logger.warning("[暗驱] 未知决策 '{}', fallback 到 goal_next_task", raw)
    return "goal_next_task"


# ---------------------------------------------------------------------------
# Provider access
# ---------------------------------------------------------------------------

_agent_name = "anqu"
_provider: Any = None


def _get_provider() -> Any:
    """Lazily obtain the LLM provider for this sixiang agent.

    Uses the per-agent config (``agents.defaults.sixiang.agents.anqu``)
    when available, falling back to the global defaults.
    """
    global _provider
    if _provider is not None:
        return _provider
    try:
        from vingobot.providers.factory import build_sixiang_provider_snapshot
        from vingobot.config.loader import load_config, resolve_config_env_vars

        config = resolve_config_env_vars(load_config())
        snapshot = build_sixiang_provider_snapshot(config, _agent_name)
        _provider = snapshot.provider
    except Exception:
        logger.warning("[暗驱] 无法初始化 provider")
    return _provider


def set_provider(provider: Any) -> None:
    global _provider
    _provider = provider
