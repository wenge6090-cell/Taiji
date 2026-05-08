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
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.core.goal_context import GoalContext
from vingobot.core.workspace import get_workspace_paths
from vingobot.goal.grid_types import CognitionEvolutionAction, CognitionUsage
from vingobot.goal.types import AnquAction, AnquDecision, RoundExecutionFact


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

    # ── Task completed → evaluate goal progress ──────────────────
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
    cognitive_context = _build_cognitive_context(cognitive_usage)

    # 列出目标目录 + 任务输出目录文件清单（相对于目标目录的路径，方便 read_file）
    goal_file_listing = _list_dir_files(goal_dir_path, "目标目录")
    task_file_listing = _list_dir_files(task_dir, "任务输出目录", base_dir=goal_dir_path) if task_dir else ""

    # 构建明觉进度提示
    progress_text = (
        f"明觉认为当前目标进度为 **{mingjue_progress_pct}%**。"
        if mingjue_progress_pct is not None and mingjue_progress_pct > 0
        else "明觉尚未做进度评估。"
    )

    system_prompt = f"""你是上爻·暗驱，目标的最高决策者。

你的唯一使命：推动目标走向完成。

你拥有验证能力——在做出决策前，你可以：
- 用 list_directory 查看目标目录和任务输出目录
- 用 read_file 读取蓝图、任务输出文件、记忆文件、认知库文件
- 用 query_capabilities 了解当前执行环境能力

认知库路径（可用 read_file 直接读取）：
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

## 决策指南
- 如果所有完成标准都已满足 → goal_completed
- 如果目标明显无法达成 → goal_failed
- 如果当前任务结果不完整或质量不高 → continue_task / verify_task / learn_task
- 如果目标还有差距 → goal_next_task 并给出清晰可执行的下一步
- **next_task_concrete_action（重要）**：如果 goal_next_task，必须给出下一任务的**第一步具体行动**。
  格式: "write_file outputs/05-xxx.py 产出..." 或 "读取 outputs/03-xxx.md 了解现状后 write_file 产出..."
  **必须包含文件路径和工具名**，不能只写抽象描述。杨会以此为第一轮的直接指令。

## 认知演化指南 (evolution 数组)

你的认知演化指令会被交给 **DMN（Default Mode Network）** — 一个专门的后台认知管家来处理。
DMN 的能力包括：
- **搜索网络**：用 web_search 查找新工具、库、解决方案
- **分析失败**：读取任务目录的执行事实（06-execution-facts.json）和输出文件，深入诊断根因
- **创建认知资产**：生成 L1 技能、L2 模型、L3 格栅、L4 真理
- **自由维护**：检查 meta.json、归档、更新认知库

**你不需要提供详细的执行上下文**。DMN 会直接读取任务目录（已自动附在演化任务中）来获取执行数据。
你只需准确判断"需要做什么"：

- 如果任务中因缺少必要技能或工具而失败 → learn_skill (priority 6-8)
- 如果任务中形成了可复用的工作模式/SOP → precipitate_skill (priority 4-6)
- 如果任务中产出了有价值的方法论或思维模型 → precipitate_model (priority 3-5)
  **提示**: 任务的执行过程本身就是最佳学习材料。即使没有显式的新方法论，
  分析执行Facts中的成功模式和失败教训也值得沉淀为经验模型。
- 如果发现需要新的认知领域格栅 → create_grid (priority 2-4)
  **提示**: 用当前目标ID作为领域名称，将沉淀的技能和模型整合为该领域的认知网格。
  对于有 2 个以上已完成任务的目标，考虑创建领域格栅。
- ✅ 如果工具/库用法有疑问、现有方案在特定平台出错 → research (priority 5-8)
  DMN 会用 web_search 查找最佳实践、替代方案或修复方法。
  例如："exec 工具在 Windows 引号转义有问题，研究跨平台最佳实践"
- ✅ 如果失败原因不明确、需要深入分析执行数据 → investigate (priority 3-6)
  DMN 会读取完整的执行事实和输出文件，诊断根因并给出改进建议。
  例如："第 5-8 轮反复失败原因不明，请深入分析 execution-facts"
- 如无需演化 → 空数组 []

**关键原则**: 让每一次任务执行都沉淀为可复用的认知资产。
任务执行数据(06-execution-facts.json)和输出文件会随演化任务传递给DMN进行分析。
所以请大胆决策，DMN会基于真实数据生成资产。

## 输出格式
调查完成后，调用 task_complete，summary 字段输出以下 JSON：

{{
  "what_was_accomplished": "本任务实际完成了什么",
  "goal_progress_pct": 0-100,
  "decision": "goal_next_task | goal_completed | goal_failed | continue_task | verify_task | learn_task",
  "next_task_description": "如果 goal_next_task，给出下一个具体任务描述",
  "next_task_concrete_action": "如果 goal_next_task，给出第一步具体行动（含文件路径和工具名）",
  "suggested_trigram": "如果 goal_next_task，建议下一任务的卦象: qian(乾/探索) | kun(坤/执行) | zhen(震/变革) | xun(巽/分析) | kan(坎/攻坚) | li(离/整理) | gen(艮/审视) | dui(兑/沟通)。根据任务性质和已发现问题选择，避免连续重复",
  "reason": "决策理由",
  "completion_note": "如果 goal_completed，总结目标成就",
  "evolution": [
    {{
      "action": "learn_skill | precipitate_skill | precipitate_model | create_grid | research | investigate",
      "target_name": "技能/模型/格栅名称（snake_case）",
      "description": "要创建或更新的内容描述",
      "priority": 1-10
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

        result = await run_anqu_loop(
            task_dir=goal_dir_path,
            system_prompt=system_prompt,
            goal_dir=goal_dir_path,
            cognition_dirs=cognition_dirs,
            signal=signal,
        )

        if result.task_completed and result.final_content:
            parsed = _parse_anqu_json(result.final_content)
        else:
            logger.warning("[暗驱] 轻量循环未完成，使用回退")
            return _fallback_anqu(final_content, facts, cognitive_usage, total_tasks=effective_total, mingjue_progress_pct=mingjue_progress_pct)

    except Exception:
        logger.exception("[暗驱] 轻量验证循环失败")
        return _fallback_anqu(final_content, facts, cognitive_usage, total_tasks=effective_total, mingjue_progress_pct=mingjue_progress_pct)

    decision_raw = parsed.get("decision", "goal_next_task")
    action = _normalize_action(decision_raw)

    # Parse evolution actions
    evolution = _parse_evolution_actions(
        parsed.get("evolution", []),
        source_task_id="",
        source_goal_id=getattr(goal_context, "goal_id", ""),
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


def _build_cognitive_context(cognitive_usage: CognitionUsage | None) -> str:
    """Build the cognitive usage context block for Anqu's prompt."""
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

    return "\n".join(parts)


def _parse_evolution_actions(
    raw_evolution: list[dict[str, Any]],
    source_task_id: str,
    source_goal_id: str,
) -> list[CognitionEvolutionAction]:
    """Parse evolution actions from Anqu's JSON output."""
    if not raw_evolution:
        return []

    valid_actions = {"learn_skill", "precipitate_skill", "precipitate_model", "create_grid", "research", "investigate"}
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


def _fallback_anqu(
    final_content: str,
    facts: list[RoundExecutionFact],
    cognitive_usage: CognitionUsage | None = None,
    total_tasks: int = 0,
    mingjue_progress_pct: int | None = None,
) -> AnquDecision:
    """LLM-less fallback — use execution statistics for guidance."""
    summary = (final_content or "")[:500]
    round_count = len(facts)
    successes = sum(1 for f in facts if f.execution_status == "success")
    failures = sum(1 for f in facts if f.execution_status in ("failure", "partial_failure", "exec_failed"))

    # ── Stop condition: too many tasks with poor performance ──
    if total_tasks >= 6 and failures > successes:
        return AnquDecision(
            action="goal_completed",
            task_summary=f"目标已运行 {total_tasks} 个任务，最近任务成功率持续偏低（{successes} 成功/{failures} 失败），建议停止。最后产出: {summary}",
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

    evolution: list[CognitionEvolutionAction] = []

    # Check for tool failures in the fallback path too
    if cognitive_usage and cognitive_usage.tools_failed:
        evolution = [
            CognitionEvolutionAction(
                action="learn_skill",
                target_name=f"fix_{t.replace('-', '_')}",
                description=f"修复/学习工具 '{t}' 的正确用法",
                priority=6,
            )
            for t in cognitive_usage.tools_failed[:2]
        ]

    return AnquDecision(
        action="goal_next_task",
        task_summary=summary,
        next_task_description=next_desc,
        next_task_concrete_action=concrete_action,
        suggested_trigram=suggested_trigram,
        goal_progress_pct=mingjue_progress_pct,
        continuation_context=context,
        evolution_actions=evolution,
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
