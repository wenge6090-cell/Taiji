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
) -> AnquDecision:
    """Evaluate the current task result and decide the goal's next step.

    Args:
        goal_context: Full goal context snapshot.
        current_task_facts: Round facts from the just-completed task.
        final_content: Yang's final content (if task_complete was called).
        task_dir: The just-completed task's working directory (for output inspection).
        signal: Optional cancellation token.
        cognitive_usage: Cognitive assets used during this task (for evolution).

    Returns:
        AnquDecision routing the outer loop, optionally with evolution_actions.
    """

    # ── Fast-path: task didn't complete ──────────────────────────
    if final_content is None:
        # Task ran out of rounds → rework
        round_count = len(current_task_facts)
        return AnquDecision(
            action="continue_task",
            task_summary=f"任务在第 {round_count} 轮后仍未完成",
            rework_instruction="请在下一轮尝试更简洁的方法，优先调用工具完成目标，避免长时间纯思考。",
        )

    # ── Task completed → evaluate goal progress ──────────────────
    return await _evaluate_goal_progress(
        goal_context,
        current_task_facts,
        final_content,
        cognitive_usage,
        task_dir=task_dir,
        signal=signal,
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
    failures = sum(1 for f in facts if f.execution_status in ("failure", "partial_failure"))
    tool_calls_total = sum(f.tool_call_count for f in facts)

    facts_summary = (
        f"任务共 {total_rounds} 轮 | "
        f"成功 {successes} 轮, 失败 {failures} 轮 | "
        f"总工具调用: {tool_calls_total} 次"
    )

    # Build cognitive usage context for evolution evaluation
    cognitive_context = _build_cognitive_context(cognitive_usage)

    # 列出目标目录 + 任务输出目录文件清单
    goal_file_listing = _list_dir_files(goal_dir_path, "目标目录")
    task_file_listing = _list_dir_files(task_dir, "任务输出目录") if task_dir else ""

    system_prompt = f"""你是上爻·暗驱，目标的最高决策者。

你的唯一使命：推动目标走向完成。

你拥有验证能力——在做出决策前，你可以：
- 用 list_directory 查看目标目录和任务输出目录
- 用 read_file 读取蓝图、任务输出文件、记忆文件
- 用 search_skills / search_models 搜索认知库
- 用 query_capabilities 了解当前执行环境能力

**不要在证据不足时猜测。先验证，再决策。**

## 当前目标上下文
- 蓝图: {blueprint}
- 记忆: {memory}
- 轨迹: {trajectory}
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

## 认知演化指南 (evolution 数组)
- 如果任务中因缺少必要技能或工具而失败 → learn_skill (priority 6-8)
- 如果任务中形成了可复用的工作模式/SOP → precipitate_skill (priority 4-6)
- 如果任务中产出了有价值的方法论或思维模型 → precipitate_model (priority 3-5)
- 如果发现需要新的认知领域格栅 → create_grid (priority 2-4)
- 如无需演化 → 空数组 []

## 输出格式
调查完成后，调用 task_complete，summary 字段输出以下 JSON：

{{
  "what_was_accomplished": "本任务实际完成了什么",
  "goal_progress_pct": 0-100,
  "decision": "goal_next_task | goal_completed | goal_failed | continue_task | verify_task | learn_task",
  "next_task_description": "如果 goal_next_task，给出下一个具体任务描述",
  "reason": "决策理由",
  "completion_note": "如果 goal_completed，总结目标成就",
  "evolution": [
    {{
      "action": "learn_skill | precipitate_skill | precipitate_model | create_grid",
      "target_name": "技能/模型/格栅名称（snake_case）",
      "description": "要创建或更新的内容描述",
      "priority": 1-10
    }}
  ]
}}

直接输出 JSON，不要包裹在 markdown 代码块中。
"""

    try:
        from vingobot.goal.lightweight_loop import run_anqu_loop

        cognition_dirs = [
            str(wp.skills),
            str(wp.models),
            str(wp.grids),
        ]

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
            return _fallback_anqu(final_content, facts, cognitive_usage)

    except Exception:
        logger.exception("[暗驱] 轻量验证循环失败")
        return _fallback_anqu(final_content, facts, cognitive_usage)

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
        task_summary=parsed.get("what_was_accomplished", final_content[:800]),
        continuation_context=parsed.get("reason", ""),
        rework_instruction=parsed.get("reason", ""),
        failure_reason=parsed.get("reason", "") if action == "goal_failed" else "",
        evolution_actions=evolution,
    )


def _list_dir_files(dir_path: str, label: str = "") -> str:
    """列出目录下的文件清单。"""
    try:
        d = Path(dir_path)
        if not d.is_dir():
            return ""
        lines = [f"## {label}文件清单"]
        for f in sorted(d.iterdir()):
            if f.is_file():
                lines.append(f"- 📄 {f.name}")
            elif f.is_dir():
                lines.append(f"- 📁 {f.name}/")
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

    valid_actions = {"learn_skill", "precipitate_skill", "precipitate_model", "create_grid"}
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
) -> AnquDecision:
    """LLM-less fallback — assume the goal continues."""
    summary = (final_content or "")[:500]
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
        next_task_description="请根据已完成的成果继续推进目标",
        continuation_context=f"上一任务完成了 {len(facts)} 轮，请基于成果继续。",
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
