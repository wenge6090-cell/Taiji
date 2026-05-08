"""
初爻·明觉 — Goal-to-task translation layer.

Mingjue translates an abstract goal or continuation instruction into a
concrete, executable task description.  It handles three entry modes:

- **initial_goal**: First-ever task for a goal; calls LLM to break down.
- **anqu_continuation**: Anqu decided the goal needs another task; uses
  Anqu's structured output directly.
- **rework**: Previous task needs re-doing; re-examines with fresh eyes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.core.goal_context import GoalContext
from vingobot.core.workspace import get_workspace_paths
from vingobot.goal.types import MingjueContextInfo, MingjueOutput, MingjueSource


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_mingjue(
    goal_context: GoalContext,
    source: MingjueSource,
    *,
    signal: asyncio.Task | None = None,
) -> MingjueOutput:
    """Translate a goal-source into a concrete task description.

    For ``initial_goal`` sources this calls the LLM to do the heavy
    lifting.  For ``anqu_continuation`` and ``rework`` it can usually
    produce an output without an extra LLM round-trip.
    """

    # --- Periodic reflection: force LLM re-assessment ---
    if source.type == "periodic_reflection":
        return await _from_initial(goal_context, source, signal)

    # --- Continuation: Anqu already provided a concrete next step ---
    if source.type == "anqu_continuation":
        return _from_continuation(goal_context, source)

    # --- Rework: re-examine the previous output ---
    if source.type == "rework":
        return _from_rework(goal_context, source, signal)

    # --- Initial: first-ever task for this goal ---
    return await _from_initial(goal_context, source, signal)


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------


def _from_continuation(goal_context: GoalContext, source: MingjueSource) -> MingjueOutput:
    """Build a MingjueOutput from Anqu's structured continuation."""
    wp = get_workspace_paths()

    # Use Anqu's suggested trigram if available, otherwise default to kun
    trigram = source.suggested_trigram or "kun"
    trigram_reason = (
        f"暗驱续接，卦象: {trigram}" if source.suggested_trigram
        else "暗驱续接，默认执行"
    )

    return MingjueOutput(
        intent="task",
        goal_id=goal_context.goal_id,
        summary=source.description or "继续推进目标",
        concrete_goal=source.description or source.previous_task_summary,
        trigram=trigram,
        trigram_reason=trigram_reason,
        initial_yao=1,
        context=MingjueContextInfo(
            workspace_root=str(wp.root),
            goal_dir=str(wp.goals / goal_context.goal_id),
            cognition_dirs={
                "skills": str(wp.skills),
                "models": str(wp.models),
                "grids": str(wp.grids),
            },
        ),
    )


def _from_rework(
    goal_context: GoalContext,
    source: MingjueSource,
    signal: asyncio.Task | None,
) -> MingjueOutput:
    """Handle rework — reuse previous output when available."""
    wp = get_workspace_paths()

    prev = source.previous_output
    if isinstance(prev, MingjueOutput):
        new_concrete = f"[回炉] {source.rework_instruction}\n\n原始任务: {prev.concrete_goal}"
        return MingjueOutput(
            intent=prev.intent,
            goal_id=goal_context.goal_id,
            summary=source.rework_instruction or "重新审题",
            concrete_goal=new_concrete,
            trigram=prev.trigram or "zhen",
            trigram_reason="回炉重审，震卦应之",
            initial_yao=getattr(prev, "initial_yao", 1),
            context=prev.context,
        )

    # Fallback
    return MingjueOutput(
        intent="task",
        goal_id=goal_context.goal_id,
        summary=source.rework_instruction or "重新执行任务",
        concrete_goal=source.rework_instruction or source.description,
        trigram="zhen",
        trigram_reason="回炉重审",
        initial_yao=1,
        context=MingjueContextInfo(
            workspace_root=str(wp.root),
            goal_dir=str(wp.goals / goal_context.goal_id),
            cognition_dirs={
                "skills": str(wp.skills),
                "models": str(wp.models),
                "grids": str(wp.grids),
            },

        ),
    )


async def _from_initial(
    goal_context: GoalContext,
    source: MingjueSource,
    signal: asyncio.Task | None,
) -> MingjueOutput:
    """Run a lightweight explore-decide loop to translate goal → task.

    Mingjue can now explore the goal directory, read phase reports,
    search cognition, and gather context BEFORE making a task-decomposition
    decision.  The loop uses only read-only tools and terminates when
    Mingjue calls ``task_complete`` with a JSON summary.
    """
    wp = get_workspace_paths()
    goal_dir_path = str(wp.goals / goal_context.goal_id)

    blueprint_snippet = (
        goal_context.blueprint_summary[:3000] if goal_context.blueprint_summary else "(无蓝图)"
    )
    memory_snippet = (
        goal_context.memory_summary[:2000] if goal_context.memory_summary else "(无记忆)"
    )
    trajectory_snippet = (
        goal_context.trajectory_snapshot[:2000] if goal_context.trajectory_snapshot else "(新目标)"
    )
    recent = goal_context.recent_task_statuses
    recent_text = (
        "\n".join(f"- {t.task_id}: {t.status} | {t.summary_snippet[:200]}" for t in recent)
        if recent
        else "(无近期任务)"
    )

    # 列出目标目录文件清单，帮助明觉快速定位
    goal_file_listing = _list_goal_files(goal_dir_path)
    phase1_text = _read_phase1(goal_dir_path)

    system_prompt = f"""你是初爻·明觉，负责将模糊目标翻译为具体可执行的任务。

你可以使用以下探索能力在决策前收集信息：
- list_directory / read_file — 查看目标目录、阶段报告、蓝图、记忆、认知库
- query_capabilities — 了解执行环境能力

认知库路径：
- skills/ — L1 技能定义
- models/ — L2 经验模型
- grids/ — L3 认知格栅（JSON 含 source_truths/models/skills 跨层链接）
- truths/ — L4 不可变底层真理

**快速决策优先**：当前上下文中已包含蓝图/记忆/轨迹/文件清单等完整信息。
最多探索 1 轮文件读取后必须调用 task_complete。

## 当前目标上下文
- 蓝图摘要: {blueprint_snippet}
- 记忆摘要: {memory_snippet}
- 轨迹快照: {trajectory_snippet}
- 近期任务:
{recent_text}

{goal_file_listing}

{phase1_text}

## 八卦路由表（根据任务性质选择一卦）
- **qian**(乾/探索) — 需要大量搜索、学习的新任务
- **kun**(坤/执行) — 明确的文件操作、批量修改
- **zhen**(震/变革) — 代码修改、重构
- **xun**(巽/分析) — 深入分析、问题定位
- **kan**(坎/攻坚) — 疑难问题、调试
- **li**(离/整理) — 文档、总结、整理
- **gen**(艮/审视) — 只读分析、评估
- **dui**(兑/沟通) — 生成报告、完成总结

## 输出格式
信息收集充分后，调用 task_complete，填入以下参数：
- **summary** — 一句话总结本任务（纯文本）
- **concrete_goal** — 详细的任务描述，包含期望成果和约束
- **trigram** — 选卦：qian|kun|zhen|xun|kan|li|gen|dui
- **trigram_reason** — 选择此卦的理由
- **goal_progress_pct** — 目标整体完成百分比（0-100）

**goal_progress_pct**: 基于当前蓝图、已完成任务和记忆，评估**目标整体**的完成百分比。
新目标从 0 开始，每个成功任务推进 10-30%。客观评估，暗驱稍后会复核。

通过函数参数传入，不要包裹在额外的 markdown 代码块中。
"""

    try:
        from vingobot.goal.lightweight_loop import run_mingjue_loop

        # 使用目标目录作为轻量循环的工作区（只读，无需独立 task_dir）
        cognition_dirs = [
            str(wp.skills),
            str(wp.models),
            str(wp.grids),
        ]

        mingjue_provider = _get_provider()

        result = await run_mingjue_loop(
            task_dir=goal_dir_path,
            system_prompt=system_prompt,
            goal_dir=goal_dir_path,
            cognition_dirs=cognition_dirs,
            signal=signal,
            provider=mingjue_provider,
        )

        if result.task_completed and result.final_content:
            parsed = _parse_mingjue_json(result.final_content)
        else:
            logger.warning("[明觉] 轻量循环未完成，使用回退")
            return _fallback_mingjue(goal_context, source)

    except Exception:
        logger.exception("[明觉] 轻量探索循环失败，使用回退")
        return _fallback_mingjue(goal_context, source)

    trigram = parsed.get("trigram", "kun")
    parsed_pct = _parse_progress_pct(parsed.get("goal_progress_pct"))

    return MingjueOutput(
        intent="task",
        goal_id=goal_context.goal_id,
        summary=parsed.get("summary") or source.description[:100],
        concrete_goal=parsed.get("concrete_goal") or source.description,
        trigram=trigram,
        trigram_reason=parsed.get("trigram_reason", f"卦象: {trigram}"),
        initial_yao=1,
        goal_progress_pct=parsed_pct,
        context=MingjueContextInfo(
            workspace_root=str(wp.root),
            goal_dir=goal_dir_path,
            cognition_dirs={
                "skills": str(wp.skills),
                "models": str(wp.models),
                "grids": str(wp.grids),
            },
        ),
    )


def _list_goal_files(goal_dir: str) -> str:
    """列出目标目录下的文件清单，帮助明觉快速定位关键文件。"""
    try:
        gd = Path(goal_dir)
        if not gd.is_dir():
            return ""
        lines = ["## 目标目录文件清单"]
        for f in sorted(gd.iterdir()):
            if f.is_file():
                lines.append(f"- 📄 {f.name}")
            elif f.is_dir():
                lines.append(f"- 📁 {f.name}/")
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception:
        return ""


def _read_phase1(goal_dir: str) -> str:
    """读取阶段报告（如果存在）。"""
    try:
        phase1 = Path(goal_dir) / "phase1-report.md"
        if phase1.is_file():
            text = phase1.read_text(encoding="utf-8")[:2500]
            return f"\n## 阶段报告（phase1-report.md）\n{text}"
    except Exception:
        pass
    return ""


def _fallback_mingjue(goal_context: GoalContext, source: MingjueSource) -> MingjueOutput:
    """LLM-less fallback — use the raw description as-is."""
    wp = get_workspace_paths()
    return MingjueOutput(
        intent="task",
        goal_id=goal_context.goal_id,
        summary=source.description[:100],
        concrete_goal=source.description,
        trigram="kun",
        trigram_reason="无法调用LLM，默认坤卦执行",
        initial_yao=1,
        context=MingjueContextInfo(
            workspace_root=str(wp.root),
            goal_dir=str(wp.goals / goal_context.goal_id),
            cognition_dirs={
                "skills": str(wp.skills),
                "models": str(wp.models),
                "grids": str(wp.grids),
            },

        ),
    )


def _parse_mingjue_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output, tolerating markdown fences."""
    # Try raw parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code block
    for fence in ("```json", "```"):
        if fence in text:
            start = text.index(fence) + len(fence)
            end = text.rfind("```")
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, TypeError):
                    pass

    logger.warning("[明觉] 无法解析 JSON，使用默认值")
    return {"trigram": "kun", "summary": "", "concrete_goal": ""}


def _parse_progress_pct(raw: Any) -> int:
    """Parse goal_progress_pct from Mingjue's JSON, returning 0 if absent/invalid."""
    if raw is None:
        return 0
    try:
        pct = int(raw)
        return max(0, min(100, pct))
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Provider lazy-loading
# ---------------------------------------------------------------------------

_agent_name = "mingjue"
_provider: Any = None


def _get_provider() -> Any:
    """Lazily obtain the LLM provider for this sixiang agent.

    Uses the per-agent config (``agents.defaults.sixiang.agents.mingjue``)
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
        logger.warning("[明觉] 无法初始化 provider")
    return _provider


def set_provider(provider: Any) -> None:
    """Explicitly set the provider used by all sixiang modules."""
    global _provider
    _provider = provider
