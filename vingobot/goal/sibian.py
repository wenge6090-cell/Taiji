"""
五爻·思变 — 连山策略引擎（Obstacle Navigation Engine）。

Sibian sits between Anqu (上爻·暗驱) and Mingjue (初爻·明觉) in the outer
loop.  When Anqu detects cross-task obstruction patterns, Sibian applies the
Lianshan Yi (连山易) framework to navigate around obstacles.

连山易框架:
- 艮(gèn)为首 — Stop first.  Observe the mountain terrain before acting.
- 六气 — Judge the goal's seasonal phase (启动/盛长/收敛/休眠).
- 六甲 — Judge the timing of obstruction (初甲/再甲/三甲).
- 三元 — Judge context freshness.
- 阴阳对峙 — Analyze the confrontation of driving vs. blocking forces.
- 方位决策 — Decide direction, timing, and posture for continued advance.

核心原则: 目标永不动，只调整推进方式。

与织(Weaver)的对偶关系:
- 织: 任务内认知姿态编排（用八卦格栅选"我是谁"）
- 思变: 目标级障碍导航（用连山策略选"怎么走"）
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.core.goal_context import GoalContext
from vingobot.core.workspace import get_workspace_paths
from vingobot.goal.types import AnquDecision, SibianDecision


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_MIN_TASKS_FOR_SIBIAN = 2
"""Minimum completed tasks before Sibian activates (need pattern data)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_sibian(
    goal_context: GoalContext,
    anqu_decision: AnquDecision,
    *,
    total_tasks: int = 0,
    goal_progress_history: list[int] | None = None,
    signal: asyncio.Task | None = None,
) -> SibianDecision:
    """Run the Lianshan obstacle-navigation engine.

    艮止 → 六气 → 六甲 → 三元 → 阴阳对峙 → 方位决策。
    目标永不动，只选择推进方式。
    """
    # ── Threshold: only activate with enough execution data ─────────
    task_statuses = goal_context.recent_task_statuses or []
    if len(task_statuses) < _MIN_TASKS_FOR_SIBIAN and total_tasks < _MIN_TASKS_FOR_SIBIAN:
        logger.debug("[思变] 任务数不足 ({}), 跳过评估", max(len(task_statuses), total_tasks))
        return SibianDecision(action="continue")

    # Only evaluate on goal_next_task (continuation decisions) or goal_failed (pre-failure rescue)
    if anqu_decision.action not in ("goal_next_task", "goal_failed"):
        return SibianDecision(action="continue")

    # ── Build Lianshan system prompt ───────────────────────────
    system_prompt = _build_lianshan_prompt(
        goal_context=goal_context,
        anqu_decision=anqu_decision,
        total_tasks=total_tasks,
        goal_progress_history=goal_progress_history,
    )

    try:
        from vingobot.goal.lightweight_loop import run_sibian_loop

        wp = get_workspace_paths()
        goal_dir_path = str(wp.goals / goal_context.goal_id)

        cognition_dirs = [
            str(wp.skills),
            str(wp.models),
            str(wp.grids),
        ]

        provider = _get_provider()

        result = await run_sibian_loop(
            task_dir=goal_dir_path,
            system_prompt=system_prompt,
            goal_dir=goal_dir_path,
            cognition_dirs=cognition_dirs,
            signal=signal,
            provider=provider,
        )

        if result.task_completed and result.final_content:
            parsed = _parse_sibian_json(result.final_content)
        else:
            logger.warning("[思变] 轻量循环未完成，跳过评估")
            return SibianDecision(action="continue")

    except Exception:
        logger.exception("[思变] 评估失败，跳过")
        return SibianDecision(action="continue")

    action = parsed.get("action", "continue")
    valid_actions = ("continue", "push_through", "navigate_around",
                     "wait_gather", "decompose", "escalate", "abort")
    if action not in valid_actions:
        logger.warning("[思变] 未知动作 '{}'，回退到 continue", action)
        action = "continue"

    timing = parsed.get("timing", "now")
    if timing not in ("now", "after_refresh", "after_one_task"):
        timing = "now"

    return SibianDecision(
        action=action,  # type: ignore[arg-type]
        liuq=parsed.get("liuq", ""),
        liujia=parsed.get("liujia", ""),
        sanyuan=parsed.get("sanyuan", ""),
        duizhi=parsed.get("duizhi", ""),
        strategy=parsed.get("strategy", ""),
        trigram_hint=parsed.get("trigram_hint", ""),
        timing=timing,  # type: ignore[arg-type]
        reason=parsed.get("reason", ""),
        blueprint_revision=parsed.get("blueprint_revision", ""),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sibian_json(text: str) -> dict[str, Any]:
    """Extract JSON from Sibian LLM output."""
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

    return {"action": "continue"}


# ---------------------------------------------------------------------------
# L2 model loader for Sibian
# ---------------------------------------------------------------------------


def _load_sibian_strategy_models() -> str:
    """Load L2 experience models referenced by 连山-策略.json.

    Reads the strategy grid's ``models`` list, then loads and formats
    each referenced L2 model file from the cognition models directory.

    Returns:
        Formatted section string for the Sibian system prompt, or empty
        string if no models found.
    """
    try:
        wp = get_workspace_paths()
        grid_path = wp.grids / "连山-策略.json"
        models_dir = wp.models

        if not grid_path.is_file() or not models_dir.is_dir():
            return ""

        raw = grid_path.read_text(encoding="utf-8")
        grid_data = json.loads(raw)
        model_names: list[str] = grid_data.get("models", [])
        if not model_names:
            return ""
    except Exception:
        logger.debug("[思变] 读取连山策略格栅失败")
        return ""

    entries: list[str] = []
    for model_name in model_names:
        if not model_name:
            continue
        model_file = None
        for ext in (".json", ".md"):
            candidate = models_dir / f"{model_name}{ext}"
            if candidate.is_file():
                model_file = candidate
                break
        if model_file is None:
            logger.debug("[思变] 连山策略引用的L2模型 '{}' 不存在", model_name)
            continue

        try:
            raw = model_file.read_text(encoding="utf-8")
            if model_file.suffix == ".json":
                content = _format_json_model(raw, model_name)
            else:
                content = raw[:800]
            entries.append(f"### {model_name}\n{content}")
        except Exception as exc:
            logger.debug("[思变] 读取L2模型 '{}' 失败: {}", model_name, exc)

    if not entries:
        return ""

    return "## 连山策略相关经验模型（L2 思维模型）\n\n" + "\n\n".join(entries)


def _format_json_model(raw: str, model_name: str) -> str:
    """Format a JSON L2 model file into readable prompt text."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:500]

    parts: list[str] = []
    desc = data.get("description", "")
    if desc:
        parts.append(f"**描述**: {desc}")

    pattern = data.get("pattern", {})
    if isinstance(pattern, dict):
        trigger = pattern.get("trigger", "")
        if trigger:
            parts.append(f"**触发条件**: {trigger}")

        mechanism = pattern.get("mechanism", "")
        if mechanism:
            parts.append(f"**机制**: {mechanism}")

        when_to_use = pattern.get("when_to_use", [])
        if isinstance(when_to_use, list) and when_to_use:
            parts.append("**适用场景**:")
            for item in when_to_use:
                parts.append(f"- {item}")

        when_not = pattern.get("when_not_to_use", [])
        if isinstance(when_not, list) and when_not:
            parts.append("**不适用场景**:")
            for item in when_not:
                parts.append(f"- {item}")

    mitigation = data.get("mitigation", {})
    if isinstance(mitigation, dict):
        strategy = mitigation.get("strategy", "")
        if strategy:
            parts.append(f"**缓解策略**: {strategy}")

    key_insight = data.get("key_insight", "")
    if key_insight:
        parts.append(f"**核心洞察**: {key_insight}")

    content = data.get("content", "")
    if content and not parts:
        parts.append(content[:500])

    return "\n".join(parts) if parts else raw[:500]


# ---------------------------------------------------------------------------
# Lianshan prompt builder
# ---------------------------------------------------------------------------


def _build_lianshan_prompt(
    goal_context: GoalContext,
    anqu_decision: AnquDecision,
    total_tasks: int = 0,
    goal_progress_history: list[int] | None = None,
) -> str:
    """Build the Lianshan obstacle-navigation system prompt.

    艮止 → 六气 → 六甲 → 三元 → 阴阳对峙 → 方位决策。
    """
    task_statuses = goal_context.recent_task_statuses or []

    # ── 执行数据 ──────────────────────────────────────
    blueprint = (
        goal_context.blueprint_summary[:3000]
        if goal_context.blueprint_summary else "(无蓝图)"
    )
    memory = (
        goal_context.memory_summary[:2000]
        if goal_context.memory_summary else "(无记忆)"
    )
    trajectory = (
        goal_context.trajectory_snapshot[:2000]
        if goal_context.trajectory_snapshot else "(新目标)"
    )
    known_traps = (
        goal_context.known_traps_text[:1500]
        if goal_context.known_traps_text else "(无已知陷阱)"
    )

    recent_tasks_text = "\n".join(
        f"- {t.task_id}: {t.status} | {t.summary_snippet[:200]}"
        for t in (task_statuses or [])
    ) or "(无近期任务)"

    # ── 进度历史 ──────────────────────────────────────
    progress_text = ""
    if goal_progress_history:
        progress_text = (
            "目标进度历史: "
            + " → ".join(f"{p}%" for p in goal_progress_history)
        )
        # 检测是否停滞
        if len(goal_progress_history) >= 3:
            last3 = goal_progress_history[-3:]
            if max(last3) - min(last3) <= 5:
                progress_text += "\n⚠️ 进度停滞：最近3个任务进度变化≤5%"

    # ── 连山格栅路径 ──────────────────────────────────
    wp = get_workspace_paths()
    lianshan_grid_path = str(wp.grids / "连山-策略.json")

    # ── 加载连山策略关联的 L2 思维模型 ─────────────────
    l2_models_text = _load_sibian_strategy_models()

    return f"""你是五爻·思变，连山策略引擎——六爻外循环的障碍导航器。

你的角色不是评估蓝图要不要改。你的角色是：**当目标推进受阻时，判断怎么继续走**。
目标永不动。你只选择推进的方式、方位、时机。

## 连山易决策框架

### 艮止（先停，再看）
不急着输出决策。先完整阅读下面的上下文，理解整座"山"的地形。

### 六气判断（目标的季节阶段）
根据任务数和进度判断目标处于什么阶段：
- **春-启动**: 任务<3且进度<30%——失败多是探索代价，不必急于改方向
- **夏-盛长**: 任务3-10且进度30-70%——失败应引起重视
- **秋-收敛**: 任务>10或进度>70%——策略偏保守，小步推进
- **冬-休眠**: 进度停滞——需要根本性调整

### 六甲判断（受阻的时序）
- **初甲**: 第一次遇阻——优先 push_through
- **再甲**: 第二次遇阻——考虑 navigate_around
- **三甲**: 第三次及以上——需要 decompose + wait_gather

### 三元判断（上下文新鲜度）
- 蓝图是否刚被修订？
- 认知库是否已刷新？
- 环境/约束是否有变化？

### 阴阳对峙（推力 vs 阻力）
- 阳（推力）: 哪些还在起作用？有进展吗？有没有未尝试的替代方案？
- 阴（阻力）: 失败集中在哪？known_traps 有匹配吗？缺什么信息/资源？

### 方位决策
根据上面的分析，选择一个推进方位：

| action | 含义 | 适用场景 |
|--------|------|---------|
| continue | 照常推进 | 无明显障碍 |
| push_through | 强攻突破 | 方向对但执行弱（换工具/加参数） |
| navigate_around | 绕行换路 | 当前路径受阻（换卦象/换方法） |
| wait_gather | 主动等待 | 信息不足（先收集、再决策） |
| decompose | 拆解降维 | 任务太大（拆成更小子任务） |
| escalate | 升级求助 | 环境不可变/依赖缺失 |
| abort | 终止 | 冬-休眠 + 三甲 + escalate 无解 |

详细策略说明见连山-策略.json。你可以在循环中 `read_file {lianshan_grid_path}` 查看完整策略格栅。

{l2_models_text}

## 当前目标上下文
- 目标ID: {goal_context.goal_id}
- 已完成任务数: {total_tasks}
- 蓝图: {blueprint}
- 记忆: {memory}
- 轨迹: {trajectory}
- 已知陷阱: {known_traps}

- 近期任务状态:
{recent_tasks_text}

{progress_text}

## 当前暗驱决策
- 动作: {anqu_decision.action}
- 下一任务: {(anqu_decision.next_task_description or '')[:300]}
- 进度评估: {anqu_decision.goal_progress_pct}%
- 任务总结: {(anqu_decision.task_summary or '')[:300]}
- 建议卦象: {anqu_decision.suggested_trigram or '(无)'}

## 输出格式

调用 task_complete，summary 输出以下 JSON（直接输出，不要包裹在 markdown 代码块中）：

{{
  "action": "continue | push_through | navigate_around | wait_gather | decompose | escalate | abort",
  "liuq": "六气判断——目标当前阶段及理由",
  "liujia": "六甲判断——第几次受阻、上次策略变更多久",
  "sanyuan": "三元判断——上下文是否陈旧",
  "duizhi": "阴阳对峙——推力 vs 阻力的对比分析",
  "strategy": "具体的推进策略描述（会注入明觉的下轮上下文中）",
  "trigram_hint": "建议下轮卦象（乾/坤/震/巽/坎/离/艮/兑，空字符串表示不改变）",
  "timing": "now | after_refresh | after_one_task",
  "reason": "决策理由",
  "blueprint_revision": "（仅 navigate_around/decompose 时可选）对任务拆解/顺序/里程碑的建议调整，不是改目标本身"
}}"""


def _get_provider() -> Any:
    """Get the configured LLM provider for Sibian."""
    try:
        from vingobot.providers.auto import get_best_provider
        return get_best_provider()
    except Exception:
        try:
            from vingobot.providers.factory import get_default_provider
            return get_default_provider()
        except Exception:
            return None
