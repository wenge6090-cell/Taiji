"""
明觉/暗驱共享的只读探索-决策轻量循环。

为明觉（目标→任务翻译）和暗驱（目标级路由决策）提供一个简化的内循环
（Weaver→Yang→Yin→Executor），仅配备只读工具。在决策前可以探索目标目
录、读取文件、搜索认知库——让"薄决策器"也能先调查再判断。

与完整 task_inner_loop 的区别：
- 无 L3 格栅发现、无 LLM 策略编织、无八卦路由
- 仅只读工具 (read_file / list_directory / load_grid / search_skills / ...)
- 最多 5 轮（vs 30 轮执行）
- 输出解析自 task_complete 的 JSON
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.goal.executor import execute_tool_calls
from vingobot.goal.yang import run_yang
from vingobot.goal.yin import approve

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_MINGJUE_MAX_ROUNDS = 5
"""明觉探索最多轮次。"""

_ANQU_MAX_ROUNDS = 5
"""暗驱验证最多轮次。"""

_LIGHTWEIGHT_TOOLS: list[str] = [
    "read_file",
    "list_directory",
    "load_grid",
    "search_skills",
    "search_models",
    "query_capabilities",
    "task_complete",
]
"""轻量循环可用的只读工具列表。"""


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------


@dataclass
class LightweightLoopResult:
    """轻量循环的输出。"""

    final_content: str | None
    """Yang 调用 task_complete 时的内容（含决策 JSON）。"""

    rounds_executed: int
    """实际执行轮次。"""

    task_completed: bool
    """是否通过 task_complete 正常结束。"""


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


async def run_mingjue_loop(
    *,
    task_dir: str | Path,
    system_prompt: str,
    goal_dir: str | None = None,
    cognition_dirs: list[str] | None = None,
    signal: asyncio.Task | None = None,
) -> LightweightLoopResult:
    """运行明觉的只读探索循环。

    明觉可以用 read_file / list_directory 探索目标目录中的文件、
    读取阶段报告、搜索认知库，在充分了解目标状态后调用 task_complete
    输出任务分解 JSON。
    """
    return await _run_lightweight_core(
        task_dir=task_dir,
        system_prompt=system_prompt,
        goal_dir=goal_dir,
        cognition_dirs=cognition_dirs,
        max_rounds=_MINGJUE_MAX_ROUNDS,
        agent_label="明觉",
        temperature=0.5,
        signal=signal,
    )


async def run_anqu_loop(
    *,
    task_dir: str | Path,
    system_prompt: str,
    goal_dir: str | None = None,
    cognition_dirs: list[str] | None = None,
    signal: asyncio.Task | None = None,
) -> LightweightLoopResult:
    """运行暗驱的只读验证循环。

    暗驱可以读取任务输出文件、验证产出质量、对照蓝图判断目标完成度，
    然后调用 task_complete 输出路由决策 JSON。
    """
    return await _run_lightweight_core(
        task_dir=task_dir,
        system_prompt=system_prompt,
        goal_dir=goal_dir,
        cognition_dirs=cognition_dirs,
        max_rounds=_ANQU_MAX_ROUNDS,
        agent_label="暗驱",
        temperature=0.4,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# 核心循环
# ---------------------------------------------------------------------------


async def _run_lightweight_core(
    *,
    task_dir: str | Path,
    system_prompt: str,
    goal_dir: str | None = None,
    cognition_dirs: list[str] | None = None,
    max_rounds: int = 5,
    agent_label: str = "",
    temperature: float = 0.5,
    signal: asyncio.Task | None = None,
) -> LightweightLoopResult:
    """共享的只读探索-决策循环引擎。

    每轮：Yang（LLM）→ 阴（审批）→ 执行器（只读工具）。工具只有
    read_file / list_directory / load_grid 等纯查询操作，阴直接放行。
    """

    task_dir = Path(task_dir)
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task_dir 不存在: {task_dir}")

    # ── 构建轻量工具定义 ──────────────────────────────────────
    tool_defs = _build_lightweight_tool_defs()

    # ── 历史记忆（跨轮传递已读内容，避免反复重读）─────────────
    previous_invoke_results = ""

    for round_num in range(1, max_rounds + 1):
        if signal is not None and signal.cancelled():
            logger.info("[{}探索] 收到取消信号，第 {} 轮退出", agent_label, round_num)
            break

        # 注入上一轮只读结果到 system prompt
        full_prompt = system_prompt
        if previous_invoke_results:
            full_prompt += f"\n\n## 上一轮查询结果\n{previous_invoke_results[:12000]}"

        # ── 阳 ────────────────────────────────────────────────
        yang_response = await run_yang(
            system_prompt=full_prompt,
            tool_definitions=tool_defs,
            task_description="探索目标状态并做出决策",
            round_facts=[],
            temperature=temperature,
            signal=signal,
        )

        if yang_response.called_task_complete:
            logger.info("[{}探索] 第 {} 轮 task_complete，收集到决策", agent_label, round_num)
            return LightweightLoopResult(
                final_content=yang_response.content,
                rounds_executed=round_num,
                task_completed=True,
            )

        if not yang_response.tool_calls:
            logger.debug("[{}探索] 第 {} 轮无工具调用，继续", agent_label, round_num)
            continue

        # ── 阴（审批）─────────────────────────────────────────
        approved_calls, _yin_decision, _yin_reason = await approve(
            yang_response.tool_calls,
            workspace_root=task_dir,
        )

        if not approved_calls:
            logger.debug("[{}探索] 第 {} 轮阴拒绝了所有调用", agent_label, round_num)
            continue

        # ── 执行器 ────────────────────────────────────────────
        results = await execute_tool_calls(
            approved_calls,
            task_dir=task_dir,
            goal_dir=goal_dir,
            cognition_dirs=cognition_dirs,
        )

        # 收集本轮只读工具输出，注入下一轮
        previous_invoke_results = _format_prev_results(approved_calls, results)

    # ── 达到最大轮次，未调用 task_complete ────────────────────
    logger.warning("[{}探索] 达到最大轮次 {}，未收到 task_complete", agent_label, max_rounds)
    return LightweightLoopResult(
        final_content=None,
        rounds_executed=min(round_num, max_rounds),  # type: ignore[possibly-used-before-assignment]
        task_completed=False,
    )


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------


def _build_lightweight_tool_defs() -> list[dict[str, Any]]:
    """从 Weaver 的 _BASE_TOOL_DEFS 中提取只读工具定义。"""
    from vingobot.goal.weaver import _BASE_TOOL_DEFS

    defs: list[dict[str, Any]] = []
    for name in _LIGHTWEIGHT_TOOLS:
        if name in _BASE_TOOL_DEFS:
            defs.append(_BASE_TOOL_DEFS[name])
    return defs


# ---------------------------------------------------------------------------
# 跨轮结果格式化
# ---------------------------------------------------------------------------

_READ_ONLY_TOOLS = frozenset({
    "read_file", "list_directory", "load_grid",
    "search_skills", "search_models", "query_capabilities",
})


def _format_prev_results(
    approved_calls: list[Any],
    results: list[Any],
) -> str:
    """格式化上一轮的只读工具输出，注入下一轮 system prompt。

    只收集成功执行的只读工具输出，每项截断到 3000 chars。
    """
    lines: list[str] = []
    for call, result in zip(approved_calls, results):
        if result.status != "success":
            continue
        if call.name not in _READ_ONLY_TOOLS:
            continue
        output = (result.output or "").strip()
        if not output:
            continue
        args_preview = ", ".join(
            f"{k}={v}" for k, v in (call.arguments or {}).items()
        )
        lines.append(f"### {call.name}({args_preview})")
        lines.append(output[:3000])

    return "\n\n".join(lines) if lines else ""
