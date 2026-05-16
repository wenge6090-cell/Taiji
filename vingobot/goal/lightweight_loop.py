"""
明觉/暗驱共享的只读探索-决策轻量循环。

为明觉（目标→任务翻译）和暗驱（目标级路由决策）提供一个简化的内循环
（Weaver→Yang→Yin→Executor），仅配备只读工具。在决策前可以探索目标目
录、读取文件、搜索认知库——让"薄决策器"也能先调查再判断。

与完整 task_inner_loop 的区别：
- 无 L3 格栅发现、无 LLM 策略编织、无八卦路由
- 仅只读工具 (read_file / list_directory / ...)
- 最多 5 轮（vs 30 轮执行）
- 输出解析自 task_complete 的 JSON
"""

from __future__ import annotations

import asyncio
import json
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

_MINGJUE_MAX_ROUNDS = 2
"""明觉探索最多轮次。"""

_ANQU_MAX_ROUNDS = 5
"""暗驱验证最多轮次。"""

_SIBIAN_MAX_ROUNDS = 2
"""思变评估最多轮次。"""


_LIGHTWEIGHT_TOOLS: list[str] = [
    "read_file",
    "list_directory",
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


async def run_sibian_loop(
    *,
    task_dir: str | Path,
    system_prompt: str,
    goal_dir: str | None = None,
    cognition_dirs: list[str] | None = None,
    signal: asyncio.Task | None = None,
    provider: Any = None,
) -> LightweightLoopResult:
    """运行思变的只读评估循环。

    思变评估跨任务执行模式（失败率、滞涨、方向偏离），
    判断蓝图是否需要修订。配有 exec 和 write_file 能力，
    可验证产出物和写入蓝图修订建议。
    """
    return await _run_lightweight_core(
        task_dir=task_dir,
        system_prompt=system_prompt,
        goal_dir=goal_dir,
        cognition_dirs=cognition_dirs,
        max_rounds=_SIBIAN_MAX_ROUNDS,
        agent_label="思变",
        temperature=0.3,
        signal=signal,
        provider=provider,
        extra_tool_names=["exec", "write_file"],
    )


async def run_mingjue_loop(
    *,
    task_dir: str | Path,
    system_prompt: str,
    goal_dir: str | None = None,
    cognition_dirs: list[str] | None = None,
    signal: asyncio.Task | None = None,
    provider: Any = None,
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
        provider=provider,
    )


async def run_anqu_loop(
    *,
    task_dir: str | Path,
    system_prompt: str,
    goal_dir: str | None = None,
    cognition_dirs: list[str] | None = None,
    signal: asyncio.Task | None = None,
    provider: Any = None,
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
        provider=provider,
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
    provider: Any = None,
    extra_tool_names: list[str] | None = None,
) -> LightweightLoopResult:
    """共享的只读探索-决策循环引擎。

    每轮：Yang（LLM）→ 阴（审批）→ 执行器（只读工具）。工具只有
    read_file / list_directory 等纯查询操作，阴直接放行。

    extra_tool_names: 额外工具名列表（如 exec, write_file），用于 Sibian
        等需要写权限的特殊节点。
    """

    task_dir = Path(task_dir)
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task_dir 不存在: {task_dir}")

    # ── 构建轻量工具定义 ──────────────────────────────────────
    tool_defs = _build_lightweight_tool_defs(extra_tool_names=extra_tool_names)

    # ── 历史跟踪 ──────────────────────────────────────────
    previous_invoke_results = ""
    previous_yang_content = ""  # 跨轮思考延续
    read_only_round_count = 0  # 连续纯读轮次计数器
    all_read_paths: set[str] = set()  # 已读文件路径集合

    for round_num in range(1, max_rounds + 1):
        if signal is not None and signal.cancelled():
            logger.info("[{}探索] 收到取消信号，第 {} 轮退出", agent_label, round_num)
            break

        # 注入轮次信息和自读检测到 system prompt
        full_prompt = system_prompt
        full_prompt += f"\n\n## 当前轮次\n第 {round_num}/{max_rounds} 轮"

        # 跨轮思考延续
        if previous_yang_content:
            full_prompt += (
                f"\n\n## 你上一轮的思考\n{previous_yang_content[:2000]}\n\n"
                "（以上是你上一轮的思考结论。无需重新读取相同文件验证，"
                "直接基于已有信息做出决策。）"
            )

        # 注入上一轮只读结果
        if previous_invoke_results:
            full_prompt += f"\n\n## 上一轮查询结果\n{previous_invoke_results[:12000]}"

        # 已读文件提醒
        if all_read_paths and read_only_round_count >= 1:
            paths_str = "\n".join(f"- {p}" for p in sorted(all_read_paths))
            full_prompt += (
                f"\n\n## 已读文件清单\n以下文件你已经读过：\n{paths_str}\n"
                "**切勿重复读取以上文件**。直接基于已有信息调用 `task_complete` 提交决策。"
            )

        # 自读循环警告
        if read_only_round_count >= 2:
            full_prompt += (
                f"\n\n## ⚠️ 自读循环警告\n"
                f"你已经连续 {read_only_round_count} 轮只读取信息。\n"
                f"**本轮必须调用 `task_complete` 提交决策**，不得再调用任何只读工具。\n"
                f"如有足够信息就做决策，信息不足也要做最佳判断——不要在有限轮次中无限读取。"
            )

        # ── 阳 ────────────────────────────────────────────────
        yang_response = await run_yang(
            system_prompt=full_prompt,
            tool_definitions=tool_defs,
            task_description="探索目标状态并做出决策",
            round_facts=[],
            temperature=temperature,
            signal=signal,
            provider=provider,
        )

        # 保存 Yang 的思考用于跨轮延续
        previous_yang_content = (yang_response.content or "")[:3000]

        # ── 轮次输出持久化 ────────────────────────────────────
        round_data: dict[str, Any] = {
            "phase": agent_label,
            "round": round_num,
            "yang_content": yang_response.content,
            "reasoning_content": yang_response.reasoning_content,
            "thinking_blocks": yang_response.thinking_blocks,
            "tool_calls": yang_response.tool_calls,
            "called_task_complete": yang_response.called_task_complete,
        }

        if yang_response.called_task_complete:
            # 始终优先 tool_call arguments（LLM 可能同时返回 text + tool_calls，
            # text 仅是推理文字，arguments 才是正式决策数据）
            final_content = None
            for tc in (yang_response.tool_calls or []):
                if tc.get("name") == "task_complete":
                    args = tc.get("arguments") or {}
                    raw = args.get("summary") or ""
                    # 兼容：summary 本身是 JSON
                    try:
                        json.loads(raw)
                        final_content = raw
                    except (json.JSONDecodeError, TypeError):
                        # 从独立参数组装 JSON
                        obj: dict[str, int | str] = {}
                        for f in ("summary", "concrete_goal", "trigram", "trigram_reason"):
                            if f in args:
                                obj[f] = args[f]
                        if "goal_progress_pct" in args and args["goal_progress_pct"] is not None:
                            obj["goal_progress_pct"] = int(args["goal_progress_pct"])
                        final_content = json.dumps(obj, ensure_ascii=False) if obj else raw
                    break

            if not final_content:
                final_content = yang_response.content or ""
            logger.info("[{}探索] 第 {} 轮 task_complete，收集到决策", agent_label, round_num)
            _save_round_output(task_dir, agent_label, round_num, round_data)
            return LightweightLoopResult(
                final_content=final_content,
                rounds_executed=round_num,
                task_completed=True,
            )

        if not yang_response.tool_calls:
            logger.debug("[{}探索] 第 {} 轮无工具调用，继续", agent_label, round_num)
            _save_round_output(task_dir, agent_label, round_num, round_data)
            continue

        # ── 阴（审批）─────────────────────────────────────────
        approved_calls, _yin_decision, _yin_reason, _ = await approve(
            yang_response.tool_calls,
            workspace_root=task_dir,
        )

        if not approved_calls:
            logger.debug("[{}探索] 第 {} 轮阴拒绝了所有调用", agent_label, round_num)
            round_data["yin_decision"] = _yin_decision
            round_data["yin_reason"] = _yin_reason
            _save_round_output(task_dir, agent_label, round_num, round_data)
            continue

        # ── 执行器 ────────────────────────────────────────────
        results = await execute_tool_calls(
            approved_calls,
            task_dir=task_dir,
            goal_dir=goal_dir,
            cognition_dirs=cognition_dirs,
        )

        # 将执行结果追加到轮次数据并保存
        round_data["yin_decision"] = _yin_decision
        round_data["yin_reason"] = _yin_reason
        round_data["approved_calls"] = [
            {"name": c.name, "arguments": c.arguments} for c in approved_calls
        ]
        round_data["results"] = [
            {"status": r.status, "output": r.output[:500], "error": r.error}
            for r in results
        ]
        _save_round_output(task_dir, agent_label, round_num, round_data)

        # 收集本轮只读工具输出，注入下一轮
        previous_invoke_results = _format_prev_results(approved_calls, results)

        # 跟踪已读文件和纯读轮次
        for call in approved_calls:
            if call.name == "read_file":
                path = call.arguments.get("path", "") if call.arguments else ""
                if path:
                    all_read_paths.add(path)

        # 检查是否全是只读工具
        is_read_only_round = all(c.name in _READ_ONLY_TOOLS for c in approved_calls)
        if is_read_only_round:
            read_only_round_count += 1
        else:
            read_only_round_count = 0

        # 自读循环自动终止：连续 3+ 轮只读
        if read_only_round_count >= 3:
            logger.warning("[{}探索] 连续 {} 轮纯读取，自动终止", agent_label, read_only_round_count)
            return LightweightLoopResult(
                final_content=None,
                rounds_executed=round_num,
                task_completed=False,
            )

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


def _build_lightweight_tool_defs(
    extra_tool_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """从 Weaver 的 _BASE_TOOL_DEFS 中提取工具定义。"""
    from vingobot.goal.weaver import _BASE_TOOL_DEFS

    tool_names = list(_LIGHTWEIGHT_TOOLS)
    if extra_tool_names:
        tool_names.extend(extra_tool_names)

    defs: list[dict[str, Any]] = []
    for name in tool_names:
        if name in _BASE_TOOL_DEFS:
            defs.append(_BASE_TOOL_DEFS[name])
    return defs


# ---------------------------------------------------------------------------
# 跨轮结果格式化
# ---------------------------------------------------------------------------

_READ_ONLY_TOOLS = frozenset({
    "read_file", "list_directory", "query_capabilities",
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


def _save_round_output(task_dir: Path, agent_label: str, round_num: int, data: dict) -> None:
    """保存单轮输出到 ``outputs/{label}-{round:03d}-round.json``。

    与 task_inner_loop 的轮次文件保存路径相同，用 agent_label 前缀区分。
    """
    out_dir = task_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    # 将中文标签转换为安全的文件前缀
    label_map = {"明觉": "mingjue", "暗驱": "anqu", "思变": "sibian"}
    prefix = label_map.get(agent_label, agent_label)
    fn = f"{prefix}-{round_num:03d}-round.json"
    try:
        (out_dir / fn).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("[{}探索] 保存轮次输出失败: {}", agent_label, fn)
