"""
五爻·行动 — Execution-phase self-reflection node.

Xingdong runs AFTER tool execution each round (conditionally, every N
rounds or on failure) to detect self-referential loops, exec walls,
and excessive read-only patterns at the round level.  It does NOT make
strategic routing decisions (that is Anqu's job) — it only asks:

    "Did what I just did actually advance the task?"

When stuck it injects a corrective warning or forces task_complete.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loguru import logger

from vingobot.goal.types import (
    ApprovedToolCall,
    ExecutionResult,
    RoundExecutionFact,
)

_RECHECK_INTERVAL = 3
"""Run self-check every N rounds (in addition to event-driven triggers)."""

_FORCE_TERMINATE_ROUND = 25
"""When remaining rounds <= this, xingdong may force task_complete."""


async def run_xingdong(
    round_num: int,
    max_rounds: int,
    facts: list[RoundExecutionFact],
    task_description: str,
    *,
    signal: asyncio.Task | None = None,
) -> str:
    """Execute tool calls and run a lightweight self-check.

    Returns a corrective message to inject into the next round's context,
    or an empty string when everything is fine.
    """
    # Only check every _RECHECK_INTERVAL rounds (or on force events)
    if round_num % _RECHECK_INTERVAL != 0 and round_num < max_rounds - 5:
        return ""

    provider = _get_provider()
    if provider is None:
        return _heuristic_check(round_num, max_rounds, facts)

    try:
        return await _llm_check(round_num, max_rounds, facts, task_description, provider, signal)
    except Exception:
        logger.exception("[行动] LLM 自省失败，使用启发式检测")
        return _heuristic_check(round_num, max_rounds, facts)


# ---------------------------------------------------------------------------
# LLM self-check
# ---------------------------------------------------------------------------


async def _llm_check(
    round_num: int,
    max_rounds: int,
    facts: list[RoundExecutionFact],
    task_description: str,
    provider: Any,
    signal: asyncio.Task | None,
) -> str:
    """Run a single-turn LLM self-check with minimal context."""
    remaining = max_rounds - round_num

    # Build compact round summary (last 5 rounds max)
    recent = facts[-5:]
    round_lines: list[str] = []
    for f in recent:
        tools = ""
        if f.tool_call_count > 0:
            tools = f"{f.tool_call_count}调用"
        status = _status_icon(f.execution_status)
        round_lines.append(
            f"- 第{f.round}轮: {f.yang_intent_summary[:80]} | "
            f"{tools} | {status} {f.execution_result_summary[:80]}"
        )
    recent_text = "\n".join(round_lines) if round_lines else "(无执行记录)"

    # Count recent write successes
    write_count = sum(
        1 for f in facts[-_RECHECK_INTERVAL:]
        if "成功" in f.execution_result_summary
        and any(
            kw in f.execution_result_summary
            for kw in ("write_file", "exec")
        )
    )
    read_only_count = sum(
        1 for f in facts[-_RECHECK_INTERVAL:]
        if f.tool_call_count > 0 and f.execution_status in ("success", "partial_failure")
        and "read_file" in (f.execution_result_summary or "")
        and "write_file" not in (f.execution_result_summary or "")
    )

    system_prompt = f"""你是五爻·行动（Xingdong），负责轮级执行自省。

**不要做全局策略决策**——那是上爻暗驱的职责。你只判断当前任务是否在轮级执行层面卡住了。

## 当前状态
- 正在执行第 {round_num} 轮 / 上限 {max_rounds} 轮（剩余 {remaining} 轮）
- 任务描述: {task_description[:200]}

## 最近 {len(recent)} 轮执行记录
{recent_text}

## 快速判断
- 最近 {_RECHECK_INTERVAL} 轮写入成功次数: {write_count}
- 最近 {_RECHECK_INTERVAL} 轮纯读次数: {read_only_count}
- 剩余轮数: {remaining}

## 输出（纯 JSON，无 markdown 包裹）
{{"self_check": "<ok|warning|force_complete>", "message": "简短原因"}}

- **ok**: 正常推进，无需干预。
- **warning**: 有风险（如连续纯读、exec 反复失败、接近上限）。message 必须以第二人称给 Yang 的警告，如 "你已经连续 N 轮只读取文件而未产出..."。
- **force_complete**: 必须立刻终止。仅当满足以下**全部**条件时使用：
  1. 剩余轮数 <= 3
  2. 最近 3 轮无任何 write_file 或 exec 成功
  3. 总执行轮数 >= 10"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "评估当前执行状态并输出 JSON。"},
    ]

    try:
        response = await provider.chat_with_retry(
            messages=messages,
            temperature=0.1,
            max_tokens=200,
        )
    except Exception:
        logger.warning("[行动] LLM 调用失败，使用启发式检测")
        return _heuristic_check(round_num, max_rounds, facts)

    content = (response.content or "").strip()
    parsed = _parse_xingdong_json(content)

    decision = parsed.get("self_check", "ok")
    msg = parsed.get("message", "")

    if decision == "force_complete":
        logger.warning("[行动] 强制终止: {}", msg)
        return f"## ⚠️ 行动节点强制终止\n{msg}\n\n请立即调用 task_complete 交付当前进度。"

    if decision == "warning":
        logger.info("[行动] 轮级警告: {}", msg)
        return f"## ⚠️ 行动节点警告\n{msg}\n\n请基于此警告调整本轮行动。"

    return ""


# ---------------------------------------------------------------------------
# Heuristic fallback (no LLM)
# ---------------------------------------------------------------------------


def _heuristic_check(
    round_num: int,
    max_rounds: int,
    facts: list[RoundExecutionFact],
) -> str:
    """Rule-based self-check when LLM is unavailable."""
    remaining = max_rounds - round_num

    # Count consecutive read-only rounds (looking back 5)
    recent = facts[-5:]
    consecutive_reads = 0
    consecutive_exec_fails = 0
    for f in reversed(recent):
        if f.tool_call_count == 0:
            break
        if f.execution_status == "exec_failed":
            consecutive_exec_fails += 1
        elif f.execution_status == "success" and any(
            kw in (f.execution_result_summary or "")
            for kw in ("write_file", "exec")
        ):
            break  # found a write — stop counting
        else:
            consecutive_reads += 1

    # Force complete: very close to limit with no writes
    if remaining <= 3 and consecutive_reads >= 3 and round_num >= 10:
        logger.warning(
            "[行动·启发式] 强制终止: 剩余 {} 轮，连续 {} 轮纯读",
            remaining, consecutive_reads,
        )
        return (
            f"## ⚠️ 行动节点强制终止（启发式检测）\n"
            f"剩余 {remaining} 轮，已连续 {consecutive_reads} 轮无产出。"
            f"请调用 task_complete 交付当前进度。"
        )

    # Warning: consecutive exec failures
    if consecutive_exec_fails >= 2:
        logger.warning(
            "[行动·启发式] exec 墙: 连续 {} 次 exec 失败",
            consecutive_exec_fails,
        )
        return (
            f"## ⚠️ 行动节点警告（exec 墙）\n"
            f"已连续 {consecutive_exec_fails} 次 exec 失败。"
            f"请切换到 write_file 模式，产出安装说明或替代方案文档。"
        )

    # Warning: extended read-only
    if consecutive_reads >= 3 and remaining <= 8:
        logger.warning(
            "[行动·启发式] 纯读警告: 连续 {} 轮，剩余 {} 轮",
            consecutive_reads, remaining,
        )
        return (
            f"## ⚠️ 行动节点警告（纯读模式）\n"
            f"已连续 {consecutive_reads} 轮只读取未产出，仅剩 {remaining} 轮。"
            f"本轮必须产出实质性交付物。"
        )

    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_icon(status: str) -> str:
    """Compact status indicator."""
    icons = {
        "success": "✅",
        "exec_failed": "❌exec",
        "failure": "❌",
        "partial_failure": "⚠️",
        "skipped": "⏭️",
    }
    return icons.get(status, "❓")


def _parse_xingdong_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output, tolerating markdown fences."""
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

    # Last resort: extract JSON-like substring
    m = re.search(r'\{[^{}]*"self_check"[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    logger.warning("[行动] 无法解析 LLM 输出: {}", text[:100])
    return {"self_check": "ok"}


# ---------------------------------------------------------------------------
# Provider lazy-loading
# ---------------------------------------------------------------------------

_agent_name = "xingdong"
_provider: Any = None


def _get_provider() -> Any:
    """Lazily obtain the LLM provider for the xingdong agent."""
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
        logger.warning("[行动] 无法初始化 provider")
    return _provider


def set_provider(provider: Any) -> None:
    """Explicitly set the provider."""
    global _provider
    _provider = provider
