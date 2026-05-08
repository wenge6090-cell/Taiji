"""
三爻·阳 — Native Function Calling layer.

Yang is the "empty" reasoning module.  It receives a woven system prompt
and a set of tool definitions, then calls the LLM via native Function
Calling.  Yang has zero knowledge of which tools are safe — it simply
outputs content and tool_calls.

Key principles:
- **Zero hardcoding**: All context comes from the system prompt.
- **Native FC only**: No text-format parsing for tool calls.
- **task_complete detection**: Monitors for the special tool call that
  signals the end of the inner loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from vingobot.goal.types import RoundExecutionFact, YangResponse


async def run_yang(
    system_prompt: str,
    tool_definitions: list[dict[str, Any]],
    task_description: str,
    round_facts: list[RoundExecutionFact],
    temperature: float = 0.7,
    *,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    signal: asyncio.Task | None = None,
    provider: Any = None,
) -> YangResponse:
    """Call the LLM with native Function Calling for one round.

    Args:
        system_prompt: Woven system prompt from the Weaver.
        tool_definitions: OpenAI-style tool definitions.
        task_description: Current task description.
        round_facts: Accumulated facts from previous rounds.
        temperature: Temperature for this round.
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling limit.
        repetition_penalty: Penalty for token repetition.
        signal: Optional cancellation token.
        provider: Optional explicit provider.  When provided, overrides
            the internally resolved yang provider.  Used by lightweight
            loops (mingjue/anqu) that need a different model.

    Returns:
        YangResponse with content and parsed tool_calls.
    """

    # Build facts text
    facts_text = _build_facts_text(round_facts)

    # Build user message
    user_content = f"当前任务：{task_description}\n\n历史执行事实汇总：\n{facts_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        effective_provider = provider or _get_provider()
        if effective_provider is None:
            logger.error("[阳] 无可用 LLM provider")
            return YangResponse(content="[错误] 无可用 LLM provider")

        response = await effective_provider.chat_with_retry(
            messages=messages,
            tools=tool_definitions if tool_definitions else None,
            tool_choice="auto",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("[阳] LLM 调用失败")
        return YangResponse(content="[错误] LLM 调用异常")

    # Parse content
    content = response.content or None
    reasoning_content = response.reasoning_content or None
    thinking_blocks = response.thinking_blocks or None

    # Parse tool calls
    tool_calls: list[dict[str, Any]] = []
    called_task_complete = False

    for tc in response.tool_calls:
        call_dict = {
            "id": tc.id,
            "name": tc.name,
            "arguments": tc.arguments,
        }
        tool_calls.append(call_dict)

        if tc.name == "task_complete":
            called_task_complete = True

    return YangResponse(
        content=content,
        reasoning_content=reasoning_content,
        thinking_blocks=thinking_blocks,
        tool_calls=tool_calls,
        called_task_complete=called_task_complete,
    )


def _build_facts_text(facts: list[RoundExecutionFact]) -> str:
    """Build a concise summary of accumulated round facts for Yang."""
    if not facts:
        return "（尚无历史记录，这是第一轮）"

    lines: list[str] = []
    for f in facts:
        tool_detail = ""
        if f.had_action_request:
            yin_detail = f"审批: {f.yin_decision}"
            if f.yin_decision in ("rejected", "modified") and f.yin_reason:
                yin_detail += f"({f.yin_reason[:120]})"
            tool_detail = f" | 工具调用: {f.tool_call_count}个 | {yin_detail} | 执行: {f.execution_status}"
        lines.append(f"第{f.round}轮: {f.yang_intent_summary}{tool_detail}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider lazy-loading — shared across all sixiang modules
# ---------------------------------------------------------------------------

_agent_name = "yang"
_provider: Any = None


def _get_provider() -> Any:
    """Lazily obtain the LLM provider for this sixiang agent.

    Uses the per-agent config (``agents.defaults.sixiang.agents.yang``)
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
        logger.warning("[阳] 无法初始化 provider")
    return _provider


def set_provider(provider: Any) -> None:
    """Explicitly set the provider used by Yang's LLM layer."""
    global _provider
    _provider = provider
