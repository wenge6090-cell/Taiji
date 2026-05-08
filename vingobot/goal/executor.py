"""
五爻·执行器 — Final protection line for tool execution.

The Executor receives the approved tool calls from Yin and executes them
with hard-coded safety checks as the last line of defense:

1. **Path traversal re-check** — Double-check against ``..`` and abs paths.
2. **Command allowlist re-verify** — Second pass on exec safety.
3. **Rate limiting** — Simple per-tool invocation cap per round.
4. **Concurrent execution** — All approved calls run concurrently via
   ``asyncio.gather``, since they target independent paths and each
   ``_execute_one`` is error-isolated.
5. **Result packaging** — Results formatted for the next round's facts.

Tool execution is delegated to ``vingobot.core.tool_executor`` (shared between
AgentLoop and sixiang processes), eliminating the previous reverse import
from ``vingobot.agent.tools.registry``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.core.tool_executor import (
    execute_builtin_tool,
)
from vingobot.goal.types import ApprovedToolCall, ExecutionResult
from vingobot.goal.types import SixiangPermissionConfig
from vingobot.goal.yin import _check_path_safety as _yin_check_path_safety
from vingobot.goal.yin import _check_exec_safety as _yin_check_exec_safety

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_MAX_CALLS_PER_ROUND = 10
"""Maximum number of approved calls per round (hard limit)."""

_MAX_CONCURRENT_TOOLS = 10
"""Maximum concurrent tool calls within a single round.

Matches _MAX_CALLS_PER_ROUND so every approved call runs immediately
(no queuing). 10 concurrent disk/network operations are well within
modern system tolerances.
"""

_tool_gate: asyncio.Semaphore | None = None


def _get_tool_gate() -> asyncio.Semaphore:
    """Lazy-init the shared semaphore (module-level singleton)."""
    global _tool_gate
    if _tool_gate is None:
        _tool_gate = asyncio.Semaphore(_MAX_CONCURRENT_TOOLS)
    return _tool_gate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_tool_calls(
    calls: list[ApprovedToolCall],
    *,
    task_dir: str | Path | None = None,
    goal_dir: str | Path | None = None,
    cognition_dirs: list[str | Path] | None = None,
    perm: SixiangPermissionConfig | None = None,
) -> list[ExecutionResult]:
    """Execute a batch of approved tool calls.

    Args:
        calls: Approved calls from Yin.
        task_dir: Optional task directory for path resolution context.
        goal_dir: Optional goal directory for read/write/exec path resolution.
        cognition_dirs: Optional cognition directories (skills, models, grids)
            for read-only path resolution.
        perm: Unified permission config.  When provided, ``task_dir`` / ``goal_dir``
            / ``cognition_dirs`` are derived from it and individual params
            are ignored.

    Returns:
        List of ``ExecutionResult``, one per call.
    """
    if not calls:
        return []

    # Rate limiting
    if len(calls) > _MAX_CALLS_PER_ROUND:
        logger.warning("[执行器] 超过单轮最大调用数 ({})，截断", _MAX_CALLS_PER_ROUND)
        calls = calls[:_MAX_CALLS_PER_ROUND]

    results: list[ExecutionResult] = []

    # Execute all approved calls concurrently (independent, error-isolated)
    gate = _get_tool_gate()

    async def _guarded(c: ApprovedToolCall) -> ExecutionResult:
        async with gate:
            return await _execute_one(
                c, task_dir=task_dir, goal_dir=goal_dir,
                cognition_dirs=cognition_dirs, perm=perm,
            )

    tasks = [_guarded(c) for c in calls]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    for result in gathered:
        if isinstance(result, BaseException):
            logger.warning("[执行器] 内部异常: {}", result)
            # Create a synthetic failure result (we lost the original call identity)
            # asyncio.gather with return_exceptions=True should not raise for a single
            # _execute_one since it catches its own errors; this is pure defense.
            results.append(ExecutionResult(
                call=ApprovedToolCall(name="<unknown>", arguments={}),
                status="error",
                output="",
                error=str(result),
            ))
        else:
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Single call execution
# ---------------------------------------------------------------------------


async def _execute_one(
    call: ApprovedToolCall,
    task_dir: str | Path | None = None,
    goal_dir: str | Path | None = None,
    cognition_dirs: list[str | Path] | None = None,
    perm: SixiangPermissionConfig | None = None,
) -> ExecutionResult:
    """Execute a single approved tool call."""
    tool_name = call.name
    args = call.arguments

    # If a permission config is provided, derive all dirs from it
    if perm is not None:
        read_allowed = perm.read_allowed_dirs
        write_allowed = perm.write_allowed_dirs
        exec_cwds = perm.exec_allowed_cwds
        yin_root = perm.yin_workspace_root or (Path(task_dir) if task_dir else None)
    else:
        # Legacy fallback
        read_allowed: list[Path] = []
        if goal_dir:
            read_allowed.append(Path(goal_dir))
        if cognition_dirs:
            read_allowed.extend(Path(d) for d in cognition_dirs)
        write_allowed = [Path(task_dir)] if task_dir else []
        exec_cwds = [Path(task_dir)] if task_dir else []
        yin_root = Path(task_dir) if task_dir else None

    # Final path safety re-check for IO tools (uses Yin's comprehensive checks)
    if tool_name in ("write_file", "read_file", "delete_file", "edit_file"):
        path_str = args.get("path", "")
        if path_str:
            is_read_op = tool_name in ("read_file",)
            if is_read_op and read_allowed:
                # Check against each read-only allowed dir
                safe = False
                for allowed in read_allowed:
                    chk_safe, _ = _yin_check_path_safety(
                        path_str,
                        workspace_root=allowed,
                        for_write=False,
                    )
                    if chk_safe:
                        safe = True
                        break
                if not safe:
                    # Fall through to task_dir check
                    chk_safe, reason = _yin_check_path_safety(
                        path_str,
                        workspace_root=yin_root,
                        for_write=False,
                    )
                    if not chk_safe:
                        return ExecutionResult(
                            call=call,
                            status="blocked",
                            output="",
                            error=f"[最终防线] {reason}",
                        )
            else:
                # Write operations: check against write_allowed targets
                safe = False
                for target in (write_allowed if write_allowed else [yin_root]):
                    chk_safe, reason = _yin_check_path_safety(
                        path_str,
                        workspace_root=target,
                        for_write=True,
                    )
                    if chk_safe:
                        safe = True
                        break
                if not safe:
                    return ExecutionResult(
                        call=call,
                        status="blocked",
                        output="",
                        error=f"[最终防线] {reason}",
                    )

    # Final exec safety re-check (with workspace boundary)
    if tool_name == "exec":
        cmd = args.get("command", "")
        cwd = args.get("cwd", "")
        if cmd:
            safe, reason = _yin_check_exec_safety(
                cmd, cwd, workspace_root=yin_root,
            )
            if not safe:
                return ExecutionResult(
                    call=call,
                    status="blocked",
                    output="",
                    error=f"[最终防线] {reason}",
                )

    # Try skill tool routing (L3 grid discovered tools)
    skill_result = await _try_skill_tool(tool_name, call, args)
    if skill_result is not None:
        return skill_result

    # Fallback: delegate to shared builtin executor
    try:
        read_only_dirs = read_allowed if tool_name in ("read_file", "list_directory") else None
        # Pass write_allowed for edit_file and delete_file too
        use_write = write_allowed if tool_name in ("write_file", "edit_file", "delete_file") else None
        output = await execute_builtin_tool(
            tool_name, args, task_dir,
            read_only_allowed_dirs=read_only_dirs,
            write_allowed_dirs=use_write,
            exec_allowed_cwds=exec_cwds,
        )
        if output.startswith("[错误]"):
            return ExecutionResult(call=call, status="error", output="", error=output)
        # ── exec 失败检测：超时或非零退出码 ──────────────────
        if tool_name == "exec":
            if "timed out" in output:
                return ExecutionResult(
                    call=call, status="exec_failed",
                    output=output[:8000], error=output[:500],
                )
            # 非零退出码（Exit code: 非0）
            if _has_nonzero_exit(output):
                return ExecutionResult(
                    call=call, status="exec_failed",
                    output=output[:8000], error=output[:500],
                )
        return ExecutionResult(
            call=call,
            status="success",
            output=output[:8000],
        )
    except Exception as exc:
        return ExecutionResult(
            call=call,
            status="error",
            output="",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_nonzero_exit(output: str) -> bool:
    """Check if exec output contains a non-zero exit code."""
    import re
    return bool(re.search(r"Exit code:\s*([1-9]\d*)", output))


# ---------------------------------------------------------------------------
# Skill tool routing
# ---------------------------------------------------------------------------


async def _try_skill_tool(
    tool_name: str,
    call: ApprovedToolCall,
    args: dict[str, Any],
) -> ExecutionResult | None:
    """Try to execute a skill-registered tool.

    Returns an ``ExecutionResult`` if the tool is executed via its
    registered executor, or ``None`` to continue normal routing
    (falling through to the shared builtin executor).
    """
    try:
        from vingobot.goal.skill_parser import get_skill_executor, get_skill_tool

        skill_tool = get_skill_tool(tool_name)
        if skill_tool is None:
            return None  # Not a skill tool → fall through

        # Prefer a dedicated executor over builtin fallback
        executor_fn = get_skill_executor(tool_name)
        if executor_fn is not None:
            logger.debug("[技能工具] 执行 '{}' 的专用执行器", tool_name)
            output = await executor_fn(**args)
            return ExecutionResult(
                call=call,
                status="success",
                output=str(output)[:8000],
            )

        # Schema-only skill tool — delegate to shared builtin executor
        logger.debug(
            "[技能工具] '{}' 已注册到 SKILL.md 但无专用执行器，回退到内置路由",
            tool_name,
        )
        return None
    except Exception as exc:
        logger.debug("[执行器] 技能工具路由失败 {}: {}", tool_name, exc)
        return None
