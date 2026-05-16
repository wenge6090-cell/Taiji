"""
五爻·行动 — Intelligent execution node with self-healing.

Action RECEIVES the approved tool calls from Yin, makes a lightweight
pre-execution judgment about whether these calls are worth running,
and then EXECUTES them.

When a side-effect call fails due to an **environment problem**
(missing directory, missing dependency, timeout), Action attempts
to auto-fix the issue and retry once — within the same round.

Action has NO authority to terminate the loop.  That power belongs
solely to Yin (via self_reflect).  Action decides:

    "Execute, or skip.  If it fails, fix the environment, or tell Yang."

**Fix boundary**: Only environment problems (dir not found, cmd not found,
timeout).  Never fix logic errors (code content, algorithm, format).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.goal.executor import execute_single_call, execute_tool_calls
from vingobot.goal.types import (
    ApprovedToolCall,
    ExecutionResult,
    RoundExecutionFact,
    SixiangPermissionConfig,
)

# ── Read-only tools (don't produce files, no retry needed) ─────────
_READ_ONLY_TOOLS = frozenset({
    "read_file", "list_directory",
    "web_search", "web_fetch", "search_codebase",
})

# ── Side-effect tools (produce files or run commands; retry-able) ──
_SIDE_EFFECT_TOOLS = frozenset({
    "write_file", "exec", "edit_file", "delete_file",
})

# ── Pre-execution check: trigger every N rounds or on stall ───────
_PRE_CHECK_INTERVAL = 4
_PRE_CHECK_FORCE = 8

# ── Retry configuration ──────────────────────────────────────────
_MAX_RETRIES = 1
"""Maximum retry attempts per side-effect call after fix."""


async def run_action(
    approved_calls: list[ApprovedToolCall],
    perm: SixiangPermissionConfig,
    facts: list[RoundExecutionFact],
    round_num: int,
    max_rounds: int,
    task_description: str,
    *,
    signal: asyncio.Task | None = None,
) -> tuple[list[ExecutionResult], str, list[str]]:
    """Execute approved tool calls with pre-execution judgment + self-healing.

    Side-effect calls (write_file, exec, etc.) run sequentially with
    automatic fix+retry for environment failures.  Read-only calls run
    concurrently without retry.

    Returns:
        (results, veto_note, fix_log)
        - results: execution results (empty if all vetoed)
        - veto_note: "" if executed normally; otherwise a note for Yang
        - fix_log: list of fix attempt summaries for Yang's context
    """
    if not approved_calls:
        return [], "", []

    # ── 1. Pre-execution check (LLM) ────────────────────────────
    remaining = max_rounds - round_num
    all_read_only = all(c.name in _READ_ONLY_TOOLS for c in approved_calls)

    veto_note = ""
    if all_read_only or round_num % _PRE_CHECK_INTERVAL == 0 or remaining <= _PRE_CHECK_FORCE:
        veto_note = await _pre_execution_check(
            approved_calls, facts, round_num, max_rounds, task_description,
            all_read_only=all_read_only, signal=signal,
        )
        if veto_note:
            logger.info("[行动] 预检判定: 跳过本轮 {} 个调用 — {}", len(approved_calls), veto_note[:100])
            return [], veto_note, []

    # ── 2. Index calls by original position (preserve order for zip) ─
    indexed_results: dict[int, ExecutionResult] = {}
    fix_log: list[str] = []

    ro_indices: list[int] = []
    se_indices: list[int] = []
    other_indices: list[int] = []

    for i, c in enumerate(approved_calls):
        if c.name in _READ_ONLY_TOOLS:
            ro_indices.append(i)
        elif c.name in _SIDE_EFFECT_TOOLS:
            se_indices.append(i)
        else:
            other_indices.append(i)

    # ── 3. Read-only + other: concurrent execution ─────────────
    batch_indices = ro_indices + other_indices
    if batch_indices:
        batch_calls = [approved_calls[i] for i in batch_indices]
        batch_results = await execute_tool_calls(batch_calls, perm=perm)
        for idx, result in zip(batch_indices, batch_results):
            indexed_results[idx] = result

    # ── 4. Side-effect: sequential with fix+retry ───────────────
    for idx in se_indices:
        call = approved_calls[idx]
        result = await execute_single_call(call, perm=perm)

        if result.status in ("exec_failed", "error"):
            failure_type = _classify_failure(call, result)
            fix_action = await _attempt_fix(call, result, failure_type, perm)

            if fix_action:
                logger.info("[行动] {}({}) 失败({}), 尝试修复后重试",
                            call.name, _call_args_preview(call), failure_type)
                retry_result = await execute_single_call(call, perm=perm)
                if retry_result.status == "success":
                    fix_log.append(
                        f"第{round_num}轮: {call.name} 因{failure_type}失败 → 自动修复后重试成功"
                    )
                    indexed_results[idx] = retry_result
                    continue
                else:
                    fix_log.append(
                        f"第{round_num}轮: {call.name} 因{failure_type}失败 → "
                        f"修复尝试后仍失败: {(retry_result.error or '')[:100]}"
                    )
                    indexed_results[idx] = retry_result
                    continue
            else:
                fix_log.append(
                    f"第{round_num}轮: {call.name} 因{failure_type}失败 → 不可自动修复"
                )

        indexed_results[idx] = result

    # ── 5. Reconstruct in original approved_calls order ─────────
    all_results = [indexed_results[i] for i in range(len(approved_calls))]

    return all_results, veto_note, fix_log


# ---------------------------------------------------------------------------
# Pre-execution LLM check
# ---------------------------------------------------------------------------


async def _pre_execution_check(
    approved_calls: list[ApprovedToolCall],
    facts: list[RoundExecutionFact],
    round_num: int,
    max_rounds: int,
    task_description: str,
    *,
    all_read_only: bool,
    signal: asyncio.Task | None,
) -> str:
    """Run LLM pre-execution judgment.

    Returns "" (execute normally) or a veto note for Yang (calls skipped).
    Action does NOT force-terminate — only Yin has that authority.
    """
    remaining = max_rounds - round_num

    # ── Compute consecutive dry-spell ───────────────────────────
    recent = facts[-5:]
    consecutive_no_write = 0
    for f in reversed(recent):
        if f.tool_call_count == 0:
            break
        if f.execution_status == "success" and any(
            kw in (f.execution_result_summary or "") for kw in ("write_file", "exec")
        ):
            break
        consecutive_no_write += 1

    # ── Grace period: first 2 rounds — allow initial info gathering ──
    if round_num <= 2:
        return ""

    # ── Tier-1 heuristic: persistent dry spell (no LLM needed) ──
    if all_read_only and consecutive_no_write >= 5:
        skipped_calls_summary = _summarize_skipped_calls(approved_calls)
        return (
            f"跳过本轮 {len(approved_calls)} 个只读调用："
            f"已连续 {consecutive_no_write} 轮无实质性产出（write_file/exec），"
            f"剩余 {remaining} 轮。\n\n"
            f"被跳过的调用：{skipped_calls_summary}\n\n"
            f"请在下一轮用 write_file 产出实质性交付物（如分析报告、代码文件），"
            f"或如果任务确已完成则调用 task_complete。"
            f"不要再提出纯读取调用——它们会被拦截。"
        )

    # If not all read-only, no need to check — let execution proceed
    if not all_read_only:
        return ""

    # ── Tier-2 LLM check: border cases (dry spell < 3 rounds) ──
    provider = _get_provider()
    if provider is None:
        # No LLM available: skip heuristic check (not critical enough)
        return ""

    # Build compact call summary
    call_lines = []
    for c in approved_calls:
        args_preview = ", ".join(f"{k}={str(v)[:40]}" for k, v in (c.arguments or {}).items())
        call_lines.append(f"- {c.name}({args_preview})")
    calls_text = "\n".join(call_lines)

    # Build recent facts summary
    fact_lines = []
    for f in recent:
        fact_lines.append(
            f"- 第{f.round}轮: {f.execution_status} | "
            f"{f.tool_call_count}调用 | {f.execution_result_summary[:80]}"
        )
    facts_text = "\n".join(fact_lines) if fact_lines else "(首轮)"

    system_prompt = f"""你是五爻·行动（Action），纯粹的执行节点。

阳（Yang）已提出以下工具调用，阴（Yin）已审批通过。你是执行者——只决定：
这些调用是否值得执行？如果全是重复读文件/诊断命令，可以拒绝执行并告诉阳原因。

**你没有权力终止循环**——那是阴的职责。你只能执行或拒绝执行。

## 当前状态
- 第 {round_num} 轮 / 上限 {max_rounds} 轮（剩余 {remaining} 轮）
- 任务: {task_description[:200]}

## 本轮待执行的调用（全部为只读）
{calls_text}

## 最近 5 轮执行记录
{facts_text}

## 判定指引
- 第 3-4 轮：如果近期已有 0-2 轮纯读且未产出，**放行**（让 Yang 收集信息）
- 第 5+ 轮：如果近期已有 2+ 轮纯读且无产出，**拒绝**（让 Yang 直接产出交付物）
- 当前是第 {round_num} 轮，连续无产出 {consecutive_no_write} 轮

## 输出（纯 JSON）
{{"action": "<execute|veto>", "message": "简短原因"}}

- **execute**: 正常执行。
- **veto**: 拒绝执行本轮调用。仅在全部是只读/诊断调用，且近期模式已反复空转时使用。
  message 要简短告诉阳为什么拒绝，并建议下一轮产出实质性交付物。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "判断本轮调用是否应该执行。"},
    ]

    try:
        response = await provider.chat_with_retry(
            messages=messages,
            temperature=0.1,
            max_tokens=200,
        )
    except Exception:
        logger.warning("[行动] LLM 预检失败，放行执行")
        return ""

    content = (response.content or "").strip()
    parsed = _parse_json(content)
    decision = parsed.get("action", "execute")
    msg = parsed.get("message", "")

    if decision == "veto":
        logger.info("[行动] 预检 LLM 判定: 拒绝执行 — {}", msg)
        return msg

    return ""


# ---------------------------------------------------------------------------
# Failure classification & auto-fix
# ---------------------------------------------------------------------------


_FAILURE_PATTERNS: list[tuple[str, str, bool]] = [
    # (label, regex pattern, is_fixable) — patterns match lowercased error text
    ("dir_not_found", r"no such file or directory", True),
    ("dir_not_found", r"enoent", True),
    ("cmd_not_found", r"command not found", True),
    ("cmd_not_found", r"not recognized as an internal or external command", True),
    ("timeout", r"timed out", True),
    ("perm_denied", r"permission denied", False),
    ("perm_denied", r"eacces", False),
    ("syntax_error", r"syntaxerror", False),
    ("syntax_error", r"invalid syntax", False),
]
"""Failure classification patterns: (label, regex, is_fixable).
Order matters — first match wins."""


def _classify_failure(call: ApprovedToolCall, result: ExecutionResult) -> str:
    """Classify an execution failure into a fixable/unfixable label.

    Returns one of: dir_not_found, cmd_not_found, timeout,
    perm_denied, syntax_error, other.
    """
    error_text = (result.error or "") + (result.output or "")
    if not error_text:
        return "other"

    error_lower = error_text.lower()
    for label, pattern, _fixable in _FAILURE_PATTERNS:
        if re.search(pattern, error_lower):
            return label

    return "other"


async def _attempt_fix(
    call: ApprovedToolCall,
    result: ExecutionResult,
    failure_type: str,
    perm: SixiangPermissionConfig,
) -> dict[str, Any] | None:
    """Attempt to fix an environment failure before retry.

    Returns a dict with fix details if fix was attempted, or None if
    the failure type is unfixable or the fix action failed.

    The fix action itself should be safe — it only operates within
    the workspace boundaries already validated by Yin and the executor.
    """
    args = call.arguments or {}

    if failure_type == "dir_not_found":
        return _fix_dir_not_found(call, args, perm)

    if failure_type == "cmd_not_found":
        return await _fix_cmd_not_found(args, perm)

    if failure_type == "timeout":
        # Retry once — transient network/load issues may resolve.
        # (Timeout multiplier would require executor changes; plain retry for now.)
        logger.info("[行动] exec 超时，直接重试")
        return {"fix": "timeout_retry", "action": "retry"}

    # perm_denied, syntax_error, other → unfixable
    return None


def _fix_dir_not_found(
    call: ApprovedToolCall,
    args: dict[str, Any],
    perm: SixiangPermissionConfig,
) -> dict[str, Any] | None:
    """Create missing parent directories for write_file / edit_file."""
    if call.name not in ("write_file", "edit_file"):
        return None

    path_str = args.get("path", "")
    if not path_str:
        return None

    try:
        parent = Path(path_str).parent
        if parent and not parent.exists():
            os.makedirs(parent, exist_ok=True)
            logger.info("[行动] 自动创建目录: {}", parent)
            return {"fix": "mkdir", "path": str(parent)}
    except OSError as exc:
        logger.warning("[行动] 创建目录失败: {} — {}", parent, exc)

    return None


async def _fix_cmd_not_found(
    args: dict[str, Any],
    perm: SixiangPermissionConfig,
) -> dict[str, Any] | None:
    """Attempt to install a missing command via pip.

    Only attempts pip install for Python-related command-not-found
    errors (e.g., 'python -m module', 'script.py').  Skips system
    commands (git, curl, etc.) to avoid destructive apt-get installs.
    """
    cmd = (args.get("command", "") or "").strip()
    if not cmd:
        return None

    # Only attempt pip install for Python tool invocations
    python_pkg = _extract_python_package(cmd)
    if not python_pkg:
        logger.info("[行动] 非 Python 命令缺失，不自动安装: {}", cmd[:80])
        return None

    # Run pip install within the task workspace
    try:
        install_cmd = f"pip install {python_pkg} 2>&1"
        logger.info("[行动] 尝试安装缺失包: {}", python_pkg)

        from vingobot.goal.executor import execute_single_call as _exec_one
        install_call = ApprovedToolCall(
            name="exec",
            arguments={"command": install_cmd},
        )
        install_result = await _exec_one(install_call, perm=perm)
        if install_result.status == "success":
            logger.info("[行动] 包安装成功: {}", python_pkg)
            return {"fix": "pip_install", "package": python_pkg}
        else:
            logger.warning("[行动] 包安装失败: {} — {}", python_pkg, install_result.error or "")[:100]
    except Exception as exc:
        logger.warning("[行动] pip install 异常: {}", exc)

    return None


def _extract_python_package(cmd: str) -> str | None:
    """Extract a pip-installable package name from a Python command.

    Examples:
        "python script.py" → None (not a package import)
        "python -m http.server" → None (stdlib)
        "python -m yt_dlp ..." → "yt-dlp"
        "python -c 'import requests'" → "requests"
        "python3 -m pip ..." → None (pip itself)
        "uv run script.py" → None
    """
    # Python -m <module>: extract module name
    m_flag = re.search(r"python\d*\.?\d*\s+-m\s+([\w.]+)", cmd)
    if m_flag:
        module = m_flag.group(1)
        # Skip stdlib modules and pip
        if module in ("pip", "http.server", "json", "unittest", "venv", "ensurepip"):
            return None
        # Convert module name to pip package name (underscore to dash)
        return module.replace("_", "-")

    # Python -c 'import X': extract import name
    import_match = re.search(r"import\s+([\w.]+)", cmd)
    if import_match and ("python" in cmd.lower()):
        pkg = import_match.group(1)
        # Skip stdlib
        if pkg in ("os", "sys", "re", "json", "pathlib", "shutil", "subprocess", "time", "math"):
            return None
        return pkg.replace("_", "-")

    return None


def _call_args_preview(call: ApprovedToolCall) -> str:
    """Compact one-line summary of a tool call for logging."""
    args = call.arguments or {}
    key_arg = ""
    if call.name in ("write_file", "read_file", "edit_file", "delete_file"):
        key_arg = args.get("path", "")
    elif call.name == "exec":
        key_arg = (args.get("command", "") or "")[:60]
    else:
        key_arg = ", ".join(f"{k}={str(v)[:30]}" for k, v in args.items())
    return key_arg or "(无参数)"


def _summarize_skipped_calls(approved_calls: list[ApprovedToolCall]) -> str:
    """Build a compact summary of vetoed calls for Yang's feedback.

    Groups by tool name and lists distinct targets so Yang knows exactly
    what was skipped and can avoid repeating the same pattern.
    """
    by_tool: dict[str, list[str]] = {}
    for c in approved_calls:
        preview = _call_args_preview(c)
        by_tool.setdefault(c.name, []).append(preview)

    lines: list[str] = []
    for tool_name, targets in by_tool.items():
        if len(targets) == 1:
            lines.append(f"- {tool_name}({targets[0]})")
        else:
            lines.append(f"- {tool_name} × {len(targets)}（{', '.join(targets[:3])}{" ..." if len(targets) > 3 else ""}）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict[str, Any]:
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

    m = re.search(r'\{[^{}]*"action"[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    logger.warning("[行动] 无法解析 LLM 输出: {}", text[:100])
    return {"action": "execute"}


# ---------------------------------------------------------------------------
# Provider lazy-loading
# ---------------------------------------------------------------------------

_agent_name = "action"
_provider: Any = None


def _get_provider() -> Any:
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
    global _provider
    _provider = provider
