"""
四爻·阴 — Two-layer tool call approval.

Yin applies a **two-layer** approval process:

1. **Front layer (hardcoded)** — Deterministic, zero LLM.  Catches clear
   violations (path traversal, dangerous commands, protected paths).

2. **LLM layer (contextual)** — Calls the LLM with L4 (truths) and L5 (soul)
   context to make semantic approval decisions for calls that pass the front
   layer but need contextual judgement.

Approval rules (front layer):

| Risk level   | Examples                  | Policy                    |
|--------------|---------------------------|---------------------------|
| ``read_only``| read_file, list_directory | Auto-approve              |
|              | web_search, web_fetch     |                           |
| ``side_effect`` | write_file, edit_file, exec, delete_file | Front check → LLM layer  |
| ``special``  | task_complete             | Auto-approve (no IO)     |

For side-effect tools, the front layer performs:
- Path traversal detection (``..``, absolute paths outside workspace)
- Command allowlist verification
- Cognitive layer protection (.taiji/)

Calls that pass the front layer are submitted to the LLM for contextual
review against L4 truths (immutable rules) and L5 soul (identity).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.goal.types import ApprovedToolCall, RoundExecutionFact, YinOutput

# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

_READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_directory",
        "web_search",
        "web_fetch",
        "search_codebase",
        "query_capabilities",
        "my",
    }
)

_SIDE_EFFECT_TOOLS: frozenset[str] = frozenset(
    {
        "write_file",
        "edit_file",
        "exec",
        "delete_file",
    }
)

_SPECIAL_TOOLS: frozenset[str] = frozenset(
    {
        "task_complete",
    }
)

# Protected cognitive-layer paths — writes to these are rejected by the
# hardcoded front layer.  The cognitive layer (L1-L5) should only be modified
# by the agent via approved flows (e.g. Anqu decision, Dream consolidation).
#
# NOTE: Only the ``cognition/`` subdirectory under ``.taiji/`` is protected.
# Writes to ``.taiji/goals/`` (task workspace) are auto-approved — they are
# the normal working directory for sixiang tasks.
_PROTECTED_COGNITION_DIRS: frozenset[str] = frozenset(
    {
        ".taiji/cognition",
    }
)

# Task workspace prefix — writes under this path are auto-approved by the
# front layer without reaching the LLM contextual layer.
_TASK_WORKSPACE_PREFIX = ".taiji/goals"

_PROTECTED_CONFIG_FILES: frozenset[str] = frozenset(
    {
        "config.json",
    }
)

# Path traversal patterns
_PATH_TRAVERSAL_RE = re.compile(r"\.\.[/\\]")
_ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[/\\]|/)")  # Windows + Unix
_DANGEROUS_CMDS: set[str] = {
    "rm",
    "rmdir",
    "del",
    "format",
    "shutdown",
    "reboot",
    "mkfs",
    "dd",
}

# Baseline dangerous commands (never removed, only augmented from L4 truths)
_DANGEROUS_CMDS_BASELINE: frozenset[str] = frozenset(_DANGEROUS_CMDS)

# Tools that produce tangible output — used for cross-round read-paralysis detection.
# Any round without at least one of these is considered "non-productive".
_PRODUCTIVE_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "exec", "delete_file"})

# Exec commands that are purely diagnostic (read-only) — they gather info
# but do not produce deliverable output.  Rounds with only these exec
# commands + read-only tools are considered "non-productive".
_DIAGNOSTIC_EXEC_COMMANDS: frozenset[str] = frozenset({
    "ls", "head", "tail", "cat", "wc", "grep", "find",
    "sort", "uniq", "cut", "echo", "printf",
    "which", "type",
    "stat", "du", "df", "file",
    "ps", "env", "printenv",
    "pwd", "date", "cal",
})


def _is_diagnostic_exec(command: str) -> bool:
    """Check if an exec command is purely diagnostic (read-only) vs productive.

    Examples:
      _is_diagnostic_exec("ls -la")          -> True
      _is_diagnostic_exec("head -3 /tmp/x")  -> True
      _is_diagnostic_exec("npm run build")   -> False
      _is_diagnostic_exec("python render.py") -> False
    """
    cmd = command.strip().split(maxsplit=1)[0] if command.strip() else ""
    return cmd in _DIAGNOSTIC_EXEC_COMMANDS


def _sync_dangerous_cmds_from_truths(truths_dir: str | Path) -> int:
    """Augment ``_DANGEROUS_CMDS`` with commands declared in L4 truth files.

    Reads every ``*.json`` in *truths_dir* and looks for:
    - A top-level ``dangerous_commands`` list (strings to add directly).
    - Individual rules with a ``dangerous_commands`` field.

    Returns the number of new commands added.
    """
    truths_path = Path(truths_dir)
    if not truths_path.is_dir():
        return 0

    added = 0
    import json as _json

    for tf in sorted(truths_path.glob("*.json")):
        try:
            data = _json.loads(tf.read_text(encoding="utf-8"))
            # Top-level dangerous_commands list
            for cmd in data.get("dangerous_commands", []):
                if isinstance(cmd, str) and cmd.lower() not in _DANGEROUS_CMDS:
                    _DANGEROUS_CMDS.add(cmd.lower())
                    added += 1
            # Per-rule dangerous_commands
            for rule in data.get("rules", []):
                for cmd in rule.get("dangerous_commands", []):
                    if isinstance(cmd, str) and cmd.lower() not in _DANGEROUS_CMDS:
                        _DANGEROUS_CMDS.add(cmd.lower())
                        added += 1
        except Exception:
            logger.warning("[阴·L4] 读取真理文件失败: {}", tf)

    if added:
        logger.info("[阴·L4] 从真理文件动态添加 {} 个危险命令", added)
    return added


# LLM layer default temperature
_LLM_TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# Public API — two-layer approval
# ---------------------------------------------------------------------------


async def approve(
    tool_calls: list[dict[str, Any]],
    workspace_root: str | Path | None = None,
    *,
    provider: Any = None,
) -> tuple[list[ApprovedToolCall], str, str, str]:
    """Single-layer safety approval: hardcoded front layer only.

    All tool calls pass through deterministic safety checks (path traversal,
    dangerous commands, protected paths, workspace boundary). Side-effect
    tools that pass are directly approved without LLM re-check.

    Args:
        tool_calls: Raw tool_calls from Yang (dicts with 'name' and 'arguments').
        workspace_root: Root directory for path-traversal checking.
        provider: Optional LLM provider (reserved for future use).

    Returns:
        Tuple of (approved_calls, decision, reason).
        - approved_calls: List of approved ``ApprovedToolCall`` instances.
        - decision: One of 'approved', 'rejected', 'modified'.
        - reason: Human-readable explanation.
    """
    if not tool_calls:
        return [], "skipped", "无工具调用", ""

    root = Path(workspace_root) if workspace_root else None

    # ── Step 1: Front layer (hardcoded) ────────────────────────
    auto_approved: list[ApprovedToolCall] = []
    front_rejected_reasons: list[str] = []

    for tc in tool_calls:
        name = tc.get("name", "") or tc.get("function", {}).get("name", "")
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}

        if not name:
            front_rejected_reasons.append("(无名称)")
            continue

        # Auto-approve read-only tools (with path check for file tools)
        if name in _READ_ONLY_TOOLS:
            if name in ("read_file", "list_directory") and root:
                path_str = args.get("path", "")
                if path_str:
                    check_ok, check_reason = _check_path_safety(
                        path_str, root, for_write=False,
                    )
                    if not check_ok:
                        front_rejected_reasons.append(f"{name}: {check_reason}")
                        logger.warning("[阴·前置] 拒绝越界读取: {} — {}", name, check_reason)
                        continue
            auto_approved.append(ApprovedToolCall(name=name, arguments=args))
            continue

        if name in _SPECIAL_TOOLS:
            auto_approved.append(ApprovedToolCall(name=name, arguments=args))
            continue

        # Skill-registered tools → check auto_approve policy
        skill_approved, skill_reason = _check_skill_tool(name)
        if skill_approved:
            auto_approved.append(ApprovedToolCall(name=name, arguments=args))
            continue
        if skill_reason == "auto_approve_false":
            # Front check → auto-approve if safe (no LLM layer needed)
            check_ok, check_reason = _check_side_effect(name, args, root)
            if check_ok:
                auto_approved.append(ApprovedToolCall(name=name, arguments=args))
            else:
                front_rejected_reasons.append(f"{name}: {check_reason}")
                logger.warning("[阴·前置] 拒绝技能工具调用: {} — {}", name, check_reason)
            continue

        # Side-effect tools → front check (always auto-approved if safe)
        if name in _SIDE_EFFECT_TOOLS:
            check_ok, check_reason = _check_side_effect(name, args, root)
            if check_ok:
                auto_approved.append(ApprovedToolCall(name=name, arguments=args))
            else:
                front_rejected_reasons.append(f"{name}: {check_reason}")
                logger.warning("[阴·前置] 拒绝工具调用: {} — {}", name, check_reason)
            continue

        # Unknown tools → reject immediately
        front_rejected_reasons.append(f"{name}: 未知工具（未在审批白名单中）")
        logger.warning("[阴·前置] 拒绝未知工具: {}", name)

    # ── Combine results ────────────────────────────────────────
    all_approved = auto_approved
    all_rejected = front_rejected_reasons

    # ── Step 2: Build suggestions for Yang ─────────────────
    suggestions = _build_suggestions(all_approved, all_rejected)

    if not all_approved:
        return [], "rejected", "全部拒绝: " + "; ".join(all_rejected), suggestions

    if all_rejected:
        return (
            all_approved,
            "modified",
            f"部分批准 ({len(all_approved)}个), 拒绝: {'; '.join(all_rejected)}",
            suggestions,
        )

    return all_approved, "approved", f"全部批准 ({len(all_approved)}个工具调用)", suggestions


# ---------------------------------------------------------------------------
# Cross-round read-paralysis detection — active rejection, not passive warning
# ---------------------------------------------------------------------------


def _detect_read_paralysis(
    recent_tool_calls: list[list[str]],
    round_num: int,
    *,
    threshold: int = 3,
    recent_exec_commands: list[list[str]] | None = None,
) -> str | None:
    """Detect read-paralysis: N consecutive rounds with no productive output.

    Unlike ``self_reflect()`` which produces a passive warning, this function
    is used by ``run_yin()`` to **actively reject** read-only calls when a
    paralysis pattern is detected.  This is the hard gate that breaks the
    read-loop at the approval level, not the advisory level.

    ``recent_exec_commands``, if provided, should align 1:1 with
    ``recent_tool_calls``: each element is the list of exec command strings
    from that round.  These are used to distinguish diagnostic exec
    (``ls``, ``head``, ``wc``, etc.) from productive exec (build, render).

    Returns a rejection reason string if paralysis is detected, or ``None``
    if the pattern is healthy.
    """
    if not recent_tool_calls:
        return None

    # Count consecutive rounds with zero productive output
    streak = 0
    for i, tools in enumerate(reversed(recent_tool_calls)):
        if not tools:
            break

        # Check productive tools (write_file, edit_file, delete_file)
        if any(t in ("write_file", "edit_file", "delete_file") for t in tools):
            break

        # exec is ambiguous — check commands if available
        if "exec" in tools and recent_exec_commands:
            idx = len(recent_tool_calls) - 1 - i
            if idx < len(recent_exec_commands):
                exec_cmds = recent_exec_commands[idx]
                # If ANY exec command is non-diagnostic, round is productive
                if exec_cmds and not all(_is_diagnostic_exec(c) for c in exec_cmds):
                    break

        # If we get here, round had only diagnostic exec + read-only tools
        streak += 1

    if streak >= threshold:
        return (
            f"检测到连续 {streak} 轮无任何文件产出"
            f"（无 write_file/edit_file/delete_file，exec 仅用于诊断命令），"
            f"本轮所有只读/诊断调用已被拒绝。"
            f"请直接用 write_file 产出交付物（如 outputs/ 下），"
            f"不要再读取文件。如果任务已完成，调用 task_complete."
        )

    return None


# ---------------------------------------------------------------------------
# Unified Yin entry point — approval + self-reflection + reweave detection
# ---------------------------------------------------------------------------


# From-scratch patterns that indicate the Worker is building a project
# from zero instead of using an existing skill template
_FROM_SCRATCH_EXEC_PATTERNS: tuple[str, ...] = (
    "npm init", "yarn init", "pnpm init",
    "npx create-", "npm create ",
    "pip install", "pip3 install",
    "cargo init", "cargo new",
    "poetry init", "poetry new",
    "django-admin startproject",
    "rails new",
    "npx @remotion/create-video",
)


def _detect_skill_bypass(
    tool_calls: list[dict[str, Any]],
    facts: list[RoundExecutionFact],
    grid_skill_names: list[str] | None,
    recent_tool_calls: list[list[str]] | None = None,
) -> str | None:
    """Detect when Yang is building from scratch while grid skills exist.

    Checks:
    1. Current round has exec calls matching "from-scratch" patterns
       (npm init, pip install, npx create-*, etc.)
    2. Grid skills exist for the current trigram
    3. Recent rounds show no evidence that skill SKILL.md files were read

    Returns a suggestion hint (not a rejection) if bypass is detected.
    """
    if not grid_skill_names:
        return None

    # ── Check current round for from-scratch exec patterns ──
    from_scratch_cmds: list[str] = []
    for tc in tool_calls:
        name = ""
        args: dict[str, Any] = {}
        if isinstance(tc, dict):
            func = tc.get("function", {})
            if isinstance(func, dict):
                name = func.get("name", "")
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(raw_args, dict):
                    args = raw_args

        if name == "exec":
            command = str(args.get("command", "")).lower()
            for pattern in _FROM_SCRATCH_EXEC_PATTERNS:
                if pattern in command:
                    from_scratch_cmds.append(command[:120])
                    break

    if not from_scratch_cmds:
        return None

    # ── Check if any skill SKILL.md was read in recent facts ──
    skill_read_detected = False
    skill_keywords = {sn.lower().replace("-", "") for sn in grid_skill_names}
    for fact in facts[-6:]:  # last 6 rounds
        summary = fact.yang_intent_summary.lower()
        # Check if any skill name appears in the intent summary
        # (crude but effective proxy for "did Yang read this skill")
        for kw in skill_keywords:
            if kw in summary or kw.replace("_", "") in summary:
                skill_read_detected = True
                break
        if skill_read_detected:
            break

    # Also check recent_tool_calls for read_file patterns
    if not skill_read_detected and recent_tool_calls:
        for tools in recent_tool_calls[-4:]:
            if "read_file" in tools:
                # read_file was called but we can't verify it was a skill file
                # from the tool name list alone — be conservative
                pass

    if skill_read_detected:
        return None  # Worker has looked at skills, let them decide

    # ── Build suggestion ──
    skill_list = ", ".join(grid_skill_names)
    cmd_snippet = from_scratch_cmds[0][:80]
    return (
        f"⚠️ 检测到从零搭建操作 (`{cmd_snippet}...`)，"
        f"但当前卦象已注入技能: {skill_list}。"
        f"建议先 `read_file` 这些技能的 SKILL.md，"
        f"优先使用已有模板/脚本而非从零搭建。"
        f"如果技能不适用，请在 execution-facts 中写明 `skill_bypass_reason`。"
    )


# ---------------------------------------------------------------------------
# Proactive arbitration — blueprint deviation & self-abandonment detection
# ---------------------------------------------------------------------------

# Keywords that indicate self-abandonment in task outputs
ABANDONMENT_KEYWORDS: tuple[str, ...] = (
    "放弃", "abandon", "give up", "gave up",
    "不可行", "not feasible", "not possible",
    "跳过", "skip", "暂时放弃", "暂时不",
    "无法完成", "cannot complete", "can't complete",
    "决定不", "decided not to", "决定放弃",
    "正式放弃", "officially abandon",
    "不再继续", "no longer continue",
    "终止此方向", "terminate this direction",
)

# Blueprint deliverable patterns — extract numeric targets from text
BLUEPRINT_TARGET_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"(\d+)\s*[个只条项件份篇张段][^\w]*\s*(?:视频|video|文件|file|文档|doc|图|image|报告|report|脚本|script|产出|deliverable|output)", re.IGNORECASE),
    re.compile(r"(?:视频|video|文件|file|文档|doc|图|image|报告|report|脚本|script|产出|deliverable|output)[^\d]*(\d+)\s*[个只条项件份篇张段]", re.IGNORECASE),
    re.compile(r"(?:create|generate|produce|make|build|创建|生成|制作|产出|构建)[^\d]*(\d+)\s+", re.IGNORECASE),
    re.compile(r"(\d+)\s+(?:videos|files|docs|images|reports|scripts|deliverables|outputs)", re.IGNORECASE),
)

# Tool names that indicate user-interaction (confirmation channel)
_USER_CONFIRMATION_TOOLS: frozenset[str] = frozenset({
    "ask_user", "ask_user_question",
})


def _build_yin_awareness(
    goal_context: Any,
    facts: list[RoundExecutionFact],
    task_dir: str | Path,
) -> dict[str, Any]:
    """Build Yin's self-awareness snapshot from goal-level data.

    Loads the blueprint summary, goal memory, recent task statuses,
    and current task execution pattern.  This is the awareness file
    that Yin uses to perform proactive arbitration.

    Returns a structured dict with keys:
    - blueprint: str (blueprint content for requirement parsing)
    - memory: str (goal memory summary)
    - trajectory: str (trajectory snapshot)
    - recent_task_statuses: list of summary strings
    - current_facts_summary: str (condensed current task execution pattern)
    - task_dir: str (for output scanning)
    """
    awareness: dict[str, Any] = {
        "blueprint": "",
        "memory": "",
        "trajectory": "",
        "recent_task_statuses": [],
        "current_facts_summary": "",
        "task_dir": str(task_dir),
    }

    if goal_context is None:
        return awareness

    # ── Load blueprint ───────────────────────────────────────
    blueprint = getattr(goal_context, "blueprint_summary", "") or ""
    awareness["blueprint"] = blueprint[:3000]

    # ── Load memory ──────────────────────────────────────────
    awareness["memory"] = (getattr(goal_context, "memory_summary", "") or "")[:2000]

    # ── Load trajectory ──────────────────────────────────────
    awareness["trajectory"] = (getattr(goal_context, "trajectory_snapshot", "") or "")[:2000]

    # ── Load recent task statuses ────────────────────────────
    recent = getattr(goal_context, "recent_task_statuses", None) or []
    status_lines: list[str] = []
    for t in recent[:10]:
        task_id = getattr(t, "task_id", "?")
        status = getattr(t, "status", "?")
        snippet = getattr(t, "summary_snippet", "") or ""
        status_lines.append(f"[{task_id}] {status}: {snippet[:200]}")
    awareness["recent_task_statuses"] = status_lines

    # ── Build current facts summary ──────────────────────────
    if facts:
        total = len(facts)
        successes = sum(1 for f in facts if f.execution_status == "success")
        failures = sum(1 for f in facts if f.execution_status in ("failure", "partial_failure", "exec_failed"))

        fact_lines: list[str] = [f"当前任务: {total} 轮 | 成功 {successes} | 失败 {failures}"]
        for f in facts[-5:]:
            fact_lines.append(
                f"  第{f.round}轮: {f.execution_status} | "
                f"{f.yang_intent_summary[:100]} | "
                f"产出: {f.execution_result_summary[:80]}"
            )
        awareness["current_facts_summary"] = "\n".join(fact_lines)

    return awareness


def _detect_blueprint_deviation(
    awareness: dict[str, Any],
) -> tuple[bool, str, str]:
    """Detect deviation between blueprint requirements and actual task outputs.

    Parses the blueprint for quantitative completion targets (e.g. "20 videos",
    "5 reports") and compares against what the task chain has actually produced.

    Also checks for self-abandonment contamination in completed task statuses.

    Returns:
        (deviation_detected, deviation_detail, fuse_prompt)
        - deviation_detected: True if a meaningful deviation was found
        - deviation_detail: Human-readable description of the deviation
        - fuse_prompt: Mandatory verification instruction for the Worker
    """
    blueprint = awareness.get("blueprint", "") or ""
    recent_statuses: list[str] = awareness.get("recent_task_statuses", []) or []
    memory = awareness.get("memory", "") or ""
    trajectory = awareness.get("trajectory", "") or ""
    current_facts = awareness.get("current_facts_summary", "") or ""

    if not blueprint:
        return False, "", ""

    # ── Extract numeric targets from blueprint ───────────────
    targets: list[tuple[str, int]] = []
    for pattern in BLUEPRINT_TARGET_PATTERNS:
        for m in pattern.finditer(blueprint):
            try:
                count = int(m.group(1))
                # Yin only cares about multi-item targets (count >= 2).
                # Anqu's _verify_blueprint_completion uses count >= 1 —
                # the difference is intentional: Yin fires on clear
                # shortfalls, Anqu reports all detectable targets.
                if count >= 2:
                    context = blueprint[max(0, m.start() - 40):m.end() + 40]
                    targets.append((context.strip(), count))
            except (ValueError, IndexError):
                continue

    # Deduplicate by context similarity
    unique_targets: dict[str, int] = {}
    for ctx, count in targets:
        key = ctx[:60]
        if key not in unique_targets or count > unique_targets[key]:
            unique_targets[key] = count

    if not unique_targets:
        return False, "", ""

    # ── Scan completed task statuses for abandonment ─────────
    abandonment_signals: list[str] = []
    for status_line in recent_statuses:
        status_lower = status_line.lower()
        for kw in ABANDONMENT_KEYWORDS:
            if kw.lower() in status_lower:
                abandonment_signals.append(status_line[:150])
                break

    # ── Check if user confirmation exists in memory/trajectory ─
    user_confirmed = (
        "ask_user" in memory.lower()
        or "ask_user" in trajectory.lower()
        or "用户确认" in memory
        or "用户确认" in trajectory
        or "user confirmed" in memory.lower()
        or "user confirmed" in trajectory.lower()
    )

    # ── Scan current facts for abandonment ──────────────────
    current_abandonment = False
    current_facts_lower = current_facts.lower()
    for kw in ABANDONMENT_KEYWORDS:
        if kw.lower() in current_facts_lower:
            current_abandonment = True
            break

    # ── Count actual deliverables in task output directory ───
    # This is the "comparison" half of the blueprint-vs-reality check.
    task_dir_path = Path(awareness.get("task_dir", ""))
    outputs_dir = task_dir_path / "outputs" if task_dir_path else None
    actual_deliverable_count = 0
    if outputs_dir and outputs_dir.is_dir():
        actual_deliverable_count = sum(
            1 for f in outputs_dir.iterdir() if f.is_file()
        )

    # ── Determine if deviation exists ────────────────────────
    deviation_detected = False
    deviation_parts: list[str] = []

    if abandonment_signals and not user_confirmed:
        deviation_detected = True
        deviation_parts.append('发现「自我放弃」污染：已完成任务中包含放弃语言，但未找到用户确认记录')
        for sig in abandonment_signals[:3]:
            deviation_parts.append(f"  - {sig}")

    if current_abandonment and not user_confirmed:
        deviation_detected = True
        deviation_parts.append("当前任务执行记录中也出现了放弃信号")

    # ── Quantitative shortfall: targets exist, count is low ──
    if unique_targets and actual_deliverable_count >= 0:
        # Find the minimum target from blueprint
        min_target = min(unique_targets.values())
        if actual_deliverable_count < min_target:
            deviation_detected = True
            deviation_parts.append(
                f"产出数量不足：蓝图预期至少 {min_target} 个交付物，"
                f"但当前 outputs/ 目录中仅有 {actual_deliverable_count} 个文件"
            )

    if not deviation_detected:
        return False, "", ""

    # ── Build fuse prompt ────────────────────────────────────
    target_summary = "; ".join(
        f"'{ctx[:40]}...' 预期 {count} 个" for ctx, count in list(unique_targets.items())[:3]
    ) if unique_targets else "（未能从蓝图中解析出量化目标）"

    detail = "\n".join(deviation_parts)
    fuse = (
        f"## ⚡ 阴·主动仲裁（熔断）\n\n"
        f"阴节点主动检查发现以下偏差，你**必须**在本轮优先处理：\n\n"
        f"### 蓝图要求\n"
        f"{blueprint[:500]}\n\n"
        f"### 蓝图解析出的量化目标\n"
        f"{target_summary}\n\n"
        f"### 检测到的偏差\n"
        f"{detail}\n\n"
        f"### 强制核实指令\n"
        f"1. **谁决定放弃的？** 请列出具体在哪个任务的哪一轮出现了放弃决策\n"
        f"2. **有用户确认吗？** 搜索 ask_user 调用记录或用户消息，确认放弃是否被用户认可\n"
        f"3. **当前实际状态是什么？** 盘点 deliverables/ 目录中已有的产出物\n"
        f"4. **不要调用 task_complete** 在以上问题得到明确答案之前\n"
        f"5. 如果用户确实确认了放弃/缩小范围 → 在 execution-facts 中记录确认证据后正常推进\n"
        f"6. 如果**没有任何用户确认** → 这就是擅自放弃，请用 write_file 产出交付物继续推进"
    )

    return True, detail, fuse


def _detect_self_abandonment(
    facts: list[RoundExecutionFact],
) -> tuple[bool, str, str]:
    """Detect self-abandonment patterns in execution facts.

    Scans execution facts (both yang_intent_summary and execution_result_summary)
    for language indicating the Worker has autonomously decided to give up on a
    path, without user confirmation.

    Also checks if user-interaction tools (ask_user, ask_user_question) were
    called — their presence suggests the user was consulted.

    Returns:
        (abandonment_found, detail, fuse_prompt)
        - abandonment_found: True if abandonment language detected
        - detail: Description of what was found
        - fuse_prompt: Mandatory verification instruction (empty if no abandonment)
    """
    if not facts:
        return False, "", ""

    # ── Scan facts for abandonment keywords ──────────────────
    abandonment_rounds: list[int] = []
    abandonment_snippets: list[str] = []

    for f in facts:
        text = (
            (f.yang_intent_summary or "") + " " +
            (f.execution_result_summary or "") + " " +
            (f.yin_reason or "")
        ).lower()

        for kw in ABANDONMENT_KEYWORDS:
            if kw.lower() in text:
                if f.round not in abandonment_rounds:
                    abandonment_rounds.append(f.round)
                    snippet = (
                        f"第{f.round}轮: 意图='{f.yang_intent_summary[:80]}' | "
                        f"产出='{f.execution_result_summary[:60]}'"
                    )
                    abandonment_snippets.append(snippet)
                break

    if not abandonment_rounds:
        return False, "", ""

    # ── Check for user confirmation tools ────────────────────
    # (We can't check actual tool call history from RoundExecutionFact alone,
    # so we check if intent summaries mention user interaction)
    user_consulted = any(
        "ask_user" in (f.yang_intent_summary or "").lower()
        or "询问用户" in (f.yang_intent_summary or "")
        or "咨询用户" in (f.yang_intent_summary or "")
        or "向用户确认" in (f.yang_intent_summary or "")
        for f in facts
    )

    if user_consulted:
        # User was consulted — abandonment may be legitimate
        return False, "", ""

    # ── Build detail and fuse prompt ─────────────────────────
    detail = (
        f"在 {len(abandonment_rounds)} 轮中检测到自我放弃信号"
        f"（第 {', '.join(str(r) for r in abandonment_rounds[:5])} 轮），"
        f"且未发现 ask_user 调用记录"
    )

    fuse = (
        f"## ⚡ 阴·主动仲裁（自我放弃检测）\n\n"
        f"阴节点检测到你的执行记录中包含放弃语言，但未找到用户确认记录。\n\n"
        f"### 检测到的放弃信号\n"
        + "\n".join(f"- {s}" for s in abandonment_snippets[:5]) +
        f"\n\n"
        f"### 强制核实指令\n"
        f"1. **是否已向用户确认？** 如果是，请在此轮提供确认证据（ask_user 调用记录或用户回复摘录）\n"
        f"2. **如果未确认** → 这是擅自放弃。你必须：\n"
        f"   - 调用 ask_user 询问用户是否同意修改计划\n"
        f"   - 或在得到用户许可前，继续按蓝图完整推进\n"
        f"3. **不要调用 task_complete** 在得到明确答案之前"
    )

    return True, detail, fuse


async def run_yin(
    tool_calls: list[dict[str, Any]],
    facts: list[RoundExecutionFact],
    round_num: int,
    max_rounds: int,
    task_description: str,
    workspace_root: str | Path | None = None,
    *,
    recent_tool_calls: list[list[str]] | None = None,
    grid_skill_names: list[str] | None = None,
    signal: asyncio.Task | None = None,
    provider: Any = None,
    recent_exec_commands: list[list[str]] | None = None,
    goal_context: Any = None,
    task_dir: str = "",
) -> YinOutput:
    """Unified Yin entry: approval + proactive arbitration + self-reflection.

    1. Runs ``approve()`` to check tool call safety.
    2. **Skill-bypass hint**: if Yang is building from scratch (npm init etc.)
       while grid skills exist but haven't been read, appends a nudge to
       suggestions (advisory, not rejection).
    3. **Read-paralysis gate**: if all approved calls are read-only AND the
       cross-round pattern shows 3+ consecutive non-productive rounds,
       actively rejects them (hard gate, not advisory).
    4. **Proactive arbitration** (NEW): Yin loads its awareness snapshot
       (blueprint, goal memory, task outputs) and actively checks for:
       - Blueprint deviation: blueprint requirements vs actual completion
       - Self-abandonment contamination: autonomous give-up without user confirmation
       When deviation is detected, a fuse instruction is generated — it does
       NOT reject current tool calls, but is injected into the next round.
    5. Runs ``self_reflect()`` to detect loop pathologies (advisory).
    6. Determines whether the cognitive posture needs re-weaving.

    Returns a single ``YinOutput`` with all results.
    """
    # ── Approval ───────────────────────────────────────────────
    approved_calls, decision, reason, suggestions = await approve(
        tool_calls, workspace_root=workspace_root, provider=provider,
    )

    # ── Skill-bypass hint: advisory nudge, not rejection ─────
    bypass_hint = _detect_skill_bypass(
        tool_calls, facts, grid_skill_names, recent_tool_calls,
    )
    if bypass_hint:
        if suggestions:
            suggestions = suggestions.rstrip() + "\n\n" + bypass_hint
        else:
            suggestions = "## 💡 阴节点技能提示\n" + bypass_hint

    # ── Cross-round read-paralysis: active rejection ────────────
    # Unlike self_reflect() which is advisory, this HARD-REJECTS
    # read-only calls when Yang is stuck in a diagnostic loop.
    # NOTE: exec with diagnostic commands (ls, head, wc, etc.) is
    # classified as read-only, NOT productive — this closes the
    # "exec bypass" hole that let read-paralysis evade detection.
    if approved_calls and recent_tool_calls and round_num >= 3:
        def _is_truly_productive(c: ApprovedToolCall) -> bool:
            """Is this call truly productive, or is exec just diagnostic?"""
            if c.name not in _READ_ONLY_TOOLS and c.name != "exec":
                return True
            if c.name == "exec":
                cmd = c.arguments.get("command", "") if c.arguments else ""
                return not _is_diagnostic_exec(cmd)
            return False

        productive_calls = [c for c in approved_calls if _is_truly_productive(c)]
        readonly_calls = [c for c in approved_calls if not _is_truly_productive(c)]
        if readonly_calls and not productive_calls:
            paralysis_reason = _detect_read_paralysis(
                recent_tool_calls, round_num,
                recent_exec_commands=recent_exec_commands,
            )
            if paralysis_reason:
                return YinOutput(
                    approved_calls=[],
                    decision="rejected",
                    reason=paralysis_reason,
                    suggestions=(
                        "## 💡 阴节点审批建议\n"
                        "本轮所有只读/诊断调用已被拒绝。\n"
                        "请用 write_file 直接产出交付物（如 outputs/ 下的成果文件），"
                        "不要再读取文件。如果任务已完成，调用 task_complete。"
                    ),
                    warning="",
                    needs_reweave=True,
                )

    # ── Proactive arbitration — blueprint deviation & self-abandonment ──
    fuse_instruction = ""
    blueprint_deviation = ""
    proactive_needs_reweave = False

    if goal_context is not None:
        # Build Yin's self-awareness snapshot
        awareness = _build_yin_awareness(goal_context, facts, task_dir)

        # Check 1: Blueprint deviation
        dev_detected, dev_detail, dev_fuse = _detect_blueprint_deviation(awareness)
        if dev_detected:
            fuse_instruction = dev_fuse
            blueprint_deviation = dev_detail
            proactive_needs_reweave = True
            logger.warning(
                "[阴·仲裁] 检测到蓝图偏差 (round={}): {}",
                round_num, dev_detail[:200],
            )

        # Check 2: Self-abandonment in current task facts
        if not dev_detected:
            # Only check self-abandonment if blueprint deviation wasn't already found
            # (blueprint deviation check already covers abandonment in recent_task_statuses)
            abd_detected, abd_detail, abd_fuse = _detect_self_abandonment(facts)
            if abd_detected:
                fuse_instruction = abd_fuse
                blueprint_deviation = abd_detail
                proactive_needs_reweave = True
                logger.warning(
                    "[阴·仲裁] 检测到自我放弃信号 (round={}): {}",
                    round_num, abd_detail[:200],
                )

    # ── Self-reflection + phase reweave detection ────────────────
    warning, needs_reweave = await self_reflect(
        facts=facts,
        round_num=round_num,
        max_rounds=max_rounds,
        task_description=task_description,
        recent_tool_calls=recent_tool_calls,
        signal=signal,
    )

    return YinOutput(
        approved_calls=approved_calls,
        decision=decision,
        reason=reason,
        suggestions=suggestions,
        warning=warning,
        needs_reweave=needs_reweave or proactive_needs_reweave,
        fuse_instruction=fuse_instruction,
        blueprint_deviation=blueprint_deviation,
    )





# ---------------------------------------------------------------------------
# LLM approval layer
# ---------------------------------------------------------------------------


async def _llm_approve(
    tool_calls: list[dict[str, Any]],
    provider: Any,
    workspace_root: Path | None,
) -> tuple[list[ApprovedToolCall], list[str]]:
    """Run LLM-based contextual approval on tool calls that passed the front layer.

    Loads L4 truths and L5 soul content to inform the LLM's decisions.

    Returns:
        (approved_calls, rejected_reasons)
    """
    # ── Load L4 truths ─────────────────────────────────────────
    truths_text = ""
    try:
        from vingobot.core.workspace import get_workspace_paths
        truths_dir = get_workspace_paths().truths
        if truths_dir.is_dir():
            # Synchronise dangerous commands from L4 truth files
            _sync_dangerous_cmds_from_truths(truths_dir)
            parts: list[str] = []
            for tf in sorted(truths_dir.glob("*.json")):
                try:
                    data = json.loads(tf.read_text(encoding="utf-8"))
                    title = data.get("title", tf.stem)
                    rules = data.get("rules", [])
                    rule_lines = [f"  - {r['statement']}" for r in rules if "statement" in r]
                    if rule_lines:
                        parts.append(f"### {title}")
                        parts.extend(rule_lines)
                except Exception:
                    logger.warning("[阴·LLM] 读取真理文件失败: {}", tf)
            truths_text = "\n".join(parts)
    except Exception:
        logger.warning("[阴·LLM] 无法加载 L4 真理层")

    # ── Load L5 soul ──────────────────────────────────────────
    soul_text = ""
    try:
        from vingobot.config.paths import get_workspace_path
        soul_file = get_workspace_path() / "SOUL.md"
        if not soul_file.is_file():
            # Fallback: bundled template
            from importlib.resources import files as pkg_files
            fallback = pkg_files("vingobot") / "templates" / "SOUL.md"
            if fallback.is_file():
                soul_file = fallback
        if soul_file.is_file():
            try:
                soul_text = soul_file.read_text(encoding="utf-8")[:2000]
            except Exception:
                pass
    except Exception:
        pass

    # ── Build system prompt ────────────────────────────────────
    system_prompt = (
        "你是四爻·阴，负责对阳提交的工具调用进行上下文审批。\n"
        "你的判断必须基于 L4 真理（不可变规则）和 L5 灵魂（身份认知）：\n\n"
        "## L4 真理层（不可违抗）\n"
        f"{truths_text or '（无已加载的真理定义）'}\n\n"
        "## L5 灵魂层（身份认知）\n"
        f"{soul_text or '（无已加载的灵魂定义）'}\n\n"
        "## 审批规则\n"
        "- 批准：该工具调用符合 L4/L5 约束，可安全执行\n"
        "- 拒绝：该工具调用违反 L4/L5 约束，需说明原因\n"
        "- 不确定时请选择拒绝（保守原则）\n"
        "- 拒绝理由需引用具体的 L4 真理编号或 L5 原则\n"
        "- **写权限例外**: `.taiji/goals/` 和 goal_dir 是任务工作区，其中的写入/编辑操作是正常且必需的，不应拒绝。"
        "仅 `.taiji/cognition/` 受保护，禁止直接写入。\n\n"
        "## 输出格式\n"
        "为每个工具调用单独输出 JSON:\n"
        "{\n"
        '  "decisions": [\n'
        '    {"tool_index": 0, "approved": true, "reason": "..."},\n'
        '    {"tool_index": 1, "approved": false, "reason": "拒绝理由"}\n'
        "  ]\n"
        "}\n"
    )

    # ── Build tool call descriptions ───────────────────────────
    call_descriptions: list[str] = []
    for i, tc in enumerate(tool_calls):
        name = tc.get("name", "") or tc.get("function", {}).get("name", "")
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        args_json = json.dumps(args, ensure_ascii=False)[:500]
        call_descriptions.append(f"## 工具调用 #{i}\n工具: {name}\n参数: {args_json}")

    user_prompt = "请审批以下工具调用：\n\n" + "\n\n".join(call_descriptions)

    # ── Call LLM ──────────────────────────────────────────────
    try:
        response = await provider.chat_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=_LLM_TEMPERATURE,
        )
    except Exception:
        logger.exception("[阴·LLM] LLM 调用失败")
        return [], [f"LLM 调用异常，拒绝 {len(tool_calls)} 个工具调用"]

    content = (response.content or "").strip()
    decisions = _parse_approval_decisions(content, len(tool_calls))

    approved: list[ApprovedToolCall] = []
    rejected: list[str] = []

    for i, tc in enumerate(tool_calls):
        name = tc.get("name", "") or tc.get("function", {}).get("name", "")
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}

        dec = decisions[i] if i < len(decisions) else None
        if dec and dec.get("approved", False):
            approved.append(ApprovedToolCall(name=name, arguments=args))
            reason = dec.get("reason", "")
            logger.info("[阴·LLM] 批准 {}: {}", name, reason[:100])
        else:
            reason = dec.get("reason", "LLM 未给出理由") if dec else "解析失败，保守拒绝"
            rejected.append(f"{name}: {reason}")
            logger.info("[阴·LLM] 拒绝 {}: {}", name, reason[:100])

    return approved, rejected


def _parse_approval_decisions(
    content: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Parse the LLM's JSON approval decisions, tolerating markdown fences."""
    # Try raw parse
    try:
        data = json.loads(content)
        return data.get("decisions", [])
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code block
    for fence in ("```json", "```"):
        if fence in content:
            start = content.index(fence) + len(fence)
            end = content.rfind("```")
            if end > start:
                try:
                    data = json.loads(content[start:end].strip())
                    return data.get("decisions", [])
                except (json.JSONDecodeError, TypeError):
                    pass

    logger.warning("[阴·LLM] 无法解析审批决策，默认全部拒绝")
    return [{"approved": False, "reason": "解析失败"} for _ in range(expected_count)]


# ---------------------------------------------------------------------------
# Front layer — hardcoded side-effect checks
# ---------------------------------------------------------------------------


def _build_suggestions(
    approved: list[ApprovedToolCall],
    rejected_reasons: list[str],
) -> str:
    """Build actionable suggestions for Yang based on approval results.

    When calls were rejected, generates specific alternative actions.
    Returns empty string if all calls were approved.
    """
    if not rejected_reasons:
        return ""

    lines: list[str] = ["## 阴节点审批建议", ""]

    # Group rejected by reason pattern
    read_rejected = [r for r in rejected_reasons if "read_file" in r or "list_directory" in r]
    write_rejected = [r for r in rejected_reasons if "write_file" in r or "edit_file" in r]
    exec_rejected = [r for r in rejected_reasons if "exec" in r]

    if read_rejected:
        lines.append("以下只读调用被拒绝（路径越界或不安全）：")
        for r in read_rejected:
            lines.append(f"- {r}")
        lines.append("建议：使用任务目录或目标目录内的相对路径重试。")
        lines.append("")

    if write_rejected:
        lines.append("以下写入调用被拒绝：")
        for r in write_rejected:
            lines.append(f"- {r}")
        lines.append("")

    if exec_rejected:
        lines.append("以下命令被拒绝（检测到危险命令或越界路径）：")
        for r in exec_rejected:
            lines.append(f"- {r}")
        lines.append("")

    # General guidance if all rejected
    approved_count = len(approved)
    if approved_count > 0:
        lines.append(f"已批准 {approved_count} 个调用，直接执行即可。")
    else:
        lines.append("本轮所有调用均被拒绝。请基于以下建议重新规划：")
        lines.append("1. 检查写入/读取的路径是否在任务目录范围内")
        lines.append("2. 避免使用危险 shell 命令（rm, del, shutdown 等）")
        lines.append("3. 先列出目标目录确认可用文件，再用 write_file 产出")

    return "\n".join(lines)


def _check_side_effect(
    name: str,
    args: dict[str, Any],
    workspace_root: Path | None,
) -> tuple[bool, str]:
    """Check a side-effect tool call for safety."""
    if name in ("write_file", "edit_file"):
        return _check_path_safety(args.get("path", ""), workspace_root, for_write=True)
    if name == "exec":
        return _check_exec_safety(
            args.get("command", ""),
            args.get("cwd", ""),
            workspace_root,
        )
    if name == "delete_file":
        return _check_path_safety(args.get("path", ""), workspace_root, for_write=True)
    return True, "ok"  # unknown tools pass to LLM contextual layer


def _check_path_safety(
    path_str: str,
    workspace_root: Path | None,
    for_write: bool = False,
) -> tuple[bool, str]:
    """Check a file path for traversal and workspace boundary violations."""
    if not path_str:
        return False, "路径为空"

    # Path traversal detection
    if _PATH_TRAVERSAL_RE.search(path_str):
        return False, f"检测到路径穿越: {path_str}"

    path = Path(path_str)

    # Absolute path on Windows
    if _ABSOLUTE_PATH_RE.match(path_str):
        if workspace_root:
            try:
                path.resolve().relative_to(workspace_root.resolve())
            except ValueError:
                return False, f"绝对路径超出工作区: {path_str}"
        else:
            # Without workspace root, reject absolute write paths
            if for_write:
                return False, f"写操作不允许绝对路径: {path_str}"
    else:
        # Relative path with workspace root check
        if workspace_root and for_write:
            try:
                resolved = (workspace_root / path).resolve()
                resolved.relative_to(workspace_root.resolve())
            except ValueError:
                return False, f"相对路径解析后超出工作区: {path_str}"

    # Cognitive-layer protection (L1-L5 / config.json)
    if for_write and workspace_root:
        resolved = (workspace_root / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            rel = resolved.relative_to(workspace_root.resolve())
            rel_str = "/" + str(rel).replace("\\", "/") + "/"

            # Auto-approve writes to task workspace (.taiji/goals/...)
            # Works regardless of whether .taiji/ is the first path component.
            # Returns "ok_task_workspace" so the front layer can skip LLM approval.
            if "/.taiji/goals/" in rel_str:
                return True, "ok_task_workspace"

            # Reject writes to cognition layer (.taiji/cognition/...)
            if "/.taiji/cognition/" in rel_str:
                return False, f"认知层路径受保护，禁止直接写入: {path_str}"
            # Check if the path targets config.json
            if rel.name in _PROTECTED_CONFIG_FILES:
                return False, f"配置文件受保护，禁止直接写入: {path_str}"
        except ValueError:
            pass  # Outside workspace — already caught above

    return True, "ok"


def _check_exec_safety(command: str, cwd: str = "",
                       workspace_root: Path | None = None) -> tuple[bool, str]:
    """Check a shell command for dangerous patterns and workspace boundary.

    When *workspace_root* is provided, validates that *cwd* resolves inside
    the workspace and that absolute file paths in the command stay within the
    workspace boundary.  Media directory paths are allowed as a read-only
    exception.
    """
    if not command:
        return False, "命令为空"

    cmd_lower = command.lower().strip()

    # Dangerous command detection
    for dangerous in _DANGEROUS_CMDS:
        if cmd_lower.startswith(dangerous) or f" {dangerous}" in cmd_lower:
            return False, f"检测到危险命令: {dangerous}"

    # Pipe to dangerous commands
    for dangerous in _DANGEROUS_CMDS:
        if f"| {dangerous}" in cmd_lower or f"|{dangerous}" in cmd_lower:
            return False, f"检测到管道到危险命令: {dangerous}"

    # Path traversal in cwd
    if cwd and _PATH_TRAVERSAL_RE.search(cwd):
        return False, f"工作目录路径穿越: {cwd}"

    # ── Workspace boundary check for cwd ───────────────────────
    if cwd and workspace_root:
        try:
            resolved_cwd = Path(cwd).expanduser().resolve()
            resolved_cwd.relative_to(workspace_root.resolve())
        except (ValueError, OSError):
            return False, f"工作目录超出工作区: {cwd}"

    # ── Workspace boundary check for absolute paths in command ─
    if workspace_root:
        media_path = _resolve_media_dir()
        abs_paths = _extract_abs_paths_from_cmd(command)
        for raw_path in abs_paths:
            try:
                expanded = os.path.expandvars(raw_path.strip())
                p = Path(expanded).expanduser().resolve()
            except Exception:
                continue
            # Allow paths inside the workspace
            try:
                p.relative_to(workspace_root.resolve())
                continue
            except ValueError:
                pass
            # Allow media_dir as a read-only exception
            if media_path and (media_path in p.parents or p == media_path):
                continue
            # Allow standard device paths (redirections like > /dev/null)
            _SAFE_DEVICE_PATHS = frozenset({
                "/dev/null", "/dev/zero", "/dev/random", "/dev/urandom",
                "/dev/tty", "/dev/stdin", "/dev/stdout", "/dev/stderr",
            })
            if str(p) in _SAFE_DEVICE_PATHS:
                continue
            # Skip non-existent paths (could be flags or partial matches)
            if not p.exists():
                continue
            return False, f"命令引用了工作区外的路径: {raw_path}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Path extraction helpers for exec safety
# ---------------------------------------------------------------------------


def _extract_abs_paths_from_cmd(command: str) -> list[str]:
    """Extract absolute file paths from a shell command string.

    Returns a list of path strings found in the command (Windows drive-root
    paths, POSIX absolute paths, and home-directory shortcuts).
    """
    paths: list[str] = []
    # Windows: drive-root paths like C:\path\to\file
    paths.extend(re.findall(r"[A-Za-z]:\\[^\s\"'|><;]*", command))
    # POSIX: absolute paths starting with /
    paths.extend(
        p.strip()
        for p in re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command)
    )
    # Home-directory shortcuts: ~/path or ~user/path
    paths.extend(
        p.strip()
        for p in re.findall(r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command)
    )
    return paths


def _resolve_media_dir() -> Path | None:
    """Resolve the media directory path, or None if unavailable."""
    try:
        from vingobot.config.paths import get_media_dir
        return Path(get_media_dir()).resolve()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Skill tool approval policy
# ---------------------------------------------------------------------------


def _check_skill_tool(name: str) -> tuple[bool, str]:
    """Check if a tool name belongs to a skill-registered tool.

    Returns:
        (True, "auto_approve") if auto-approve is set.
        (False, "auto_approve_false") if found but not auto-approved.
        (False, "not_found") if not a skill tool at all.
    """
    try:
        from vingobot.goal.skill_parser import get_skill_tool

        skill_tool = get_skill_tool(name)
        if skill_tool is None:
            return False, "not_found"

        if skill_tool.tool_auto_approve:
            return True, "auto_approve"

        return False, "auto_approve_false"

    except Exception:
        return False, "not_found"


# ---------------------------------------------------------------------------
# Provider lazy-loading — shared across all sixiang modules
# ---------------------------------------------------------------------------

_provider: Any = None


def _get_provider() -> Any:
    """Lazily obtain the LLM provider for this sixiang agent.

    Uses the per-agent config (``agents.defaults.sixiang.agents.yin``)
    when available, falling back to the global defaults.
    """
    global _provider
    if _provider is not None:
        return _provider

    try:
        from vingobot.providers.factory import build_sixiang_provider_snapshot
        from vingobot.config.loader import load_config, resolve_config_env_vars

        config = resolve_config_env_vars(load_config())
        snapshot = build_sixiang_provider_snapshot(config, "yin")
        _provider = snapshot.provider
    except Exception:
        logger.warning("[阴] 无法初始化 provider")
    return _provider


def set_provider(provider: Any) -> None:
    """Explicitly set the provider used by Yin's LLM approval layer."""
    global _provider
    _provider = provider


# ---------------------------------------------------------------------------
# Round-level self-reflection (moved from Action node)
# ---------------------------------------------------------------------------

_SELF_CHECK_INTERVAL = 4


async def self_reflect(
    facts: list[RoundExecutionFact],
    round_num: int,
    max_rounds: int,
    task_description: str,
    *,
    recent_tool_calls: list[list[str]] | None = None,
    signal: asyncio.Task | None = None,
) -> tuple[str, bool]:
    """Round-level self-check: detect read-only loops, exec walls, stagnation,
    and cross-round patterns (诊断成瘾, 轮次空耗).

    recent_tool_calls: past N rounds' approved tool call names, used for
        cross-round pattern detection by the heuristic checker.

    Returns (warning, needs_reweave).  needs_reweave signals the inner
    loop that the cognitive posture should be re-woven.
    """
    if round_num % _SELF_CHECK_INTERVAL != 0 and round_num < max_rounds - 5:
        return ("", False)

    # ── Pre-execution check (absorbed from old action.py) ────────
    pre_warning, pre_reweave = _pre_exec_check(facts, round_num, max_rounds,
                                recent_tool_calls=recent_tool_calls)
    if pre_warning:
        return (pre_warning, pre_reweave)

    # Try LLM first, fall back to heuristic
    provider = _get_provider()
    if provider is None:
        return _self_reflect_heuristic(facts, round_num, max_rounds,
                                        recent_tool_calls=recent_tool_calls)

    try:
        return await _self_reflect_llm(facts, round_num, max_rounds,
                                         task_description, provider, signal,
                                         recent_tool_calls=recent_tool_calls)
    except Exception:
        logger.exception("[阴·自省] LLM 失败，使用启发式")
        return _self_reflect_heuristic(facts, round_num, max_rounds,
                                        recent_tool_calls=recent_tool_calls)


async def _self_reflect_llm(
    facts: list[RoundExecutionFact],
    round_num: int,
    max_rounds: int,
    task_description: str,
    provider: Any,
    signal: asyncio.Task | None,
    recent_tool_calls: list[list[str]] | None = None,
) -> tuple[str, bool]:
    """LLM-based round self-reflection.

    Returns (warning, needs_reweave).  needs_reweave signals that the
    cognitive posture should be re-woven because the task phase has changed
    or is stuck in a repeating pattern.
    """
    remaining = max_rounds - round_num

    recent = facts[-5:]
    round_lines: list[str] = []
    for f in recent:
        tools = f"{f.tool_call_count}调用" if f.tool_call_count > 0 else ""
        status = _self_reflect_icon(f.execution_status)
        round_lines.append(
            f"- 第{f.round}轮: {f.yang_intent_summary[:80]} | "
            f"{tools} | {status} {f.execution_result_summary[:80]}"
        )
    recent_text = "\n".join(round_lines) if round_lines else "(无记录)"

    write_count = sum(
        1 for f in facts[-_SELF_CHECK_INTERVAL:]
        if any(kw in (f.execution_result_summary or "") for kw in ("write_file", "exec", "成功"))
    )

    system_prompt = f"""你是四爻·阴（Yin），在审批完本轮工具调用后做轮级自省。

## 当前状态
- 第 {round_num} 轮 / 上限 {max_rounds}（剩 {remaining} 轮）
- 任务: {task_description[:200]}

## 最近 {len(recent)} 轮
{recent_text}

## 判断
- 最近 {_SELF_CHECK_INTERVAL} 轮产出次数: {write_count}
- 剩余: {remaining} 轮

## 阶段变化检测 (needs_reweave)
判断当前任务的「阶段」是否已经变化或卡住，是否需要重新编织(Weaver)认知姿态:
- 如果阶段已变化(从探索→执行、从执行→验证) → needs_reweave: true
- 如果重复同一模式(连续纯读、连续同工具失败) → needs_reweave: true
- 如果 force_complete → needs_reweave: true
- 正常推进 → needs_reweave: false

## 输出（纯 JSON）
{{"self_check": "ok|warning|force_complete", "message": "原因", "needs_reweave": true|false}}
- ok: 正常
- warning: 连续纯读/exec失败，给 Yang 警告
- force_complete: 剩余<=3且连续5轮无产出且总轮>=15"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "评估当前任务是否卡住。"},
    ]

    try:
        response = await provider.chat_with_retry(messages=messages, temperature=0.1, max_tokens=200)
    except Exception:
        logger.warning("[阴·自省] LLM 调用失败")
        return _self_reflect_heuristic(facts, round_num, max_rounds,
                                        recent_tool_calls=recent_tool_calls)

    content = (response.content or "").strip()
    parsed = _parse_self_reflect_json(content)
    decision = parsed.get("self_check", "ok")
    msg = parsed.get("message", "")
    needs_reweave = parsed.get("needs_reweave", False)
    if not isinstance(needs_reweave, bool):
        needs_reweave = False

    if decision == "force_complete":
        logger.warning("[阴·自省] 强制终止: {}", msg)
        return (
            f"## ⚠️ 阴节点强制终止\n{msg}\n\n请立即调用 task_complete。",
            True,
        )

    if decision == "warning":
        logger.info("[阴·自省] 警告: {}", msg)
        return (
            f"## ⚠️ 阴节点自省警告\n{msg}\n\n请基于此调整本轮行动。",
            needs_reweave,
        )

    return ("", needs_reweave)


def _self_reflect_heuristic(
    facts: list[RoundExecutionFact],
    round_num: int,
    max_rounds: int,
    *,
    recent_tool_calls: list[list[str]] | None = None,
) -> tuple[str, bool]:
    """Heuristic round self-reflection (no LLM).

    Returns (warning, needs_reweave).  needs_reweave=True when a stuck
    pattern is detected that requires a phase change.
    """
    remaining = max_rounds - round_num
    recent = facts[-5:]
    consecutive_reads = 0
    consecutive_exec_fails = 0

    for f in reversed(recent):
        if f.tool_call_count == 0:
            break
        if f.execution_status == "exec_failed":
            consecutive_exec_fails += 1
        elif f.execution_status == "success" and any(
            kw in (f.execution_result_summary or "") for kw in ("write_file", "exec")
        ):
            break
        else:
            consecutive_reads += 1

    # ── 诊断成瘾检测: 连续3+轮全是 read_file/grep/ls ─
    if recent_tool_calls:
        tc_list = list(recent_tool_calls)
        diagnose_streak = 0
        diagnose_only_tools = {"read_file", "list_directory", "grep_code",
                                 "search_codebase", "search_symbol", "search_file"}
        for tools in reversed(tc_list):
            if tools and all(t in diagnose_only_tools for t in tools):
                diagnose_streak += 1
            else:
                break
        if diagnose_streak >= 3:
            return (
                _build_stuck_warning(
                    "诊断成瘾", diagnose_streak, remaining,
                    action="**你必须直接产出交付物**，不要再读取更多文件。"
                           "用 write_file 写入成果文件（如 outputs/ 下），"
                           "然后调用 task_complete 结束任务。"
                ),
                True,
            )

    # ── 轮次空耗检测: 5轮内无任何文件写入/工具产出 ─
    recent_5 = facts[-5:]
    has_output = any(
        f.execution_status == "success" and
        any(kw in (f.execution_result_summary or "")
            for kw in ("write_file", "edit_file", "exec"))
        for f in recent_5
    )
    if len(recent_5) >= 5 and not has_output and round_num >= 8:
        no_write_rounds = sum(
            1 for f in recent_5
            if f.tool_call_count > 0
            and not any(kw in (f.execution_result_summary or "")
                        for kw in ("write_file", "edit_file", "exec"))
        )
        return (
            _build_stuck_warning(
                "轮次空耗", no_write_rounds, remaining,
                action=f"已连续 {no_write_rounds} 轮无文件产出。"
                       f"用 write_file 写入 outputs/ 下的成果文件，"
                       f"然后调用 task_complete。"
            ),
            True,
        )

    if remaining <= 3 and consecutive_reads >= 5 and round_num >= 15:
        logger.warning("[阴·自省] 强制终止: 剩 {} 轮，连续 {} 轮纯读", remaining, consecutive_reads)
        return (
            f"## ⚠️ 阴节点强制终止\n"
            f"剩余 {remaining} 轮，已连续 {consecutive_reads} 轮无产出。请调用 task_complete。",
            True,
        )

    if consecutive_exec_fails >= 3:
        logger.warning("[阴·自省] exec 墙: 连续 {} 次失败", consecutive_exec_fails)
        return (
            f"## ⚠️ 阴节点自省警告（exec 墙）\n"
            f"已连续 {consecutive_exec_fails} 次 exec 失败。请切换到 write_file 模式。",
            True,
        )

    if consecutive_reads >= 5 and remaining <= 6:
        logger.warning("[阴·自省] 纯读: 连续 {} 轮，剩 {} 轮", consecutive_reads, remaining)
        return (
            f"## ⚠️ 阴节点自省警告（纯读）\n"
            f"已连续 {consecutive_reads} 轮纯读，仅剩 {remaining} 轮。本轮必须产出。",
            True,
        )

    return ("", False)


def _build_stuck_warning(
    pattern: str, streak: int, remaining: int, action: str,
) -> str:
    """Build a concrete stuck-pattern warning with explicit action."""
    return (
        f"## ⚠️ 阴节点自省警告（{pattern}）\n"
        f"检测到连续 {streak} 轮的{pattern}模式。\n"
        f"剩余 {remaining} 轮。\n\n"
        f"{action}"
    )


def _pre_exec_check(
    facts: list[RoundExecutionFact],
    round_num: int,
    max_rounds: int,
    *,
    recent_tool_calls: list[list[str]] | None = None,
) -> tuple[str, bool]:
    """Pre-execution check absorbed from old action.py.

    Detects read-only spiral patterns at specific checkpoints (rounds 3, 5, 8,
    11, 15...) to prevent Yang from falling into pure-read cycles.

    Returns (warning, needs_reweave).  needs_reweave=True when a phase change
    is needed (e.g. stuck in read loop needs reweave to shift to write mode).
    """
    # Round 3 check: first intervention point
    if round_num == 3:
        all_read = all(
            f.execution_status == "success" and f.tool_call_count > 0
            and not any(kw in (f.execution_result_summary or "")
                        for kw in ("write_file", "edit_file", "exec"))
            for f in facts[-3:]
        )
        if all_read:
            return (
                f"## ⚠️ 阴节点自省警告（启动停滞）\n"
                f"前 3 轮均为纯读取，没有任何产出。\n"
                f"请立即用 write_file 写入初始成果文件（如 outputs/00-plan.md），"
                f"然后继续后续工作。",
                True,
            )

    # Round 5 and every 3rd thereafter: check最近3轮
    if round_num >= 5 and round_num % 3 == 2:
        recent_3 = facts[-3:]
        all_read_recent = all(
            f.tool_call_count > 0 and
            not any(kw in (f.execution_result_summary or "")
                    for kw in ("write_file", "edit_file"))
            for f in recent_3
        )
        if all_read_recent:
            return (
                f"## ⚠️ 阴节点自省警告（产出停滞）\n"
                f"最近 3 轮（第 {round_num - 2}-{round_num} 轮）均为纯读取，无文件产出。\n"
                f"请在 outputs/ 目录下 write_file 写入阶段性成果，不要无限阅读。\n"
                f"轮次上限 {max_rounds}，已消耗 {round_num} 轮。",
                True,
            )

    return ("", False)


def _self_reflect_icon(status: str) -> str:
    icons = {"success": "✅", "exec_failed": "❌exec", "failure": "❌", "partial_failure": "⚠️", "skipped": "⏭️"}
    return icons.get(status, "❓")


def _parse_self_reflect_json(text: str) -> dict[str, Any]:
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
    m = re.search(r'\{[^{}]*"self_check"[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
    return {"self_check": "ok"}
