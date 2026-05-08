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

import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.goal.types import ApprovedToolCall

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
    "chmod 777",
    "sudo",
    "su",
}

# Baseline dangerous commands (never removed, only augmented from L4 truths)
_DANGEROUS_CMDS_BASELINE: frozenset[str] = frozenset(_DANGEROUS_CMDS)


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

# L4 truths directory path relative to workspace root
_TRUTHS_DIR = "cognition/truths"


# ---------------------------------------------------------------------------
# Public API — two-layer approval
# ---------------------------------------------------------------------------


async def approve(
    tool_calls: list[dict[str, Any]],
    workspace_root: str | Path | None = None,
    *,
    provider: Any = None,
) -> tuple[list[ApprovedToolCall], str, str]:
    """Two-layer approval: hardcoded front layer then LLM contextual layer.

    Args:
        tool_calls: Raw tool_calls from Yang (dicts with 'name' and 'arguments').
        workspace_root: Root directory for path-traversal checking.
        provider: Optional LLM provider for contextual approval layer.  If
            None, falls back to the module-level shared provider.

    Returns:
        Tuple of (approved_calls, decision, reason).
        - approved_calls: List of approved ``ApprovedToolCall`` instances.
        - decision: One of 'approved', 'rejected', 'modified', 'need_user_approval'.
        - reason: Human-readable explanation.
    """
    if not tool_calls:
        return [], "skipped", "无工具调用"

    root = Path(workspace_root) if workspace_root else None

    # ── Step 1: Front layer (hardcoded) ────────────────────────
    auto_approved: list[ApprovedToolCall] = []
    needs_llm_check: list[dict[str, Any]] = []
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
            # Front check → auto-approve task workspace, LLM layer for others
            check_ok, check_reason = _check_side_effect(name, args, root)
            if check_ok:
                if check_reason == "ok_task_workspace":
                    auto_approved.append(ApprovedToolCall(name=name, arguments=args))
                else:
                    needs_llm_check.append(tc)
            else:
                front_rejected_reasons.append(f"{name}: {check_reason}")
                logger.warning("[阴·前置] 拒绝技能工具调用: {} — {}", name, check_reason)
            continue

        # Side-effect tools → front check (auto-approve task workspace, LLM for others)
        if name in _SIDE_EFFECT_TOOLS:
            check_ok, check_reason = _check_side_effect(name, args, root)
            if check_ok:
                if check_reason == "ok_task_workspace":
                    auto_approved.append(ApprovedToolCall(name=name, arguments=args))
                else:
                    needs_llm_check.append(tc)
            else:
                front_rejected_reasons.append(f"{name}: {check_reason}")
                logger.warning("[阴·前置] 拒绝工具调用: {} — {}", name, check_reason)
            continue

        # Unknown tools → reject immediately
        front_rejected_reasons.append(f"{name}: 未知工具（未在审批白名单中）")
        logger.warning("[阴·前置] 拒绝未知工具: {}", name)

    # ── Step 2: LLM layer (contextual approval) ────────────────
    llm_approved: list[ApprovedToolCall] = []
    llm_rejected_reasons: list[str] = []

    if needs_llm_check:
        try:
            llm_provider = provider or _get_provider()
            if llm_provider is not None:
                llm_approved, llm_rejected_reasons = await _llm_approve(
                    needs_llm_check,
                    llm_provider,
                    root,
                )
            else:
                # No provider available — conservative fallback: reject
                for tc in needs_llm_check:
                    name = tc.get("name", "") or tc.get("function", {}).get("name", "")
                    llm_rejected_reasons.append(f"{name}: 无 LLM provider，保守拒绝")
        except Exception:
            logger.exception("[阴·LLM] LLM 审批层异常")
            # Conservative fallback: reject all
            for tc in needs_llm_check:
                name = tc.get("name", "") or tc.get("function", {}).get("name", "")
                llm_rejected_reasons.append(f"{name}: LLM 审批异常，保守拒绝")

    # ── Combine results ────────────────────────────────────────
    all_approved = auto_approved + llm_approved
    all_rejected = front_rejected_reasons + llm_rejected_reasons

    if not all_approved:
        return [], "rejected", "全部拒绝: " + "; ".join(all_rejected)

    if all_rejected:
        return (
            all_approved,
            "modified",
            f"部分批准 ({len(all_approved)}个), 拒绝: {'; '.join(all_rejected)}",
        )

    return all_approved, "approved", f"全部批准 ({len(all_approved)}个工具调用)"


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
        "你的判断必须基于以下 L4 真理（不可变规则）和 L5 灵魂（身份认知）：\n\n"
        "## L4 真理层（不可违抗）\n"
        f"{truths_text or '（无已加载的真理定义）'}\n\n"
        "## L5 灵魂层（身份认知）\n"
        f"{soul_text or '（无已加载的灵魂定义）'}\n\n"
        "## 审批规则\n"
        "- 批准：该工具调用符合 L4/L5 约束，可安全执行\n"
        "- 拒绝：该工具调用违反 L4/L5 约束，需说明原因\n"
        "- 不确定时请选择拒绝（保守原则）\n"
        "- 拒绝理由需引用具体的 L4 真理编号或 L5 原则\n"
        "- **重要**: `.taiji/goals/` 是任务工作区，其中的写入操作是正常且必需的，不应拒绝。"
        "goal_dir（目标目录）同样在写权限范围内，对其中的写入/编辑操作也是正常且必需的。"
        "只有 `.taiji/cognition/` 受保护，禁止直接写入。\n\n"
        "请为以下每个工具调用单独输出 JSON:\n"
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
