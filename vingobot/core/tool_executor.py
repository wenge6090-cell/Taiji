"""
Shared builtin tool executor for both AgentLoop and sixiang processes.

Provides the core tool execution functions that the sixiang process needs
(read_file, write_file, list_directory, exec, etc.). Both
processes can import and use this module, eliminating duplicate implementations.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_builtin_tool(
    name: str,
    args: dict[str, Any],
    task_dir: str | Path | None = None,
    *,
    read_only_allowed_dirs: list[Path] | None = None,
    write_allowed_dirs: list[Path] | None = None,
    exec_allowed_cwds: list[Path] | None = None,
) -> str:
    """Execute a builtin tool by name. Returns result string (or error string).

    Args:
        name: Tool name.
        args: Tool arguments.
        task_dir: Primary task directory (fallback for path resolution).
        read_only_allowed_dirs: Extra directories allowed for **read** operations.
        write_allowed_dirs: Extra directories allowed for **write** operations.
        exec_allowed_cwds: Extra directories where shell commands may run.
    """
    if name == "read_file":
        return _read_file(args, task_dir, read_only_allowed_dirs=read_only_allowed_dirs)
    if name == "write_file":
        return _write_file(args, task_dir, write_allowed_dirs=write_allowed_dirs)
    if name == "edit_file":
        return _edit_file(args, task_dir, write_allowed_dirs=write_allowed_dirs)
    if name == "delete_file":
        return _delete_file(args, task_dir, write_allowed_dirs=write_allowed_dirs)
    if name == "list_directory":
        return _list_directory(args, task_dir, read_only_allowed_dirs=read_only_allowed_dirs)
    if name == "exec":
        return await _exec_command(args, task_dir, allowed_cwds=exec_allowed_cwds)
    if name == "task_complete":
        return f"任务完成标记: {args.get('summary', '')}"
    if name == "query_capabilities":
        return _query_capabilities()

    return f"[错误] 未知工具: {name}"


# ---------------------------------------------------------------------------
# Read file
# ---------------------------------------------------------------------------


def _read_file(args: dict[str, Any], task_dir: str | Path | None = None, *, read_only_allowed_dirs: list[Path] | None = None) -> str:
    path_str = args.get("path", "")
    if not path_str:
        return "[错误] 缺少 path 参数"

    path = _resolve_path(path_str, task_dir, read_only_allowed_dirs=read_only_allowed_dirs)
    if not path.exists():
        return f"[错误] 文件不存在: {path}"
    if not path.is_file():
        return f"[错误] 不是文件: {path}"

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        start = args.get("start_line")
        end = args.get("end_line")
        if start or end:
            lines = content.splitlines()
            s = max(0, (start or 1) - 1)
            e = min(len(lines), end) if end else len(lines)
            content = "\n".join(lines[s:e])
        return content[:8000]
    except Exception as exc:
        return f"[错误] 读取失败: {exc}"


# ---------------------------------------------------------------------------
# Write file
# ---------------------------------------------------------------------------


def _write_file(args: dict[str, Any], task_dir: str | Path | None = None, *, write_allowed_dirs: list[Path] | None = None) -> str:
    path_str = args.get("path", "")
    content = args.get("content", "")
    if not path_str:
        return "[错误] 缺少 path 参数"

    path = _resolve_path(path_str, task_dir, write_allowed_dirs=write_allowed_dirs)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"写入成功: {path} ({len(content)} 字符)"
    except Exception as exc:
        return f"[错误] 写入失败: {exc}"


# ---------------------------------------------------------------------------
# Delete file
# ---------------------------------------------------------------------------


def _delete_file(args: dict[str, Any], task_dir: str | Path | None = None, *, write_allowed_dirs: list[Path] | None = None) -> str:
    path_str = args.get("path", "")
    if not path_str:
        return "[错误] 缺少 path 参数"

    path = _resolve_path(path_str, task_dir, write_allowed_dirs=write_allowed_dirs)
    if not path.exists():
        return f"[错误] 文件不存在: {path}"
    try:
        path.unlink()
        return f"删除成功: {path}"
    except Exception as exc:
        return f"[错误] 删除失败: {exc}"


# ---------------------------------------------------------------------------
# Edit file
# ---------------------------------------------------------------------------


def _edit_file(args: dict[str, Any], task_dir: str | Path | None = None, *, write_allowed_dirs: list[Path] | None = None) -> str:
    """Edit a file by replacing *old_string* with *new_string*.

    This is a surgical alternative to write_file: read the target file,
    find-and-replace the first occurrence of *old_string*, and write back.
    Raises an error if *old_string* is not found.
    """
    path_str = args.get("path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")

    if not path_str:
        return "[错误] 缺少 path 参数"
    if not old_string:
        return "[错误] 缺少 old_string 参数"

    path = _resolve_path(path_str, task_dir, write_allowed_dirs=write_allowed_dirs)

    if not path.exists():
        return f"[错误] 文件不存在: {path}"
    if not path.is_file():
        return f"[错误] 不是文件: {path}"

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[错误] 读取失败: {exc}"

    if old_string not in content:
        return f"[错误] 未找到待替换的文本: {old_string[:100]}"

    new_content = content.replace(old_string, new_string, 1)

    try:
        path.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        return f"[错误] 写入失败: {exc}"

    lines_changed = new_content.count("\n") - content.count("\n")
    return (
        f"编辑成功: {path} ({len(content)} → {len(new_content)} 字符, "
        f"{abs(lines_changed)} 行变化)"
    )


# ---------------------------------------------------------------------------
# List directory
# ---------------------------------------------------------------------------


def _list_directory(args: dict[str, Any], task_dir: str | Path | None = None, *, read_only_allowed_dirs: list[Path] | None = None) -> str:
    path_str = args.get("path", ".")
    path = _resolve_path(path_str, task_dir, read_only_allowed_dirs=read_only_allowed_dirs)
    if not path.exists():
        return f"[错误] 目录不存在: {path}"
    if not path.is_dir():
        return f"[错误] 不是目录: {path}"

    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines: list[str] = []
        for e in entries[:200]:
            kind = "📁" if e.is_dir() else "📄"
            size = ""
            if e.is_file():
                try:
                    size = f" ({e.stat().st_size} B)"
                except OSError:
                    pass
            lines.append(f"{kind} {e.name}{size}")
        return "\n".join(lines) if lines else "(空目录)"
    except Exception as exc:
        return f"[错误] 列表失败: {exc}"


# ---------------------------------------------------------------------------
# Shell exec
# ---------------------------------------------------------------------------


async def _exec_command(
    args: dict[str, Any],
    task_dir: str | Path | None = None,
    *,
    allowed_cwds: list[Path] | None = None,
) -> str:
    command = args.get("command", "")
    cwd = args.get("cwd", "")

    if not command:
        return "[错误] 缺少 command 参数"

    # Resolve cwd to an absolute path (allow within exec_allowed_cwds)
    try:
        work_dir = str(_resolve_path(
            cwd or ".", task_dir, write_allowed_dirs=allowed_cwds,
        ))
    except ValueError as exc:
        return f"[错误] {exc}"

    # Verify cwd is inside one of the allowed directories
    allowed = list(allowed_cwds) if allowed_cwds else []
    if task_dir:
        allowed.append(Path(task_dir).resolve())

    cwd_ok = False
    if allowed:
        try:
            wp = Path(work_dir).resolve()
            for d in allowed:
                try:
                    wp.relative_to(d.resolve())
                    cwd_ok = True
                    break
                except ValueError:
                    continue
        except (ValueError, OSError):
            pass
    else:
        cwd_ok = True  # no restrictions

    if not cwd_ok:
        return f"[错误] 工作目录超出允许范围: {cwd}"

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "[错误] 命令执行超时 (30s)"

        output = (stdout.decode("utf-8", errors="replace"))[:4000]
        if stderr:
            err_text = stderr.decode("utf-8", errors="replace")[:2000]
            output += f"\n[stderr]\n{err_text}"
        if proc.returncode and proc.returncode != 0:
            output += f"\n[退出码: {proc.returncode}]"
        return output or "(无输出)"
    except OSError as exc:
        return f"[错误] 执行失败: {exc}"


# ---------------------------------------------------------------------------
# Safety checks (shared)
# ---------------------------------------------------------------------------


def check_path_safety(path_str: str) -> tuple[bool, str]:
    """Check for path traversal attacks."""
    if ".." in path_str.replace("\\", "/").split("/"):
        return False, f"路径穿越: {path_str}"
    return True, "ok"


def check_exec_safety(command: str) -> tuple[bool, str]:
    """Check for dangerous commands."""
    dangerous = {
        "rm -rf", "rm -r", "del /f", "format",
        "shutdown", "reboot", "mkfs", "sudo", "su -",
    }
    cmd = command.lower().strip()
    for d in dangerous:
        if d in cmd:
            return False, f"危险命令: {d}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Path resolution — unified with write_allowed_dirs
# ---------------------------------------------------------------------------


def _resolve_path(
    path_str: str,
    task_dir: str | Path | None = None,
    *,
    read_only_allowed_dirs: list[Path] | None = None,
    write_allowed_dirs: list[Path] | None = None,
) -> Path:
    """Resolve a path relative to the task directory with configurable boundaries.

    Resolution order:
    1. If absolute, check **write_allowed_dirs** first (for write ops).
    2. Fall back to **task_dir**.
    3. Then check **read_only_allowed_dirs** (read-only ops only).
    4. If relative, resolve against **task_dir** (or **write_allowed_dirs**).

    Raises ``ValueError`` when the resolved path crosses all allowed boundaries.
    """
    path = Path(path_str)

    def _in_any(resolved: Path, dirs: list[Path]) -> bool:
        for d in dirs:
            try:
                resolved.relative_to(d.resolve())
                return True
            except ValueError:
                continue
        return False

    if path.is_absolute():
        resolved = path.resolve()

        # No boundaries configured → no restriction
        if not task_dir and not write_allowed_dirs and not read_only_allowed_dirs:
            return resolved

        # 1. Check write_allowed_dirs
        if write_allowed_dirs and _in_any(resolved, write_allowed_dirs):
            return resolved

        # 2. Check task_dir
        if task_dir:
            try:
                resolved.relative_to(Path(task_dir).resolve())
                return resolved
            except ValueError:
                pass

        # 3. Check read_only_allowed_dirs
        if read_only_allowed_dirs and _in_any(resolved, read_only_allowed_dirs):
            return resolved

        raise ValueError(f"路径超出允许范围: {path_str}")

    # Relative path: resolve against write_allowed_dirs first, then task_dir
    if write_allowed_dirs:
        for d in write_allowed_dirs:
            candidate = (d / path).resolve()
            if _in_any(candidate, write_allowed_dirs):
                return candidate

    if task_dir:
        return (Path(task_dir) / path).resolve()

    return path.resolve()


# ---------------------------------------------------------------------------
# Query capabilities
# ---------------------------------------------------------------------------


def _query_capabilities() -> str:
    """Return a summary of the current execution environment capabilities.

    This provides the agent with runtime information about resource limits,
    permissions, and available mechanisms so it can plan tool usage effectively.
    """
    return (
        "## 执行环境能力\n\n"
        "**并发与配额**\n"
        "- 单轮最大工具调用数: 10\n"
        "- 最大并发执行数: 10（所有调用同时执行）\n"
        "- 单任务最大轮次: 30\n"
        "- 命令执行超时: 30 秒\n\n"
        "**可用工具**\n"
        "- read_file, write_file, edit_file, delete_file, list_directory\n"
        "- exec, search_codebase\n"
        "- web_search, web_fetch, query_capabilities, task_complete\n\n"
        "**读写权限**\n"
        "- 写权限: 任务目录 (task_dir) + 目标目录 (goal_dir)\n"
        "- 读权限: 任务目录 + 目标目录 + 认知库 (skills/models/grids)\n"
        "- 绝对路径: 允许，但必须在允许的目录范围内\n"
        "- edit_file: 支持在 task_dir 和 goal_dir 范围内编辑文件\n\n"
        "**命令执行**\n"
        "- exec: 可在 task_dir 和 goal_dir 下执行命令\n"
        "- 超时: 30 秒\n"
        "- 危险命令限制: rm -rf, sudo, shutdown 等被禁止\n\n"
        "**上下文与信息**\n"
        "- 文件读取上限: 8,000 chars/次\n"
        "- 跨轮传递: 上一轮只读工具结果自动注入下一轮系统提示\n"
        "- 目标蓝图/记忆/轨迹: 已在系统提示中提供\n"
        "- 认知导航: 用 list_directory + read_file 直接读取认知库文件\n\n"
        "**效率建议**\n"
        "- 需要读取多个文件时，一次性发送多个 read_file 调用（并发执行）\n"
        "- 先用 list_directory 了解目录结构，再批量读取\n"
        "- 已读取的内容下一轮会自动注入，无需反复重读"
    )
