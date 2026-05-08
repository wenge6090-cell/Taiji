"""Tests for workspace/safety boundary checking.

Covers the **front-layer** and **executor final-defense** enforcement added
during the workspace boundary security hardening:

- ``_check_exec_safety`` with ``workspace_root`` — cwd and command path checks
- ``_check_path_safety`` for read-only tools (read_file, list_directory)
- ``_resolve_path`` — absolute-path boundary enforcement
- ``_exec_command`` — cwd boundary enforcement
- Executor final-defense — exec / file writes blocked on boundary violation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vingobot.goal.types import ApprovedToolCall


# ---------------------------------------------------------------------------
# Mock provider (same as test_yin_llm_approval.py)
# ---------------------------------------------------------------------------


class MockDecisionProvider:
    """Mock LLM provider that returns configurable approval decisions."""

    def __init__(self, decisions: list[dict] | None = None) -> None:
        self.decisions = decisions or [
            {"tool_index": 0, "approved": True, "reason": "Mock approved"},
        ]

    async def chat_with_retry(
        self,
        messages: list[dict],
        **kwargs,
    ) -> object:
        import json

        content = json.dumps({"decisions": self.decisions}, ensure_ascii=False)

        class _MockResponse:
            def __init__(self, content: str) -> None:
                self.content = content

        return _MockResponse(content)


# ---------------------------------------------------------------------------
# Helper — import yin's private helpers for unit-level tests
# ---------------------------------------------------------------------------


def _import_yin_helpers():
    """Lazy-import yin internals to avoid module-level side effects."""
    from vingobot.goal.yin import (
        _check_exec_safety,
        _check_path_safety,
        _extract_abs_paths_from_cmd,
    )
    return _check_exec_safety, _check_path_safety, _extract_abs_paths_from_cmd


# ===================================================================
# _extract_abs_paths_from_cmd
# ===================================================================


class TestExtractAbsPathsFromCmd:
    """Extracting absolute file paths from shell command strings."""

    def test_windows_absolute_path(self) -> None:
        """Extracts a Windows drive-root path."""
        _, _, extract = _import_yin_helpers()
        paths = extract("cat C:\\Users\\file.txt")
        assert any("C:\\Users\\file.txt" in p for p in paths)

    def test_posix_absolute_path(self) -> None:
        """Extracts a POSIX absolute path."""
        _, _, extract = _import_yin_helpers()
        paths = extract("cat /etc/passwd")
        assert any("/etc/passwd" in p for p in paths)

    def test_home_directory_path(self) -> None:
        """Extracts a ~/ prefixed path."""
        _, _, extract = _import_yin_helpers()
        paths = extract("cat ~/file.txt")
        assert any("~/file.txt" in p for p in paths)

    def test_multiple_paths(self) -> None:
        """Extracts multiple absolute paths from a single command."""
        _, _, extract = _import_yin_helpers()
        paths = extract("cp /src/a.txt /dst/b.txt")
        assert len(paths) >= 2

    def test_no_paths_returns_empty(self) -> None:
        """Commands without absolute paths return an empty list."""
        _, _, extract = _import_yin_helpers()
        paths = extract("ls -la | grep foo")
        assert paths == []


# ===================================================================
# _check_exec_safety — workspace boundary
# ===================================================================


class TestCheckExecSafetyWorkspaceBoundary:
    """Direct calls to _check_exec_safety with workspace_root."""

    def test_cwd_outside_workspace_rejected(self, tmp_path: Path) -> None:
        """cwd outside workspace_root is rejected."""
        check, _, _ = _import_yin_helpers()
        safe, reason = check("ls", cwd=str(tmp_path.parent), workspace_root=tmp_path)
        assert not safe
        assert "工作目录超出工作区" in reason

    def test_cwd_inside_workspace_accepted(self, tmp_path: Path) -> None:
        """cwd inside workspace_root is accepted."""
        check, _, _ = _import_yin_helpers()
        safe, reason = check("ls", cwd=str(tmp_path), workspace_root=tmp_path)
        assert safe
        assert reason == "ok"

    def test_cwd_subdir_inside_workspace_accepted(self, tmp_path: Path) -> None:
        """cwd pointing to a subdirectory of workspace_root is accepted."""
        check, _, _ = _import_yin_helpers()
        sub = tmp_path / "sub"
        sub.mkdir()
        safe, reason = check("ls", cwd=str(sub), workspace_root=tmp_path)
        assert safe

    def test_command_path_outside_workspace_rejected(self, tmp_path: Path) -> None:
        """Command referencing an absolute path outside workspace is rejected."""
        check, _, _ = _import_yin_helpers()
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("data")
        safe, reason = check(f"cat {outside}", workspace_root=tmp_path)
        assert not safe
        assert "工作区外的路径" in reason

    def test_command_path_inside_workspace_accepted(self, tmp_path: Path) -> None:
        """Command referencing an absolute path inside workspace is accepted."""
        check, _, _ = _import_yin_helpers()
        inside = tmp_path / "data.txt"
        inside.write_text("data")
        safe, reason = check(f"cat {inside}", workspace_root=tmp_path)
        assert safe
        assert reason == "ok"

    def test_command_non_existent_path_ignored(self, tmp_path: Path) -> None:
        """Non-existent absolute paths in commands are skipped (no false positive)."""
        check, _, _ = _import_yin_helpers()
        # This file does not exist — should be ignored
        safe, reason = check("cat C:\\NonExistent\\path\\to\\file.txt", workspace_root=tmp_path)
        assert safe

    def test_no_workspace_root_skips_boundary_checks(self, tmp_path: Path) -> None:
        """Without workspace_root, boundary checks are skipped."""
        check, _, _ = _import_yin_helpers()
        safe, reason = check("ls", cwd=str(tmp_path.parent), workspace_root=None)
        assert safe
        assert reason == "ok"

    def test_dangerous_command_still_rejected_first(self, tmp_path: Path) -> None:
        """Dangerous command is rejected even if paths are valid."""
        check, _, _ = _import_yin_helpers()
        safe, reason = check("rm -rf /", workspace_root=tmp_path)
        assert not safe
        assert "危险命令" in reason


# ===================================================================
# approve() — front layer: read-only tool path check
# ===================================================================


class TestApproveReadOnlyPathBoundary:
    """read_file / list_directory path boundary checks in front layer."""

    async def test_read_file_absolute_outside_rejected(self, tmp_path: Path) -> None:
        """read_file with absolute path outside workspace is rejected."""
        from vingobot.goal.yin import approve

        outside = tmp_path.parent / "secret.txt"
        outside.write_text("secret")
        calls = [{"name": "read_file", "arguments": {"path": str(outside)}}]
        _, decision, reason = await approve(calls, workspace_root=tmp_path)
        assert decision == "rejected"
        assert "超出工作区" in reason

    async def test_read_file_relative_inside_auto_approved(self, tmp_path: Path) -> None:
        """read_file with relative path inside workspace is auto-approved."""
        from vingobot.goal.yin import approve

        inside = tmp_path / "notes.txt"
        inside.write_text("hello")
        calls = [{"name": "read_file", "arguments": {"path": "notes.txt"}}]
        result, decision, reason = await approve(calls, workspace_root=tmp_path)
        assert decision == "approved"
        assert len(result) == 1
        assert result[0].name == "read_file"

    async def test_read_file_absolute_inside_auto_approved(self, tmp_path: Path) -> None:
        """read_file with absolute path inside workspace is auto-approved."""
        from vingobot.goal.yin import approve

        inside = tmp_path / "readme.md"
        inside.write_text("# Readme")
        calls = [{"name": "read_file", "arguments": {"path": str(inside)}}]
        result, decision, reason = await approve(calls, workspace_root=tmp_path)
        assert decision == "approved"
        assert len(result) == 1

    async def test_list_directory_absolute_outside_rejected(self, tmp_path: Path) -> None:
        """list_directory with absolute path outside workspace is rejected."""
        from vingobot.goal.yin import approve

        calls = [{"name": "list_directory", "arguments": {"path": str(tmp_path.parent)}}]
        _, decision, reason = await approve(calls, workspace_root=tmp_path)
        assert decision == "rejected"
        assert "超出工作区" in reason

    async def test_list_directory_relative_inside_auto_approved(self, tmp_path: Path) -> None:
        """list_directory with relative path inside workspace is auto-approved."""
        from vingobot.goal.yin import approve

        calls = [{"name": "list_directory", "arguments": {"path": "."}}]
        result, decision, reason = await approve(calls, workspace_root=tmp_path)
        assert decision == "approved"
        assert len(result) == 1

    async def test_read_file_traversal_rejected_before_boundary(self, tmp_path: Path) -> None:
        """read_file with '..' is caught by traversal check before boundary."""
        from vingobot.goal.yin import approve

        calls = [{"name": "read_file", "arguments": {"path": "../outside.txt"}}]
        _, decision, reason = await approve(calls, workspace_root=tmp_path)
        assert decision == "rejected"
        assert "路径穿越" in reason or "超出工作区" in reason

    async def test_no_workspace_root_skips_read_path_check(self, tmp_path: Path) -> None:
        """Without workspace_root, read_file absolute path is not checked."""
        from vingobot.goal.yin import approve

        calls = [{"name": "read_file", "arguments": {"path": "C:\\Windows\\win.ini"}}]
        result, decision, reason = await approve(calls, workspace_root=None)
        # No workspace_root → front layer skip → auto-approved
        assert decision == "approved"
        assert len(result) == 1


# ===================================================================
# approve() — front layer: exec workspace boundary
# ===================================================================


class TestApproveExecWorkspaceBoundary:
    """exec cwd / command path boundary checks in front layer."""

    async def test_exec_cwd_outside_workspace_rejected(self, tmp_path: Path) -> None:
        """exec with cwd outside workspace is rejected by front layer."""
        from vingobot.goal.yin import approve

        calls = [{"name": "exec", "arguments": {"command": "ls", "cwd": str(tmp_path.parent)}}]
        _, decision, reason = await approve(calls, workspace_root=tmp_path)
        assert decision == "rejected"
        assert "工作目录超出工作区" in reason

    async def test_exec_cwd_inside_workspace_passes_front_check(self, tmp_path: Path) -> None:
        """exec with cwd inside workspace passes front check and goes to LLM."""
        from vingobot.goal.yin import approve

        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": True, "reason": "safe"},
        ])
        sub = tmp_path / "workdir"
        sub.mkdir()
        calls = [{"name": "exec", "arguments": {"command": "ls", "cwd": str(sub)}}]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert decision == "approved"
        assert len(result) == 1

    async def test_exec_command_path_outside_workspace_rejected(self, tmp_path: Path) -> None:
        """exec command referencing file outside workspace is rejected."""
        from vingobot.goal.yin import approve

        outside = tmp_path.parent / "report.txt"
        outside.write_text("data")
        calls = [{"name": "exec", "arguments": {"command": f"cat {outside}"}}]
        _, decision, reason = await approve(calls, workspace_root=tmp_path)
        assert decision == "rejected"
        assert "工作区外的路径" in reason

    async def test_exec_command_path_inside_workspace_passes_front(self, tmp_path: Path) -> None:
        """exec command referencing file inside workspace passes front check."""
        from vingobot.goal.yin import approve

        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": True, "reason": "safe"},
        ])
        inside = tmp_path / "data.txt"
        inside.write_text("data")
        calls = [{"name": "exec", "arguments": {"command": f"cat {inside}"}}]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert decision == "approved"
        assert len(result) == 1


# ===================================================================
# _resolve_path — absolute path boundary enforcement
# ===================================================================


class TestResolvePathBoundary:
    """_resolve_path absolute-path boundary enforcement."""

    def test_absolute_outside_task_dir_raises(self, tmp_path: Path) -> None:
        """Absolute path outside task_dir raises ValueError."""
        from vingobot.core.tool_executor import _resolve_path

        outside = tmp_path.parent / "outside.txt"
        with pytest.raises(ValueError, match="路径超出允许范围"):
            _resolve_path(str(outside.resolve()), task_dir=tmp_path)

    def test_absolute_inside_task_dir_succeeds(self, tmp_path: Path) -> None:
        """Absolute path inside task_dir resolves normally."""
        from vingobot.core.tool_executor import _resolve_path

        inside = tmp_path / "data.txt"
        result = _resolve_path(str(inside.resolve()), task_dir=tmp_path)
        assert result == inside.resolve()

    def test_relative_path_stays_inside(self, tmp_path: Path) -> None:
        """Relative path resolves inside task_dir."""
        from vingobot.core.tool_executor import _resolve_path

        result = _resolve_path("sub/file.txt", task_dir=tmp_path)
        assert result == (tmp_path / "sub/file.txt").resolve()
        # Verify it's inside task_dir
        result.relative_to(tmp_path.resolve())  # should not raise

    def test_no_task_dir_returns_absolute_as_is(self, tmp_path: Path) -> None:
        """Without task_dir, absolute paths are returned as-is."""
        from vingobot.core.tool_executor import _resolve_path

        p = (tmp_path.parent / "anywhere.txt").resolve()
        result = _resolve_path(str(p), task_dir=None)
        assert result == p


# ===================================================================
# Executor — final defense: exec workspace boundary
# ===================================================================


class TestExecutorFinalDefense:
    """Executor's final-defense layer blocks workspace boundary violations."""

    async def test_exec_cwd_outside_task_dir_blocked(self, tmp_path: Path) -> None:
        """Executor blocks exec with cwd outside task_dir."""
        from vingobot.goal.executor import execute_tool_calls

        calls = [
            ApprovedToolCall(
                name="exec",
                arguments={"command": "echo hello", "cwd": str(tmp_path.parent)},
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        assert results[0].status == "blocked"
        assert "工作目录超出工作区" in results[0].error

    async def test_exec_path_outside_task_dir_blocked(self, tmp_path: Path) -> None:
        """Executor blocks exec referencing file outside task_dir."""
        from vingobot.goal.executor import execute_tool_calls

        outside = tmp_path.parent / "userdata.txt"
        outside.write_text("secret")
        calls = [
            ApprovedToolCall(
                name="exec",
                arguments={"command": f"cat {outside}"},
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        assert results[0].status == "blocked"
        assert "工作区外的路径" in results[0].error

    async def test_file_write_outside_task_dir_blocked(self, tmp_path: Path) -> None:
        """Executor blocks write_file with path outside task_dir."""
        from vingobot.goal.executor import execute_tool_calls

        outside = tmp_path.parent / "evil.txt"
        calls = [
            ApprovedToolCall(
                name="write_file",
                arguments={"path": str(outside), "content": "pwned"},
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        assert results[0].status == "blocked"
        assert "超出工作区" in results[0].error

    async def test_file_read_outside_task_dir_blocked(self, tmp_path: Path) -> None:
        """Executor blocks read_file with path outside task_dir."""
        from vingobot.goal.executor import execute_tool_calls

        outside = tmp_path.parent / "config.txt"
        outside.write_text("should not be readable")
        calls = [
            ApprovedToolCall(
                name="read_file",
                arguments={"path": str(outside)},
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        assert results[0].status == "blocked"
        assert "超出工作区" in results[0].error

    async def test_exec_inside_task_dir_succeeds(self, tmp_path: Path) -> None:
        """Executor allows exec with safe cwd and paths inside task_dir."""
        from vingobot.goal.executor import execute_tool_calls

        calls = [
            ApprovedToolCall(
                name="exec",
                arguments={"command": "echo hello"},
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        # Should execute successfully (status=success) — the command returns output
        assert results[0].status == "success"

    async def test_file_write_inside_task_dir_succeeds(self, tmp_path: Path) -> None:
        """Executor allows write_file inside task_dir."""
        from vingobot.goal.executor import execute_tool_calls

        inside = tmp_path / "safe.txt"
        calls = [
            ApprovedToolCall(
                name="write_file",
                arguments={"path": str(inside), "content": "safe data"},
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        assert results[0].status == "success"
        assert inside.is_file()
