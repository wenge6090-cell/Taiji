"""Integration tests for unified permission management.

Covers the new sixiang permission features:

- ``SixiangPermissionConfig`` — unified config dataclass
- ``_edit_file`` — surgical file editing (new builtin tool)
- ``execute_builtin_tool`` with ``write_allowed_dirs`` / ``exec_allowed_cwds``
- ``execute_tool_calls`` with ``perm`` parameter (goal_dir write/exec)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vingobot.goal.types import ApprovedToolCall, SixiangPermissionConfig


# ===================================================================
# SixiangPermissionConfig — unit tests
# ===================================================================


class TestSixiangPermissionConfig:
    """SixiangPermissionConfig derives correct permission lists."""

    def test_full_config(self, tmp_path: Path) -> None:
        """All fields populated produces expected permission lists."""
        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        ws_root = tmp_path
        cog_dirs = [str(tmp_path / "skills"), str(tmp_path / "models")]

        cfg = SixiangPermissionConfig(
            task_dir=str(task_dir),
            goal_dir=str(goal_dir),
            workspace_root=str(ws_root),
            cognition_dirs=cog_dirs,
        )

        assert task_dir in cfg.read_allowed_dirs
        assert goal_dir in cfg.read_allowed_dirs
        assert Path(tmp_path / "skills") in cfg.read_allowed_dirs
        assert Path(tmp_path / "models") in cfg.read_allowed_dirs

        assert task_dir in cfg.write_allowed_dirs
        assert goal_dir in cfg.write_allowed_dirs
        assert Path(tmp_path / "skills") not in cfg.write_allowed_dirs

        assert task_dir in cfg.exec_allowed_cwds
        assert goal_dir in cfg.exec_allowed_cwds

        assert cfg.yin_workspace_root == ws_root

    def test_empty_config(self) -> None:
        """Empty config returns empty permission lists."""
        cfg = SixiangPermissionConfig()
        assert cfg.read_allowed_dirs == []
        assert cfg.write_allowed_dirs == []
        assert cfg.exec_allowed_cwds == []
        assert cfg.yin_workspace_root is None

    def test_task_dir_only(self, tmp_path: Path) -> None:
        """Config with only task_dir — goal_dir is not included."""
        cfg = SixiangPermissionConfig(task_dir=str(tmp_path))
        assert cfg.read_allowed_dirs == [tmp_path]
        assert cfg.write_allowed_dirs == [tmp_path]
        assert cfg.exec_allowed_cwds == [tmp_path]
        assert cfg.yin_workspace_root is None

    def test_cognition_dirs_not_in_write(self, tmp_path: Path) -> None:
        """Cognition directories are read-only — not in write_allowed."""
        cfg = SixiangPermissionConfig(
            cognition_dirs=[str(tmp_path / "cog")],
        )
        assert Path(tmp_path / "cog") in cfg.read_allowed_dirs
        assert Path(tmp_path / "cog") not in cfg.write_allowed_dirs
        assert Path(tmp_path / "cog") not in cfg.exec_allowed_cwds


# ===================================================================
# _edit_file — unit tests for surgical file editing
# ===================================================================


class TestEditFile:
    """_edit_file: surgical find-and-replace on existing files."""

    def _call_edit(
        self, path: str, old_string: str, new_string: str,
        task_dir: str | Path | None = None,
        write_allowed_dirs: list[Path] | None = None,
    ) -> str:
        from vingobot.core.tool_executor import _edit_file

        return _edit_file(
            {"path": path, "old_string": old_string, "new_string": new_string},
            task_dir=task_dir,
            write_allowed_dirs=write_allowed_dirs,
        )

    def test_basic_edit(self, tmp_path: Path) -> None:
        """Replace first occurrence of old_string in a file."""
        f = tmp_path / "notes.txt"
        f.write_text("hello world\nhello again")
        result = self._call_edit(str(f), old_string="hello", new_string="hi", task_dir=tmp_path)
        assert "编辑成功" in result
        assert f.read_text() == "hi world\nhello again"

    def test_old_string_not_found(self, tmp_path: Path) -> None:
        """Error returned when old_string is not present."""
        f = tmp_path / "data.txt"
        f.write_text("some content")
        result = self._call_edit(str(f), old_string="nonexistent", new_string="x", task_dir=tmp_path)
        assert "[错误]" in result
        assert "未找到" in result

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Error returned when file does not exist."""
        result = self._call_edit(
            str(tmp_path / "missing.txt"), old_string="x", new_string="y", task_dir=tmp_path,
        )
        assert "[错误]" in result
        assert "不存在" in result

    def test_edit_via_write_allowed_dirs(self, tmp_path: Path) -> None:
        """Edit a file under goal_dir via write_allowed_dirs."""
        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()

        f = goal_dir / "blueprint.md"
        f.write_text("# Blueprint v1")

        result = self._call_edit(
            str(f), old_string="v1", new_string="v2",
            task_dir=task_dir,
            write_allowed_dirs=[task_dir, goal_dir],
        )
        assert "编辑成功" in result
        assert f.read_text() == "# Blueprint v2"

    def test_edit_outside_write_allowed_raises(self, tmp_path: Path) -> None:
        """Edit outside write_allowed_dirs raises ValueError."""
        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()

        outside = tmp_path.parent / "outside.txt"
        outside.write_text("should not be editable")

        with pytest.raises(ValueError, match="路径超出允许范围"):
            self._call_edit(
                str(outside), old_string="should", new_string="nope",
                task_dir=task_dir,
                write_allowed_dirs=[task_dir, goal_dir],
            )

    def test_missing_path_param(self, tmp_path: Path) -> None:
        """Missing path parameter returns error."""
        from vingobot.core.tool_executor import _edit_file

        result = _edit_file({"old_string": "x", "new_string": "y"}, task_dir=tmp_path)
        assert "[错误]" in result
        assert "缺少 path" in result

    def test_missing_old_string_param(self, tmp_path: Path) -> None:
        """Missing old_string parameter returns error."""
        from vingobot.core.tool_executor import _edit_file

        f = tmp_path / "test.txt"
        f.write_text("content")
        result = _edit_file({"path": str(f), "new_string": "y"}, task_dir=tmp_path)
        assert "[错误]" in result
        assert "缺少 old_string" in result


# ===================================================================
# execute_builtin_tool — write_allowed_dirs & exec_allowed_cwds
# ===================================================================


class TestExecuteBuiltinWithPermissions:
    """execute_builtin_tool with new write_allowed_dirs and exec_allowed_cwds."""

    @pytest.mark.asyncio
    async def test_write_file_to_goal_dir(self, tmp_path: Path) -> None:
        """write_file to goal_dir succeeds with write_allowed_dirs."""
        from vingobot.core.tool_executor import execute_builtin_tool

        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()

        target = goal_dir / "goal_file.txt"
        result = await execute_builtin_tool(
            "write_file",
            {"path": str(target), "content": "goal data"},
            task_dir=task_dir,
            write_allowed_dirs=[task_dir, goal_dir],
        )
        assert "写入成功" in result
        assert target.read_text() == "goal data"

    @pytest.mark.asyncio
    async def test_edit_file_in_goal_dir(self, tmp_path: Path) -> None:
        """edit_file in goal_dir succeeds with write_allowed_dirs."""
        from vingobot.core.tool_executor import execute_builtin_tool

        goal_dir = tmp_path / "goal"
        goal_dir.mkdir()
        target = goal_dir / "plan.md"
        target.write_text("## Step 1")

        result = await execute_builtin_tool(
            "edit_file",
            {"path": str(target), "old_string": "Step 1", "new_string": "Step 2"},
            task_dir=tmp_path,
            write_allowed_dirs=[tmp_path, goal_dir],
        )
        assert "编辑成功" in result
        assert target.read_text() == "## Step 2"

    @pytest.mark.asyncio
    async def test_exec_in_goal_dir(self, tmp_path: Path) -> None:
        """Exec with cwd=goal_dir succeeds with exec_allowed_cwds."""
        from vingobot.core.tool_executor import execute_builtin_tool

        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()

        result = await execute_builtin_tool(
            "exec",
            {"command": "echo hello", "cwd": str(goal_dir)},
            task_dir=task_dir,
            exec_allowed_cwds=[task_dir, goal_dir],
        )
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_exec_outside_allowed_cwd_blocked(self, tmp_path: Path) -> None:
        """Exec with cwd outside allowed_cwds is blocked."""
        from vingobot.core.tool_executor import execute_builtin_tool

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        outside = tmp_path.parent

        result = await execute_builtin_tool(
            "exec",
            {"command": "echo hello", "cwd": str(outside)},
            task_dir=task_dir,
            exec_allowed_cwds=[task_dir],
        )
        assert "[错误]" in result
        assert "超出允许范围" in result


# ===================================================================
# execute_tool_calls — perm parameter integration
# ===================================================================


class TestExecuteToolCallsWithPerm:
    """execute_tool_calls with SixiangPermissionConfig."""

    @pytest.mark.asyncio
    async def test_write_to_goal_dir_succeeds(self, tmp_path: Path) -> None:
        """Write to goal_dir via perm succeeds."""
        from vingobot.goal.executor import execute_tool_calls

        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()

        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            goal_dir=str(goal_dir),
        )

        target = goal_dir / "artifact.txt"
        calls = [
            ApprovedToolCall(
                name="write_file",
                arguments={"path": str(target), "content": "goal artifact"},
            ),
        ]
        results = await execute_tool_calls(calls, perm=perm)
        assert results[0].status == "success"
        assert target.read_text() == "goal artifact"

    @pytest.mark.asyncio
    async def test_edit_file_in_goal_dir_succeeds(self, tmp_path: Path) -> None:
        """Edit file in goal_dir via perm succeeds."""
        from vingobot.goal.executor import execute_tool_calls

        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()

        target = goal_dir / "plan.md"
        target.write_text("## Initial plan")

        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            goal_dir=str(goal_dir),
        )

        calls = [
            ApprovedToolCall(
                name="edit_file",
                arguments={
                    "path": str(target),
                    "old_string": "Initial plan",
                    "new_string": "Updated plan",
                },
            ),
        ]
        results = await execute_tool_calls(calls, perm=perm)
        assert results[0].status == "success"
        assert target.read_text() == "## Updated plan"

    @pytest.mark.asyncio
    async def test_write_outside_all_boundaries_blocked(self, tmp_path: Path) -> None:
        """Write outside all perm boundaries is blocked."""
        from vingobot.goal.executor import execute_tool_calls

        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()

        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            goal_dir=str(goal_dir),
        )

        outside = tmp_path.parent / "malicious.txt"
        calls = [
            ApprovedToolCall(
                name="write_file",
                arguments={"path": str(outside), "content": "pwned"},
            ),
        ]
        results = await execute_tool_calls(calls, perm=perm)
        assert results[0].status == "blocked"
        assert "超出工作区" in results[0].error or "超出允许范围" in results[0].error

    @pytest.mark.asyncio
    async def test_exec_in_goal_dir_succeeds(self, tmp_path: Path) -> None:
        """Exec with cwd=goal_dir via perm succeeds."""
        from vingobot.goal.executor import execute_tool_calls

        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()

        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            goal_dir=str(goal_dir),
        )

        calls = [
            ApprovedToolCall(
                name="exec",
                arguments={"command": "echo goal_works", "cwd": str(goal_dir)},
            ),
        ]
        results = await execute_tool_calls(calls, perm=perm)
        assert results[0].status == "success"
        assert "goal_works" in results[0].output

    @pytest.mark.asyncio
    async def test_exec_outside_all_boundaries_blocked(self, tmp_path: Path) -> None:
        """Exec with cwd outside all perm boundaries is blocked."""
        from vingobot.goal.executor import execute_tool_calls

        task_dir = tmp_path / "task"
        task_dir.mkdir()

        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            workspace_root=str(tmp_path),
        )

        calls = [
            ApprovedToolCall(
                name="exec",
                arguments={"command": "echo evil", "cwd": str(tmp_path.parent)},
            ),
        ]
        results = await execute_tool_calls(calls, perm=perm)
        assert results[0].status == "blocked"
        assert "工作目录超出工作区" in results[0].error

    @pytest.mark.asyncio
    async def test_read_from_cognition_dirs_succeeds(self, tmp_path: Path) -> None:
        """Read from cognition_dirs via perm succeeds."""
        from vingobot.goal.executor import execute_tool_calls

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        cog_dir = tmp_path / "cognition" / "skills"
        cog_dir.mkdir(parents=True)
        skill_file = cog_dir / "guide.md"
        skill_file.write_text("# Skill guide")

        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            cognition_dirs=[str(cog_dir)],
        )

        calls = [
            ApprovedToolCall(
                name="read_file",
                arguments={"path": str(skill_file)},
            ),
        ]
        results = await execute_tool_calls(calls, perm=perm)
        assert results[0].status == "success"
        assert "# Skill guide" in results[0].output

    @pytest.mark.asyncio
    async def test_delete_file_in_goal_dir_succeeds(self, tmp_path: Path) -> None:
        """Delete file in goal_dir via perm succeeds."""
        from vingobot.goal.executor import execute_tool_calls

        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()
        target = goal_dir / "temporary.txt"
        target.write_text("delete me")

        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            goal_dir=str(goal_dir),
        )

        calls = [
            ApprovedToolCall(
                name="delete_file",
                arguments={"path": str(target)},
            ),
        ]
        results = await execute_tool_calls(calls, perm=perm)
        assert results[0].status == "success"
        assert not target.exists()


# ===================================================================
# Legacy compat — old-style params still work
# ===================================================================


class TestLegacyCompat:
    """Old execute_tool_calls signature (without perm) still works."""

    @pytest.mark.asyncio
    async def test_legacy_task_dir_write(self, tmp_path: Path) -> None:
        """Old-style task_dir write still works."""
        from vingobot.goal.executor import execute_tool_calls

        target = tmp_path / "legacy.txt"
        calls = [
            ApprovedToolCall(
                name="write_file",
                arguments={"path": str(target), "content": "legacy"},
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        assert results[0].status == "success"
        assert target.read_text() == "legacy"

    @pytest.mark.asyncio
    async def test_legacy_goal_dir_read(self, tmp_path: Path) -> None:
        """Old-style goal_dir read still works."""
        from vingobot.goal.executor import execute_tool_calls

        goal_dir = tmp_path / "goal"
        goal_dir.mkdir()
        target = goal_dir / "readme.md"
        target.write_text("# Legacy read")

        calls = [
            ApprovedToolCall(
                name="read_file",
                arguments={"path": str(target)},
            ),
        ]
        results = await execute_tool_calls(
            calls, task_dir=tmp_path, goal_dir=goal_dir,
        )
        assert results[0].status == "success"
        assert "# Legacy read" in results[0].output

    @pytest.mark.asyncio
    async def test_legacy_read_outside_blocked(self, tmp_path: Path) -> None:
        """Old-style blocks read outside task_dir + goal_dir."""
        from vingobot.goal.executor import execute_tool_calls

        calls = [
            ApprovedToolCall(
                name="read_file",
                arguments={"path": str(tmp_path.parent / "config.txt")},
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        assert results[0].status == "blocked"
        assert "超出工作区" in results[0].error


# ===================================================================
# edit_file — executor final defense (integration)
# ===================================================================


class TestEditFileExecutorIntegration:
    """edit_file through the full executor path."""

    @pytest.mark.asyncio
    async def test_edit_file_within_task_dir_succeeds(self, tmp_path: Path) -> None:
        """edit_file within task_dir succeeds through executor."""
        from vingobot.goal.executor import execute_tool_calls

        target = tmp_path / "data.txt"
        target.write_text("original content")

        calls = [
            ApprovedToolCall(
                name="edit_file",
                arguments={
                    "path": str(target),
                    "old_string": "original",
                    "new_string": "updated",
                },
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        assert results[0].status == "success"
        assert target.read_text() == "updated content"

    @pytest.mark.asyncio
    async def test_edit_file_outside_task_dir_blocked(self, tmp_path: Path) -> None:
        """edit_file outside task_dir is blocked by executor."""
        from vingobot.goal.executor import execute_tool_calls

        outside = tmp_path.parent / "protected.txt"
        outside.write_text("should not be editable")

        calls = [
            ApprovedToolCall(
                name="edit_file",
                arguments={
                    "path": str(outside),
                    "old_string": "should",
                    "new_string": "nope",
                },
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        assert results[0].status == "blocked"
        assert "超出工作区" in results[0].error or "超出允许范围" in results[0].error

    @pytest.mark.asyncio
    async def test_edit_file_with_goal_dir_via_perm(self, tmp_path: Path) -> None:
        """edit_file in goal_dir via perm succeeds through executor."""
        from vingobot.goal.executor import execute_tool_calls

        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()

        target = goal_dir / "goal_edit.md"
        target.write_text("## Draft")

        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            goal_dir=str(goal_dir),
        )

        calls = [
            ApprovedToolCall(
                name="edit_file",
                arguments={
                    "path": str(target),
                    "old_string": "Draft",
                    "new_string": "Final",
                },
            ),
        ]
        results = await execute_tool_calls(calls, perm=perm)
        assert results[0].status == "success"
        assert target.read_text() == "## Final"

    @pytest.mark.asyncio
    async def test_list_directory_within_goal_dir_via_perm(self, tmp_path: Path) -> None:
        """list_directory in goal_dir via perm succeeds."""
        from vingobot.goal.executor import execute_tool_calls

        task_dir = tmp_path / "task"
        goal_dir = tmp_path / "goal"
        task_dir.mkdir()
        goal_dir.mkdir()
        (goal_dir / "file1.txt").write_text("a")
        (goal_dir / "file2.txt").write_text("b")

        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            goal_dir=str(goal_dir),
        )

        calls = [
            ApprovedToolCall(
                name="list_directory",
                arguments={"path": str(goal_dir)},
            ),
        ]
        results = await execute_tool_calls(calls, perm=perm)
        assert results[0].status == "success"
        assert "file1.txt" in results[0].output
        assert "file2.txt" in results[0].output

    @pytest.mark.asyncio
    async def test_query_capabilities_reflects_permissions(self, tmp_path: Path) -> None:
        """query_capabilities tool works correctly through executor."""
        from vingobot.goal.executor import execute_tool_calls

        calls = [
            ApprovedToolCall(
                name="query_capabilities",
                arguments={},
            ),
        ]
        results = await execute_tool_calls(calls, task_dir=tmp_path)
        assert results[0].status == "success"
        assert "edit_file" in results[0].output
        assert "goal_dir" in results[0].output
