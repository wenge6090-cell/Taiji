"""Integration tests for the goal system — SKILL.md → discovery → registration → routing → approval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vingobot.goal.executor import _try_skill_tool
from vingobot.goal.skill_parser import (
    SkillMeta,
    SkillToolDef,
    SkillToolParam,
    get_skill_tool,
    parse_skill_md,
    register_skill_tool,
    register_skill_tools_from_meta,
)
from vingobot.goal.yin import _check_skill_tool

# ── Helpers ───────────────────────────────────────────────────────────────────


def _clear_registry() -> None:
    """Clear the skill tool registry."""
    from vingobot.goal.skill_parser import _skill_tool_registry

    _skill_tool_registry.clear()


def _create_minimal_skill_md(tmp_path: Path, name: str = "integration-skill") -> Path:
    """Create a minimal SKILL.md for integration testing."""
    md = tmp_path / "SKILL.md"
    md.write_text(
        f"""---
name: {name}
version: "1.0"
description: Integration test skill
tools:
  - name: greet
    description: Greet the user
    parameters:
      - name: name
        type: string
        description: Name to greet
        required: true
    tool_auto_approve: true
  - name: dangerous_op
    description: A dangerous operation
    parameters:
      - name: cmd
        type: string
        description: Command to run
        required: true
    tool_auto_approve: false
---

# {name}
""",
        encoding="utf-8",
    )
    return md


# ── Flow: Parse → Register → Retrieve ───────────────────────────────────────


class TestParseRegisterRetrieve:
    """Tests the full parse → register → retrieve cycle."""

    def teardown_method(self) -> None:
        _clear_registry()

    def test_full_cycle(self, tmp_path: Path) -> None:
        """SKILL.md is parsed, tools registered, tools retrieved."""
        skill_md = _create_minimal_skill_md(tmp_path)
        meta = parse_skill_md(skill_md)
        assert meta is not None
        assert meta.name == "integration-skill"

        register_skill_tools_from_meta(meta)

        greet = get_skill_tool("greet")
        assert greet is not None
        assert greet.tool_auto_approve is True

        dangerous = get_skill_tool("dangerous_op")
        assert dangerous is not None
        assert dangerous.tool_auto_approve is False

    def test_register_then_parse_overwrites(self, tmp_path: Path) -> None:
        """Manually registering a tool, then parsing, overwrites correctly."""
        # Manually register a tool
        manual = SkillToolDef(
            name="greet",
            description="Manual greet",
            tool_auto_approve=False,
        )
        register_skill_tool(manual)

        # Parse SKILL.md — should overwrite
        skill_md = _create_minimal_skill_md(tmp_path)
        meta = parse_skill_md(skill_md)
        register_skill_tools_from_meta(meta)

        greet = get_skill_tool("greet")
        assert greet is not None
        assert greet.description == "Greet the user"  # overwritten


# ── Flow: Executor routing for skill tools ───────────────────────────────────


class TestExecutorSkillToolRouting:
    """Tests that executor routes skill tool calls correctly."""

    def teardown_method(self) -> None:
        _clear_registry()

    async def test_executor_routes_to_skill_tool(self) -> None:
        """Executor falls through when ToolRegistry is not set."""
        tool = SkillToolDef(
            name="echo",
            description="Echo the input",
            parameters=[
                SkillToolParam(name="message", type="string", description="Message to echo", required=True),
            ],
        )
        register_skill_tool(tool)

        # Without a real ToolRegistry, _try_skill_tool returns None so the
        # call falls through to the built-in fallback (_builtin_execute).
        result = await _try_skill_tool("echo", None, {"message": "hello"})
        assert result is None

    async def test_executor_skips_unregistered(self) -> None:
        """Executor returns None for unregistered skill tool."""
        result = await _try_skill_tool("nonexistent", None, {})
        assert result is None


# ── Flow: Yin approval for skill tools ───────────────────────────────────────


class TestYinSkillToolApproval:
    """Tests that Yin approves/denies skill tools correctly."""

    def teardown_method(self) -> None:
        _clear_registry()

    def test_auto_approved_tool(self) -> None:
        """Tool with tool_auto_approve: true is auto-approved."""
        tool = SkillToolDef(
            name="safe_tool",
            description="Safe tool",
            tool_auto_approve=True,
        )
        register_skill_tool(tool)

        approved, reason = _check_skill_tool("safe_tool")
        assert approved is True
        assert reason == "auto_approve"

    def test_non_auto_approved_tool(self) -> None:
        """Tool without auto_approve returns auto_approve_false."""
        tool = SkillToolDef(
            name="risky_tool",
            description="Risky tool",
            tool_auto_approve=False,
        )
        register_skill_tool(tool)

        approved, reason = _check_skill_tool("risky_tool")
        assert approved is False
        assert reason == "auto_approve_false"

    def test_unregistered_tool(self) -> None:
        """Unregistered tool returns not_found."""
        approved, reason = _check_skill_tool("nonexistent")
        assert approved is False
        assert reason == "not_found"


# ── Full end-to-end: SKILL.md → Weaver → Executor → Yin ──────────────────────


class TestEndToEnd:
    """Complete flow: SKILL.md discovery through Yin approval."""

    def teardown_method(self) -> None:
        _clear_registry()

    def test_auto_approved_tool_flow(self, tmp_path: Path) -> None:
        """Auto-approved tool goes through parse → register → Yin auto-approve."""
        skill_md = _create_minimal_skill_md(tmp_path, "e2e-skill")
        meta = parse_skill_md(skill_md)
        assert meta is not None
        register_skill_tools_from_meta(meta)

        # Yin check for auto-approved tool
        approved, reason = _check_skill_tool("greet")
        assert approved is True
        assert reason == "auto_approve"

        # OpenAI schema generation
        greet = get_skill_tool("greet")
        assert greet is not None
        schema = greet.to_openai_tool_def()
        assert schema["function"]["name"] == "greet"
        assert "name" in schema["function"]["parameters"]["properties"]

    def test_non_auto_approved_tool_flow(self, tmp_path: Path) -> None:
        """Non-auto-approved tool goes through parse → register → Yin side-effect path."""
        skill_md = _create_minimal_skill_md(tmp_path, "e2e-skill-2")
        meta = parse_skill_md(skill_md)
        assert meta is not None
        register_skill_tools_from_meta(meta)

        # Yin check for non-auto-approved tool
        approved, reason = _check_skill_tool("dangerous_op")
        assert approved is False
        assert reason == "auto_approve_false"  # will go through side-effect check

    def test_weaver_discovers_tools(self, tmp_path: Path) -> None:
        """Weaver's _load_skill_tools discovers and registers tools from SKILL.md."""
        from vingobot.goal.weaver import _load_skill_tools

        skill_md = _create_minimal_skill_md(tmp_path, "weaver-skill")
        skill_dir = skill_md.parent

        tool_defs = _load_skill_tools("weaver-skill", skill_dir)
        assert len(tool_defs) == 2

        # Verify OpenAI schema format
        assert tool_defs[0]["type"] == "function"
        assert tool_defs[0]["function"]["name"] == "greet"
        assert tool_defs[1]["function"]["name"] == "dangerous_op"

        # Verify tools are registered in the global registry
        greet = get_skill_tool("greet")
        assert greet is not None
        assert greet.tool_auto_approve is True

    def test_weaver_returns_empty_for_no_skill(self, tmp_path: Path) -> None:
        """Weaver returns empty list when no SKILL.md exists."""
        from vingobot.goal.weaver import _load_skill_tools

        tool_defs = _load_skill_tools("empty-skill", tmp_path)
        assert tool_defs == []

    def test_weaver_loads_skill_tools_from_skill_dir(self, tmp_path: Path) -> None:
        """Weaver discovers tools from nested skill directories."""
        # Create a skill directory structure
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: test-skill
version: "1.0"
description: A nested skill
tools:
  - name: nested_tool
    description: A tool in a nested dir
    parameters:
      - name: input
        type: string
        description: Input
        required: true
    tool_auto_approve: true
---

# Nested Skill
""",
            encoding="utf-8",
        )

        from vingobot.goal.weaver import _load_skill_tools

        tool_defs = _load_skill_tools("test-skill", skill_dir)
        assert len(tool_defs) == 1
        assert tool_defs[0]["function"]["name"] == "nested_tool"

        # Also check registry
        nested = get_skill_tool("nested_tool")
        assert nested is not None
