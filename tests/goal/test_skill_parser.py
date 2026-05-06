"""Tests for vingobot.goal.skill_parser — SKILL.md frontmatter parsing and tool registration."""

from __future__ import annotations

from pathlib import Path

import pytest

from vingobot.goal.skill_parser import (
    SkillMeta,
    SkillToolDef,
    SkillToolParam,
    get_skill_tool,
    parse_skill_md,
    register_skill_tool,
    register_skill_tools_from_meta,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_skill_md(tmp_path: Path) -> Path:
    """Creates a minimal SKILL.md with one tool."""
    md = tmp_path / "SKILL.md"
    md.write_text(
        """---
name: test-skill
version: "1.0"
description: A test skill
tools:
  - name: test_tool
    description: A test tool
    parameters:
      - name: input
        type: string
        description: The input value
        required: true
    tool_auto_approve: true
---

# Test Skill

This is a test skill.
""",
        encoding="utf-8",
    )
    return md


@pytest.fixture
def skill_md_no_tools(tmp_path: Path) -> Path:
    """SKILL.md with no tools section."""
    md = tmp_path / "SKILL.md"
    md.write_text(
        """---
name: no-tools-skill
version: "1.0"
description: A skill with no tools
---

# No Tools Skill
""",
        encoding="utf-8",
    )
    return md


@pytest.fixture
def skill_md_multi_tool(tmp_path: Path) -> Path:
    """SKILL.md with multiple tools and complex parameters."""
    md = tmp_path / "SKILL.md"
    md.write_text(
        """---
name: multi-tool-skill
version: "2.0"
description: A skill with multiple tools
tools:
  - name: read_data
    description: Read data from storage
    parameters:
      - name: key
        type: string
        description: The storage key
        required: true
    tool_auto_approve: true
  - name: write_data
    description: Write data to storage
    parameters:
      - name: key
        type: string
        description: The storage key
        required: true
      - name: value
        type: string
        description: The value to write
        required: true
    tool_auto_approve: false
  - name: delete_data
    description: Delete data from storage
    parameters:
      - name: key
        type: string
        description: The storage key
        required: true
---

# Multi-Tool Skill
""",
        encoding="utf-8",
    )
    return md


@pytest.fixture
def skill_md_no_frontmatter(tmp_path: Path) -> Path:
    """SKILL.md with no YAML frontmatter."""
    md = tmp_path / "SKILL.md"
    md.write_text(
        "# Just a heading\n\nNo frontmatter here.\n",
        encoding="utf-8",
    )
    return md


@pytest.fixture
def skill_md_empty(tmp_path: Path) -> Path:
    """Empty SKILL.md."""
    md = tmp_path / "SKILL.md"
    md.write_text("", encoding="utf-8")
    return md


# ── parse_skill_md tests ─────────────────────────────────────────────────────


class TestParseSkillMd:
    """Tests for parse_skill_md()."""

    def test_parse_basic_skill(self, sample_skill_md: Path) -> None:
        """Parses a SKILL.md with one tool correctly."""
        meta = parse_skill_md(sample_skill_md)
        assert meta is not None
        assert meta.name == "test-skill"
        assert meta.version == "1.0"
        assert meta.description == "A test skill"
        assert len(meta.tools) == 1

        tool = meta.tools[0]
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.tool_auto_approve is True
        assert len(tool.parameters) == 1

        param = tool.parameters[0]
        assert param.name == "input"
        assert param.type == "string"
        assert param.description == "The input value"
        assert param.required is True

    def test_parse_no_tools(self, skill_md_no_tools: Path) -> None:
        """Parses a SKILL.md with no tools section."""
        meta = parse_skill_md(skill_md_no_tools)
        assert meta is not None
        assert meta.name == "no-tools-skill"
        assert len(meta.tools) == 0

    def test_parse_multi_tool(self, skill_md_multi_tool: Path) -> None:
        """Parses a SKILL.md with multiple tools."""
        meta = parse_skill_md(skill_md_multi_tool)
        assert meta is not None
        assert meta.name == "multi-tool-skill"
        assert meta.version == "2.0"
        assert len(meta.tools) == 3

        # read_data — auto_approve
        t0 = meta.tools[0]
        assert t0.name == "read_data"
        assert t0.tool_auto_approve is True
        assert len(t0.parameters) == 1

        # write_data — no auto_approve, 2 params
        t1 = meta.tools[1]
        assert t1.name == "write_data"
        assert t1.tool_auto_approve is False
        assert len(t1.parameters) == 2

        # delete_data — auto_approve default (False), 1 param
        t2 = meta.tools[2]
        assert t2.name == "delete_data"
        assert t2.tool_auto_approve is False
        assert len(t2.parameters) == 1

    def test_parse_no_frontmatter(self, skill_md_no_frontmatter: Path) -> None:
        """Returns None when SKILL.md has no frontmatter."""
        meta = parse_skill_md(skill_md_no_frontmatter)
        assert meta is None

    def test_parse_empty_file(self, skill_md_empty: Path) -> None:
        """Returns None for empty file."""
        meta = parse_skill_md(skill_md_empty)
        assert meta is None

    def test_parse_nonexistent_file(self, tmp_path: Path) -> None:
        """Returns None when file does not exist."""
        meta = parse_skill_md(tmp_path / "NONEXISTENT.md")
        assert meta is None

    def test_parse_directory(self, tmp_path: Path) -> None:
        """Returns None when path is a directory."""
        meta = parse_skill_md(tmp_path)
        assert meta is None


# ── SkillToolDef.to_openai_tool_def tests ────────────────────────────────────


class TestSkillToolDefToOpenai:
    """Tests for SkillToolDef.to_openai_tool_def()."""

    def test_basic_conversion(self) -> None:
        """Converts a basic tool to OpenAI function calling schema."""
        tool = SkillToolDef(
            name="test_tool",
            description="A test tool",
            parameters=[
                SkillToolParam(name="input", type="string", description="Input value", required=True),
            ],
            tool_auto_approve=True,
        )
        result = tool.to_openai_tool_def()

        assert result["type"] == "function"
        assert result["function"]["name"] == "test_tool"
        assert result["function"]["description"] == "A test tool"
        assert result["function"]["parameters"]["type"] == "object"
        assert "input" in result["function"]["parameters"]["properties"]
        assert result["function"]["parameters"]["required"] == ["input"]

    def test_conversion_no_required_params(self) -> None:
        """Omits 'required' field when no parameters are required."""
        tool = SkillToolDef(
            name="optional_tool",
            description="A tool with optional params only",
            parameters=[
                SkillToolParam(name="opt", type="string", description="Optional", required=False),
            ],
        )
        result = tool.to_openai_tool_def()

        # required should be an empty list if no required params
        assert result["function"]["parameters"]["required"] == []

    def test_conversion_no_parameters(self) -> None:
        """Handles tools with no parameters."""
        tool = SkillToolDef(
            name="noop",
            description="Does nothing",
            parameters=[],
        )
        result = tool.to_openai_tool_def()

        assert result["function"]["parameters"]["type"] == "object"
        assert result["function"]["parameters"]["properties"] == {}
        assert result["function"]["parameters"]["required"] == []

    def test_conversion_mixed_types(self) -> None:
        """Handles different parameter types."""
        tool = SkillToolDef(
            name="typed_tool",
            description="A tool with various types",
            parameters=[
                SkillToolParam(name="s", type="string", description="String param", required=True),
                SkillToolParam(name="n", type="number", description="Number param", required=True),
                SkillToolParam(name="b", type="boolean", description="Boolean param", required=False),
                SkillToolParam(name="a", type="array", description="Array param", required=False),
            ],
        )
        result = tool.to_openai_tool_def()
        props = result["function"]["parameters"]["properties"]

        assert props["s"]["type"] == "string"
        assert props["n"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["a"]["type"] == "array"
        assert result["function"]["parameters"]["required"] == ["s", "n"]


# ── Registration tests ───────────────────────────────────────────────────────


class TestRegistration:
    """Tests for register_skill_tool and get_skill_tool."""

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        from vingobot.goal.skill_parser import _skill_tool_registry

        _skill_tool_registry.clear()

    def test_register_and_get(self) -> None:
        """Registers and retrieves a tool."""
        tool = SkillToolDef(
            name="my_tool",
            description="My tool",
            parameters=[SkillToolParam(name="x", type="string", description="X", required=True)],
        )
        register_skill_tool(tool)

        retrieved = get_skill_tool("my_tool")
        assert retrieved is not None
        assert retrieved.name == "my_tool"

    def test_register_twice_overwrites(self) -> None:
        """Registering a tool with the same name overwrites."""
        tool1 = SkillToolDef(name="same_name", description="First")
        tool2 = SkillToolDef(name="same_name", description="Second")
        register_skill_tool(tool1)
        register_skill_tool(tool2)

        retrieved = get_skill_tool("same_name")
        assert retrieved is not None
        assert retrieved.description == "Second"  # overwritten

    def test_get_nonexistent(self) -> None:
        """Returns None for unregistered tool."""
        assert get_skill_tool("nonexistent") is None


class TestRegisterFromMeta:
    """Tests for register_skill_tools_from_meta."""

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        from vingobot.goal.skill_parser import _skill_tool_registry

        _skill_tool_registry.clear()

    def test_register_multiple(self) -> None:
        """Registers all tools from a SkillMeta."""
        meta = SkillMeta(
            name="test-skill",
            tools=[
                SkillToolDef(name="tool_a", description="Tool A"),
                SkillToolDef(name="tool_b", description="Tool B"),
                SkillToolDef(name="tool_c", description="Tool C"),
            ],
        )
        register_skill_tools_from_meta(meta)

        assert get_skill_tool("tool_a") is not None
        assert get_skill_tool("tool_b") is not None
        assert get_skill_tool("tool_c") is not None

    def test_register_empty(self) -> None:
        """Registers nothing from an empty tools list."""
        meta = SkillMeta(name="empty-skill", tools=[])
        register_skill_tools_from_meta(meta)

        # No crash, nothing registered
        assert get_skill_tool("tool_a") is None


# ── Integration: parse then register ─────────────────────────────────────────


class TestParseThenRegister:
    """End-to-end: parse SKILL.md then register tools."""

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        from vingobot.goal.skill_parser import _skill_tool_registry

        _skill_tool_registry.clear()

    def test_parse_and_register(self, sample_skill_md: Path) -> None:
        """Parses a SKILL.md and registers its tool."""
        meta = parse_skill_md(sample_skill_md)
        assert meta is not None

        register_skill_tools_from_meta(meta)

        tool = get_skill_tool("test_tool")
        assert tool is not None
        assert tool.description == "A test tool"
        assert tool.tool_auto_approve is True

        # Verify OpenAI schema
        schema = tool.to_openai_tool_def()
        assert schema["function"]["name"] == "test_tool"
        assert "input" in schema["function"]["parameters"]["properties"]

    def test_parse_register_multi(self, skill_md_multi_tool: Path) -> None:
        """Parses multi-tool SKILL.md and registers all tools."""
        meta = parse_skill_md(skill_md_multi_tool)
        assert meta is not None

        register_skill_tools_from_meta(meta)

        assert get_skill_tool("read_data") is not None
        assert get_skill_tool("write_data") is not None
        assert get_skill_tool("delete_data") is not None

        # Verify auto_approve flags
        assert get_skill_tool("read_data").tool_auto_approve is True
        assert get_skill_tool("write_data").tool_auto_approve is False
        assert get_skill_tool("delete_data").tool_auto_approve is False
