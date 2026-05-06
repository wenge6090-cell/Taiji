"""
SKILL.md frontmatter parser — L1 Skill auto-registration.

Each L1 Skill in ``<workspace>/cognition/skills/<name>/`` has a
``SKILL.md`` file.  The YAML frontmatter (between ``---`` markers)
declares tool schemas that Weaver can discover and register as
OpenAI Function Calling tools for Yang to invoke directly.

Example SKILL.md::

    ---
    name: browser-automation
    version: 1.0
    description: "Browser automation with Playwright"
    tools:
      - name: browser_navigate
        description: "Navigate to a URL"
        parameters:
          url:
            type: string
            description: "Target URL"
        tool_auto_approve: true
      - name: browser_screenshot
        description: "Take a screenshot of current page"
        parameters:
          path:
            type: string
            description: "Save path"
        tool_auto_approve: true
    ---

    # browser-automation

    Full description and usage guide...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# Regex to extract YAML frontmatter between --- markers
_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SkillToolParam:
    """A single parameter of a skill-registered tool."""

    name: str
    type: str = "string"  # "string" | "integer" | "boolean" | "number"
    description: str = ""
    required: bool = True


@dataclass
class SkillToolDef:
    """A tool definition parsed from SKILL.md frontmatter."""

    name: str
    description: str = ""
    parameters: list[SkillToolParam] = field(default_factory=list)
    tool_auto_approve: bool = False
    executor: Any = None
    """Optional async callable (``fn(**kwargs) -> str``) that implements
    this skill tool's execution logic.  When set, the Executor routes
    calls directly to this function instead of falling back to builtins."""

    def to_openai_tool_def(self) -> dict[str, Any]:
        """Convert to an OpenAI Function Calling tool definition dict."""
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


@dataclass
class SkillMeta:
    """Parsed metadata from a single SKILL.md file."""

    name: str
    version: str = "1.0"
    description: str = ""
    tools: list[SkillToolDef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_skill_md(path: str | Path) -> SkillMeta | None:
    """Parse a SKILL.md file and return its structured metadata.

    Args:
        path: Absolute or relative path to a ``SKILL.md`` file.

    Returns:
        ``SkillMeta`` if the file exists and has valid frontmatter,
        ``None`` if the file is missing or has no frontmatter.
    """
    p = Path(path)
    if not p.is_file():
        logger.debug("[技能解析] 文件不存在: {}", p)
        return None

    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("[技能解析] 读取失败 {}: {}", p, exc)
        return None

    m = _FRONTMATTER_RE.match(text)
    if not m:
        logger.debug("[技能解析] 无 frontmatter: {}", p.name)
        return None

    try:
        import yaml

        raw = yaml.safe_load(m.group(1))
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    tools_raw = raw.get("tools", []) or []
    tools = []
    for t in tools_raw:
        if not isinstance(t, dict):
            continue
        params_raw = t.get("parameters", {}) or {}
        params = _parse_parameters(params_raw)
        tools.append(
            SkillToolDef(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=params,
                tool_auto_approve=t.get("tool_auto_approve", False),
            )
        )

    return SkillMeta(
        name=raw.get("name", p.parent.name),
        version=str(raw.get("version", "1.0")),
        description=raw.get("description", ""),
        tools=tools,
    )


def _parse_parameters(raw: Any) -> list[SkillToolParam]:
    """Parse parameters from either list or dict format.

    Supports two YAML formats:

    **List format** (recommended for explicit control)::

        parameters:
          - name: input
            type: string
            description: "The input"
            required: true

    **Dict format** (shorthand)::

        parameters:
          input:
            type: string
            description: "The input"
    """
    if isinstance(raw, list):
        result: list[SkillToolParam] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            result.append(
                SkillToolParam(
                    name=item.get("name", ""),
                    type=item.get("type", "string"),
                    description=item.get("description", ""),
                    required=item.get("required", True),
                )
            )
        return result

    if isinstance(raw, dict):
        return [
            SkillToolParam(
                name=k,
                type=v.get("type", "string") if isinstance(v, dict) else "string",
                description=v.get("description", "") if isinstance(v, dict) else str(v),
            )
            for k, v in raw.items()
        ]

    return []


# ---------------------------------------------------------------------------
# Registry (runtime in-memory)
# ---------------------------------------------------------------------------

_skill_tool_registry: dict[str, SkillToolDef] = {}


def register_skill_tool(tool_def: SkillToolDef) -> None:
    """Register a single skill tool definition for executor routing."""
    _skill_tool_registry[tool_def.name] = tool_def


def get_skill_tool(name: str) -> SkillToolDef | None:
    """Look up a registered skill tool by name."""
    return _skill_tool_registry.get(name)


def register_skill_executor(name: str, fn: Any) -> None:
    """Register an async callable as the executor for a skill tool.

    Args:
        name: Skill tool name (must already be registered).
        fn: An async callable ``fn(**kwargs) -> str``.
    """
    tool = _skill_tool_registry.get(name)
    if tool is not None:
        tool.executor = fn


def get_skill_executor(name: str) -> Any | None:
    """Return the registered async executor for a skill tool, or None."""
    tool = _skill_tool_registry.get(name)
    return tool.executor if tool is not None else None


def register_skill_tools_from_meta(meta: SkillMeta) -> None:
    """Register all tools from a ``SkillMeta`` into the global registry."""
    for t in meta.tools:
        register_skill_tool(t)


def clear_skill_tool_registry() -> None:
    """Clear all registered skill tools (useful for testing)."""
    _skill_tool_registry.clear()


def list_registered_skill_tools() -> list[str]:
    """Return names of all currently registered skill tools."""
    return list(_skill_tool_registry.keys())
