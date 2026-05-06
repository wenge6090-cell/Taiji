"""
认知演化执行器 — Executes Anqu-driven cognitive evolution actions.

When Anqu determines that the cognition layer needs updating, it produces
0-N ``CognitionEvolutionAction`` items.  These are enqueued as pending
tasks under the ``cognition-evolution`` goal.  This module provides the
actual implementation handlers:

- ``create_skill()`` — Create a new L1 skill directory + SKILL.md
- ``create_model()`` — Create a new L2 experience model
- ``create_domain_grid()`` — Create a new L3 domain grid JSON
- ``execute_evolution_action()`` — Unified dispatcher

All handlers check for existing assets before creating new ones, and
update the navigation index after successful creation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from vingobot.goal.grid_types import (
    CognitionEvolutionAction,
    GridFile,
    GridModelRef,
    GridSkillRef,
)

# ---------------------------------------------------------------------------
# L3 Grid creation
# ---------------------------------------------------------------------------


async def create_domain_grid(
    domain: str,
    description: str = "",
    skills: list[GridSkillRef] | None = None,
    models: list[GridModelRef] | None = None,
) -> bool:
    """Create a new L3 domain grid JSON file.

    Args:
        domain: Domain name (also used as the filename stem).
        description: What this grid is for.
        skills: L1 skill references to include.
        models: L2 model references to include.

    Returns:
        True if a new grid was created, False if it already existed.
    """
    from vingobot.core.workspace import get_workspace_paths
    from vingobot.goal.cognition_tools import save_grid, update_nav_index

    wp = get_workspace_paths()
    grid_path = wp.grids / f"{domain}.json"

    if grid_path.exists():
        logger.info("[认知演化] 格栅已存在，跳过创建: {}", domain)
        return False

    grid = GridFile(
        domain=domain,
        description=description,
        version="1.0",
        last_used=datetime.now(timezone.utc).isoformat(),
        skills=skills or [],
        models=models or [],
    )

    save_grid(grid)
    logger.info("[认知演化] 创建 L3 领域格栅: {}", domain)

    # Update navigation index
    try:
        update_nav_index()
    except Exception:
        pass

    return True


# ---------------------------------------------------------------------------
# L1 Skill creation
# ---------------------------------------------------------------------------


async def create_skill(
    name: str,
    description: str = "",
    tools: list[dict[str, Any]] | None = None,
) -> bool:
    """Create a new L1 skill directory with SKILL.md.

    Args:
        name: Skill directory name (snake_case).
        description: Short description of what the skill does.
        tools: Optional list of tool definitions to include in the
            SKILL.md frontmatter. Each tool dict must have ``name``
            and ``description``, optionally ``parameters`` (dict of
            ``{param_name: {type, description}}``) and
            ``tool_auto_approve`` (bool).

    Returns:
        True if a new skill was created, False if it already existed.
    """
    from vingobot.core.workspace import get_workspace_paths

    wp = get_workspace_paths()
    skill_dir = wp.skills / name

    if skill_dir.is_dir():
        logger.info("[认知演化] Skill 已存在，跳过创建: {}", name)
        return False

    skill_dir.mkdir(parents=True, exist_ok=True)

    # Build tools YAML block
    tools_yaml_lines: list[str] = []
    if tools:
        tools_yaml_lines.append("tools:")
        for t in tools:
            tool_name = t.get("name", "")
            tool_desc = t.get("description", "")
            tools_yaml_lines.append(f"  - name: {tool_name}")
            tools_yaml_lines.append(f'    description: "{tool_desc}"')
            params = t.get("parameters", {})
            if params:
                tools_yaml_lines.append("    parameters:")
                for pname, pinfo in params.items():
                    ptype = pinfo.get("type", "string") if isinstance(pinfo, dict) else "string"
                    pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else str(pinfo)
                    tools_yaml_lines.append(f"      {pname}:")
                    tools_yaml_lines.append(f"        type: {ptype}")
                    tools_yaml_lines.append(f'        description: "{pdesc}"')
            auto_approve = t.get("tool_auto_approve", False)
            if auto_approve:
                tools_yaml_lines.append("    tool_auto_approve: true")

    tools_yaml = "\n".join(tools_yaml_lines)

    skill_md_content = f"""---
name: {name}
version: 1.0
description: "{description}"
{tools_yaml}
---

# {name}

{description}

> 由认知演化系统于 {datetime.now(timezone.utc).isoformat()} 自动创建。
"""
    (skill_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")
    logger.info("[认知演化] 创建 L1 Skill: {}", name)
    return True


# ---------------------------------------------------------------------------
# L2 Model creation
# ---------------------------------------------------------------------------


async def create_model(
    name: str,
    content: str = "",
) -> bool:
    """Create a new L2 experience model markdown file.

    Args:
        name: Model name (also used as the filename stem).
        content: Markdown content describing the model/pattern.

    Returns:
        True if a new model was created, False if it already existed.
    """
    from vingobot.core.workspace import get_workspace_paths

    wp = get_workspace_paths()
    model_path = wp.models / f"{name}.md"

    if model_path.exists():
        logger.info("[认知演化] 模型已存在，跳过创建: {}", name)
        return False

    model_path.write_text(
        f"# {name}\n\n{content}\n\n"
        f"---\n"
        f"_由认知演化系统于 {datetime.now(timezone.utc).isoformat()} 自动创建_\n",
        encoding="utf-8",
    )
    logger.info("[认知演化] 创建 L2 思维模型: {}", name)
    return True


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


async def execute_evolution_action(ea: CognitionEvolutionAction) -> bool:
    """Execute a single cognitive evolution action.

    Dispatches to the appropriate creator based on ``action`` type.
    Returns ``True`` if an asset was created or updated.

    Args:
        ea: The evolution action to execute.

    Returns:
        True if the action resulted in a new/updated cognitive asset.
    """
    action = ea.action
    target = ea.target_name
    ctx = ea.context or {}

    if action == "learn_skill":
        return await create_skill(
            name=target,
            description=ctx.get("description", ea.description),
            tools=ctx.get("suggested_tools"),
        )

    if action == "precipitate_skill":
        return await create_skill(
            name=target,
            description=ctx.get("description", ea.description),
            tools=ctx.get("suggested_tools"),
        )

    if action == "precipitate_model":
        return await create_model(
            name=target,
            content=ctx.get("insight", ea.description),
        )

    if action == "create_grid":
        return await create_domain_grid(
            domain=target,
            description=ea.description,
            skills=ctx.get("skills"),
            models=ctx.get("models"),
        )

    logger.warning("[认知演化] 未知动作类型: {}", action)
    return False
