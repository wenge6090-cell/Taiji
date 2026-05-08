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

import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from vingobot.goal.grid_types import (
    CognitionEvolutionAction,
    GridFile,
    GridModelRef,
    GridSkillRef,
    GridWorkflowStep,
    TruthFile,
)

# ---------------------------------------------------------------------------
# L3 Grid creation
# ---------------------------------------------------------------------------


async def create_domain_grid(
    domain: str,
    description: str = "",
    skills: list[GridSkillRef] | None = None,
    models: list[GridModelRef] | None = None,
    llm_analysis: str = "",
) -> bool:
    """Create a new L3 domain grid JSON file.

    When ``llm_analysis`` is provided, it is parsed as JSON to extract
    full metadata (trigram, workflow, gaps, proficiency, skills with
    relevance, models with relevance).  Falls back to the simple path
    when parsing fails or the string is empty.

    Args:
        domain: Domain name (also used as the filename stem).
        description: What this grid is for.
        skills: L1 skill references to include (fallback when no LLM data).
        models: L2 model references to include (fallback when no LLM data).
        llm_analysis: Optional LLM-generated JSON with full grid structure.

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

    # ── Parse LLM analysis JSON ───────────────────────────────────
    parsed_trigram = ""
    parsed_proficiency = 0.0
    parsed_workflow: list[GridWorkflowStep] = []
    parsed_gaps: list[str] = []
    parsed_skills: list[GridSkillRef] = skills or []
    parsed_models: list[GridModelRef] = models or []

    if llm_analysis:
        try:
            # Strip markdown code fences if present (safety net)
            cleaned = llm_analysis.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            if isinstance(data, dict):
                parsed_trigram = data.get("trigram", "")
                parsed_proficiency = float(data.get("proficiency", 0.0))

                # Skills with relevance
                llm_skills = data.get("skills", [])
                if isinstance(llm_skills, list) and llm_skills:
                    parsed_skills = []
                    for s in llm_skills:
                        if isinstance(s, dict):
                            parsed_skills.append(
                                GridSkillRef(
                                    name=s.get("name", ""),
                                    relevance=s.get("relevance", "supporting"),
                                )
                            )
                        elif isinstance(s, str):
                            parsed_skills.append(GridSkillRef(name=s))

                # Models with relevance
                llm_models = data.get("models", [])
                if isinstance(llm_models, list) and llm_models:
                    parsed_models = []
                    for m in llm_models:
                        if isinstance(m, dict):
                            parsed_models.append(
                                GridModelRef(
                                    name=m.get("name", ""),
                                    relevance=m.get("relevance", "supporting"),
                                )
                            )
                        elif isinstance(m, str):
                            parsed_models.append(GridModelRef(name=m))

                # Workflow
                llm_workflow = data.get("workflow", [])
                if isinstance(llm_workflow, list):
                    for w in llm_workflow:
                        if isinstance(w, dict):
                            parsed_workflow.append(
                                GridWorkflowStep(
                                    step=int(w.get("step", 0)),
                                    description=w.get("description", ""),
                                    skills=w.get("skills", []),
                                )
                            )

                # Gaps
                llm_gaps = data.get("gaps", [])
                if isinstance(llm_gaps, list):
                    parsed_gaps = [str(g) for g in llm_gaps]

                # source_models (L2→L3 lineage)
                parsed_source_models = data.get("source_models", [])
                if not isinstance(parsed_source_models, list):
                    parsed_source_models = []

                # emergence_score
                parsed_emergence = float(data.get("emergence_score", 0.0))

            logger.info(
                "[认知演化] 从 LLM JSON 解析格栅元数据: trigram={}, skills={}, models={}, workflow_steps={}, source_models={}",
                parsed_trigram, len(parsed_skills), len(parsed_models), len(parsed_workflow), len(parsed_source_models),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("[认知演化] LLM JSON 解析失败，回退到简单模式: {}", exc)

    grid = GridFile(
        domain=domain,
        description=description,
        version="1.0",
        trigram=parsed_trigram,
        proficiency=parsed_proficiency,
        last_used=datetime.now(timezone.utc).isoformat(),
        skills=parsed_skills,
        models=parsed_models,
        source_models=parsed_source_models,
        emergence_score=parsed_emergence,
        workflow=parsed_workflow,
        gaps=parsed_gaps,
    )

    save_grid(grid)
    logger.info("[认知演化] 创建 L3 领域格栅: {} (trigram={}, skills={}, models={}, workflow_steps={})",
               domain, parsed_trigram, len(parsed_skills), len(parsed_models), len(parsed_workflow))

    # ── Reverse-update: add this grid to each source model's source_grids ──
    if parsed_source_models:
        for model_name in parsed_source_models:
            try:
                model_path = wp.models / f"{model_name}.json"
                if model_path.is_file():
                    model_data = json.loads(model_path.read_text(encoding="utf-8"))
                    model_grids: list[str] = list(model_data.get("source_grids", []))
                    if domain not in model_grids:
                        model_grids.append(domain)
                        model_data["source_grids"] = model_grids
                        model_path.write_text(
                            json.dumps(model_data, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        logger.info("[认知演化] 反向更新模型 {} source_grids += {}", model_name, domain)
            except Exception as exc:
                logger.warning("[认知演化] 反向更新模型 {} source_grids 失败: {}", model_name, exc)

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
    source_skills: list[str] | None = None,
    source_grids: list[str] | None = None,
) -> bool:
    """Create a new L2 experience model JSON file.

    Args:
        name: Model name (also used as the filename stem).
        content: Markdown content describing the model/pattern.
        source_skills: Names of L1 skills this model was abstracted from.
        source_grids: Names of L3 grids that reference this model (L3→L2 back-link).

    Returns:
        True if a new model was created, False if it already existed.
    """
    from vingobot.core.workspace import get_workspace_paths

    wp = get_workspace_paths()
    model_path = wp.models / f"{name}.json"

    if model_path.exists():
        logger.info("[认知演化] 模型已存在，跳过创建: {}", name)
        return False

    model_data: dict[str, Any] = {
        "name": name,
        "description": content[:500],
        "confidence": 0.5,
        "source_skills": source_skills or [],
        "source_grids": source_grids or [],
        "content": content,
        "version": "1.0",
        "created": datetime.now(timezone.utc).isoformat(),
    }

    model_path.write_text(
        json.dumps(model_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[认知演化] 创建 L2 思维模型: {} (source_skills={}, source_grids={})", name, len(source_skills or []), len(source_grids or []))
    return True


# ---------------------------------------------------------------------------
# L4 Truth creation
# ---------------------------------------------------------------------------


async def create_truth(
    name: str,
    title: str = "",
    truth_type: str = "pattern",
    rules: list[dict[str, Any]] | None = None,
    source_grids: list[str] | None = None,
    confidence: float = 0.0,
) -> bool:
    """Create a new L4 immutable truth JSON file.

    Args:
        name: Truth filename stem (e.g. 'truth_pattern_xxx').
        title: Short display title.
        truth_type: Category (identity / safety / pattern).
        rules: List of {id, statement} truth rule dicts.
        source_grids: L3 grid names this truth was distilled from.
        confidence: Confidence score 0.0–1.0.

    Returns:
        True if a new truth was created, False if it already existed.
    """
    from vingobot.core.workspace import get_workspace_paths
    from vingobot.goal.grid_types import truth_file_to_dict
    from vingobot.goal.cognition_tools import parse_grid, save_grid

    wp = get_workspace_paths()
    truth_path = wp.truths / f"{name}.json"

    if truth_path.exists():
        logger.info("[认知演化] 真理已存在（不可变），跳过创建: {}", name)
        return False

    truth = TruthFile(
        title=title or name,
        type=truth_type,
        version=1,
        immutable=True,
        confidence=confidence,
        source_grids=source_grids or [],
        rules=rules or [],
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

    truth_path.write_text(
        json.dumps(truth_file_to_dict(truth), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "[认知演化] 创建 L4 真理: {} (type={}, rules={}, source_grids={})",
        name, truth.type, len(truth.rules), len(truth.source_grids),
    )

    # ── Reverse-update: add this truth to each source grid's source_truths ──
    _source_grids = source_grids or []
    if _source_grids:
        for grid_name in _source_grids:
            try:
                grid_path = wp.grids / f"{grid_name}.json"
                if grid_path.is_file():
                    grid = parse_grid(grid_path)
                    if grid is not None and name not in grid.source_truths:
                        grid.source_truths.append(name)
                        save_grid(grid)
                        logger.info("[认知演化] 反向更新格栅 {} source_truths += {}", grid_name, name)
            except Exception as exc:
                logger.warning("[认知演化] 反向更新格栅 {} source_truths 失败: {}", grid_name, exc)

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
            source_skills=ctx.get("source_skills"),
            source_grids=ctx.get("source_grids"),
        )

    if action == "create_grid":
        return await create_domain_grid(
            domain=target,
            description=ea.description,
            skills=ctx.get("skills"),
            models=ctx.get("models"),
            llm_analysis=ctx.get("llm_analysis", ""),
        )

    if action == "precipitate_truth":
        return await create_truth(
            name=target,
            title=ctx.get("title", ea.description),
            truth_type=ctx.get("type", "pattern"),
            rules=ctx.get("rules"),
            source_grids=ctx.get("source_grids"),
            confidence=float(ctx.get("confidence", 0.0)),
        )

    logger.warning("[认知演化] 未知动作类型: {}", action)
    return False
