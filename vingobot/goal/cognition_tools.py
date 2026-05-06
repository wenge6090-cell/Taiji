"""
L1/L2/L3 认知检索工具 — On-demand cognitive navigation.

These are meta-tools registered in Yang's tool_definitions.  Yang can
call them to load relevant knowledge on-demand rather than having
everything pre-injected into the context:

- **L1 search_skills**: Search the skill library for relevant reusable
  skill definitions.
- **L2 search_models**: Search the experience model library for
  patterns and past solutions.
- **L3 load_grid**: Load a cognitive grid (structured thinking framework)
  by name.

These tools are READ-ONLY and always auto-approved by Yin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.core.workspace import get_workspace_paths

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI schema)
# ---------------------------------------------------------------------------


SEARCH_SKILLS_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_skills",
        "description": (
            "搜索 L1 技能库，返回匹配的技能定义。"
            "技能是封装好的任务执行步骤（如 'test-runner'、'code-reviewer'）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "技能搜索关键词，如 '测试'、'代码审查'、'部署'",
                },
            },
            "required": ["query"],
        },
    },
}


SEARCH_MODELS_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_models",
        "description": (
            "搜索 L2 经验模型库，返回过去解决类似问题的参考模式。"
            "模型是对成功经验的抽象，可帮助加快当前任务的解决速度。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "模型搜索关键词，如 '错误处理模式'、'数据库迁移'",
                },
            },
            "required": ["query"],
        },
    },
}


LOAD_GRID_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "load_grid",
        "description": (
            "加载 L3 认知网格。认知网格是一种结构化的思维框架，"
            "帮助以系统化的方式分析问题。如 'exploration'（探索）、"
            "'debugging'（调试）、'refactor'（重构）等。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要加载的认知网格名称",
                },
            },
            "required": ["name"],
        },
    },
}


_COGNITION_TOOL_DEFS: list[dict[str, Any]] = [
    SEARCH_SKILLS_DEF,
    SEARCH_MODELS_DEF,
    LOAD_GRID_DEF,
]


def get_cognition_tool_definitions() -> list[dict[str, Any]]:
    """Return the three cognition meta-tool definitions."""
    return _COGNITION_TOOL_DEFS


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def search_skills(query: str, **kwargs: Any) -> str:
    """Search the L1 skill library."""
    wp = get_workspace_paths()
    skills_dir = wp.skills

    if not skills_dir.is_dir():
        return "[L1] 技能库目录不存在，请先初始化工作区。"

    matches: list[str] = []
    query_lower = query.lower()
    keywords = query_lower.split()

    try:
        for item in sorted(skills_dir.iterdir()):
            if not item.is_dir():
                continue
            skill_name = item.name.lower()
            score = sum(1 for kw in keywords if kw in skill_name)
            if score > 0:
                # Try to read a description
                desc = ""
                for desc_file in ("README.md", "skill.md", "description.md"):
                    df = item / desc_file
                    if df.is_file():
                        try:
                            desc = df.read_text(encoding="utf-8")[:300]
                        except Exception:
                            pass
                        break
                matches.append(f"## {item.name} (匹配度: {score})\n{desc or '(无描述)'}")
    except OSError:
        return "[L1] 无法读取技能库目录。"

    if not matches:
        # Fallback: list all available skills
        try:
            all_skills = [item.name for item in sorted(skills_dir.iterdir()) if item.is_dir()]
            if all_skills:
                return "[L1] 可用技能:\n" + "\n".join(f"- {s}" for s in all_skills)
        except OSError:
            pass
        return f"[L1] 未找到匹配 '{query}' 的技能。"

    return "[L1] 技能搜索结果:\n\n" + "\n\n".join(matches[:5])


async def search_models(query: str, **kwargs: Any) -> str:
    """Search the L2 experience model library."""
    wp = get_workspace_paths()
    models_dir = wp.models

    if not models_dir.is_dir():
        return "[L2] 经验模型库目录不存在，请先初始化工作区。"

    matches: list[str] = []
    query_lower = query.lower()
    keywords = query_lower.split()

    try:
        for item in sorted(models_dir.iterdir()):
            if item.suffix not in (".md", ".json", ".yaml", ".yml"):
                continue
            name = item.stem.lower()
            score = sum(1 for kw in keywords if kw in name)
            if score > 0:
                try:
                    content = item.read_text(encoding="utf-8")[:500]
                except Exception:
                    content = "(无法读取)"
                matches.append(f"## {item.stem} (匹配度: {score})\n{content}")
    except OSError:
        return "[L2] 无法读取模型库目录。"

    if not matches:
        try:
            all_models = sorted(
                item.stem
                for item in models_dir.iterdir()
                if item.suffix in (".md", ".json", ".yaml", ".yml")
            )
            if all_models:
                return "[L2] 可用经验模型:\n" + "\n".join(f"- {m}" for m in all_models)
        except OSError:
            pass
        return f"[L2] 未找到匹配 '{query}' 的模型。"

    return "[L2] 经验模型搜索结果:\n\n" + "\n\n".join(matches[:5])


async def load_grid(name: str, **kwargs: Any) -> str:
    """Load an L3 cognitive grid by name."""
    wp = get_workspace_paths()
    grids_dir = wp.grids

    if not grids_dir.is_dir():
        return "[L3] 认知网格目录不存在，请先初始化工作区。"

    # Try exact match first
    for ext in (".md", ".json", ".yaml", ".yml"):
        grid_file = grids_dir / f"{name}{ext}"
        if grid_file.is_file():
            try:
                return f"[L3] 认知网格 '{name}':\n\n{grid_file.read_text(encoding='utf-8')[:3000]}"
            except Exception as exc:
                return f"[L3] 读取网格 '{name}' 失败: {exc}"

    # Fuzzy match
    name_lower = name.lower()
    try:
        for item in sorted(grids_dir.iterdir()):
            if item.suffix not in (".md", ".json", ".yaml", ".yml"):
                continue
            if name_lower in item.stem.lower():
                try:
                    return (
                        f"[L3] 认知网格 '{item.stem}':\n\n{item.read_text(encoding='utf-8')[:3000]}"
                    )
                except Exception as exc:
                    return f"[L3] 读取网格 '{item.stem}' 失败: {exc}"
    except OSError:
        pass

    # List available
    try:
        all_grids = sorted(
            item.stem
            for item in grids_dir.iterdir()
            if item.suffix in (".md", ".json", ".yaml", ".yml")
        )
        if all_grids:
            return f"[L3] 未找到网格 '{name}'。可用网格:\n" + "\n".join(f"- {g}" for g in all_grids)
    except OSError:
        pass

    return f"[L3] 未找到网格 '{name}'。"


# ---------------------------------------------------------------------------
# Execution dispatcher
# ---------------------------------------------------------------------------


async def execute_cognition_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a cognition tool call to the correct handler."""
    if name == "search_skills":
        return await search_skills(**arguments)
    if name == "search_models":
        return await search_models(**arguments)
    if name == "load_grid":
        return await load_grid(**arguments)
    return f"[错误] 未知认知工具: {name}"


# ---------------------------------------------------------------------------
# L3 Grid file I/O (structured JSON format)
# ---------------------------------------------------------------------------


def parse_grid(file_path: str | Path) -> "GridFile | None":
    """Parse an L3 grid JSON file into a ``GridFile`` instance.

    Supports both the new standardised format and the legacy (old taiji)
    format with Chinese field names.  Returns ``None`` if the file cannot
    be parsed.
    """
    from vingobot.goal.grid_types import GridFile, dict_to_grid_file

    path = Path(file_path)
    if not path.is_file():
        return None

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("[认知] 无法读取格栅文件 {}: {}", path.name, exc)
        return None

    # ── New format (domain key present) ───────────────────────────────
    if "domain" in data:
        return dict_to_grid_file(data)

    # ── Legacy format: Chinese keys ───────────────────────────────────
    domain = data.get("领域") or data.get("网格类型") or path.stem
    description = data.get("描述", "")
    version = str(data.get("版本", "1.0"))
    trigram = data.get("trigram", "")

    skills: list[Any] = []
    models: list[Any] = []
    workflow: list[Any] = []

    # Legacy: ``nodes`` with ``skills``, ``models`` sub-keys
    nodes = data.get("nodes") or data.get("modes") or {}
    nodes_meta: dict[str, dict[str, Any]] = {}
    if isinstance(nodes, dict):
        for node_name, node_data in nodes.items():
            if isinstance(node_data, dict):
                node_skills = node_data.get("skills") or node_data.get("preferred_actions") or []
                if isinstance(node_skills, list):
                    for s in node_skills:
                        if isinstance(s, str):
                            skills.append({"name": s, "relevance": "supporting"})
                # Preserve sixiang-specific node metadata for Weaver
                node_meta: dict[str, Any] = {}
                for key in (
                    "phase",
                    "rule",
                    "execution_hint",
                    "temperature_override",
                    "符号",
                    "位置",
                    "认知映射",
                    "典型状态",
                ):
                    if key in node_data:
                        node_meta[key] = node_data[key]
                if node_meta:
                    nodes_meta[node_name] = node_meta

    # Legacy: top-level ``skills`` / ``models`` arrays
    for raw_s in data.get("skills", []):
        if isinstance(raw_s, str):
            skills.append({"name": raw_s, "relevance": "supporting"})
        elif isinstance(raw_s, dict):
            skills.append(raw_s)

    for raw_m in data.get("models", []):
        if isinstance(raw_m, str):
            models.append({"name": raw_m, "relevance": "supporting"})
        elif isinstance(raw_m, dict):
            models.append(raw_m)

    # Legacy: ``sub_branches`` / ``子分支``
    legacy_branches = data.get("子分支") or data.get("sub_branches") or []
    if isinstance(legacy_branches, list):
        for branch in legacy_branches:
            name = branch.get("名称") or branch.get("name", "")
            if name:
                skills.append({"name": name, "relevance": "supporting"})

    return GridFile(
        domain=domain,
        description=description,
        version=version,
        trigram=trigram,
        skills=[
            {"name": s["name"], "relevance": s.get("relevance", "supporting")}
            if isinstance(s, dict)
            else {"name": str(s), "relevance": "supporting"}
            for s in skills
        ],
        models=[
            {"name": m["name"], "relevance": m.get("relevance", "supporting")}
            if isinstance(m, dict)
            else {"name": str(m), "relevance": "supporting"}
            for m in models
        ],
        workflow=workflow,
        gaps=data.get("gaps", []),
        nodes_meta=nodes_meta,
    )


def save_grid(grid: "GridFile", grids_dir: str | Path | None = None) -> Path:
    """Save a ``GridFile`` to disk as a JSON file.

    Args:
        grid: The grid to persist.
        grids_dir: Target directory (defaults to workspace grids dir).

    Returns:
        The path of the saved file.
    """
    from vingobot.goal.grid_types import grid_file_to_dict

    if grids_dir is None:
        from vingobot.core.workspace import get_workspace_paths

        grids_dir = get_workspace_paths().grids

    grids_dir = Path(grids_dir)
    grids_dir.mkdir(parents=True, exist_ok=True)

    data = grid_file_to_dict(grid)
    file_path = grids_dir / f"{grid.domain}.json"

    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[认知] 保存格栅文件: {}", file_path)
    return file_path


def list_all_grids(grids_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """List all L3 cognitive grids with metadata.

    Args:
        grids_dir: Grids directory (defaults to workspace grids dir).

    Returns:
        List of ``{domain, trigram, proficiency, skill_count, model_count, gap_count}``.
    """
    if grids_dir is None:
        from vingobot.core.workspace import get_workspace_paths

        grids_dir = get_workspace_paths().grids

    grids_dir = Path(grids_dir)
    if not grids_dir.is_dir():
        return []

    result: list[dict[str, Any]] = []
    for fpath in sorted(grids_dir.glob("*.json")):
        grid = parse_grid(fpath)
        if grid is None:
            continue
        result.append(
            {
                "domain": grid.domain,
                "trigram": grid.trigram,
                "proficiency": grid.proficiency,
                "skill_count": len(grid.skills),
                "model_count": len(grid.models),
                "gap_count": len(grid.gaps),
                "version": grid.version,
                "file": fpath.name,
            }
        )

    return result


def update_nav_index(grids_dir: str | Path | None = None) -> "GridFile | None":
    """Update (or create) the master navigation index grid — ``认知导航.json``.

    This index aggregates metadata from all other grids into a single
    navigation file with proficiency, gap, and dependency info.

    Args:
        grids_dir: Grids directory (defaults to workspace grids dir).

    Returns:
        The updated ``GridFile``, or ``None`` if no grids exist.
    """
    from datetime import datetime, timezone

    from vingobot.goal.grid_types import GridFile, GridModelRef, GridSkillRef

    if grids_dir is None:
        from vingobot.core.workspace import get_workspace_paths

        grids_dir = get_workspace_paths().grids

    grids_dir = Path(grids_dir)
    if not grids_dir.is_dir():
        return None

    all_skills: dict[str, set[str]] = {}
    all_models: dict[str, set[str]] = {}
    all_gaps: list[str] = []
    proficiency_sum = 0.0
    grid_count = 0

    for fpath in sorted(grids_dir.glob("*.json")):
        if fpath.stem == "认知导航":
            continue
        grid = parse_grid(fpath)
        if grid is None:
            continue

        for s in grid.skills:
            all_skills.setdefault(s["name"] if isinstance(s, dict) else s, set()).add(grid.domain)
        for m in grid.models:
            all_models.setdefault(m["name"] if isinstance(m, dict) else m, set()).add(grid.domain)
        all_gaps.extend(grid.gaps)
        proficiency_sum += grid.proficiency
        grid_count += 1

    if grid_count == 0:
        return None

    nav = GridFile(
        domain="认知导航",
        description=(
            "认知导航主索引 — 聚合所有L3格栅的元数据，"
            "提供各领域的能力覆盖、技能依赖和学习队列概览。"
        ),
        version="1.0",
        trigram="",
        proficiency=round(proficiency_sum / grid_count, 2),
        last_used=datetime.now(timezone.utc).isoformat(),
        skills=[
            GridSkillRef(name=sname, relevance="supporting") for sname in sorted(all_skills.keys())
        ],
        models=[
            GridModelRef(name=mname, relevance="supporting") for mname in sorted(all_models.keys())
        ],
        gaps=list(set(all_gaps)),
    )

    save_grid(nav, grids_dir)
    return nav
