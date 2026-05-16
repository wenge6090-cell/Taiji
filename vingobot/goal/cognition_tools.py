"""
L3 认知网格 I/O — 结构化网格文件解析、保存、索引。

本模块仅负责 L3 网格文件的数据操作，不提供 Agent 工具定义。
Agent 应使用 read_file / list_directory 直接读取认知库文件。

- **parse_grid**: 解析格栅 JSON 为 GridFile 实例
- **save_grid**: 持久化 GridFile 到磁盘
- **list_all_grids**: 列出所有格栅的元数据摘要
- **update_nav_index**: 更新/创建 认知导航.json 主索引
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.core.workspace import get_workspace_paths


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
    if isinstance(nodes, dict):
        for node_name, node_data in nodes.items():
            if isinstance(node_data, dict):
                node_skills = node_data.get("skills") or node_data.get("preferred_actions") or []
                if isinstance(node_skills, list):
                    for s in node_skills:
                        if isinstance(s, str):
                            skills.append({"name": s, "relevance": "supporting"})

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
