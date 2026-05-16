"""
认知资产更新器 — Update (not just create) cognitive assets.

Supports revising L1 skills, L2 models, and L4 truths with versioning
and confidence adjustment.  Unlike the creator functions (create_skill,
create_model, create_truth) which skip when assets exist, these updaters
intentionally modify existing assets.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# L1 Skill update
# ---------------------------------------------------------------------------


def update_skill(
    name: str,
    *,
    updated_description: str = "",
    updated_tools: list[dict[str, Any]] | None = None,
    revision_reason: str = "",
) -> bool:
    """Update an existing L1 skill's SKILL.md with version bump.

    Reads the existing SKILL.md, increments the version (1.0→1.1→1.2...),
    updates description/tools if provided, and appends a revision_history
    section.

    Args:
        name: Skill directory name.
        updated_description: New description (empty = keep existing).
        updated_tools: Updated tool list (None = keep existing).
        revision_reason: Why this revision was made.

    Returns:
        True if the skill was updated, False if it doesn't exist.
    """
    from vingobot.core.workspace import get_workspace_paths

    wp = get_workspace_paths()
    skill_dir = wp.skills / name

    if not skill_dir.is_dir():
        logger.warning("[认知更新] Skill 不存在，无法更新: {}", name)
        return False

    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.is_file():
        logger.warning("[认知更新] SKILL.md 不存在: {}", skill_md_path)
        return False

    content = skill_md_path.read_text(encoding="utf-8")

    # ── Parse existing version ─────────────────────────
    version_match = re.search(r"^version:\s*([\d.]+)", content, re.MULTILINE)
    old_version = version_match.group(1) if version_match else "1.0"
    try:
        parts = old_version.split(".")
        new_version = f"{parts[0]}.{int(parts[1]) + 1}" if len(parts) >= 2 else "1.1"
    except (ValueError, IndexError):
        new_version = "1.1"

    # ── Update version line ────────────────────────────
    if version_match:
        content = content.replace(
            f"version: {old_version}", f"version: {new_version}", 1
        )

    # ── Update description if provided ─────────────────
    if updated_description:
        desc_match = re.search(r'^description:\s*"(.*)"', content, re.MULTILINE)
        if desc_match:
            content = content.replace(
                f'description: "{desc_match.group(1)}"',
                f'description: "{updated_description}"',
                1,
            )

    # ── Update tools YAML block if provided ────────────
    if updated_tools:
        tools_yaml = _build_tools_yaml(updated_tools)
        # Replace existing tools block between frontmatter (--- and ---)
        content = _replace_tools_in_frontmatter(content, tools_yaml)

    # ── Append or update revision_history ──────────────
    timestamp = datetime.now(timezone.utc).isoformat()
    revision_entry = (
        f"\n\n## 修订历史\n\n"
        f"- **v{new_version}** ({timestamp}): {revision_reason or '认知演化自动更新'}"
    )

    # Check if revision_history section already exists
    if "## 修订历史" in content:
        # Insert after the header
        rev_header_idx = content.index("## 修订历史")
        next_nl = content.index("\n", rev_header_idx + len("## 修订历史"))
        content = (
            content[:next_nl + 1]
            + f"\n- **v{new_version}** ({timestamp}): {revision_reason or '认知演化自动更新'}\n"
            + content[next_nl + 1:]
        )
    else:
        content = content.rstrip() + revision_entry + "\n"

    skill_md_path.write_text(content, encoding="utf-8")
    logger.info(
        "[认知更新] 更新 L1 Skill: {} v{} → v{} (reason: {})",
        name, old_version, new_version, revision_reason[:60],
    )
    return True


def _build_tools_yaml(tools: list[dict[str, Any]]) -> str:
    """Build a YAML tools block from tool definitions."""
    lines = ["tools:"]
    for t in tools:
        tool_name = t.get("name", "")
        tool_desc = t.get("description", "")
        lines.append(f"  - name: {tool_name}")
        lines.append(f'    description: "{tool_desc}"')
        params = t.get("parameters", {})
        if params:
            lines.append("    parameters:")
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "string") if isinstance(pinfo, dict) else "string"
                pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else str(pinfo)
                lines.append(f"      {pname}:")
                lines.append(f"        type: {ptype}")
                lines.append(f'        description: "{pdesc}"')
        auto_approve = t.get("tool_auto_approve", False)
        if auto_approve:
            lines.append("    tool_auto_approve: true")
    return "\n".join(lines)


def _replace_tools_in_frontmatter(content: str, new_tools_yaml: str) -> str:
    """Replace the tools block within YAML frontmatter (between --- markers)."""
    # Find frontmatter bounds
    first_dash = content.find("---")
    if first_dash == -1:
        return content
    second_dash = content.find("---", first_dash + 3)
    if second_dash == -1:
        return content

    before = content[:first_dash + 3]
    frontmatter = content[first_dash + 3:second_dash]
    after = content[second_dash:]

    # Remove existing tools block from frontmatter
    tools_start = frontmatter.find("\ntools:")
    if tools_start != -1:
        # Find where tools block ends (next top-level key without indent)
        rest = frontmatter[tools_start + 1:]
        lines = rest.split("\n")
        end_idx = 1
        for line in lines[1:]:
            if line and not line.startswith(" ") and not line.startswith("\t"):
                break
            end_idx += 1
        frontmatter = frontmatter[:tools_start] + "\n".join(lines[end_idx:])

    # Append new tools block
    frontmatter = frontmatter.rstrip() + "\n" + new_tools_yaml + "\n"

    return before + frontmatter + after


# ---------------------------------------------------------------------------
# L2 Model update
# ---------------------------------------------------------------------------


def update_model(
    name: str,
    *,
    content: str = "",
    confidence_adjust: float = 0.0,
    source_skills: list[str] | None = None,
    source_grids: list[str] | None = None,
    revision_reason: str = "",
) -> bool:
    """Update an existing L2 model JSON file.

    Adjusts confidence by *confidence_adjust* (clamped to [0, 1]),
    merges source_skills / source_grids, increments version,
    and appends revision_history.

    Args:
        name: Model filename stem.
        content: Updated markdown content (empty = keep existing).
        confidence_adjust: Delta to add to confidence (+0.1 for success, -0.05 for failure).
        source_skills: New source skills to merge.
        source_grids: New source grids to merge.
        revision_reason: Why this revision was made.

    Returns:
        True if updated, False if model doesn't exist.
    """
    from vingobot.core.workspace import get_workspace_paths

    wp = get_workspace_paths()
    model_path = wp.models / f"{name}.json"

    if not model_path.is_file():
        logger.warning("[认知更新] 模型不存在，无法更新: {}", name)
        return False

    try:
        data = json.loads(model_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("[认知更新] 模型文件损坏: {}", model_path)
        return False

    # ── Version bump ──────────────────────────────────
    old_version = str(data.get("version", "1.0"))
    try:
        parts = old_version.split(".")
        new_version = f"{parts[0]}.{int(parts[1]) + 1}" if len(parts) >= 2 else "1.1"
    except (ValueError, IndexError):
        new_version = "1.1"
    data["version"] = new_version

    # ── Confidence adjustment ─────────────────────────
    if confidence_adjust != 0.0:
        old_conf = float(data.get("confidence", 0.5))
        new_conf = max(0.0, min(1.0, old_conf + confidence_adjust))
        data["confidence"] = round(new_conf, 3)

    # ── Content update ────────────────────────────────
    if content:
        data["content"] = content

    # ── Merge source lists ────────────────────────────
    if source_skills:
        existing = set(data.get("source_skills", []))
        existing.update(source_skills)
        data["source_skills"] = sorted(existing)
    if source_grids:
        existing = set(data.get("source_grids", []))
        existing.update(source_grids)
        data["source_grids"] = sorted(existing)

    # ── Revision history ──────────────────────────────
    timestamp = datetime.now(timezone.utc).isoformat()
    rev_entry = {
        "version": new_version,
        "timestamp": timestamp,
        "reason": revision_reason or "认知演化自动更新",
        "confidence_before": round(old_conf, 3) if "old_conf" in dir() else None,
        "confidence_after": data.get("confidence"),
    }
    history = data.get("revision_history", [])
    if not isinstance(history, list):
        history = []
    history.append(rev_entry)
    data["revision_history"] = history

    data["updated_at"] = timestamp

    model_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "[认知更新] 更新 L2 模型: {} v{} → v{} (conf: {:.2f}, reason: {})",
        name, old_version, new_version,
        data.get("confidence", 0.5), revision_reason[:60],
    )
    return True


# ---------------------------------------------------------------------------
# L4 Truth revision (controlled)
# ---------------------------------------------------------------------------


def revise_truth(
    name: str,
    *,
    title: str = "",
    rules: list[dict[str, Any]] | None = None,
    confidence: float = 0.0,
    revision_reason: str = "",
) -> bool:
    """Revise an existing L4 truth JSON file under controlled conditions.

    Truth revision requires:
    - confidence >= 0.7 (high certainty only)
    - An explicit revision_reason must be provided

    Args:
        name: Truth filename stem.
        title: Updated title (empty = keep existing).
        rules: Updated rules list (None = keep existing).
        confidence: New confidence value (must be >= 0.7).
        revision_reason: Mandatory explanation for the revision.

    Returns:
        True if revised, False if conditions not met or file doesn't exist.
    """
    from vingobot.core.workspace import get_workspace_paths

    if confidence < 0.7:
        logger.warning(
            "[认知更新] 真理修订被拒: 置信度 {:.2f} < 0.7 门槛 (name={})",
            confidence, name,
        )
        return False

    if not revision_reason:
        logger.warning("[认知更新] 真理修订被拒: 缺少修订理由 (name={})", name)
        return False

    wp = get_workspace_paths()
    truth_path = wp.truths / f"{name}.json"

    if not truth_path.is_file():
        logger.warning("[认知更新] 真理文件不存在: {}", truth_path)
        return False

    try:
        data = json.loads(truth_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("[认知更新] 真理文件损坏: {}", truth_path)
        return False

    # ── Version increment ─────────────────────────────
    old_version = data.get("version", 1)
    if not isinstance(old_version, int):
        old_version = 1
    data["version"] = old_version + 1

    # ── Update fields ─────────────────────────────────
    if title:
        data["title"] = title
    if rules is not None:
        data["rules"] = rules
    if confidence > 0:
        data["confidence"] = round(confidence, 3)

    # ── Revision history ──────────────────────────────
    timestamp = datetime.now(timezone.utc).isoformat()
    rev_entry = {
        "version": data["version"],
        "timestamp": timestamp,
        "reason": revision_reason,
        "confidence": data.get("confidence"),
    }
    history = data.get("revision_history", [])
    if not isinstance(history, list):
        history = []
    history.append(rev_entry)
    data["revision_history"] = history
    data["updated_at"] = timestamp

    # Keep immutable flag but log the exception
    data["immutable"] = True
    data["immutable_revised"] = True
    data["immutable_revision_reason"] = revision_reason

    truth_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "[认知更新] 修订 L4 真理: {} v{} → v{} (conf={:.2f}, reason: {})",
        name, old_version, data["version"],
        data.get("confidence", 0), revision_reason[:60],
    )
    return True
