"""
quick_validate — Validate a skill directory structure and metadata.
"""

from __future__ import annotations

import re
from pathlib import Path

_ALLOWED_ROOT_ENTRIES = frozenset({
    "SKILL.md",
    "scripts",
    "references",
    "assets",
    "widgets",
})


def validate_skill(skill_dir: str | Path) -> tuple[bool, str]:
    """Validate a skill directory.

    Checks performed:
      1. ``SKILL.md`` exists.
      2. Description is not a placeholder (``[TODO: fill me in]``).
      3. No unexpected files/directories at the root.

    Args:
        skill_dir: Path to the skill directory.

    Returns:
        ``(True, "")`` on success, ``(False, reason)`` on failure.
    """
    path = Path(skill_dir).resolve()
    if not path.is_dir():
        return False, f"Not a directory: {path}"

    # 1. SKILL.md exists
    skill_md = path / "SKILL.md"
    if not skill_md.is_file():
        return False, "Missing SKILL.md"

    # 2. Parse front matter for description
    front = _parse_frontmatter(skill_md)
    desc = front.get("description", "")
    if _is_placeholder(desc):
        return False, "Description contains a TODO placeholder"

    # 3. No unexpected root entries
    for entry in path.iterdir():
        if entry.name not in _ALLOWED_ROOT_ENTRIES:
            return False, f"Unexpected file or directory in skill root: {entry.name}"

    return True, ""


def _parse_frontmatter(path: Path) -> dict[str, str]:
    """Minimal frontmatter parser (no external dep)."""
    text = path.read_text(encoding="utf-8")
    result: dict[str, str] = {}
    if not text.startswith("---"):
        return result
    end = text.find("---", 3)
    if end == -1:
        return result
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


_PLACEHOLDER_RE = re.compile(r"\[TODO[\s\S]*?\]", re.IGNORECASE)


def _is_placeholder(description: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(description))
