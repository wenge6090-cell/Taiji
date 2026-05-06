"""
init_skill — Scaffold a new skill directory with SKILL.md.
"""

from __future__ import annotations

from pathlib import Path


def init_skill(
    name: str,
    parent_dir: str | Path,
    subdirs: list[str] | None = None,
    include_examples: bool = False,
) -> Path:
    """Create a new skill directory tree.

    Args:
        name: Skill name (also used as the directory name).
        parent_dir: Parent directory to create the skill under.
        subdirs: Subdirectory names to create (e.g. ``["scripts", "references"]``).
        include_examples: If true, create example files in each subdirectory.

    Returns:
        Path to the created skill directory.
    """
    skill_dir = Path(parent_dir).resolve() / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    # SKILL.md
    skel = _skill_skeleton(name)
    (skill_dir / "SKILL.md").write_text(skel, encoding="utf-8")

    # Subdirectories
    for sub in (subdirs or []):
        sub_path = skill_dir / sub
        sub_path.mkdir(parents=True, exist_ok=True)
        if include_examples:
            _write_example(sub_path, sub)

    return skill_dir


def _skill_skeleton(name: str) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f'description: "[TODO: fill me in]"\n'
        "version: 0.1.0\n"
        "---\n"
        f"\n"
        f"# {name}\n"
        f"\n"
        f"TODO: describe what this skill does.\n"
    )


_EXAMPLE_BY_DIR: dict[str, str] = {
    "scripts": "scripts/example.py",
    "references": "references/api_reference.md",
    "assets": "assets/example_asset.txt",
}

_EXAMPLE_CONTENT: dict[str, str] = {
    "scripts/example.py": '"""Example script for the skill."""\n\ndef main() -> None:\n    print("Hello from skill!")\n\n\nif __name__ == "__main__":\n    main()\n',
    "references/api_reference.md": "# API Reference\n\nTODO: document the API endpoints or interfaces used by this skill.\n",
    "assets/example_asset.txt": "Example asset file for the skill.\n",
}


def _write_example(sub_path: Path, dir_name: str) -> None:
    key = _EXAMPLE_BY_DIR.get(dir_name)
    if key is None:
        return
    content = _EXAMPLE_CONTENT.get(key, "")
    if content:
        (sub_path / Path(key).name).write_text(content, encoding="utf-8")
