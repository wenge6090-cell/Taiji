"""
package_skill — Package a skill directory into a ``.skill`` zip archive.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


def package_skill(
    skill_dir: str | Path,
    output_dir: str | Path,
) -> Path | None:
    """Create a ``.skill`` zip archive from a skill directory.

    The archive preserves the relative structure under ``skill_dir``.
    Symlinks are rejected (returns ``None``).

    Args:
        skill_dir: Path to the skill directory.
        output_dir: Directory where the archive will be written.

    Returns:
        Path to the created archive, or ``None`` on failure.
    """
    src = Path(skill_dir).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    skill_name = src.name
    archive_path = out / f"{skill_name}.skill"
    files_to_archive: list[Path] = []

    # Collect files, rejecting symlinks
    for entry in src.rglob("*"):
        if entry.is_file():
            if entry.is_symlink():
                return None
            files_to_archive.append(entry)

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in files_to_archive:
            arcname = str(file_path.relative_to(src.parent))
            archive.write(file_path, arcname)

    return archive_path
