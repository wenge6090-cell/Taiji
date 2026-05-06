"""Tests for vingobot.agent.skills.SkillsLoader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vingobot.agent.skills import SkillsLoader


def _write_skill(
    base: Path,
    name: str,
    *,
    metadata_json: dict | None = None,
    body: str = "# Skill\n",
) -> Path:
    """Create ``base / name / SKILL.md`` with optional vingobot metadata JSON."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    lines = ["---"]
    if metadata_json is not None:
        payload = json.dumps({"vingobot": metadata_json}, separators=(",", ":"))
        lines.append(f'metadata: {payload}')
    lines.extend(["---", "", body])
    path = skill_dir / "SKILL.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _skills_loader(tmp_path: Path) -> SkillsLoader:
    """Helper: patch _resolve_builtin_skills_dir and return a SkillsLoader."""
    import vingobot.agent.skills as sk_mod

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    sk_mod._resolve_builtin_skills_dir = lambda: skills_dir
    return SkillsLoader()


def test_list_skills_empty_when_skills_dir_missing(tmp_path: Path) -> None:
    """Skills dir doesn't exist → empty list."""
    import vingobot.agent.skills as sk_mod

    missing = tmp_path / "no_such_skills"
    sk_mod._resolve_builtin_skills_dir = lambda: missing
    loader = SkillsLoader()
    assert loader.list_skills(filter_unavailable=False) == []


def test_list_skills_empty_when_skills_dir_exists_but_empty(tmp_path: Path) -> None:
    loader = _skills_loader(tmp_path)
    assert loader.list_skills(filter_unavailable=False) == []


def test_list_skills_entry_shape_and_source(tmp_path: Path) -> None:
    loader = _skills_loader(tmp_path)
    skill_path = _write_skill(loader.skills_dir, "alpha", body="# Alpha")
    entries = loader.list_skills(filter_unavailable=False)
    assert entries == [
        {"name": "alpha", "path": str(skill_path), "source": "taiji"},
    ]


def test_list_skills_skips_non_directories_and_missing_skill_md(tmp_path: Path) -> None:
    loader = _skills_loader(tmp_path)
    (loader.skills_dir / "not_a_dir.txt").write_text("x", encoding="utf-8")
    (loader.skills_dir / "no_skill_md").mkdir()
    ok_path = _write_skill(loader.skills_dir, "ok", body="# Ok")
    entries = loader.list_skills(filter_unavailable=False)
    names = {entry["name"] for entry in entries}
    assert names == {"ok"}
    assert entries[0]["path"] == str(ok_path)


def test_list_skills_unavailable_requirement_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _skills_loader(tmp_path)
    _write_skill(
        loader.skills_dir,
        "needs_bin",
        metadata_json={"requires": {"bins": ["VINGOBOT_test_fake_binary"]}},
    )

    def fake_which(cmd: str) -> str | None:
        if cmd == "VINGOBOT_test_fake_binary":
            return None
        return "/usr/bin/true"

    monkeypatch.setattr("vingobot.agent.skills.shutil.which", fake_which)
    assert loader.list_skills(filter_unavailable=True) == []


def test_list_skills_available_when_requirement_met(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _skills_loader(tmp_path)
    skill_path = _write_skill(
        loader.skills_dir,
        "has_bin",
        metadata_json={"requires": {"bins": ["VINGOBOT_test_fake_binary"]}},
    )

    def fake_which(cmd: str) -> str | None:
        if cmd == "VINGOBOT_test_fake_binary":
            return "/fake/VINGOBOT_test_fake_binary"
        return None

    monkeypatch.setattr("vingobot.agent.skills.shutil.which", fake_which)
    entries = loader.list_skills(filter_unavailable=True)
    assert entries == [{"name": "has_bin", "path": str(skill_path), "source": "taiji"}]


def test_list_skills_unavailable_false_keeps_unmet_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _skills_loader(tmp_path)
    skill_path = _write_skill(
        loader.skills_dir,
        "blocked",
        metadata_json={"requires": {"bins": ["VINGOBOT_test_fake_binary"]}},
    )

    monkeypatch.setattr("vingobot.agent.skills.shutil.which", lambda _cmd: None)
    entries = loader.list_skills(filter_unavailable=False)
    assert entries == [{"name": "blocked", "path": str(skill_path), "source": "taiji"}]


def test_list_skills_env_requirement_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _skills_loader(tmp_path)
    _write_skill(
        loader.skills_dir,
        "needs_env",
        metadata_json={"requires": {"env": ["VINGOBOT_SKILLS_TEST_ENV_VAR"]}},
    )

    monkeypatch.delenv("VINGOBOT_SKILLS_TEST_ENV_VAR", raising=False)
    assert loader.list_skills(filter_unavailable=True) == []


def test_disabled_skills_excluded_from_list(tmp_path: Path) -> None:
    loader = _skills_loader(tmp_path)
    _write_skill(loader.skills_dir, "alpha", body="# Alpha")
    beta_path = _write_skill(loader.skills_dir, "beta", body="# Beta")

    loader = SkillsLoader(disabled_skills={"alpha"})
    entries = loader.list_skills(filter_unavailable=False)
    assert len(entries) == 1
    assert entries[0]["name"] == "beta"
    assert entries[0]["path"] == str(beta_path)


def test_disabled_skills_empty_set_no_effect(tmp_path: Path) -> None:
    loader = _skills_loader(tmp_path)
    _write_skill(loader.skills_dir, "alpha", body="# Alpha")
    _write_skill(loader.skills_dir, "beta", body="# Beta")

    loader = SkillsLoader(disabled_skills=set())
    entries = loader.list_skills(filter_unavailable=False)
    assert len(entries) == 2


def test_disabled_skills_excluded_from_build_skills_summary(tmp_path: Path) -> None:
    loader = _skills_loader(tmp_path)
    _write_skill(loader.skills_dir, "alpha", body="# Alpha")
    _write_skill(loader.skills_dir, "beta", body="# Beta")

    loader = SkillsLoader(disabled_skills={"alpha"})
    summary = loader.build_skills_summary()
    assert "alpha" not in summary
    assert "beta" in summary


def test_disabled_skills_excluded_from_get_always_skills(tmp_path: Path) -> None:
    loader = _skills_loader(tmp_path)
    _write_skill(loader.skills_dir, "alpha", metadata_json={"always": True}, body="# Alpha")
    _write_skill(loader.skills_dir, "beta", metadata_json={"always": True}, body="# Beta")

    loader = SkillsLoader(disabled_skills={"alpha"})
    always = loader.get_always_skills()
    assert "alpha" not in always
    assert "beta" in always


# -- multiline description tests (YAML folded > and literal |) -----------------


def test_build_skills_summary_folded_description(tmp_path: Path) -> None:
    """description: > (YAML folded scalar) should be parsed correctly."""
    loader = _skills_loader(tmp_path)
    skill_dir = loader.skills_dir / "pdf"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: pdf\n"
        "description: >\n"
        "  Use this skill when visual quality and design identity matter for a PDF.\n"
        "  CREATE (generate from scratch): \"make a PDF\".\n"
        "---\n\n# PDF Skill\n",
        encoding="utf-8",
    )

    summary = loader.build_skills_summary()
    assert "pdf" in summary
    assert "visual quality" in summary


def test_build_skills_summary_literal_description(tmp_path: Path) -> None:
    """description: | (YAML literal scalar) should be parsed correctly."""
    loader = _skills_loader(tmp_path)
    skill_dir = loader.skills_dir / "multi"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: multi\n"
        "description: |\n"
        "  Line one of description.\n"
        "  Line two of description.\n"
        "---\n\n# Multi\n",
        encoding="utf-8",
    )

    meta = loader.get_skill_metadata("multi")
    assert meta is not None
    desc = meta.get("description")
    assert isinstance(desc, str)
    assert "Line one" in desc
    assert "Line two" in desc


def test_get_skill_metadata_handles_yaml_types(tmp_path: Path) -> None:
    """yaml.safe_load returns native types; always should be True, not 'true'."""
    loader = _skills_loader(tmp_path)
    payload = json.dumps({"vingobot": {"requires": {"bins": ["gh"]}, "always": True}}, separators=(",", ":"))
    skill_dir = loader.skills_dir / "typed"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: typed\n"
        f"metadata: {payload}\n"
        "always: true\n"
        "---\n\n# Typed\n",
        encoding="utf-8",
    )

    meta = loader.get_skill_metadata("typed")
    assert meta is not None
    # YAML parsed 'true' to Python True
    assert meta.get("always") is True
    # metadata is a parsed dict, not a JSON string
    assert isinstance(meta.get("metadata"), dict)
