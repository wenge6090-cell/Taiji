"""Tests for L4 truths layer — seeding, idempotency, content validation.

Covers:
- ``_seed_l4_truths()`` creates truth files from bundled templates
- Idempotency: existing files are never overwritten
- Content validation: created files contain valid JSON with required fields
- Integration: ``init_workspace(seed=True)`` creates truths directory
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vingobot.core.workspace import _seed_l4_truths, init_workspace


# ---------------------------------------------------------------------------
# _seed_l4_truths — direct unit tests
# ---------------------------------------------------------------------------


class TestSeedL4Truths:
    """Direct calls to ``_seed_l4_truths()``."""

    def test_creates_truth_files(self, tmp_path: Path) -> None:
        """Seeds both truth_identity.json and truth_safety.json from templates."""
        truths_dir = tmp_path / "truths"
        truths_dir.mkdir(parents=True)
        _seed_l4_truths(truths_dir)

        identity = truths_dir / "truth_identity.json"
        safety = truths_dir / "truth_safety.json"

        assert identity.is_file(), "truth_identity.json should exist"
        assert safety.is_file(), "truth_safety.json should exist"

    def test_files_are_valid_json(self, tmp_path: Path) -> None:
        """Seeded truth files contain valid JSON."""
        truths_dir = tmp_path / "truths"
        truths_dir.mkdir(parents=True)
        _seed_l4_truths(truths_dir)

        for fname in ("truth_identity.json", "truth_safety.json"):
            data = json.loads((truths_dir / fname).read_text(encoding="utf-8"))
            assert "version" in data, f"{fname} missing version"
            assert "title" in data, f"{fname} missing title"
            assert "type" in data, f"{fname} missing type"
            assert "rules" in data, f"{fname} missing rules"
            assert isinstance(data["rules"], list), f"{fname} rules must be a list"
            assert len(data["rules"]) > 0, f"{fname} must have at least one rule"

    def test_rules_have_required_fields(self, tmp_path: Path) -> None:
        """Each rule must have id and statement."""
        truths_dir = tmp_path / "truths"
        truths_dir.mkdir(parents=True)
        _seed_l4_truths(truths_dir)

        for fname in ("truth_identity.json", "truth_safety.json"):
            data = json.loads((truths_dir / fname).read_text(encoding="utf-8"))
            for rule in data["rules"]:
                assert "id" in rule, f"{fname} rule missing id"
                assert "statement" in rule, f"{fname} rule missing statement"

    def test_workspace_seeded_at_added(self, tmp_path: Path) -> None:
        """Seeded file gets workspace_seeded_at timestamp."""
        truths_dir = tmp_path / "truths"
        truths_dir.mkdir(parents=True)
        _seed_l4_truths(truths_dir)

        for fname in ("truth_identity.json", "truth_safety.json"):
            data = json.loads((truths_dir / fname).read_text(encoding="utf-8"))
            assert "workspace_seeded_at" in data, f"{fname} missing workspace_seeded_at"

    def test_idempotent_does_not_overwrite(self, tmp_path: Path) -> None:
        """Existing files are NOT overwritten by _seed_l4_truths."""
        truths_dir = tmp_path / "truths"
        truths_dir.mkdir(parents=True)

        # Create a custom truth file first
        custom = truths_dir / "truth_identity.json"
        custom.write_text(
            json.dumps({"version": 999, "title": "Custom", "type": "custom", "rules": []}, ensure_ascii=False),
            encoding="utf-8",
        )

        _seed_l4_truths(truths_dir)

        # Verify content is still custom (not overwritten)
        data = json.loads(custom.read_text(encoding="utf-8"))
        assert data["version"] == 999
        assert data["title"] == "Custom"

    def test_truth_types_are_correct(self, tmp_path: Path) -> None:
        """truth_identity is type 'identity', truth_safety is type 'safety'."""
        truths_dir = tmp_path / "truths"
        truths_dir.mkdir(parents=True)
        _seed_l4_truths(truths_dir)

        identity = json.loads((truths_dir / "truth_identity.json").read_text(encoding="utf-8"))
        assert identity["type"] == "identity"

        safety = json.loads((truths_dir / "truth_safety.json").read_text(encoding="utf-8"))
        assert safety["type"] == "safety"

    def test_identity_rules_have_correct_structure(self, tmp_path: Path) -> None:
        """Identity truths should include self-awareness invariants."""
        truths_dir = tmp_path / "truths"
        truths_dir.mkdir(parents=True)
        _seed_l4_truths(truths_dir)

        data = json.loads((truths_dir / "truth_identity.json").read_text(encoding="utf-8"))
        statements = [r["statement"] for r in data["rules"]]

        # Must include at least identity/core principles
        assert any("vingobot" in s.lower() for s in statements)
        assert any("诚实" in s for s in statements)

    def test_safety_rules_have_correct_structure(self, tmp_path: Path) -> None:
        """Safety truths should include file system and command restrictions."""
        truths_dir = tmp_path / "truths"
        truths_dir.mkdir(parents=True)
        _seed_l4_truths(truths_dir)

        data = json.loads((truths_dir / "truth_safety.json").read_text(encoding="utf-8"))
        statements = [r["statement"] for r in data["rules"]]

        assert any("路径穿越" in s for s in statements)
        assert any("rm" in s.lower() or "format" in s.lower() for s in statements)
        assert any("sudo" in s.lower() for s in statements)

    def test_immutable_flag_set(self, tmp_path: Path) -> None:
        """All truth files should have immutable: true."""
        truths_dir = tmp_path / "truths"
        truths_dir.mkdir(parents=True)
        _seed_l4_truths(truths_dir)

        for fname in ("truth_identity.json", "truth_safety.json"):
            data = json.loads((truths_dir / fname).read_text(encoding="utf-8"))
            assert data.get("immutable") is True, f"{fname} should be immutable"


# ---------------------------------------------------------------------------
# init_workspace integration tests
# ---------------------------------------------------------------------------


class TestInitWorkspaceTruths:
    """Integration tests: init_workspace() creates L4 truths directory."""

    def test_init_workspace_creates_truths_dir(self, tmp_path: Path) -> None:
        """init_workspace(seed=True) creates the truths subdirectory under cognition."""
        root = tmp_path / ".taiji"
        wp = init_workspace(root, seed=True)

        assert wp.truths.is_dir()
        assert wp.truths.name == "truths"
        assert wp.truths.parent.name == "cognition"

    def test_init_workspace_seeds_truths(self, tmp_path: Path) -> None:
        """init_workspace(seed=True) populates L4 truth files."""
        root = tmp_path / ".taiji"
        wp = init_workspace(root, seed=True)

        assert (wp.truths / "truth_identity.json").is_file()
        assert (wp.truths / "truth_safety.json").is_file()

    def test_init_workspace_no_seed_skips_truths(self, tmp_path: Path) -> None:
        """init_workspace(seed=False) creates directory but does not seed files."""
        root = tmp_path / ".taiji"
        wp = init_workspace(root, seed=False)

        assert wp.truths.is_dir()
        assert not (wp.truths / "truth_identity.json").exists()

    def test_workspace_paths_has_truths_field(self, tmp_path: Path) -> None:
        """WorkspacePaths dataclass includes the 'truths' field."""
        wp = init_workspace(tmp_path / ".taiji", seed=True)
        assert hasattr(wp, "truths")
        assert str(wp.truths).endswith("cognition\\truths") or str(wp.truths).endswith("cognition/truths")


# ---------------------------------------------------------------------------
# Graceful error handling
# ---------------------------------------------------------------------------


class TestSeedL4TruthsErrorHandling:
    """Seeding error handling — invalid target, missing templates."""

    def test_nonexistent_truths_dir_does_not_crash(self, tmp_path: Path) -> None:
        """Calling with non-existent directory fails silently (except to logger)."""
        truths_dir = tmp_path / "nonexistent" / "truths"
        # No error should be raised
        _seed_l4_truths(truths_dir)
        # Directory still doesn't exist
        assert not truths_dir.is_dir()
