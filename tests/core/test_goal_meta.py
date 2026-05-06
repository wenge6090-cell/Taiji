"""Tests for GoalMeta — field serialisation, backward compatibility, and updates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vingobot.core.goal_meta import (
    GoalMeta,
    SelfDrivenConfig,
    read_goal_meta,
    update_goal_meta,
    write_goal_meta,
)
from vingobot.core.workspace import init_workspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_test_workspace(tmp_path: Path) -> Path:
    """Initialise a .taiji workspace and return its root."""
    root = tmp_path / ".taiji"
    init_workspace(root, seed=False)
    return root


# ---------------------------------------------------------------------------
# Tests — GoalMeta serialisation
# ---------------------------------------------------------------------------


class TestGoalMetaSerialisation:
    """GoalMeta to_dict / from_dict round-trip consistency."""

    def test_round_trip_full(self) -> None:
        """Full GoalMeta survives to_dict → from_dict unchanged."""
        original = GoalMeta(
            id="test-goal",
            name="测试目标",
            description="A test goal for serialisation",
            status="active",
            priority=7,
            created_at="2026-01-01T00:00:00",
            last_active="2026-01-02T00:00:00",
            self_driven=SelfDrivenConfig(enabled=True, interval_minutes=15),
            last_anqu_at="2026-01-03T00:00:00",
            warnings=["low balance", "timeout"],
            rounds_completed=42,
            auto_managed=False,
        )

        restored = GoalMeta.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.status == original.status
        assert restored.priority == original.priority
        assert restored.created_at == original.created_at
        assert restored.last_active == original.last_active
        assert restored.self_driven.enabled == original.self_driven.enabled
        assert restored.self_driven.interval_minutes == original.self_driven.interval_minutes
        assert restored.last_anqu_at == original.last_anqu_at
        assert restored.warnings == original.warnings
        assert restored.rounds_completed == original.rounds_completed
        assert restored.auto_managed == original.auto_managed

    def test_round_trip_minimal(self) -> None:
        """Minimal GoalMeta (only id) survives to_dict → from_dict."""
        original = GoalMeta(id="minimal")
        restored = GoalMeta.from_dict(original.to_dict())

        assert restored.id == "minimal"
        assert restored.status == "active"  # default
        assert restored.priority == 5  # default
        assert restored.warnings == []  # default
        assert restored.last_anqu_at == ""  # default
        assert restored.auto_managed is True  # default
        assert restored.rounds_completed == 0  # default

    def test_to_dict_is_json_serialisable(self) -> None:
        """to_dict output can be serialised to JSON."""
        meta = GoalMeta(
            id="json-test",
            warnings=["error1", "error2"],
            self_driven=SelfDrivenConfig(enabled=True, interval_minutes=10),
        )
        d = meta.to_dict()
        # Should not raise
        json_str = json.dumps(d, ensure_ascii=False, indent=2)
        assert "json-test" in json_str
        assert "error1" in json_str


# ---------------------------------------------------------------------------
# Tests — Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """GoalMeta.from_dict handles old meta.json files missing new fields."""

    def test_old_meta_no_control_fields(self) -> None:
        """Old meta.json without control-plane fields → defaults applied."""
        old_data = {
            "id": "old-goal",
            "name": "Old Goal",
            "description": "Created before control plane existed",
            "status": "active",
            "priority": 5,
            "created_at": "2025-01-01T00:00:00",
            "last_active": "2025-06-01T00:00:00",
            "self_driven": {"enabled": False, "interval_minutes": 30},
            # No last_anqu_at, warnings, rounds_completed, auto_managed
        }

        meta = GoalMeta.from_dict(old_data)

        assert meta.id == "old-goal"
        assert meta.status == "active"
        # Control-plane defaults
        assert meta.last_anqu_at == ""
        assert meta.warnings == []
        assert meta.rounds_completed == 0
        assert meta.auto_managed is True

    def test_old_meta_partial_control_fields(self) -> None:
        """Old meta.json with some but not all control fields."""
        old_data = {
            "id": "partial-goal",
            "name": "Partial",
            "status": "active",
            "priority": 3,
            "created_at": "2025-01-01T00:00:00",
            "last_active": "",
            "self_driven": {},
            "warnings": ["existing warning"],
            # last_anqu_at, rounds_completed, auto_managed missing
        }

        meta = GoalMeta.from_dict(old_data)

        assert meta.warnings == ["existing warning"]  # preserved
        assert meta.last_anqu_at == ""  # default
        assert meta.rounds_completed == 0  # default
        assert meta.auto_managed is True  # default

    def test_old_meta_null_warnings(self) -> None:
        """Warnings field is null/None in old data → treated as empty list."""
        old_data = {
            "id": "null-warn",
            "name": "Null Warnings",
            "status": "active",
            "priority": 5,
            "created_at": "",
            "last_active": "",
            "self_driven": {},
            "warnings": None,  # JSON null
        }

        meta = GoalMeta.from_dict(old_data)
        assert meta.warnings == []

    def test_old_meta_empty_self_driven(self) -> None:
        """Empty self_driven dict → defaults applied."""
        old_data = {
            "id": "no-sd",
            "name": "No SelfDriven",
            "status": "active",
            "priority": 5,
            "created_at": "",
            "last_active": "",
            "self_driven": {},
        }

        meta = GoalMeta.from_dict(old_data)
        assert meta.self_driven.enabled is False
        assert meta.self_driven.interval_minutes == 30


# ---------------------------------------------------------------------------
# Tests — update_goal_meta
# ---------------------------------------------------------------------------


class TestUpdateGoalMeta:
    """update_goal_meta patches fields atomically."""

    def test_update_single_field(self, tmp_path: Path) -> None:
        """update_goal_meta modifies only the specified field."""
        root = _init_test_workspace(tmp_path)
        meta = GoalMeta(
            id="update-test",
            name="Update Test",
            status="active",
            priority=5,
            last_anqu_at="2026-01-01T00:00:00",
        )
        write_goal_meta("update-test", meta)

        updated = update_goal_meta("update-test", status="paused")
        assert updated is not None
        assert updated.status == "paused"
        assert updated.priority == 5  # unchanged
        assert updated.name == "Update Test"  # unchanged
        assert updated.last_anqu_at == "2026-01-01T00:00:00"  # unchanged

    def test_update_multiple_fields(self, tmp_path: Path) -> None:
        """update_goal_meta can modify multiple fields at once."""
        root = _init_test_workspace(tmp_path)
        meta = GoalMeta(id="multi-test", status="active", priority=3,
                        warnings=["old warning"])
        write_goal_meta("multi-test", meta)

        updated = update_goal_meta(
            "multi-test",
            status="archived",
            warnings=["new warning"],
            rounds_completed=10,
        )
        assert updated is not None
        assert updated.status == "archived"
        assert updated.warnings == ["new warning"]
        assert updated.rounds_completed == 10
        assert updated.priority == 3  # unchanged

    def test_update_persists_to_disk(self, tmp_path: Path) -> None:
        """update_goal_meta writes changes to meta.json on disk."""
        root = _init_test_workspace(tmp_path)
        meta = GoalMeta(id="disk-test", status="active")
        write_goal_meta("disk-test", meta)

        update_goal_meta("disk-test", status="paused")

        # Re-read from disk
        reloaded = read_goal_meta("disk-test")
        assert reloaded is not None
        assert reloaded.status == "paused"

    def test_update_nonexistent_goal(self, tmp_path: Path) -> None:
        """update_goal_meta returns None for non-existent goal."""
        root = _init_test_workspace(tmp_path)
        result = update_goal_meta("no-such-goal", status="paused")
        assert result is None

    def test_update_preserves_control_fields(self, tmp_path: Path) -> None:
        """update_goal_meta does not overwrite unmentioned control fields."""
        root = _init_test_workspace(tmp_path)
        meta = GoalMeta(
            id="ctrl-test",
            status="active",
            last_anqu_at="2026-01-01T00:00:00",
            warnings=["w1"],
            auto_managed=False,
            rounds_completed=5,
        )
        write_goal_meta("ctrl-test", meta)

        # Update only status
        update_goal_meta("ctrl-test", status="completed")

        reloaded = read_goal_meta("ctrl-test")
        assert reloaded is not None
        assert reloaded.status == "completed"
        assert reloaded.last_anqu_at == "2026-01-01T00:00:00"
        assert reloaded.warnings == ["w1"]
        assert reloaded.auto_managed is False
        assert reloaded.rounds_completed == 5


# ---------------------------------------------------------------------------
# Tests — File I/O
# ---------------------------------------------------------------------------


class TestGoalMetaFileIO:
    """read_goal_meta / write_goal_meta disk operations."""

    def test_write_and_read(self, tmp_path: Path) -> None:
        """write_goal_meta + read_goal_meta round-trip via disk."""
        root = _init_test_workspace(tmp_path)
        meta = GoalMeta(
            id="io-test",
            name="IO Test",
            status="active",
            priority=8,
            last_anqu_at="2026-05-04T00:00:00",
            warnings=["disk warning"],
            auto_managed=False,
        )
        write_goal_meta("io-test", meta)

        loaded = read_goal_meta("io-test")
        assert loaded is not None
        assert loaded.id == "io-test"
        assert loaded.priority == 8
        assert loaded.last_anqu_at == "2026-05-04T00:00:00"
        assert loaded.warnings == ["disk warning"]
        assert loaded.auto_managed is False

    def test_read_nonexistent(self, tmp_path: Path) -> None:
        """read_goal_meta returns None for non-existent goal."""
        root = _init_test_workspace(tmp_path)
        assert read_goal_meta("no-such-goal") is None

    def test_read_corrupted_json(self, tmp_path: Path) -> None:
        """read_goal_meta returns None for corrupted meta.json."""
        root = _init_test_workspace(tmp_path)
        goal_dir = root / "goals" / "corrupt"
        goal_dir.mkdir(parents=True, exist_ok=True)
        (goal_dir / "meta.json").write_text("not valid json{{{", encoding="utf-8")

        assert read_goal_meta("corrupt") is None
