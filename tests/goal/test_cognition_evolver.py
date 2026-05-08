"""Unit tests for cognition_evolver — reverse-update (双向链接) logic.

Tests two reverse-update paths:

1. **create_domain_grid**: after creating a grid, reverse-updates each model in
   ``source_models`` to append the grid domain to the model's ``source_grids``.
2. **create_truth**: after creating a truth, reverse-updates each grid in
   ``source_grids`` to append the truth name to the grid's ``source_truths``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vingobot.core.workspace import get_workspace_paths, init_workspace
from vingobot.goal.cognition_evolver import create_domain_grid, create_truth
from vingobot.goal.grid_types import grid_file_to_dict


def _init_ws(tmp_path: Path) -> Path:
    """Initialise a test workspace without seeding."""
    root = tmp_path / ".taiji"
    init_workspace(root, seed=False)
    return root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(wp, name: str, *, source_grids: list[str] | None = None) -> Path:
    """Write a minimal L2 model JSON file and return its path."""
    wp.models.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "name": name,
        "description": f"Model {name}",
        "confidence": 0.5,
        "source_skills": [],
        "source_grids": source_grids or [],
        "content": f"Content of {name}",
        "version": "1.0",
    }
    p = wp.models / f"{name}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _make_grid(wp, domain: str, *, source_truths: list[str] | None = None) -> Path:
    """Write a minimal L3 grid JSON file and return its path."""
    wp.grids.mkdir(parents=True, exist_ok=True)
    grid_data = grid_file_to_dict(
        __import__("vingobot.goal.grid_types", fromlist=["GridFile"]).GridFile(
            domain=domain,
            description=f"Grid {domain}",
            source_truths=source_truths or [],
        )
    )
    p = wp.grids / f"{domain}.json"
    p.write_text(json.dumps(grid_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# create_model  — create_model writes source_grids correctly
# ---------------------------------------------------------------------------


class TestCreateModelSourceGrids:
    """create_model() writes source_grids into the model JSON."""

    @pytest.mark.asyncio
    async def test_writes_source_grids(self, tmp_path: Path) -> None:
        """Model JSON includes source_grids when provided."""
        from vingobot.goal.cognition_evolver import create_model

        _init_ws(tmp_path)
        wp = get_workspace_paths()

        result = await create_model("test-model", source_grids=["grid-a", "grid-b"])
        assert result is True

        data = json.loads((wp.models / "test-model.json").read_text(encoding="utf-8"))
        assert data["source_grids"] == ["grid-a", "grid-b"]

    @pytest.mark.asyncio
    async def test_defaults_to_empty_list(self, tmp_path: Path) -> None:
        """source_grids defaults to [] when not provided."""
        from vingobot.goal.cognition_evolver import create_model

        _init_ws(tmp_path)
        wp = get_workspace_paths()

        await create_model("default-model")
        data = json.loads((wp.models / "default-model.json").read_text(encoding="utf-8"))
        assert data["source_grids"] == []


# ---------------------------------------------------------------------------
# create_domain_grid — reverse-updates models' source_grids
# ---------------------------------------------------------------------------


class TestGridReverseUpdateModels:
    """create_domain_grid() reverse-updates each source model's source_grids."""

    @pytest.mark.asyncio
    async def test_reverse_update_single_model(self, tmp_path: Path) -> None:
        """Grid domain appended to the model's source_grids."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        _make_model(wp, "research-pattern", source_grids=["existing-grid"])

        llm_json = json.dumps({"source_models": ["research-pattern"]})
        result = await create_domain_grid("new-domain", llm_analysis=llm_json)
        assert result is True

        updated = json.loads((wp.models / "research-pattern.json").read_text(encoding="utf-8"))
        assert "new-domain" in updated["source_grids"]
        assert "existing-grid" in updated["source_grids"]  # preserved

    @pytest.mark.asyncio
    async def test_no_duplicate_appended(self, tmp_path: Path) -> None:
        """Grid domain NOT appended if already present in source_grids."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        _make_model(wp, "dup-model", source_grids=["already-there"])

        llm_json = json.dumps({"source_models": ["dup-model"]})
        await create_domain_grid("already-there", llm_analysis=llm_json)

        updated = json.loads((wp.models / "dup-model.json").read_text(encoding="utf-8"))
        assert updated["source_grids"].count("already-there") == 1

    @pytest.mark.asyncio
    async def test_model_file_missing_no_crash(self, tmp_path: Path) -> None:
        """Reverse-update silently skips non-existent model files."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        llm_json = json.dumps({"source_models": ["nonexistent-model"]})
        # Must not raise
        result = await create_domain_grid("survivor-grid", llm_analysis=llm_json)
        assert result is True

    @pytest.mark.asyncio
    async def test_multiple_source_models_all_updated(self, tmp_path: Path) -> None:
        """All models in source_models get reverse-updated."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        _make_model(wp, "model-x")
        _make_model(wp, "model-y")
        _make_model(wp, "model-z")

        llm_json = json.dumps({"source_models": ["model-x", "model-y"]})
        await create_domain_grid("multi-grid", llm_analysis=llm_json)

        for name in ("model-x", "model-y"):
            data = json.loads((wp.models / f"{name}.json").read_text(encoding="utf-8"))
            assert "multi-grid" in data["source_grids"], f"{name} should have multi-grid"

        # model-z was NOT in source_models
        data_z = json.loads((wp.models / "model-z.json").read_text(encoding="utf-8"))
        assert "multi-grid" not in data_z["source_grids"]

    @pytest.mark.asyncio
    async def test_empty_source_models_noop(self, tmp_path: Path) -> None:
        """No reverse-update when source_models is empty."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        _make_model(wp, "untouched-model")
        llm_json = json.dumps({"source_models": []})
        await create_domain_grid("solo-grid", llm_analysis=llm_json)

        data = json.loads((wp.models / "untouched-model.json").read_text(encoding="utf-8"))
        assert data["source_grids"] == []


# ---------------------------------------------------------------------------
# create_truth — reverse-updates grids' source_truths
# ---------------------------------------------------------------------------


class TestTruthReverseUpdateGrids:
    """create_truth() reverse-updates each source grid's source_truths."""

    @pytest.mark.asyncio
    async def test_reverse_update_single_grid(self, tmp_path: Path) -> None:
        """Truth name appended to the grid's source_truths."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        _make_grid(wp, "refactor-grid", source_truths=["existing-truth"])

        result = await create_truth(
            "truth_new_pattern",
            source_grids=["refactor-grid"],
            rules=[{"id": "r1", "statement": "test"}],
        )
        assert result is True

        grid_data = json.loads((wp.grids / "refactor-grid.json").read_text(encoding="utf-8"))
        assert "truth_new_pattern" in grid_data["source_truths"]
        assert "existing-truth" in grid_data["source_truths"]  # preserved

    @pytest.mark.asyncio
    async def test_no_duplicate_appended(self, tmp_path: Path) -> None:
        """Truth name NOT appended if already in source_truths."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        _make_grid(wp, "dup-grid", source_truths=["truth_dup"])

        await create_truth(
            "truth_dup",
            source_grids=["dup-grid"],
            rules=[{"id": "r1", "statement": "dup"}],
            # truth exists? No — it's a new truth named "truth_dup"
            # But we pre-set source_truths=["truth_dup"] on the grid
        )

        grid_data = json.loads((wp.grids / "dup-grid.json").read_text(encoding="utf-8"))
        assert grid_data["source_truths"].count("truth_dup") == 1

    @pytest.mark.asyncio
    async def test_grid_file_missing_no_crash(self, tmp_path: Path) -> None:
        """Reverse-update silently skips non-existent grid files."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        # Must not raise
        result = await create_truth(
            "truth_survivor",
            source_grids=["ghost-grid"],
            rules=[{"id": "r1", "statement": "alive"}],
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_multiple_source_grids_all_updated(self, tmp_path: Path) -> None:
        """All grids in source_grids get reverse-updated."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        _make_grid(wp, "grid-alpha")
        _make_grid(wp, "grid-beta")
        _make_grid(wp, "grid-gamma")

        await create_truth(
            "truth_multi",
            source_grids=["grid-alpha", "grid-beta"],
            rules=[{"id": "r1", "statement": "multi"}],
        )

        for name in ("grid-alpha", "grid-beta"):
            data = json.loads((wp.grids / f"{name}.json").read_text(encoding="utf-8"))
            assert "truth_multi" in data["source_truths"], f"{name} should have truth_multi"

        # grid-gamma was NOT in source_grids
        data_g = json.loads((wp.grids / "grid-gamma.json").read_text(encoding="utf-8"))
        assert "truth_multi" not in data_g["source_truths"]

    @pytest.mark.asyncio
    async def test_empty_source_grids_noop(self, tmp_path: Path) -> None:
        """No reverse-update when source_grids is empty."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        _make_grid(wp, "untouched-grid")
        await create_truth(
            "truth_solo",
            source_grids=[],
            rules=[{"id": "r1", "statement": "solo"}],
        )

        data = json.loads((wp.grids / "untouched-grid.json").read_text(encoding="utf-8"))
        assert data["source_truths"] == []

    @pytest.mark.asyncio
    async def test_truth_idempotent_no_reverse_update(self, tmp_path: Path) -> None:
        """When truth already exists, reverse-update is NOT triggered."""
        _init_ws(tmp_path)
        wp = get_workspace_paths()

        _make_grid(wp, "idem-grid")

        # First creation succeeds
        result1 = await create_truth(
            "truth_idem",
            source_grids=["idem-grid"],
            rules=[{"id": "r1", "statement": "first"}],
        )
        assert result1 is True

        # Second creation is idempotent (returns False)
        result2 = await create_truth(
            "truth_idem",
            source_grids=["idem-grid"],
            rules=[{"id": "r1", "statement": "second"}],
        )
        assert result2 is False

        # Grid should only have one entry
        grid_data = json.loads((wp.grids / "idem-grid.json").read_text(encoding="utf-8"))
        assert grid_data["source_truths"].count("truth_idem") == 1
