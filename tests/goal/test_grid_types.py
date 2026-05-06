"""Tests for vingobot.goal.grid_types — Grid data structures and CognitionEvolutionAction."""

from __future__ import annotations

import pytest

from vingobot.goal.grid_types import (
    CognitionEvolutionAction,
    CognitionUsage,
    GridFile,
    GridModelRef,
    GridSkillRef,
    GridWorkflowStep,
    dict_to_grid_file,
    grid_file_to_dict,
)


# ── GridSkillRef ──────────────────────────────────────────────────────────────


class TestGridSkillRef:
    """Tests for GridSkillRef dataclass."""

    def test_minimal(self) -> None:
        """Creates a minimal GridSkillRef."""
        ref = GridSkillRef(name="test-skill")
        assert ref.name == "test-skill"
        assert ref.path == ""
        assert ref.relevance == "supporting"

    def test_full(self) -> None:
        """Creates a fully populated GridSkillRef."""
        ref = GridSkillRef(
            name="browser-skill",
            path="browser-automation",
            relevance="core",
        )
        assert ref.name == "browser-skill"
        assert ref.path == "browser-automation"
        assert ref.relevance == "core"


# ── GridModelRef ──────────────────────────────────────────────────────────────


class TestGridModelRef:
    """Tests for GridModelRef dataclass."""

    def test_minimal(self) -> None:
        """Creates a minimal GridModelRef."""
        ref = GridModelRef(name="gpt-4")
        assert ref.name == "gpt-4"
        assert ref.path == ""
        assert ref.relevance == "supporting"

    def test_full(self) -> None:
        """Creates a fully populated GridModelRef."""
        ref = GridModelRef(
            name="research-methodology",
            path="research/methodology.md",
            relevance="core",
        )
        assert ref.name == "research-methodology"
        assert ref.path == "research/methodology.md"
        assert ref.relevance == "core"


# ── GridWorkflowStep ──────────────────────────────────────────────────────────


class TestGridWorkflowStep:
    """Tests for GridWorkflowStep dataclass."""

    def test_minimal(self) -> None:
        """Creates a minimal GridWorkflowStep."""
        step = GridWorkflowStep(step=1, description="Analyze the problem")
        assert step.step == 1
        assert step.description == "Analyze the problem"
        assert step.skills == []

    def test_full(self) -> None:
        """Creates a fully populated GridWorkflowStep."""
        step = GridWorkflowStep(
            step=2,
            description="Perform web research",
            skills=["web_search", "data_extract"],
        )
        assert step.step == 2
        assert step.skills == ["web_search", "data_extract"]


# ── GridFile ──────────────────────────────────────────────────────────────────


class TestGridFile:
    """Tests for GridFile dataclass."""

    def test_minimal(self) -> None:
        """Creates a minimal GridFile."""
        grid = GridFile(domain="test")
        assert grid.domain == "test"
        assert grid.version == "1.0"
        assert grid.description == ""
        assert grid.trigram == ""
        assert grid.proficiency == 0.0
        assert grid.last_used == ""
        assert grid.skills == []
        assert grid.models == []
        assert grid.workflow == []
        assert grid.gaps == []

    def test_with_skills_and_models(self) -> None:
        """Creates GridFile with skills and models."""
        grid = GridFile(
            domain="research",
            description="Research domain grid",
            trigram="xun",
            proficiency=0.7,
            skills=[
                GridSkillRef(name="web-search", relevance="core"),
                GridSkillRef(name="data-analysis", relevance="frequent"),
            ],
            models=[
                GridModelRef(name="research-methodology", relevance="core"),
            ],
            workflow=[
                GridWorkflowStep(step=1, description="Search", skills=["web-search"]),
                GridWorkflowStep(step=2, description="Analyze", skills=["data-analysis"]),
            ],
            gaps=["Missing PDF parser skill"],
        )
        assert len(grid.skills) == 2
        assert len(grid.models) == 1
        assert len(grid.workflow) == 2
        assert len(grid.gaps) == 1
        assert grid.trigram == "xun"
        assert grid.proficiency == 0.7

    def test_to_dict_roundtrip(self) -> None:
        """Tests grid_file_to_dict and dict_to_grid_file produce a roundtrip."""
        original = GridFile(
            domain="test-domain",
            description="Test",
            trigram="qian",
            skills=[GridSkillRef(name="sk", relevance="core")],
            models=[GridModelRef(name="md", relevance="frequent")],
            workflow=[GridWorkflowStep(step=1, description="Do work")],
            gaps=["gap1"],
        )
        d = grid_file_to_dict(original)

        assert d["domain"] == "test-domain"
        assert d["version"] == "1.0"
        assert d["trigram"] == "qian"
        assert len(d["skills"]) == 1
        assert d["skills"][0]["name"] == "sk"
        assert d["skills"][0]["relevance"] == "core"
        assert d["models"][0]["name"] == "md"
        assert d["gaps"] == ["gap1"]

        # Roundtrip
        restored = dict_to_grid_file(d)
        assert restored.domain == original.domain
        assert restored.trigram == original.trigram
        assert restored.skills[0].name == original.skills[0].name


# ── CognitionUsage ────────────────────────────────────────────────────────────


class TestCognitionUsage:
    """Tests for CognitionUsage dataclass."""

    def test_defaults(self) -> None:
        """Creates a CognitionUsage with default values."""
        usage = CognitionUsage()
        assert usage.grids_loaded == []
        assert usage.skills_used == []
        assert usage.models_loaded == []
        assert usage.tools_failed == []
        assert usage.tool_calls_total == 0

    def test_full(self) -> None:
        """Creates a CognitionUsage with all fields."""
        usage = CognitionUsage(
            grids_loaded=["research", "development"],
            skills_used=["web-search", "code-review"],
            models_loaded=["research-methodology"],
            tools_failed=["broken_tool"],
            tool_calls_total=15,
        )
        assert usage.tool_calls_total == 15
        assert "web-search" in usage.skills_used
        assert len(usage.grids_loaded) == 2


# ── CognitionEvolutionAction ────────────────────────────────────────────────────


class TestCognitionEvolutionAction:
    """Tests for CognitionEvolutionAction dataclass."""

    def test_learn_skill(self) -> None:
        """Creates a 'learn_skill' action."""
        action = CognitionEvolutionAction(
            action="learn_skill",
            target_name="web-scraper",
            description="A web scraping skill",
        )
        assert action.action == "learn_skill"
        assert action.target_name == "web-scraper"
        assert action.description == "A web scraping skill"
        assert action.priority == 5
        assert action.context == {}

    def test_precipitate_skill(self) -> None:
        """Creates a 'precipitate_skill' action."""
        action = CognitionEvolutionAction(
            action="precipitate_skill",
            target_name="refactor-helper",
            description="Refactoring helper skill",
            priority=8,
        )
        assert action.action == "precipitate_skill"
        assert action.priority == 8

    def test_precipitate_model(self) -> None:
        """Creates a 'precipitate_model' action."""
        action = CognitionEvolutionAction(
            action="precipitate_model",
            target_name="error-patterns",
            description="Common error patterns model",
            source_task_id="task_123",
        )
        assert action.action == "precipitate_model"
        assert action.source_task_id == "task_123"

    def test_create_grid(self) -> None:
        """Creates a 'create_grid' action."""
        action = CognitionEvolutionAction(
            action="create_grid",
            target_name="web-dev",
            description="Web development domain grid",
        )
        assert action.action == "create_grid"

    def test_with_context(self) -> None:
        """Creates an action with extra context."""
        action = CognitionEvolutionAction(
            action="learn_skill",
            target_name="test",
            context={"reason": "capability gap detected", "logs": ["error1"]},
        )
        assert action.context["reason"] == "capability gap detected"
        assert action.context["logs"] == ["error1"]

    def test_default_action_type(self) -> None:
        """Default action type is 'learn_skill'."""
        action = CognitionEvolutionAction()
        assert action.action == "learn_skill"
        assert action.target_name == ""

    def test_custom_priority(self) -> None:
        """Priority defaults to 5."""
        action = CognitionEvolutionAction(priority=10)
        assert action.priority == 10
