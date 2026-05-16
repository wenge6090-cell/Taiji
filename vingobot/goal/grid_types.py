"""
L3 认知格栅类型定义 — Standardised Grid file format + Evolution types.

All data structures for the three-layer cognitive architecture:

- **L1 Skills**: Reusable executable steps (e.g. test-runner, code-reviewer).
- **L2 Models**: Abstracted experience patterns (e.g. error-handling-pattern).
- **L3 Grids**: Domain-specific registries connecting L1 + L2 to task
  domains (e.g. research-report.json).

Each grid is a JSON file on disk under ``<workspace>/.taiji/cognition/grids/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Grid JSON data structures
# ---------------------------------------------------------------------------


@dataclass
class GridSkillRef:
    """Reference to an L1 skill within an L3 grid."""

    name: str
    """Skill directory name (e.g. 'browser-automation', 'pdf-parser')."""

    path: str = ""
    """Relative path from the skills directory; auto-resolved if empty."""

    relevance: Literal["core", "frequent", "supporting", "mandatory"] = "supporting"
    """How relevant this skill is to the grid's domain."""


@dataclass
class GridModelRef:
    """Reference to an L2 experience model within an L3 grid."""

    name: str
    """Model filename (stem only, e.g. 'research-methodology')."""

    path: str = ""
    """Relative path from the models directory; auto-resolved if empty."""

    relevance: Literal["core", "frequent", "supporting", "mandatory"] = "supporting"
    """How relevant this model is to the grid's domain."""


@dataclass
class GridWorkflowStep:
    """A recommended workflow step within an L3 grid."""

    step: int
    description: str
    skills: list[str] = field(default_factory=list)
    """Skill names relevant to this step."""


@dataclass
class GridFile:
    """Standardised L3 cognitive grid JSON format.

    Serialised to/from ``<workspace>/.taiji/cognition/grids/<domain>.json``.
    """

    domain: str
    """Domain name — also used as the filename stem."""

    description: str = ""
    """What this grid is for (1-2 sentences)."""

    version: str = "1.0"
    """Semantic version for evolution tracking."""

    trigram: str = ""
    """Associated trigram (qian/kun/zhen/xun/kan/li/gen/dui) if applicable."""

    proficiency: float = 0.0
    """Estimated mastery 0.0–1.0, updated after each use."""

    last_used: str = ""
    """ISO-8601 timestamp of last load."""

    models: list[GridModelRef] = field(default_factory=list)
    """L2 model references relevant to this domain."""

    skills: list[GridSkillRef] = field(default_factory=list)
    """L1 skill references relevant to this domain."""

    source_models: list[str] = field(default_factory=list)
    """Names of L2 models that this grid was built from (L2→L3 lineage)."""

    source_truths: list[str] = field(default_factory=list)
    """Names of L4 truths that reference this grid (L4→L3 back-link)."""

    emergence_score: float = 0.0
    """Emergence score 0.0–1.0 from L2→L3 compression (reserved)."""

    workflow: list[GridWorkflowStep] = field(default_factory=list)
    """Recommended execution workflow."""

    gaps: list[str] = field(default_factory=list)
    """Identified capability gaps (triggers for learn_skill)."""



# ---------------------------------------------------------------------------
# Cognitive usage tracking
# ---------------------------------------------------------------------------


@dataclass
class CognitionUsage:
    """Records which cognitive assets were used during a task inner loop.

    Populated by Weaver during ``weave()`` and passed to Anqu for
    evolution decision-making.
    """

    grids_loaded: list[str] = field(default_factory=list)
    """Names of grids Yang loaded via read_file."""

    skills_used: list[str] = field(default_factory=list)
    """Skill names that Yang discovered and used."""

    models_loaded: list[str] = field(default_factory=list)
    """Model names that Yang loaded via read_file."""

    tools_failed: list[str] = field(default_factory=list)
    """Tool names that were called but failed during execution."""

    tool_calls_total: int = 0
    """Total number of tool calls made this round."""


# ---------------------------------------------------------------------------
# Cognitive evolution types
# ---------------------------------------------------------------------------

EvolutionActionType = Literal[
    "learn_skill",
    "precipitate_skill",
    "precipitate_model",
    "create_grid",
    "research",
    "investigate",
    "review_blueprint",
]


@dataclass
class CognitionEvolutionAction:
    """A single evolution action decided by Anqu.

    These are enqueued under the special ``cognition-evolution`` goal and
    processed by the WorkerPool asynchronously.
    """

    action: EvolutionActionType = "learn_skill"
    """What kind of evolution to perform."""

    target_name: str = ""
    """Name of the skill/model/grid to create or update."""

    description: str = ""
    """Natural-language description of what to build."""

    source_task_id: str = ""
    """The task that triggered this evolution."""

    source_goal_id: str = ""
    """The goal that contained the triggering task."""

    priority: int = 5
    """Priority 1–10 (higher = more urgent)."""

    context: dict[str, Any] = field(default_factory=dict)
    """Extra context for the evolution worker (e.g. failure logs, SOP text)."""


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def grid_file_to_dict(g: GridFile) -> dict[str, Any]:
    """Convert a ``GridFile`` to a JSON-serialisable dict."""
    return {
        "domain": g.domain,
        "description": g.description,
        "version": g.version,
        "trigram": g.trigram,
        "proficiency": g.proficiency,
        "last_used": g.last_used,
        "models": [
            {
                "name": m.name,
                "path": m.path,
                "relevance": m.relevance,
            }
            for m in g.models
        ],
        "skills": [
            {
                "name": s.name,
                "path": s.path,
                "relevance": s.relevance,
            }
            for s in g.skills
        ],
        "source_models": g.source_models,
        "source_truths": g.source_truths,
        "emergence_score": g.emergence_score,
        "workflow": [
            {
                "step": w.step,
                "description": w.description,
                "skills": w.skills,
            }
            for w in g.workflow
        ],
        "gaps": g.gaps,
    }


def dict_to_grid_file(d: dict[str, Any]) -> GridFile:
    """Parse a dict (from JSON) into a ``GridFile`` instance."""
    return GridFile(
        domain=d.get("domain", ""),
        description=d.get("description", ""),
        version=d.get("version", "1.0"),
        trigram=d.get("trigram", ""),
        proficiency=float(d.get("proficiency", 0.0)),
        last_used=d.get("last_used", ""),
        models=[
            GridModelRef(
                name=m if isinstance(m, str) else m.get("name", ""),
                path="" if isinstance(m, str) else m.get("path", ""),
                relevance="supporting" if isinstance(m, str) else m.get("relevance", "supporting"),
            )
            for m in d.get("models", [])
        ],
        skills=[
            GridSkillRef(
                name=s if isinstance(s, str) else s.get("name", ""),
                path="" if isinstance(s, str) else s.get("path", ""),
                relevance="supporting" if isinstance(s, str) else s.get("relevance", "supporting"),
            )
            for s in d.get("skills", [])
        ],
        source_models=d.get("source_models", []),
        source_truths=d.get("source_truths", []),
        emergence_score=float(d.get("emergence_score", 0.0)),
        workflow=[
            GridWorkflowStep(
                step=w.get("step", 0),
                description=w.get("description", ""),
                skills=w.get("skills", []),
            )
            for w in d.get("workflow", [])
        ],
        gaps=list(d.get("gaps", [])),
    )


# ---------------------------------------------------------------------------
# L4 Truth data structures
# ---------------------------------------------------------------------------


@dataclass
class TruthFile:
    """Standardised L4 immutable truth JSON format.

    Serialised to/from ``<workspace>/.taiji/cognition/truths/<name>.json``.
    """

    title: str
    """Short title for this truth."""

    type: str = "pattern"
    """Truth category: identity / safety / pattern."""

    version: int = 1
    """Monotonic version number."""

    immutable: bool = True
    """Whether this truth is immutable once written."""

    confidence: float = 0.0
    """Confidence score 0.0–1.0."""

    source_grids: list[str] = field(default_factory=list)
    """Names of L3 grids this truth was distilled from (L3→L4 lineage)."""

    rules: list[dict[str, Any]] = field(default_factory=list)
    """Truth statements as list of {id, statement} dicts."""

    updated_at: str = ""
    """ISO-8601 timestamp of last update."""


# ---------------------------------------------------------------------------
# L4 Truth serialisation helpers
# ---------------------------------------------------------------------------


def truth_file_to_dict(t: TruthFile) -> dict[str, Any]:
    """Convert a ``TruthFile`` to a JSON-serialisable dict."""
    return {
        "title": t.title,
        "type": t.type,
        "version": t.version,
        "immutable": t.immutable,
        "confidence": t.confidence,
        "source_grids": t.source_grids,
        "rules": t.rules,
        "updated_at": t.updated_at,
    }


def dict_to_truth_file(d: dict[str, Any]) -> TruthFile:
    """Parse a dict (from JSON) into a ``TruthFile`` instance."""
    return TruthFile(
        title=d.get("title", ""),
        type=d.get("type", "pattern"),
        version=int(d.get("version", 1)),
        immutable=bool(d.get("immutable", True)),
        confidence=float(d.get("confidence", 0.0)),
        source_grids=list(d.get("source_grids", [])),
        rules=list(d.get("rules", [])),
        updated_at=d.get("updated_at", ""),
    )


# ---------------------------------------------------------------------------
# 六爻 / 四象 / 八卦 meta-cognitive grid dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SixiangMode:
    """A single cognitive mode within the four-sixiang grid (太极-四象)."""

    name: str = ""
    temperature: float = 0.5
    top_p: float = 0.85
    role_prompt: str = ""
    description: str = ""
    act_phase: str = ""


@dataclass
class SixiangGrid:
    """The complete four-sixiang grid (太极-四象.json)."""

    modes: dict[str, SixiangMode] = field(default_factory=dict)


@dataclass
class YaoNode:
    """A single yao within the six-yao grid (太极-六爻)."""

    name: str = ""
    phase: str = ""
    rule: str = ""
    execution_hint: str = ""
    temperature_override: float | None = None


@dataclass
class LiuyaoGrid:
    """The complete six-yao grid (太极-六爻.json)."""

    nodes: dict[str, YaoNode] = field(default_factory=dict)


@dataclass
class TrigramNode:
    """A single trigram within the bagua grid (太极-八卦)."""

    name: str = ""
    context: str = ""
    prompt_prefix: str = ""
    default_sixiang: str = "少阴"
    preferred_actions: list[str] = field(default_factory=list)


@dataclass
class BaguaGrid:
    """The complete bagua grid (太极-八卦.json)."""

    nodes: dict[str, TrigramNode] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Meta-grid parsers
# ---------------------------------------------------------------------------

# 四象 top_p 填值（太极-四象.json 中无 top_p 字段）
_SIXIANG_DEFAULT_TOP_P: dict[str, float] = {
    "老阳": 0.95,
    "少阳": 0.85,
    "少阴": 0.75,
    "老阴": 0.65,
}


def parse_sixiang_grid(raw: dict[str, Any]) -> SixiangGrid:
    """Parse the 太极-四象.json raw dict into a SixiangGrid."""
    modes: dict[str, SixiangMode] = {}
    for name, data in (raw.get("modes") or {}).items():
        if not isinstance(data, dict):
            continue
        modes[name] = SixiangMode(
            name=name,
            temperature=float(data.get("temperature", 0.5)),
            top_p=_SIXIANG_DEFAULT_TOP_P.get(name, 0.85),
            role_prompt=str(data.get("role_prompt", "")),
            description=str(data.get("description", "")),
            act_phase=str(data.get("act_phase", "")),
        )
    return SixiangGrid(modes=modes)


def parse_liuyao_grid(raw: dict[str, Any]) -> LiuyaoGrid:
    """Parse the 太极-六爻.json raw dict into a LiuyaoGrid."""
    nodes: dict[str, YaoNode] = {}
    for name, data in (raw.get("nodes") or {}).items():
        if not isinstance(data, dict):
            continue
        override = data.get("temperature_override")
        nodes[name] = YaoNode(
            name=name,
            phase=str(data.get("phase", "")),
            rule=str(data.get("rule", "")),
            execution_hint=str(data.get("execution_hint", "")),
            temperature_override=float(override) if override is not None else None,
        )
    return LiuyaoGrid(nodes=nodes)


def parse_bagua_grid(raw: dict[str, Any]) -> BaguaGrid:
    """Parse the 太极-八卦.json raw dict into a BaguaGrid."""
    nodes: dict[str, TrigramNode] = {}
    for name, data in (raw.get("nodes") or {}).items():
        if not isinstance(data, dict):
            continue
        nodes[name] = TrigramNode(
            name=name,
            context=str(data.get("context", "")),
            prompt_prefix=str(data.get("prompt_prefix", "")),
            default_sixiang=str(data.get("default_sixiang", "少阴")),
            preferred_actions=list(data.get("preferred_actions", [])),
        )
    return BaguaGrid(nodes=nodes)
