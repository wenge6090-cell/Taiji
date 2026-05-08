"""
Shared type definitions for the sixiang (六爻) goal-driven loop.

All data structures used across Mingjue, Weaver, Yang, Yin, Executor,
and Anqu are declared here to avoid circular imports and provide a
single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pathlib import Path

from vingobot.goal.grid_types import CognitionEvolutionAction

# ---------------------------------------------------------------------------
# SixiangPermissionConfig — unified permission config for sixiang tools
# ---------------------------------------------------------------------------


@dataclass
class SixiangPermissionConfig:
    """Unified permission configuration for sixiang (六爻) tool execution.

    Consolidates path boundaries and tool-access policies that were
    previously scattered across weaver (tool listing), yin (approval),
    executor (path checks), and tool_executor (path resolution).

    All paths use ``/`` as separator regardless of OS.
    """

    task_dir: str = ""
    """Current task directory — primary write target."""

    goal_dir: str = ""
    """Current goal directory — read + exec (cwd) allowed."""

    workspace_root: str = ""
    """Project root directory — expanded read + write + exec scope."""

    cognition_dirs: list[str] = field(default_factory=list)
    """Read-only cognition directories (L1 skills, L2 models, L3 grids)."""

    def __post_init__(self) -> None:
        self._task_dir_p = Path(self.task_dir) if self.task_dir else None
        self._goal_dir_p = Path(self.goal_dir) if self.goal_dir else None
        self._ws_p = Path(self.workspace_root) if self.workspace_root else None
        self._cog_dirs = [Path(d) for d in self.cognition_dirs if d]

    # ── Derived helper properties (cached in __post_init__) ────────────

    @property
    def read_allowed_dirs(self) -> list[Path]:
        """Directories accessible for **read** operations."""
        dirs: list[Path] = []
        if self._task_dir_p:
            dirs.append(self._task_dir_p)
        if self._goal_dir_p:
            dirs.append(self._goal_dir_p)
        dirs.extend(self._cog_dirs)
        return dirs

    @property
    def write_allowed_dirs(self) -> list[Path]:
        """Directories accessible for **write** operations.

        Goal directory is included so edit_file can modify goal-level
        artifacts (blueprint, memory files).
        """
        dirs: list[Path] = []
        if self._task_dir_p:
            dirs.append(self._task_dir_p)
        if self._goal_dir_p:
            dirs.append(self._goal_dir_p)
        return dirs

    @property
    def exec_allowed_cwds(self) -> list[Path]:
        """Directories where shell commands may run (cwd)."""
        dirs: list[Path] = []
        if self._task_dir_p:
            dirs.append(self._task_dir_p)
        if self._goal_dir_p:
            dirs.append(self._goal_dir_p)
        return dirs

    @property
    def yin_workspace_root(self) -> Path | None:
        """Workspace root passed to Yin's path-safety checks."""
        return self._ws_p


# ---------------------------------------------------------------------------
# Mingjue (初爻·明觉) — goal → first-task translation
# ---------------------------------------------------------------------------


@dataclass
class MingjueSource:
    """What triggered Mingjue this time."""

    type: Literal["initial_goal", "anqu_continuation", "rework", "periodic_reflection"]
    description: str = ""
    previous_task_summary: str = ""
    continuation_context: str = ""
    rework_instruction: str = ""
    previous_output: Any = None  # MingjueOutput from previous iteration
    suggested_trigram: str = ""  # Anqu's suggested gua for the next task


@dataclass
class MingjueContextInfo:
    workspace_root: str = ""
    goal_dir: str = ""
    task_dir: str = ""
    cognition_dirs: dict[str, str] = field(default_factory=dict)


@dataclass
class MingjueOutput:
    """Structured output from Mingjue — translation of goal into first/next task."""

    intent: Literal["task"] = "task"
    goal_id: str = ""
    summary: str = ""
    concrete_goal: str = ""
    trigram: str = ""  # 八卦卦象 (qian/kun/zhen/xun/kan/li/gen/dui)
    trigram_reason: str = ""
    initial_yao: int = 1  # 初始爻位 (1-6), 默认初爻
    goal_progress_pct: int = 0  # 明觉评估的当前目标完成百分比 (0-100)
    context: MingjueContextInfo = field(default_factory=MingjueContextInfo)


# ---------------------------------------------------------------------------
# RoundExecutionFact (轮次客观事实)
# ---------------------------------------------------------------------------

YinDecision = Literal["approved", "rejected", "modified", "need_user_approval", "skipped"]
ExecStatus = Literal["success", "partial_failure", "failure", "exec_failed", "skipped"]


@dataclass
class RoundExecutionFact:
    """Objective facts about one round — consumed by Weaver & Anqu."""

    round: int = 0
    yang_intent_summary: str = ""
    had_action_request: bool = False
    yin_decision: YinDecision = "skipped"
    yin_reason: str = ""
    execution_result_summary: str = ""
    execution_status: ExecStatus = "skipped"
    tool_call_count: int = 0
    yao: int = 0
    sixiang: str = ""
    current_gua: str = ""
    had_failable_op: bool = False
    """True if any approved call was exec/web_search/write_file/edit_file —
    i.e. an operation whose outcome cannot be predicted in advance."""


# ---------------------------------------------------------------------------
# Weaver (二爻·编织器) — cognitive profile + orchestration output
# ---------------------------------------------------------------------------


@dataclass
class CognitiveProfile:
    """Weaver LLM 动态决定的认知画像.

    The Weaver LLM reads the three meta-cognitive grids (六爻/八卦/四象)
    and execution history, then outputs this structured profile that
    describes the cognitive posture for the next round.
    """

    current_yao: int = 1
    current_gua: str = "乾"
    sixiang_selected: str = "少阳"
    temperature: float = 0.7
    top_p: float = 0.85
    top_k: int = 40
    repetition_penalty: float = 1.1
    yao_reasoning: str = ""
    sixiang_reasoning: str = ""
    gua_reasoning: str = ""


@dataclass
class WeaverOutput:
    """Complete output from the Weaver for one round."""

    system_prompt: str = ""
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    cognitive_profile: CognitiveProfile = field(default_factory=CognitiveProfile)
    grid_domain: str = ""
    grid_skills: list[str] = field(default_factory=list)


@dataclass
class MetaCognitionState:
    """Internal cross-round tracking state (not exposed to Yang)."""

    current_yao: int = 1
    current_gua: str = "乾"
    current_sixiang: str = "少阳"
    loop_count: int = 0


# ---------------------------------------------------------------------------
# Yang (三爻·阳) — native Function Calling response
# ---------------------------------------------------------------------------


@dataclass
class YangResponse:
    """Raw output from Yang's LLM call — native tool_calls + optional content."""

    content: str | None = None
    reasoning_content: str | None = None
    """LLM's chain-of-thought / reasoning text (DeepSeek, Kimi, Gemini, etc.)."""
    thinking_blocks: list[dict] | None = None
    """Extended thinking blocks from Anthropic Claude."""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    called_task_complete: bool = False


# ---------------------------------------------------------------------------
# Yin (四爻·阴) — approved tool calls
# ---------------------------------------------------------------------------


@dataclass
class ApprovedToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Executor (五爻·执行器) — execution result
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    call: ApprovedToolCall
    status: Literal["success", "error", "blocked", "exec_failed"] = "success"
    output: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Anqu (上爻·暗驱) — goal-level decisions
# ---------------------------------------------------------------------------

AnquAction = Literal[
    "goal_next_task",
    "goal_completed",
    "goal_failed",
    "continue_task",
    "verify_task",
    "learn_task",
]


@dataclass
class AnquDecision:
    """The highest-level routing decision for the goal outer-loop."""

    action: AnquAction = "goal_completed"
    next_task_description: str = ""
    next_task_concrete_action: str = ""
    """Concrete first-step action for the next task (e.g. 'write_file outputs/05-x.py').
    Injected by Weaver into round-1 system prompt so Yang starts with a specific action."""
    task_summary: str = ""
    continuation_context: str = ""
    rework_instruction: str = ""
    failure_reason: str = ""
    suggested_trigram: str = ""
    """Anqu's suggested gua (卦象) for the next task. Carried into MingjueSource."""
    goal_progress_pct: int | None = None
    """Goal completion percentage (0-100) as assessed by Anqu. None means not evaluated."""
    evolution_actions: list[CognitionEvolutionAction] = field(default_factory=list)
    """0-N cognitive evolution actions to enqueue after routing."""


# ---------------------------------------------------------------------------
# GoalResult — final outcome
# ---------------------------------------------------------------------------


@dataclass
class GoalResult:
    status: Literal["completed", "failed", "aborted"]
    goal_id: str = ""
    reason: str = ""
