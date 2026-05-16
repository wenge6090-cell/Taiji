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
# ExecutionInsight — aggregated cross-task analytics
# ---------------------------------------------------------------------------


@dataclass
class CognitiveStat:
    """Aggregated execution stats for a cognitive posture (gua/sixiang/yao)."""

    count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_rounds: int = 0
    total_tool_calls: int = 0


@dataclass
class ToolStatItem:
    """Per-tool execution statistics."""

    call_count: int = 0
    failure_count: int = 0
    exec_failed_count: int = 0
    top_errors: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class YinDecisionStat:
    """Aggregated Yin approval/denial patterns."""

    total: int = 0
    approved: int = 0
    rejected: int = 0
    modified: int = 0
    top_rejection_reasons: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class StuckLoopRecord:
    """Record of a self-referential stuck loop detected across rounds."""

    goal_id: str = ""
    task_id: str = ""
    round_range: str = ""
    consecutive_read_rounds: int = 0
    detection_reason: str = ""


@dataclass
class ExecutionInsight:
    """Aggregated execution analytics across all sixiang goals/tasks.

    Scans every goal's task directories, reads ``06-execution-facts.json``
    and round output files, then produces structured statistics on:

    - Cognitive posture effectiveness (gua/sixiang/yao)
    - Tool execution success/failure patterns
    - Yin approval/denial trends
    - Self-referential stuck loop detection
    """

    generated_at: str = ""
    total_goals: int = 0
    total_tasks: int = 0
    total_rounds: int = 0
    task_status_breakdown: dict[str, int] = field(default_factory=dict)

    # ── Per-cognitive-pattern stats ──────────────────────────────────────
    gua_stats: dict[str, CognitiveStat] = field(default_factory=dict)
    sixiang_stats: dict[str, CognitiveStat] = field(default_factory=dict)
    yao_stats: dict[str, CognitiveStat] = field(default_factory=dict)

    # ── Tool execution analysis ──────────────────────────────────────────
    tool_stats: dict[str, ToolStatItem] = field(default_factory=dict)

    # ── Yin approval analysis ────────────────────────────────────────────
    yin_stats: YinDecisionStat = field(default_factory=YinDecisionStat)

    # ── Self-referential loop detection ──────────────────────────────────
    stuck_loops: list[StuckLoopRecord] = field(default_factory=list)
    total_stuck_loops: int = 0

    # ── Display ───────────────────────────────────────────────────────────

    def to_text(self) -> str:
        """Render as a human-readable analytics report."""

        parts: list[str] = [
            "# 🧠 六爻系统执行洞察\n",
        ]

        # ── Overview ─────────────────────────────────────────────────
        parts.append("## 总览")
        parts.append(f"- 目标数: {self.total_goals}")
        parts.append(f"- 任务数: {self.total_tasks}")
        parts.append(f"- 总轮数: {self.total_rounds}")
        status_parts = [f"{k}={v}" for k, v in sorted(self.task_status_breakdown.items())]
        if status_parts:
            parts.append(f"- 任务状态分布: {', '.join(status_parts)}")
        parts.append(f"- 检测到自读循环: {self.total_stuck_loops} 次")
        parts.append("")

        # ── Gua stats ────────────────────────────────────────────────
        if self.gua_stats:
            parts.append("## 卦象效率")
            parts.append(f"| 卦象 | 次数 | 成功率 | 平均轮数 | 平均工具调用 |")
            parts.append(f"|------|------|--------|----------|-------------|")
            for gua, stat in sorted(
                self.gua_stats.items(),
                key=lambda x: -x[1].count,
            ):
                sr = stat.success_count / max(stat.count, 1) * 100
                avg_r = stat.total_rounds / max(stat.count, 1)
                avg_t = stat.total_tool_calls / max(stat.count, 1)
                parts.append(
                    f"| {gua} | {stat.count} | {sr:.0f}% | {avg_r:.1f} | {avg_t:.1f} |"
                )
            parts.append("")

        # ── Sixiang stats ────────────────────────────────────────────
        if self.sixiang_stats:
            parts.append("## 四象模式")
            parts.append(f"| 四象 | 次数 | 成功率 | 平均轮数 | 平均工具调用 |")
            parts.append(f"|------|------|--------|----------|-------------|")
            for sixiang, stat in sorted(
                self.sixiang_stats.items(),
                key=lambda x: -x[1].count,
            ):
                sr = stat.success_count / max(stat.count, 1) * 100
                avg_r = stat.total_rounds / max(stat.count, 1)
                avg_t = stat.total_tool_calls / max(stat.count, 1)
                parts.append(
                    f"| {sixiang} | {stat.count} | {sr:.0f}% | {avg_r:.1f} | {avg_t:.1f} |"
                )
            parts.append("")

        # ── Yao stats ────────────────────────────────────────────────
        if self.yao_stats:
            parts.append("## 爻位分布")
            parts.append(f"| 爻位 | 次数 | 成功率 | 平均轮数 | 平均工具调用 |")
            parts.append(f"|------|------|--------|----------|-------------|")
            for yao, stat in sorted(
                self.yao_stats.items(),
                key=lambda x: -x[1].count,
            ):
                sr = stat.success_count / max(stat.count, 1) * 100
                avg_r = stat.total_rounds / max(stat.count, 1)
                avg_t = stat.total_tool_calls / max(stat.count, 1)
                parts.append(
                    f"| {yao} | {stat.count} | {sr:.0f}% | {avg_r:.1f} | {avg_t:.1f} |"
                )
            parts.append("")

        # ── Tool stats ───────────────────────────────────────────────
        if self.tool_stats:
            sorted_tools = sorted(
                self.tool_stats.items(), key=lambda x: -x[1].call_count
            )
            total_calls = sum(s.call_count for _, s in sorted_tools)
            total_fails = sum(s.failure_count for _, s in sorted_tools)
            parts.append("## 工具执行分析")
            parts.append(
                f"- 总调用: {total_calls} | 总失败: {total_fails} "
                f"({total_fails / max(total_calls, 1) * 100:.0f}%)"
            )
            parts.append("")
            parts.append(f"| 工具 | 调用数 | 失败数 | 失败率 | 执行失败 |")
            parts.append(f"|------|--------|--------|--------|----------|")
            for name, stat in sorted_tools:
                fr = stat.failure_count / max(stat.call_count, 1) * 100
                parts.append(
                    f"| {name} | {stat.call_count} | {stat.failure_count} | "
                    f"{fr:.0f}% | {stat.exec_failed_count} |"
                )
            # Show top errors for tools with failures
            for name, stat in sorted_tools:
                if stat.top_errors:
                    for err, count in stat.top_errors[:2]:
                        parts.append(f"  - [{name}] `{err}` (×{count})")
            parts.append("")

        # ── Yin stats ───────────────────────────────────────────────
        y = self.yin_stats
        parts.append("## 阴（审批）模式")
        parts.append(f"- 总审批决策: {y.total}")
        if y.total > 0:
            parts.append(f"- ✅ 批准: {y.approved} ({y.approved / y.total * 100:.0f}%)")
            parts.append(f"- ❌ 拒绝: {y.rejected} ({y.rejected / y.total * 100:.0f}%)")
            parts.append(f"- ✏️ 修改: {y.modified}")
        if y.top_rejection_reasons:
            parts.append("- 常见拒绝原因:")
            for reason, cnt in y.top_rejection_reasons[:5]:
                parts.append(f"  - `{reason}` (×{cnt})")
        parts.append("")

        # ── Stuck loops ─────────────────────────────────────────────
        if self.stuck_loops:
            parts.append("## 自读循环")
            for sl in self.stuck_loops:
                parts.append(
                    f"- [{sl.goal_id}/{sl.task_id}] {sl.round_range}: "
                    f"{sl.consecutive_read_rounds} 轮连续读操作 → {sl.detection_reason}"
                )
            parts.append("")

        parts.append(f"---")
        parts.append(f"生成时间: {self.generated_at}")

        return "\n".join(parts)


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
    # ── Anqu evaluation context (for 思变) ──────────────────────
    anqu_task_summary: str = ""
    """Anqu's summary of what the previous task accomplished."""
    anqu_reason: str = ""
    """Anqu's reasoning for the routing decision."""
    anqu_goal_progress_pct: int | None = None
    """Anqu's assessment of overall goal progress (0-100)."""


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
    rework_attempt: int = 0
    """Rework attempt number (0 = first attempt, 1+ = Anqu-ordered rework)."""
    is_verification_round: bool = False
    """True if this round's exec failures should be counted as verification
    (expected to fail), not production failures that trigger degradation."""


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
# Yin (四爻·阴) — approval + introspection output
# ---------------------------------------------------------------------------


@dataclass
class YinOutput:
    """Unified output from Yin's approval, self-reflection, and proactive arbitration.

    Merges the former ``approve()`` + ``self_reflect()`` into a single
    return value.  ``needs_reweave`` signals the inner loop that the
    cognitive posture should be re-woven (Weaver triggered by Yin).

    ``fuse_instruction`` is the proactive arbitration output: when Yin
    detects blueprint deviation or self-abandonment contamination, it
    injects a mandatory verification instruction (熔断) into the next
    round's context — the Worker MUST address it before calling
    task_complete.  This is NOT a rejection of current tool calls.

    ``blueprint_deviation`` records the specific deviation found, for
    diagnostic and logging purposes.
    """

    approved_calls: list[ApprovedToolCall] = field(default_factory=list)
    decision: str = "skipped"
    """'approved' | 'rejected' | 'modified' | 'skipped'"""
    reason: str = ""
    suggestions: str = ""
    """Suggestions for Yang's next round."""
    warning: str = ""
    """Self-reflection warning (e.g. force-termination detected)."""
    needs_reweave: bool = False
    """Yin detected phase staleness — inner loop should re-weave posture."""
    fuse_instruction: str = ""
    """Proactive arbitration: mandatory verification instruction (熔断).

    When Yin detects blueprint deviation or self-abandonment contamination,
    this field contains a concrete instruction that the Worker MUST address
    before calling task_complete.  Injected into the next round's context.

    This is NOT a rejection — current round's approved tool calls still
    execute normally.  The fuse takes effect in the NEXT round.
    """
    blueprint_deviation: str = ""
    """Diagnostic record of what triggered the fuse.

    When ``_detect_blueprint_deviation`` triggers: contains structured
    detail about blueprint targets vs actual task outputs.
    When ``_detect_self_abandonment`` triggers: contains the specific
    abandonment signals found in execution facts.

    Used for logging and debugging — the field name reflects the
    primary use case (blueprint target comparison).
    """


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
    known_traps_proposal: list[dict] = field(default_factory=list)
    """Anqu-detected failure patterns to propose as known_traps.

    Each proposal is a dict with:
    - name (str): trap name in snake_case
    - description (str): what the failure pattern is
    - trigger (str): when it triggers
    - response (str): how to avoid it

    Proposals must be confirmed by the main loop before writing to meta.json."""
    needs_sibian: bool = False
    """Anqu determines that the blueprint needs SiBian review.

    When True, the outer loop invokes ``run_sibian()`` to evaluate
    cross-task failure patterns and potentially revise the blueprint.
    Defaults to False — SiBian is only triggered when Anqu detects
    cross-task stagnation or systematic failure modes."""


# ---------------------------------------------------------------------------
# SibianDecision — 思变节点输出（连山策略引擎）
# ---------------------------------------------------------------------------


@dataclass
class SibianDecision:
    """思变节点的连山策略决策——艮止观照后的方位、时机、姿态。

    思变位于 Anqu 的 goal_next_task 之后、Mingjue 的下次迭代之前。
    它不是"蓝图审查器"，而是"障碍导航器"——目标永不动，只调整推进方式。

    连山易框架：
    - 艮止 → 停下观山势
    - 六气 → 判断目标处于什么阶段（启动/盛长/收敛/休眠）
    - 六甲 → 判断受阻的时序状态（第几次遇阻、上次策略变更多久）
    - 三元 → 判断上下文新鲜度
    - 阴阳对峙 → 推力与阻力的对比分析
    - 方位决策 → 从哪个方向、什么时机、以什么姿态继续推进
    """

    # ── 方位决策 ──────────────────────────────────────
    action: Literal[
        "continue",          # 无障碍，照常推进
        "push_through",      # 强攻——正面突破（换工具/加力度）
        "navigate_around",   # 绕行——换条路（换卦象/换方法）
        "wait_gather",       # 主动等待——暂停强攻、收集信息
        "decompose",         # 拆解——目标太大，拆成更小子目标
        "escalate",          # 升级——需要外部干预（人类介入/环境变更）
        "abort",             # 终止——目标在当下约束内确实无法推进
    ] = "continue"

    # ── 连山时序分析 ──────────────────────────────────
    liuq: str = ""
    """六气判断——目标当前处于什么季节阶段（春-启动/夏-盛长/秋-收敛/冬-休眠）。"""

    liujia: str = ""
    """六甲判断——时序状态：第几次受阻、距上次策略变更多久、是否有恶性循环。"""

    sanyuan: str = ""
    """三元判断——上下文新鲜度：蓝图是否刚更新、认知库是否刷新、环境是否变化。"""

    duizhi: str = ""
    """阴阳对峙分析——推力（有效策略/工具/进展）vs 阻力（失败模式/缺失能力/环境约束）。"""

    # ── 推进策略 ──────────────────────────────────────
    strategy: str = ""
    """具体的推进策略描述，注入到 Mingjue 的 continuation_context。"""

    trigram_hint: str = ""
    """建议下轮 Mingjue 使用的卦象。空字符串表示不改变当前卦象。"""

    timing: Literal["now", "after_refresh", "after_one_task"] = "now"
    """时机判断：
    - now: 立即行动
    - after_refresh: 等待上下文刷新后再动（如刷新认知库）
    - after_one_task: 等下一任务完成后再评估
    """

    reason: str = ""
    """决策理由——注入到 Mingjue 下一轮的 continuation_context 中。"""

    blueprint_revision: str = ""
    """[DEPRECATED] 仅 navigate_around 时如果涉及蓝图可调层微调，填写修订内容。
    不是改写目标，是调整任务拆解方式/执行顺序/中间里程碑。"""


@dataclass
class GoalResult:
    status: Literal["completed", "failed", "aborted"]
    goal_id: str = ""
    reason: str = ""
