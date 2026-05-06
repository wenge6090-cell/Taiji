"""goal/__init__.py — Sixiang (六爻) goal-driven loop public API."""

from vingobot.goal.anqu import run_anqu
from vingobot.goal.coroutine import WorkerPool
from vingobot.goal.dialogue_target import (
    ensure_default_goal,
    push_dialogue_task,
    run_dialogue_goal,
    run_goal,
    start_sixiang_workers,
)
from vingobot.goal.executor import execute_tool_calls
from vingobot.goal.mingjue import run_mingjue
from vingobot.goal.sixiang_loop import execute_complete_sixiang_loop
from vingobot.goal.types import (
    AnquAction,
    AnquDecision,
    ApprovedToolCall,
    ExecutionResult,
    GoalResult,
    MingjueOutput,
    MingjueSource,
    RoundExecutionFact,
    YangResponse,
)
from vingobot.goal.weaver import weave
from vingobot.goal.yang import run_yang
from vingobot.goal.yin import approve

__all__ = [
    "MingjueSource",
    "MingjueOutput",
    "RoundExecutionFact",
    "YangResponse",
    "ApprovedToolCall",
    "AnquAction",
    "AnquDecision",
    "GoalResult",
    "ExecutionResult",
    "WorkerPool",
    "execute_complete_sixiang_loop",
    "ensure_default_goal",
    "push_dialogue_task",
    "run_dialogue_goal",
    "run_goal",
    "start_sixiang_workers",
    "weave",
    "run_yang",
    "approve",
    "execute_tool_calls",
    "run_mingjue",
    "run_anqu",
]
