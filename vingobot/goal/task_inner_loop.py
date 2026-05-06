"""
Task inner loop — per-task Weaver → Yang → Yin → Executor cycle.

Each round:
1. Weaver generates the system prompt, tool definitions, and temperature.
2. Yang calls the LLM with native Function Calling.
3. If Yang calls ``task_complete``, the inner loop ends.
4. Yin approves / rejects Yang's tool calls.
5. Executor runs approved calls and collects results.
6. Facts are accumulated for the next round.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.config.paths import get_workspace_path
from vingobot.core.workspace import get_workspace_paths
from vingobot.goal.executor import execute_tool_calls
from vingobot.goal.grid_types import CognitionUsage
from vingobot.goal.types import (
    ApprovedToolCall,
    ExecutionResult,
    MingjueOutput,
    RoundExecutionFact,
    YangResponse,
)
from vingobot.goal.weaver import weave
from vingobot.goal.yang import run_yang
from vingobot.goal.yin import approve

_DEFAULT_MAX_ROUNDS = 30
"""逐轮执行的最大轮数上限（30轮）"""

_FACTS_FILE = "06-execution-facts.json"
"""每轮的结构化执行事实持久化文件，位于 task_dir 下"""


@dataclass
class InnerLoopResult:
    """Output of the task inner loop."""

    facts: list[RoundExecutionFact] = field(default_factory=list)
    final_content: str | None = None
    task_completed: bool = False
    rounds_executed: int = 0
    cognitive_usage: CognitionUsage | None = None
    """Cognitive assets used during this inner loop (populated by Weaver)."""


async def execute_task_inner_loop(
    task_dir: str | Path,
    mingjue_output: MingjueOutput,
    goal_context: Any,
    signal: asyncio.Task | None = None,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
) -> InnerLoopResult:
    """Run the Weaver→Yang→Yin→Executor cycle for a single task.

    Continues until Yang invokes ``task_complete`` or ``max_rounds`` is
    exhausted.
    """
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "outputs").mkdir(exist_ok=True)

    # ── Load existing facts for recovery ──────────────────────
    facts: list[RoundExecutionFact] = _load_facts(task_dir)

    task_description = mingjue_output.concrete_goal or mingjue_output.summary

    # Init cognitive usage tracker
    cognitive_usage = CognitionUsage()

    # Cross-round invoke results: previous round's read-only tool outputs
    previous_invoke_results = ""

    # Cross-round Yang thinking: previous round's content (preserves chain of thought)
    previous_yang_content = ""

    # Consecutive read-only round counter (for termination detection)
    read_only_round_count = 0
    # Whether the previous round had a successful write/exec tool
    had_successful_write = False

    for round_num in range(1, max_rounds + 1):
        if signal is not None and signal.cancelled():
            break

        # ── 1. 编织器 ─────────────────────────────────────────
        weaver_output = await weave(
            mingjue_output, facts, goal_context, round_num,
            previous_invoke_results=previous_invoke_results,
            previous_yang_content=previous_yang_content,
            read_only_round_count=read_only_round_count,
            had_successful_write=had_successful_write,
        )

        # Track grid/skills discovered from L3 grid
        grid_domain = weaver_output.grid_domain
        grid_skills = weaver_output.grid_skills
        if grid_domain and grid_domain not in cognitive_usage.grids_loaded:
            cognitive_usage.grids_loaded.append(grid_domain)
        for skill_name in grid_skills:
            if skill_name not in cognitive_usage.skills_used:
                cognitive_usage.skills_used.append(skill_name)

        # ── 2. 阳 (native FC) ──────────────────────────────────
        profile = weaver_output.cognitive_profile
        yang_response = await run_yang(
            system_prompt=weaver_output.system_prompt,
            tool_definitions=weaver_output.tool_definitions,
            task_description=task_description,
            round_facts=facts,
            temperature=profile.temperature,
            top_p=profile.top_p,
            top_k=profile.top_k,
            repetition_penalty=profile.repetition_penalty,
            signal=signal,
        )

        # Persist Yang response
        _save_round_output(
            task_dir,
            round_num,
            "yang",
            {
                "round": round_num,
                "content": yang_response.content,
                "reasoning_content": yang_response.reasoning_content,
                "thinking_blocks": yang_response.thinking_blocks,
                "tool_calls": yang_response.tool_calls,
                "called_task_complete": yang_response.called_task_complete,
            },
        )

        # Save Yang's thinking for cross-round continuity
        previous_yang_content = yang_response.content or ""

        # If Yang called task_complete → inner loop done
        if yang_response.called_task_complete:
            fact = _build_round_fact(
                round_num, yang_response, "skipped", "", "skipped",
                yao=profile.current_yao,
                sixiang=profile.sixiang_selected,
                current_gua=profile.current_gua,
            )
            facts.append(fact)
            _persist_facts(task_dir, facts)
            # Track any cognition tools called in this final round
            _track_cognition_usage(cognitive_usage, yang_response, [])
            return InnerLoopResult(
                facts=facts,
                final_content=yang_response.content,
                task_completed=True,
                rounds_executed=round_num,
                cognitive_usage=cognitive_usage,
            )

        # No tool calls → text-only response, continue
        if not yang_response.tool_calls:
            fact = _build_round_fact(
                round_num, yang_response, "skipped", "", "skipped",
                yao=profile.current_yao,
                sixiang=profile.sixiang_selected,
                current_gua=profile.current_gua,
            )
            facts.append(fact)
            _persist_facts(task_dir, facts)
            # Reset write/read tracking for pure-text rounds — no tools executed
            had_successful_write = False
            read_only_round_count = 0
            continue

        # Track cognition tools called by Yang
        _track_cognition_usage(cognitive_usage, yang_response, [])

        # ── 3. 阴 (approval) — two-layer: hardcoded front + LLM contextual ──
        approved_calls, yin_decision, yin_reason = await approve(
            yang_response.tool_calls,
            workspace_root=get_workspace_path().parent,  # 项目根目录，覆盖 .vingobot/.taiji 及项目文件
        )

        # ── 4. 执行器 ──────────────────────────────────────────
        goal_dir_path = mingjue_output.context.goal_dir if mingjue_output.context.goal_dir else None
        cognition_dirs: list[str] | None = None
        if mingjue_output.context and mingjue_output.context.cognition_dirs:
            cognition_dirs = list(mingjue_output.context.cognition_dirs.values())
        results = await execute_tool_calls(
            approved_calls,
            task_dir=task_dir,
            goal_dir=goal_dir_path,
            cognition_dirs=cognition_dirs,
        )

        # Persist round
        _save_round_output(
            task_dir,
            round_num,
            "exec",
            {
                "approved_calls": [
                    {"name": c.name, "arguments": c.arguments} for c in approved_calls
                ],
                "yin_decision": yin_decision,
                "yin_reason": yin_reason,
                "results": [
                    {"status": r.status, "output": r.output[:500], "error": r.error}
                    for r in results
                ],
            },
        )

        # Track failed tools for cognition usage
        _track_tool_failures(cognitive_usage, results)

        # Build fact
        fact = _build_round_fact(
            round_num, yang_response, yin_decision, yin_reason, results,
            yao=profile.current_yao,
            sixiang=profile.sixiang_selected,
            current_gua=profile.current_gua,
        )
        facts.append(fact)
        _persist_facts(task_dir, facts)

        # Build previous invoke results for next round
        previous_invoke_results = _format_prev_invoke_results(approved_calls, results)

        # Detect if this round had a successful write/exec (for next round's force-completion check)
        had_successful_write = any(
            r.status == "success" and c.name in _WRITE_TOOLS
            for c, r in zip(approved_calls, results)
        )

        # Update consecutive read-only round counter
        if approved_calls and all(c.name in _READ_ONLY_TOOLS for c in approved_calls):
            read_only_round_count += 1
        else:
            read_only_round_count = 0

        # ── Auto-termination: self-read loop detected ──────────────
        if read_only_round_count >= _AUTO_TERMINATE_FLOOR and round_num >= _AUTO_TERMINATE_THRESHOLD:
            _auto_complete(
                task_dir, facts, round_num,
                f"自动终止：连续 {read_only_round_count} 轮纯读取，已达轮次上限。",
            )
            return InnerLoopResult(
                facts=facts,
                final_content="任务因自读循环自动终止。",
                task_completed=True,
                rounds_executed=round_num,
                cognitive_usage=cognitive_usage,
            )

    # ── Max rounds exhausted (non-auto-terminated fallback) ──────
    return InnerLoopResult(
        facts=facts,
        final_content=None,
        task_completed=False,
        rounds_executed=min(round_num, max_rounds),
        cognitive_usage=cognitive_usage,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_round_fact(
    round_num: int,
    yang: YangResponse,
    yin_decision: str,
    yin_reason: str,
    results: list[ExecutionResult] | str,
    yao: int = 0,
    sixiang: str = "",
    current_gua: str = "",
) -> RoundExecutionFact:
    """Build a ``RoundExecutionFact`` from Yang response and execution state."""
    # Extract intent summary from content
    intent = "进行了思考"
    if yang.content:
        clean = yang.content.strip().split("\n")[0][:100]
        if clean:
            intent = clean

    had_action = len(yang.tool_calls) > 0

    exec_status = "skipped"
    exec_summary = ""
    if isinstance(results, list):
        successes = sum(1 for r in results if r.status == "success")
        failures = sum(1 for r in results if r.status in ("error", "blocked"))
        if failures == 0 and successes > 0:
            exec_status = "success"
            exec_summary = f"{successes} 个工具调用成功"
        elif successes == 0 and failures > 0:
            exec_status = "failure"
            exec_summary = f"{failures} 个工具调用失败"
        elif successes > 0:
            exec_status = "partial_failure"
            exec_summary = f"{successes} 成功 / {failures} 失败"

    return RoundExecutionFact(
        round=round_num,
        yang_intent_summary=intent,
        had_action_request=had_action,
        yin_decision=yin_decision,  # type: ignore[arg-type]
        yin_reason=yin_reason[:200],
        execution_result_summary=exec_summary,
        execution_status=exec_status,  # type: ignore[arg-type]
        tool_call_count=len(yang.tool_calls),
        yao=yao,
        sixiang=sixiang,
        current_gua=current_gua,
    )


def _save_round_output(task_dir: Path, round_num: int, phase: str, data: dict[str, Any]) -> None:
    """Persist a round's output to the task directory for audit / recovery."""
    out_dir = task_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    fn = f"{round_num:03d}-{phase}.json"
    try:
        (out_dir / fn).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("Failed to persist round output: {}", fn)


def _load_facts(task_dir: Path) -> list[RoundExecutionFact]:
    """Load existing execution facts from disk for recovery."""
    facts_path = task_dir / _FACTS_FILE
    if not facts_path.exists():
        return []
    try:
        raw = json.loads(facts_path.read_text(encoding="utf-8"))
        facts = [RoundExecutionFact(**f) for f in raw]
        logger.info("Recovered {} execution facts from {}", len(facts), facts_path)
        return facts
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Failed to load existing facts from {}, starting fresh: {}", facts_path, e)
        return []


def _persist_facts(task_dir: Path, facts: list[RoundExecutionFact]) -> None:
    """Persist structured execution facts to 06-execution-facts.json."""
    facts_path = task_dir / _FACTS_FILE
    try:
        facts_path.write_text(
            json.dumps([asdict(f) for f in facts], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("Failed to persist execution facts")


def _track_cognition_usage(
    usage: CognitionUsage,
    yang: YangResponse,
    _results: list[ExecutionResult],
) -> None:
    """Track which cognitive tools Yang called this round."""
    if not yang.tool_calls:
        return

    for tc in yang.tool_calls:
        name = ""
        if isinstance(tc, dict):
            name = (
                tc.get("function", {}).get("name", "")
                if isinstance(tc.get("function"), dict)
                else ""
            )
            if not name:
                name = tc.get("name", "")
        elif hasattr(tc, "function"):
            name = tc.function.name if hasattr(tc.function, "name") else ""

        if name == "load_grid":
            # Try to extract which grid was loaded
            args = {}
            if isinstance(tc, dict):
                try:
                    raw = (
                        tc.get("function", {}).get("arguments", "{}")
                        if isinstance(tc.get("function"), dict)
                        else "{}"
                    )
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    pass
            grid_name = args.get("name", "")
            if grid_name and grid_name not in usage.grids_loaded:
                usage.grids_loaded.append(grid_name)

        elif name == "search_skills":
            usage.skills_used.append("__searched__")

        elif name == "search_models":
            usage.models_loaded.append("__searched__")

    usage.tool_calls_total += len(yang.tool_calls)


def _track_tool_failures(
    usage: CognitionUsage,
    results: list[ExecutionResult],
) -> None:
    """Track which tools failed during execution."""
    for r in results:
        if r.status in ("error", "blocked"):
            tool_name = r.call.name if hasattr(r, "call") else ""
            if tool_name and tool_name not in usage.tools_failed:
                usage.tools_failed.append(tool_name)


_READ_ONLY_TOOLS = frozenset({
    "read_file", "list_directory", "load_grid",
    "search_skills", "search_models", "web_search", "web_fetch",
    "search_codebase",
})
"""Tools whose outputs carry valuable information across rounds."""

_WRITE_TOOLS = frozenset({
    "write_file", "exec",
})
"""Tools that produce actionable outputs worth summarizing across rounds."""


def _format_prev_invoke_results(
    approved_calls: list[ApprovedToolCall],
    results: list[ExecutionResult],
) -> str:
    """Format previous round's tool outputs for cross-round injection.

    Includes both read-only tool outputs (file contents, search results)
    and write/exec tool results (what was written, command output), so
    Yang knows what it actually accomplished in earlier rounds.
    """
    lines: list[str] = []
    for call, result in zip(approved_calls, results):
        if result.status != "success":
            continue
        output = (result.output or "").strip()
        args_preview = ", ".join(
            f"{k}={v}" for k, v in (call.arguments or {}).items()
        )
        if call.name in _READ_ONLY_TOOLS:
            if not output:
                continue
            lines.append(f"### {call.name}({args_preview})")
            lines.append(output[:4000])
        elif call.name in _WRITE_TOOLS:
            summary = output[:1000] if output else "(空输出)"
            lines.append(f"### {call.name}({args_preview}) → ✅ 成功")
            lines.append(summary)
        else:
            summary = output[:500] if output else "(空输出)"
            lines.append(f"### {call.name}({args_preview}) → ✅ {summary}")

    if not lines:
        return ""

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-termination: break out of self-read infinite loop
# ---------------------------------------------------------------------------

_AUTO_TERMINATE_FLOOR = 5
"""连续纯读轮次超过此值，结合轮次上限启用自动终结。需与 weaver._TERMINATION_AUTO_FLOOR 同步。"""

_AUTO_TERMINATE_THRESHOLD = 12
"""轮次达到此值且纯读轮次超过下限时自动终结。需与 weaver._TERMINATION_AUTO_THRESHOLD 同步。"""


async def _auto_complete(
    task_dir: Path,
    facts: list[RoundExecutionFact],
    round_num: int,
    reason: str,
) -> None:
    """Save a synthetic final fact, persist it, and write a read summary."""
    fact = RoundExecutionFact(
        round=round_num,
        yang_intent_summary=f"[自动终止] {reason}",
        had_action_request=False,
        yin_decision="skipped",
        execution_result_summary="无需执行，自动终止",
        execution_status="skipped",
        tool_call_count=0,
    )
    facts.append(fact)
    _persist_facts(task_dir, facts)
    _save_round_output(
        task_dir, round_num, "yang",
        {
            "round": round_num,
            "content": f"[自动终止] {reason}",
            "tool_calls": [],
            "called_task_complete": True,
        },
    )

    # Save a read summary for traceability
    _save_read_summary(task_dir, facts, reason)

    logger.info("[自动终止] {} (round={}, reason={})", task_dir.name, round_num, reason)


_READ_SUMMARY_FILE = "05-read-summary.json"
"""Summary of what was read during a terminated read-only loop."""


def _save_read_summary(
    task_dir: Path,
    facts: list[RoundExecutionFact],
    reason: str,
) -> None:
    """Save a summary of what was explored during the read-only loop.

    Builds a deduplicated list of tool calls and their intents from the
    execution facts so that even auto-terminated tasks leave audit trail.
    """
    calls_log: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    read_files: list[str] = []
    list_dirs: list[str] = []

    # Scan all output files for tool call details
    out_dir = task_dir / "outputs"
    if out_dir.is_dir():
        for fn in sorted(out_dir.iterdir()):
            if fn.suffix != ".json" or "-exec" not in fn.name:
                continue
            try:
                data = json.loads(fn.read_text(encoding="utf-8"))
                approved = data.get("approved_calls", [])
                for call in approved:
                    name = call.get("name", "")
                    args = call.get("arguments", {})
                    path_key = args.get("path", "") or args.get("command", "")
                    if path_key and path_key not in seen_paths:
                        seen_paths.add(path_key)
                        if name == "read_file":
                            read_files.append(path_key)
                        elif name == "list_directory":
                            list_dirs.append(path_key)
                    calls_log.append({"round": fn.stem.split("-")[0], "tool": name, "args": args})
            except (OSError, json.JSONDecodeError):
                pass

    summary = {
        "reason": reason,
        "total_rounds": len(facts),
        "total_tool_calls": sum(f.tool_call_count for f in facts),
        "files_read": read_files,
        "directories_listed": list_dirs,
        "round_facts": [
            {
                "round": f.round,
                "intent": f.yang_intent_summary[:150],
                "tools": f.tool_call_count,
                "status": f.execution_status,
            }
            for f in facts
        ],
    }

    try:
        (task_dir / _READ_SUMMARY_FILE).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("Failed to save read summary to {}", _READ_SUMMARY_FILE)
