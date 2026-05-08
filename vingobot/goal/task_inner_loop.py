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
import os
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
    SixiangPermissionConfig,
    YangResponse,
)
from vingobot.goal.weaver import weave
from vingobot.goal.yang import run_yang
from vingobot.goal.yin import approve

_DEFAULT_MAX_ROUNDS = 30
"""逐轮执行的最大轮数上限（30轮）"""

_FACTS_FILE = "06-execution-facts.json"
"""每轮的结构化执行事实持久化文件，位于 task_dir 下"""

_FACTS_WRITE_INTERVAL = 5
"""每 N 轮写入一次 facts checkpoint（减少 I/O 次数）"""


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

    # Cross-round invoke results: rolling window of tool outputs (up to _RECENT_INVOKE_WINDOW rounds)
    recent_invoke_results: list[str] = []

    # Cross-round Yang thinking: previous round's content (preserves chain of thought)
    previous_yang_content = ""

    # Consecutive read-only round counter (for termination detection)
    read_only_round_count = 0
    # Whether the previous round had a successful write/exec tool
    had_successful_write = False
    # Self-referential read counter: Yang reading its own execution records
    self_ref_round_count = 0

    for round_num in range(1, max_rounds + 1):
        if signal is not None and signal.cancelled():
            break

        # ── 1. 编织器 ─────────────────────────────────────────
        # Format recent invoke results as a single text blob for Weaver
        invoke_text_for_weaver = "\n\n---\n\n".join(recent_invoke_results) if recent_invoke_results else ""
        weaver_output = await weave(
            mingjue_output, facts, goal_context, round_num,
            previous_invoke_results=invoke_text_for_weaver,
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
        # Inject previous Yang thinking for cross-round continuity
        if previous_yang_content:
            weaver_output.system_prompt += (
                "\n\n## 你上一轮的思考\n"
                + (previous_yang_content or "")[:3000]
                + "\n\n"
                "（以上是你上一轮的思考结论。无需重新读取相同文件验证，"
                "直接基于已有信息推进任务。）"
            )

        # Inject recent invoke results rolling window
        if recent_invoke_results:
            parts = []
            for i, result in enumerate(recent_invoke_results):
                round_label = round_num - len(recent_invoke_results) + i
                parts.append(f"### 第{round_label}轮工具执行结果\n{result}")
            invoke_text = "\n\n---\n\n".join(parts)
            weaver_output.system_prompt += f"\n\n## 跨轮工具执行结果（最近 {len(recent_invoke_results)} 轮）\n{invoke_text}"

        # Inject execution history path reference (let Yang self-read)
        weaver_output.system_prompt += (
            f"\n\n## 完整执行历史\n"
            f"文件: {task_dir / _FACTS_FILE}\n"
            f"包含所有 {len(facts)} 轮的结构化执行事实（意图、审批、执行状态）。"
            f"如需回顾前期轮次的详细决策和失败原因，用 read_file 读取此文件。"
        )

        # ── Self-referential read warning ───────────────────────
        if self_ref_round_count >= 1:
            weaver_output.system_prompt += (
                f"\n\n## ⚠️ 自指涉读取警告\n"
                f"你已经连续 {self_ref_round_count} 轮只读取自己的执行记录"
                f"（06-execution-facts.json 或 outputs/ 目录下的文件）。\n"
                f"读取自己的执行记录不会推进任务——它只是观察，不是行动。\n"
                f"**本轮必须产出实质性交付物**：调用 write_file 写入成果文件，"
                f"或调用 exec 执行任务脚本。\n"
            )
            if self_ref_round_count >= 2:
                weaver_output.system_prompt += (
                    f"**这是第二次警告。如果再有一轮自指涉读取，系统将强制终止此任务。**\n"
                )

        # ── Auto-termination: self-referential loop (aggressive) ─
        if self_ref_round_count >= 3:
            await _auto_complete(
                task_dir, facts, round_num,
                f"自动终止：连续 {self_ref_round_count} 轮自指涉读取（读自己的执行记录），无实质性产出。",
            )
            return InnerLoopResult(
                facts=facts,
                final_content="任务因自指涉读取循环自动终止。",
                task_completed=True,
                rounds_executed=round_num,
                cognitive_usage=cognitive_usage,
            )

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

        # Build per-round output data (思考内容放在最前面)
        round_data: dict[str, Any] = {
            "round": round_num,
            "yang_content": yang_response.content,
            "reasoning_content": yang_response.reasoning_content,
            "thinking_blocks": yang_response.thinking_blocks,
            "tool_calls": yang_response.tool_calls,
            "called_task_complete": yang_response.called_task_complete,
        }

        # Save Yang's thinking for cross-round continuity
        previous_yang_content = yang_response.content or ""

        # If Yang called task_complete → inner loop done
        if yang_response.called_task_complete:
            _save_round_output(task_dir, round_num, "round", dict(round_data))
            fact = _build_round_fact(
                round_num, yang_response, "skipped", "", "skipped",
                yao=profile.current_yao,
                sixiang=profile.sixiang_selected,
                current_gua=profile.current_gua,
            )
            facts.append(fact)
            _checkpoint_facts(task_dir, facts, round_num, force=True)
            # Track any cognition tools called in this final round
            _track_cognition_usage(cognitive_usage, yang_response, [])
            # 当 LLM 通过 tool_calls 调用 task_complete 时 content 可能为空
            final_content = yang_response.content
            if not final_content:
                for tc in (yang_response.tool_calls or []):
                    if tc.get("name") == "task_complete":
                        args = tc.get("arguments") or {}
                        final_content = args.get("summary") or args.get("content") or ""
                        break
            return InnerLoopResult(
                facts=facts,
                final_content=final_content,
                task_completed=True,
                rounds_executed=round_num,
                cognitive_usage=cognitive_usage,
            )

        # No tool calls → text-only response, continue
        if not yang_response.tool_calls:
            _save_round_output(task_dir, round_num, "round", dict(round_data))
            fact = _build_round_fact(
                round_num, yang_response, "skipped", "", "skipped",
                yao=profile.current_yao,
                sixiang=profile.sixiang_selected,
                current_gua=profile.current_gua,
            )
            facts.append(fact)
            _checkpoint_facts(task_dir, facts, round_num, force=(round_num == 1))
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

        # Build unified permission config
        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            goal_dir=str(goal_dir_path) if goal_dir_path else "",
            workspace_root=str(get_workspace_path().parent),
            cognition_dirs=[str(d) for d in cognition_dirs] if cognition_dirs else [],
        )

        results = await execute_tool_calls(
            approved_calls,
            perm=perm,
        )

        # Add exec data to round output and persist (merge exec into same file)
        round_data["yin_decision"] = yin_decision
        round_data["yin_reason"] = yin_reason
        round_data["approved_calls"] = [
            {"name": c.name, "arguments": c.arguments} for c in approved_calls
        ]
        round_data["results"] = [
            {"status": r.status, "output": r.output[:500], "error": r.error}
            for r in results
        ]
        _save_round_output(task_dir, round_num, "round", dict(round_data))

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
        _checkpoint_facts(task_dir, facts, round_num, force=(round_num == 1))

        # Build previous invoke results for next round (rolling window)
        round_invoke = _format_prev_invoke_results(approved_calls, results)
        if round_invoke:
            recent_invoke_results.append(round_invoke)
            if len(recent_invoke_results) > _RECENT_INVOKE_WINDOW:
                recent_invoke_results.pop(0)

        # Detect if this round had a successful write/exec (for next round's force-completion check)
        had_successful_write = any(
            r.status == "success" and c.name in _WRITE_TOOLS
            for c, r in zip(approved_calls, results)
        )

        # Update consecutive read-only round counter
        if approved_calls and all(c.name in _READ_ONLY_TOOLS for c in approved_calls):
            # Check if ALL reads are self-referential (reading own execution records)
            if all(_is_self_referential(c) for c in approved_calls):
                self_ref_round_count += 1
                # Don't count self-referential reads as regular read-only —
                # reading your own execution facts doesn't advance the task.
            else:
                read_only_round_count += 1
                self_ref_round_count = 0
        else:
            read_only_round_count = 0
            self_ref_round_count = 0

        # ── Auto-termination: self-read loop detected ──────────────
        if read_only_round_count >= _AUTO_TERMINATE_FLOOR and round_num >= _AUTO_TERMINATE_THRESHOLD:
            await _auto_complete(
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
    had_failable = any(
        tc.get("name") in ("exec", "web_search", "web_fetch", "write_file", "edit_file")
        for tc in (yang.tool_calls or [])
    )

    exec_status = "skipped"
    exec_summary = ""
    if isinstance(results, list):
        successes = sum(1 for r in results if r.status == "success")
        failures = sum(1 for r in results if r.status in ("error", "blocked", "exec_failed"))
        exec_failed_count = sum(1 for r in results if r.status == "exec_failed")
        if failures == 0 and successes > 0:
            exec_status = "success"
            exec_summary = f"{successes} 个工具调用成功"
        elif successes == 0 and failures > 0 and exec_failed_count == failures:
            exec_status = "exec_failed"
            exec_summary = f"{exec_failed_count} 个 exec 执行失败（超时或非零退出码）"
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
        had_failable_op=had_failable,
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
        facts: list[RoundExecutionFact] = []
        for f in raw:
            # Migration: old facts lack had_failable_op
            if "had_failable_op" not in f:
                status = f.get("execution_status", "skipped")
                # These statuses only arise from attempted failable actions
                if status in ("exec_failed", "failure", "partial_failure"):
                    f["had_failable_op"] = True
                elif status == "success" and f.get("tool_call_count", 0) > 0:
                    f["had_failable_op"] = True
                else:
                    f["had_failable_op"] = False
            facts.append(RoundExecutionFact(**f))
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


def _checkpoint_facts(
    task_dir: Path,
    facts: list[RoundExecutionFact],
    round_num: int,
    force: bool = False,
) -> None:
    """Write facts to disk only every ``_FACTS_WRITE_INTERVAL`` rounds (or on force).

    Reduces I/O from N writes per task to ~N/interval writes.
    """
    if force or round_num % _FACTS_WRITE_INTERVAL == 0:
        _persist_facts(task_dir, facts)


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

        if name == "read_file":
            # Check if reading a cognition file (grid/model/skill)
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
            path = args.get("path", "")
            # Detect grid reads — check for cognition/grids/ in path
            if "/grids/" in path or "\\grids\\" in path:
                grid_name = os.path.splitext(os.path.basename(path))[0]
                if grid_name and grid_name not in usage.grids_loaded:
                    usage.grids_loaded.append(grid_name)
            # Detect model reads
            if "/models/" in path or "\\models\\" in path:
                usage.models_loaded.append("__read__")
            # Detect skill reads
            if "/skills/" in path or "\\skills\\" in path:
                usage.skills_used.append("__read__")

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


_RECENT_INVOKE_WINDOW = 5
"""跨轮工具输出滚动窗口大小——Yang 可以看到最近 N 轮的执行结果。"""

_READ_ONLY_TOOLS = frozenset({
    "read_file", "list_directory",
    "web_search", "web_fetch", "search_codebase",
})
"""Tools whose outputs carry valuable information across rounds."""

_WRITE_TOOLS = frozenset({
    "write_file", "exec",
})
"""Tools that produce actionable outputs worth summarizing across rounds."""

_SELF_REFERENTIAL_PATTERNS = (
    "06-execution-facts.json",
    "outputs/",
)
"""File patterns that indicate Yang is reading its own execution records."""


def _is_self_referential(call: ApprovedToolCall) -> bool:
    """Check if a read_file call is reading the task's own execution records."""
    if call.name != "read_file":
        return False
    path = (call.arguments or {}).get("path", "")
    return any(p in path for p in _SELF_REFERENTIAL_PATTERNS)


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
    """Save a synthetic final fact and write checkpoint."""
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
    _checkpoint_facts(task_dir, facts, round_num, force=True)

    logger.info("[自动终止] {} (round={}, reason={})", task_dir.name, round_num, reason)



