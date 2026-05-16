"""
Task inner loop — per-task 织→[阳→阴] micro-cycle.

Phase entry:
1. Weaver generates the system prompt, tool definitions, and temperature (once).

Per round (micro-cycle):
2. Yang (阳) calls the LLM with native Function Calling.
3. Yin (阴) approves / rejects tool calls + proactive arbitration + self-reflection.
   Yin is the closer of every round (symmetric counterpart of 暗驱 in the outer loop).
4. Approved tool calls are executed mechanically (not a separate 爻).
5. If Yang called ``task_complete`` and Yin approved it, the inner loop ends.
6. Yin may signal ``needs_reweave`` → Weaver re-woven for next round.
7. Facts / signals are accumulated for the next round.
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
from vingobot.goal.grid_types import CognitionUsage
from vingobot.goal.types import (
    ApprovedToolCall,
    ExecutionResult,
    MingjueOutput,
    RoundExecutionFact,
    SixiangPermissionConfig,
    WeaverOutput,
    YangResponse,
    YinOutput,
)
from vingobot.goal.weaver import weave
from vingobot.goal.executor import execute_tool_calls
from vingobot.goal.yang import run_yang
from vingobot.goal.yin import run_yin, _is_diagnostic_exec

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


# ---------------------------------------------------------------------------
# Context injection — builds cross-round continuity text for Yang
# ---------------------------------------------------------------------------


def _build_context_injection(
    previous_yang_content: str,
    recent_invoke_results: list[str],
    round_num: int,
    task_dir: Path,
    facts: list[RoundExecutionFact],
    action_warning: str,
    self_ref_round_count: int,
) -> str:
    """Build cross-round context text to inject into Yang's system prompt.

    Replaces the former per-round Weaver-level prompt append block.
    This keeps the micro-cycle (阳→阴) light — Weaver only runs
    at phase entry or when Yin triggers a reweave.
    """
    parts: list[str] = []

    # ── Previous Yang thinking ─────────────────────────────────
    if previous_yang_content:
        parts.append(
            f"\n\n## 你上一轮的思考\n"
            f"{previous_yang_content[:3000]}\n\n"
            "（以上是你上一轮的思考结论。无需重新读取相同文件验证，"
            "直接基于已有信息推进任务。）"
        )

    # ── Recent invoke results (rolling window) ─────────────────
    if recent_invoke_results:
        invoke_parts = []
        for i, result in enumerate(recent_invoke_results):
            round_label = round_num - len(recent_invoke_results) + i
            invoke_parts.append(f"### 第{round_label}轮工具执行结果\n{result}")
        invoke_text = "\n\n---\n\n".join(invoke_parts)
        parts.append(
            f"\n\n## 跨轮工具执行结果（最近 {len(recent_invoke_results)} 轮）\n{invoke_text}"
        )

    # ── Execution history path reference ───────────────────────
    parts.append(
        f"\n\n## 完整执行历史\n"
        f"文件: {task_dir / _FACTS_FILE}\n"
        f"包含所有 {len(facts)} 轮的结构化执行事实（意图、审批、执行状态）。"
        f"如需回顾前期轮次的详细决策和失败原因，用 read_file 读取此文件。"
    )

    # ── Action warning from previous round ─────────────────────
    if action_warning:
        parts.append(f"\n\n{action_warning}")

    # ── Self-referential read warning ──────────────────────────
    if self_ref_round_count >= 1:
        parts.append(
            f"\n\n## ⚠️ 自指涉读取警告\n"
            f"你已经连续 {self_ref_round_count} 轮只读取自己的执行记录"
            f"（06-execution-facts.json 或 outputs/ 目录下的文件）。\n"
            f"读取自己的执行记录不会推进任务——它只是观察，不是行动。\n"
            f"**本轮必须产出实质性交付物**：调用 write_file 写入成果文件，"
            f"或调用 exec 执行任务脚本。\n"
        )
        if self_ref_round_count >= 2:
            parts.append(
                f"**这是第二次警告。如果再有一轮自指涉读取，系统将强制终止此任务。**\n"
            )

    return "".join(parts)


async def execute_task_inner_loop(
    task_dir: str | Path,
    mingjue_output: MingjueOutput,
    goal_context: Any,
    signal: asyncio.Task | None = None,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    rework_attempt: int = 0,
    rework_action: str | None = None,
) -> InnerLoopResult:
    """Run the Weaver→Yang→Yin→Executor cycle for a single task.

    Continues until Yang invokes ``task_complete`` or ``max_rounds`` is
    exhausted.

    Args:
        rework_attempt: 0 = first attempt, 1+ = Anqu-ordered rework.
            On rework, round output files are prefixed to avoid collisions
            with previous attempts (``r{rework_attempt}-NNN-round.json``).
        rework_action: The Anqu action that triggered this rework
            ("continue_task" | "verify_task" | "learn_task").
            Used to tag facts and read ``05-anqu-instruction.md``.
    """
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "outputs").mkdir(exist_ok=True)

    # ── Load existing facts for recovery ──────────────────────
    facts: list[RoundExecutionFact] = _load_facts(task_dir)

    # ── Detect rework: read Anqu instruction if present ───────
    _IS_VERIFICATION = (rework_action == "verify_task")
    anqu_instruction_text = ""
    instruction_path = task_dir / "05-anqu-instruction.md"
    if instruction_path.exists():
        try:
            anqu_instruction_text = instruction_path.read_text(encoding="utf-8")[:5000]
            logger.info("[内循环] 读取暗驱回炉指令 ({} 字符)", len(anqu_instruction_text))
        except OSError:
            pass

    task_description = mingjue_output.concrete_goal or mingjue_output.summary

    # Init cognitive usage tracker
    cognitive_usage = CognitionUsage()

    # Cross-round invoke results: rolling window of tool outputs
    recent_invoke_results: list[str] = []

    # Cross-round Yang thinking: previous round's content
    previous_yang_content = ""

    # Consecutive read-only round counter (for termination detection)
    read_only_round_count = 0
    had_successful_write = False
    self_ref_round_count = 0
    action_warning = ""
    recent_tool_calls: list[list[str]] = []
    recent_exec_commands: list[list[str]] = []
    """Per-round list of exec command strings, aligned 1:1 with recent_tool_calls."""

    # ── 织 (入口一次) ────────────────────────────────────────
    invoke_text = ""
    weaver_output = await weave(
        mingjue_output, facts, goal_context, 1,
        previous_invoke_results=invoke_text,
        read_only_round_count=0,
        had_successful_write=False,
    )
    _save_weaver_output(task_dir, weaver_output, 1, is_reweave=False)

    grid_domain = weaver_output.grid_domain
    grid_skills = weaver_output.grid_skills
    if grid_domain and grid_domain not in cognitive_usage.grids_loaded:
        cognitive_usage.grids_loaded.append(grid_domain)
    for skill_name in grid_skills:
        if skill_name not in cognitive_usage.skills_used:
            cognitive_usage.skills_used.append(skill_name)

    # Inject Anqu rework instruction at entry
    if anqu_instruction_text:
        weaver_output.system_prompt = (
            f"## ⚠️ 暗驱回炉指令（本任务已重置，这是第 {rework_attempt} 次回炉）\n\n"
            f"{anqu_instruction_text}\n\n"
            f"---\n\n"
            + weaver_output.system_prompt
        )

    for round_num in range(1, max_rounds + 1):
        if signal is not None and signal.cancelled():
            break

        # ── Build cross-round context injection ──────────────
        context_injection = _build_context_injection(
            previous_yang_content=previous_yang_content,
            recent_invoke_results=recent_invoke_results,
            round_num=round_num,
            task_dir=task_dir,
            facts=facts,
            action_warning=action_warning,
            self_ref_round_count=self_ref_round_count,
        )
        if had_successful_write:
            action_warning = ""  # reverse signal: lifted when Yang produces

        # ── 阳¹ (LLM call) ─────────────────────────────────
        effective_prompt = weaver_output.system_prompt + context_injection
        profile = weaver_output.cognitive_profile

        # Auto-termination: self-referential loop
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

        yang_response = await run_yang(
            system_prompt=effective_prompt,
            tool_definitions=weaver_output.tool_definitions,
            task_description=task_description,
            round_facts=facts,
            temperature=profile.temperature,
            top_p=profile.top_p,
            top_k=profile.top_k,
            repetition_penalty=profile.repetition_penalty,
            signal=signal,
        )

        # Build per-round output data
        round_data: dict[str, Any] = {
            "round": round_num,
            "yang_content": yang_response.content,
            "reasoning_content": yang_response.reasoning_content,
            "thinking_blocks": yang_response.thinking_blocks,
            "tool_calls": yang_response.tool_calls,
            "called_task_complete": yang_response.called_task_complete,
        }

        previous_yang_content = yang_response.content or ""

        # ── task_complete / no-tool-calls: defer to 阴 for closing judgement ──
        # 阴 is the symmetric counterpart of 暗驱 in the inner loop.
        # Just like 暗驱 always runs at the end of every task, 阴 always runs
        # at the end of every round — including the final one.

        # Track cognition tools called by Yang
        _track_cognition_usage(cognitive_usage, yang_response, [])

        # ── 阴 (unified approval + proactive arbitration + self-reflection) ─────────
        recent_tc = [list(r) for r in recent_tool_calls[-6:]]
        recent_exec = [list(e) for e in recent_exec_commands[-6:]]
        yin_output = await run_yin(
            tool_calls=yang_response.tool_calls,
            facts=facts,
            round_num=round_num,
            max_rounds=max_rounds,
            task_description=task_description,
            workspace_root=get_workspace_path().parent,
            recent_tool_calls=recent_tc,
            grid_skill_names=grid_skills,
            signal=signal,
            recent_exec_commands=recent_exec,
            goal_context=goal_context,
            task_dir=str(task_dir),
        )

        # ── Execute approved calls (mechanical, not a 爻) ──
        goal_dir_path = mingjue_output.context.goal_dir if mingjue_output.context.goal_dir else None
        cognition_dirs: list[str] | None = None
        if mingjue_output.context and mingjue_output.context.cognition_dirs:
            cognition_dirs = list(mingjue_output.context.cognition_dirs.values())

        perm = SixiangPermissionConfig(
            task_dir=str(task_dir),
            goal_dir=str(goal_dir_path) if goal_dir_path else "",
            workspace_root=str(get_workspace_path().parent),
            cognition_dirs=[str(d) for d in cognition_dirs] if cognition_dirs else [],
        )

        results = await execute_tool_calls(yin_output.approved_calls, perm=perm)

        # ── Yin suggestions / warnings / fuse ───────────────
        # Fuse instruction (active arbitration) gets highest priority — prepend first
        if yin_output.fuse_instruction:
            action_warning = yin_output.fuse_instruction
            logger.warning(
                "[内循环] 阴触发熔断 (round={}): {}",
                round_num, yin_output.blueprint_deviation[:200] if yin_output.blueprint_deviation else "(unknown)",
            )
        if yin_output.suggestions:
            existing = action_warning
            action_warning = f"## 💡 阴节点建议\n{yin_output.suggestions}"
            if existing:
                action_warning = existing + "\n\n" + action_warning
        if yin_output.warning:
            existing = action_warning
            action_warning = yin_output.warning
            if existing:
                action_warning = existing + "\n\n" + action_warning

        # Track tool call names for cross-round pattern detection
        round_tool_names = [c.name for c in yin_output.approved_calls]
        recent_tool_calls.append(round_tool_names)
        if len(recent_tool_calls) > 10:
            recent_tool_calls = recent_tool_calls[-10:]

        # Track exec command strings for diagnostic-vs-productive classification
        round_exec_cmds = [
            (c.arguments.get("command", "") or "")
            for c in yin_output.approved_calls
            if c.name == "exec"
        ]
        recent_exec_commands.append(round_exec_cmds)
        if len(recent_exec_commands) > 10:
            recent_exec_commands = recent_exec_commands[-10:]

        # Add exec data to round output and persist
        round_data["yin_decision"] = yin_output.decision
        round_data["yin_reason"] = yin_output.reason
        round_data["approved_calls"] = [
            {"name": c.name, "arguments": c.arguments} for c in yin_output.approved_calls
        ]
        round_data["results"] = [
            {"status": r.status, "output": r.output[:500], "error": r.error}
            for r in results
        ]
        _save_round_output(task_dir, round_num, "round", dict(round_data), rework_attempt=rework_attempt)

        # Track failed tools for cognition usage
        _track_tool_failures(cognitive_usage, results)

        # Build fact
        fact = _build_round_fact(
            round_num, yang_response,
            yin_output.decision, yin_output.reason, results,
            yao=profile.current_yao,
            sixiang=profile.sixiang_selected,
            current_gua=profile.current_gua,
            rework_attempt=rework_attempt,
            is_verification_round=_IS_VERIFICATION,
        )
        facts.append(fact)
        _checkpoint_facts(task_dir, facts, round_num, force=(round_num == 1))

        # Build previous invoke results for next round
        round_invoke = _format_prev_invoke_results(yin_output.approved_calls, results)
        if round_invoke:
            recent_invoke_results.append(round_invoke)
            if len(recent_invoke_results) > _RECENT_INVOKE_WINDOW:
                recent_invoke_results.pop(0)

        had_successful_write = any(
            r.status == "success" and c.name in _WRITE_TOOLS
            for c, r in zip(yin_output.approved_calls, results)
        )

        # Update consecutive read-only round counter
        # NOTE: exec with diagnostic commands (ls, head, wc, etc.)
        # is classified as read-only — this prevents the "exec bypass"
        # where Yang uses exec instead of read_file to evade detection.
        if yin_output.approved_calls and all(
            _is_read_only_call(c) for c in yin_output.approved_calls
        ):
            if all(_is_self_referential(c) for c in yin_output.approved_calls):
                self_ref_round_count += 1
            else:
                read_only_round_count += 1
                self_ref_round_count = 0
        else:
            read_only_round_count = 0
            self_ref_round_count = 0

        # ── Auto-termination: read-only loop ──────────────────
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

        # ── 阴·强制终止检测 ─────────────────────────────────
        if yin_output.warning and yin_output.warning.startswith("## ⚠️ 阴节点强制终止"):
            await _auto_complete(
                task_dir, facts, round_num,
                f"阴节点强制终止: {yin_output.warning[:200]}",
            )
            return InnerLoopResult(
                facts=facts,
                final_content=yin_output.warning,
                task_completed=True,
                rounds_executed=round_num,
                cognitive_usage=cognitive_usage,
            )

        # ── task_complete: 阴收束后终止内循环 ─────────────────
        # 阴是内循环的暗驱对应——暗驱在每任务结束时运行，阴在每轮结束时运行。
        # task_complete 必须经过阴的审批和仲裁才能终止内循环。
        if yang_response.called_task_complete:
            task_complete_approved = any(
                c.name == "task_complete" for c in yin_output.approved_calls
            )
            if task_complete_approved:
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
            # 阴拒绝了 task_complete（如读瘫门禁清空 approved_calls）→ 继续循环
            logger.info("[内循环] 阴拒绝 task_complete (轮次 {})，继续循环", round_num)

        # ── 阴触发 织 (phase re-weave) ─────────────────────
        if yin_output.needs_reweave:
            logger.info("[内循环] 阴触发姿态重织 (轮次 {})", round_num)
            invoke_text_for_weave = (
                "\n\n---\n\n".join(recent_invoke_results)
                if recent_invoke_results else ""
            )
            weaver_output = await weave(
                mingjue_output, facts, goal_context, round_num,
                previous_invoke_results=invoke_text_for_weave,
                read_only_round_count=read_only_round_count,
                had_successful_write=had_successful_write,
            )
            _save_weaver_output(task_dir, weaver_output, round_num, is_reweave=True)
            grid_domain = weaver_output.grid_domain
            grid_skills = weaver_output.grid_skills
            if grid_domain and grid_domain not in cognitive_usage.grids_loaded:
                cognitive_usage.grids_loaded.append(grid_domain)
            for skill_name in grid_skills:
                if skill_name not in cognitive_usage.skills_used:
                    cognitive_usage.skills_used.append(skill_name)

    # ── Max rounds exhausted ──────────────────────────────────
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
    rework_attempt: int = 0,
    is_verification_round: bool = False,
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
        rework_attempt=rework_attempt,
        is_verification_round=is_verification_round,
    )


def _save_round_output(task_dir: Path, round_num: int, phase: str, data: dict[str, Any], *, rework_attempt: int = 0) -> None:
    """Persist a round's output to the task directory for audit / recovery.

    On rework (rework_attempt > 0), round files are prefixed with
    ``r{rework_attempt}-`` to avoid overwriting previous attempts.
    """
    out_dir = task_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    prefix = f"r{rework_attempt}-" if rework_attempt > 0 else ""
    fn = f"{prefix}{round_num:03d}-{phase}.json"
    try:
        (out_dir / fn).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("Failed to persist round output: {}", fn)


def _save_weaver_output(
    task_dir: Path,
    weaver_output: WeaverOutput,
    round_num: int,
    *,
    is_reweave: bool = False,
) -> None:
    """Persist Weaver's cognitive profile and grid info to disk.

    Saves to ``outputs/weaver-{initial|reweave}-r{round_num:03d}.json``.
    The full system_prompt and tool_definitions are not persisted (too large).
    """
    out_dir = task_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    profile = weaver_output.cognitive_profile
    data = {
        "type": "reweave" if is_reweave else "initial",
        "round_num": round_num,
        "cognitive_profile": {
            "current_yao": profile.current_yao,
            "current_gua": profile.current_gua,
            "sixiang_selected": profile.sixiang_selected,
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "top_k": profile.top_k,
            "repetition_penalty": profile.repetition_penalty,
            "yao_reasoning": profile.yao_reasoning,
            "sixiang_reasoning": profile.sixiang_reasoning,
            "gua_reasoning": profile.gua_reasoning,
        },
        "grid_domain": weaver_output.grid_domain,
        "grid_skills": weaver_output.grid_skills,
    }

    tag = "reweave" if is_reweave else "initial"
    fn = f"weaver-{tag}-r{round_num:03d}.json"
    try:
        (out_dir / fn).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("Failed to persist weaver output: {}", fn)


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


def _is_read_only_call(call: ApprovedToolCall) -> bool:
    """Check if a tool call is read-only, accounting for diagnostic exec.

    ``exec ls -la`` / ``exec head`` / ``exec wc`` are **not** productive
    even though ``exec`` itself is nominally a write tool.  This closes
    the "exec bypass" hole in read-paralysis detection.
    """
    if call.name in _READ_ONLY_TOOLS:
        return True
    if call.name == "exec":
        cmd = (call.arguments or {}).get("command", "")
        return _is_diagnostic_exec(cmd)
    return False

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


def _build_skipped_calls_context(
    approved_calls: list[ApprovedToolCall],
    round_num: int,
    reject_reason: str,
) -> str:
    """Build a cross-round context block showing what was rejected and why.

    This gets injected into ``recent_invoke_results`` so Yang can see
    exactly which calls were skipped in the next round, preventing it
    from proposing the same rejected pattern again.

    Used when Yin partially/fully rejects a round's tool calls — Yang
    needs to know its proposed calls didn't get executed.
    """
    by_tool: dict[str, list[str]] = {}
    for c in approved_calls:
        args = c.arguments or {}
        if c.name in ("read_file", "write_file", "edit_file", "delete_file"):
            preview = args.get("path", "")
        elif c.name == "exec":
            preview = (args.get("command", "") or "")[:60]
        elif c.name == "list_directory":
            preview = args.get("path", "")
        else:
            preview = ", ".join(f"{k}={str(v)[:30]}" for k, v in args.items())
        by_tool.setdefault(c.name, []).append(preview or "(无参数)")

    call_lines: list[str] = []
    for tool_name, targets in by_tool.items():
        if len(targets) == 1:
            call_lines.append(f"- {tool_name}({targets[0]})")
        else:
            suffix = " ..." if len(targets) > 3 else ""
            call_lines.append(
                f"- {tool_name} × {len(targets)}（{', '.join(targets[:3])}{suffix}）"
            )

    calls_text = "\n".join(call_lines)
    reason_short = reject_reason[:200]

    return (
        f"## ⚠️ 第{round_num}轮被拒绝的调用\n"
        f"阴节点拒绝了以下 {len(approved_calls)} 个调用，原因：\n"
        f"> {reason_short}\n\n"
        f"被拒绝的具体调用：\n{calls_text}\n"
    )


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



