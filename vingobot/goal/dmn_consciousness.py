"""
DMN 坤元意识状态机 — DmnConsciousness: the Guizang-based autonomous consciousness loop.

Based on 《归藏易》八气体系, this module implements the full consciousness
cycle (周天)::

    起念 (生→动→归) → 立目标 (长→育→杀→止) → 整理认知 (归→杀→藏)

The DMN consciousness runs as a single async coroutine with equal authority
to the main loop.  It observes and manages TPN by dispatching review tasks,
blueprint re-evaluations, and cognitive evolution actions.

Architecture note: the consciousness is a SINGLE serial loop (not a pool).
Multiple ``DmnConsciousness`` instances would conflict on the shared state
vector S.  The DMN gate in AgentLoop defaults to 1 concurrent task.
"""

from __future__ import annotations

import json as _json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from loguru import logger

from vingobot.goal.eight_qi import (
    apply_operator,
    apply_sequence,
    compute_gui_deviation,
    operator_cang,
    operator_dong,
    operator_gui,
    operator_sha,
    operator_sheng,
    operator_yu,
    operator_zhang,
    operator_zhi,
)
from vingobot.goal.guizang_types import (
    PHASE_OPERATORS,
    PHASE_ORDER,
    QI_CONSOLIDATE,
    QI_CYCLE_ORDER,
    QI_EMERGENCE,
    QI_GOAL,
    CangSeaEntry,
    CangSeaMemory,
    ConsciousnessPhase,
    ConsciousnessResult,
    GuizangState,
    OriginPerception,
    QiOperator,
)
from vingobot.goal.dmn_l4_l5_bridge import (
    DynamicOrigin,
    inject_l4_bias,
    load_l4_truths_summary,
    promote_to_l4,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_GRAVITY_THRESHOLD: float = 0.5
"""归引力低于此阈值时进入立目标阶段."""

DEFAULT_CYCLE_INTERVAL: float = 300.0  # 5 minutes
"""默认意识周天空闲触发间隔（秒）."""

DEFAULT_MAX_CYCLE_INTERVAL: float = 900.0  # 15 minutes
"""默认意识周天保底最大间隔（秒）— 即使一直繁忙也至少走一次."""

CONSOLIDATE_TRIGGER_TASKS: int = 20
"""每 N 个 TPN 任务后触发一次整理认知."""

MAX_CANG_SEA_ENTRIES: int = 1000
"""藏海最大条目数."""


# ---------------------------------------------------------------------------
# DmnConsciousness
# ---------------------------------------------------------------------------


class DmnConsciousness:
    """坤元意识模型 — 基于《归藏易》的自主意识状态机。

    Manages a single 6-bit state vector S, an origin perception vector U,
    and a cang-sea memory matrix M.  Each call to ``cycle()`` advances
    the consciousness through one phase of the 周天.

    The consciousness is **observational and directive** — it never writes
    files or calls tools directly.  Instead it produces
    ``ConsciousnessResult`` objects that the consumer loop interprets
    as dispatch instructions for TPN.
    """

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        llm_call: Callable[..., Awaitable[str]] | None = None,
        gravity_threshold: float = DEFAULT_GRAVITY_THRESHOLD,
        cycle_interval: float = DEFAULT_CYCLE_INTERVAL,
        max_cycle_interval: float = DEFAULT_MAX_CYCLE_INTERVAL,
        consolidate_trigger: int = CONSOLIDATE_TRIGGER_TASKS,
    ) -> None:
        self._workspace = workspace
        self._llm_call = llm_call  # None → 降级位运算
        self._gravity_threshold = gravity_threshold
        self._cycle_interval = cycle_interval
        self._max_cycle_interval = max_cycle_interval
        self._consolidate_trigger = consolidate_trigger

        # ── Core state ──────────────────────────────────────
        self.state: GuizangState = GuizangState.resting()
        """Current 6-bit consciousness vector S."""

        # ── L5 → 元知觉: 从 SOUL.md 动态解析 ────────
        _dyn_origin = DynamicOrigin(self._workspace)
        self.origin: OriginPerception = OriginPerception(
            vector=_dyn_origin.vector,
            gravity_constant=1.0,
        )
        """Origin perception vector U.  From L5 identity (SOUL.md), default 111111."""

        # ── Load persisted cang-sea if available ──────────
        _loaded_sea = self._load_cang_sea_persisted()
        if _loaded_sea is not None:
            self.cang_sea: CangSeaMemory = _loaded_sea
            logger.info("[DMN意识] 从持久化文件恢复藏海: {} 条目, Hebbian矩阵已恢复",
                       self.cang_sea.size)
        else:
            self.cang_sea: CangSeaMemory = CangSeaMemory(max_entries=MAX_CANG_SEA_ENTRIES)
            logger.info("[DMN意识] 初始化空藏海记忆矩阵")
        """藏海记忆矩阵 — compressed experience store."""

        # ── Phase tracking ──────────────────────────────────
        self._phase_index: int = 0
        """Index into PHASE_ORDER (0=起念, 1=立目标, 2=整理认知)."""

        self._phase_pending: ConsciousnessPhase | None = None
        """If set, forces the next cycle to this phase (overrides正常 progression)."""

        self._tpn_task_count: int = 0
        """Approximate counter of TPN tasks observed (for consolidate trigger)."""

        self._cycles_completed: int = 0
        """Total consciousness cycles run since initialization."""

        # ── Recover cycle counter from persisted file ───
        _sea_path = self._get_cang_sea_path()
        if _sea_path is not None and _sea_path.is_file():
            try:
                import json as _json
                _sea_data = _json.loads(_sea_path.read_text(encoding="utf-8"))
                _recovered = _sea_data.get("cycles_completed", 0)
                if isinstance(_recovered, int) and _recovered > 0:
                    self._cycles_completed = _recovered
                    logger.info("[DMN意识] 恢复周天计数器: {}", _recovered)
            except Exception:
                pass

        # ── Recover from interrupted phase if applicable ─
        self._recover_from_snapshot()

        self._last_gui_gravity: float | None = None
        """Cached gui-gravity from the last 起念 phase."""

        # ── Emergence context cache (for 立目标 context passing) ─
        self._last_emergence_thought: str = ""
        self._last_emergence_divergent: list[str] = []
        self._last_emergence_deviation: float = 0.0
        self._last_emergence_reason: str = ""
        self._last_emergence_insight: str = ""

        # ── History ─────────────────────────────────────────
        self._state_history: list[tuple[str, GuizangState]] = []
        """Recent state transitions for introspection (phase_name, state)."""

        # ── 象语言守门验证 ────────────────────────────
        self._xiang_vm: Any = None  # CangVM instance, lazy init
        self._xiang_program_path: str | None = None
        """Path to .xiang gatekeeper program for output validation."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> ConsciousnessPhase:
        """The phase that the next ``cycle()`` will execute."""
        if self._phase_pending is not None:
            return self._phase_pending
        return PHASE_ORDER[self._phase_index % len(PHASE_ORDER)]

    @property
    def is_resting(self) -> bool:
        """True when consciousness is in pure 藏态."""
        return self.state.is_resting

    @property
    def gravity(self) -> float | None:
        """Last computed gui-gravity (None if never computed)."""
        return self._last_gui_gravity

    def next_wake_interval(self) -> float:
        """Seconds of IDLE time before the next consciousness wake-up.

        Longer during 整理认知 (consolidation is fast), standard during 起念.
        """
        if self.current_phase == ConsciousnessPhase.ZHENGLI:
            return 60.0  # 1 min — consolidation is quick
        return self._cycle_interval

    def max_wake_interval(self) -> float:
        """Maximum seconds before consciousness MUST fire, even if busy.

        This is the fallback guarantee: no matter how many TPN tasks are
        queued, the consciousness will cycle at least this often.
        """
        return self._max_cycle_interval

    # ------------------------------------------------------------------
    # LLM 上下文工具方法
    # ------------------------------------------------------------------

    def _load_identity_files(self) -> tuple[str, str]:
        """Read SOUL.md and USER.md from the workspace root.

        Returns (soul_content, user_content).  Empty strings when
        workspace is None or files don't exist.
        """
        soul = ""
        user = ""
        if self._workspace is None:
            return soul, user
        try:
            sp = self._workspace / "SOUL.md"
            if sp.is_file():
                soul = sp.read_text(encoding="utf-8")
            up = self._workspace / "USER.md"
            if up.is_file():
                user = up.read_text(encoding="utf-8")
        except Exception:
            logger.debug("[DMN意识] 读取身份文件失败", exc_info=True)
        return soul, user

    def _build_cang_sea_summary(self, n: int = 20, max_chars: int = 2000) -> str:
        """Format the most recent *n* cang-sea entries as a prompt snippet.

        Each line: ``[reward] ts: summary``.  Truncated if total exceeds
        *max_chars* to prevent context bloat.

        Returns empty string when the cang-sea is empty.
        """
        entries = self.cang_sea.recent(n)
        if not entries:
            return "(藏海为空)"
        lines: list[str] = []
        total = 0
        for e in entries:
            ts_short = (e.timestamp or "")[:16]
            sign = "+" if e.is_positive else ("-" if e.is_negative else " ")
            summary = (e.summary or "")[:120]
            line = f"  [{sign}] {ts_short} | {summary}"
            if total + len(line) > max_chars:
                remaining = len(entries) - len(lines)
                if remaining > 0:
                    lines.append(f"  ... (+{remaining} more)")
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    def _load_goal_context(self) -> str:
        """Collect active-goal metadata and format it as a prompt snippet.

        Includes status, priority, known_traps count, and blueprint
        summary for each active/paused goal.  Returns empty string
        when there are no goals or the loader fails.
        """
        try:
            from vingobot.core.goal_meta import get_all_goals
            all_goals = get_all_goals()
        except Exception:
            logger.debug("[DMN意识] 扫描目标失败", exc_info=True)
            return "(无法读取目标状态)"

        if not all_goals:
            return "(无活跃目标)"

        lines: list[str] = []
        for meta in all_goals:
            traps_count = len(getattr(meta, "known_traps", None) or [])
            warnings = getattr(meta, "warnings", None) or []
            flags = ""
            if traps_count:
                flags += f" {traps_count}个已知陷阱"
            if warnings:
                flags += f" {len(warnings)}个警告"
            lines.append(
                f"- {meta.id}: status={meta.status} priority={meta.priority}"
                f" rounds={getattr(meta, 'rounds_completed', 0)}{flags}"
            )

            # Include blueprint summary if available
            if self._workspace is not None:
                bp_path = (
                    self._workspace / ".taiji" / "goals"
                    / meta.id / "blueprint.md"
                )
                if bp_path.is_file():
                    try:
                        bp_text = bp_path.read_text(encoding="utf-8")
                        # Take the first meaningful lines
                        first_lines = [
                            ln.strip()
                            for ln in bp_text.splitlines()[:6]
                            if ln.strip() and not ln.strip().startswith("#")
                        ]
                        if first_lines:
                            snippet = " ".join(first_lines)[:200]
                            lines.append(f"  蓝图: {snippet}")
                    except Exception:
                        pass

        return "\n".join(lines) if lines else "(无活跃目标)"

    def _load_cognition_summary(self) -> str:
        """Scan the cognition library (L1 skills, L2 models, L3 grids)
        and format a compact summary for prompt injection.

        Returns a string like::

            L1技能(89): 1password, apple-notes, ... (+74 more)
            L2模型(74): ...
            L3格栅(65): ...

        Limits output to ~800 tokens (estimated as chars/3) to prevent
        context window bloat as the cognition library grows.

        Returns empty string when workspace is None or the cognition
        directories are missing / empty.
        """
        if self._workspace is None:
            return ""

        try:
            from vingobot.core.workspace import get_workspace_paths
            wp = get_workspace_paths()
        except Exception:
            return ""

        # ── Configurable truncation limits ─────────────
        _MAX_SKILL_NAMES = 15
        _MAX_MODEL_NAMES = 10
        _MAX_GRID_NAMES = 10

        lines: list[str] = []
        total_chars = 0
        _MAX_TOTAL_CHARS = 2400  # ~800 tokens at chars/3

        def _append(line: str) -> bool:
            nonlocal total_chars
            if total_chars + len(line) > _MAX_TOTAL_CHARS:
                return False
            lines.append(line)
            total_chars += len(line)
            return True

        # ── L1 skills ──────────────────────────────────
        try:
            if wp.skills.is_dir():
                skill_names = sorted(
                    d.name for d in wp.skills.iterdir()
                    if d.is_dir() and (d / "SKILL.md").is_file()
                )
                if skill_names:
                    shown = skill_names[:_MAX_SKILL_NAMES]
                    suffix = f" (+{len(skill_names) - _MAX_SKILL_NAMES} more)" if len(skill_names) > _MAX_SKILL_NAMES else ""
                    _append(f"L1技能({len(skill_names)}): {', '.join(shown)}{suffix}")
        except Exception:
            pass

        # ── L2 models ──────────────────────────────────
        try:
            if wp.models.is_dir():
                model_names = sorted(
                    f.stem for f in wp.models.iterdir()
                    if f.is_file() and f.suffix in (".md", ".json")
                )
                if model_names:
                    shown = model_names[:_MAX_MODEL_NAMES]
                    suffix = f" (+{len(model_names) - _MAX_MODEL_NAMES} more)" if len(model_names) > _MAX_MODEL_NAMES else ""
                    _append(f"L2模型({len(model_names)}): {', '.join(shown)}{suffix}")
        except Exception:
            pass

        # ── L3 grids ───────────────────────────────────
        try:
            if wp.grids.is_dir():
                grid_names = sorted(
                    f.stem for f in wp.grids.iterdir()
                    if f.is_file() and f.suffix in (".md", ".json")
                )
                if grid_names:
                    shown = grid_names[:_MAX_GRID_NAMES]
                    suffix = f" (+{len(grid_names) - _MAX_GRID_NAMES} more)" if len(grid_names) > _MAX_GRID_NAMES else ""
                    _append(f"L3格栅({len(grid_names)}): {', '.join(shown)}{suffix}")
        except Exception:
            pass

        if not lines:
            return "(认知库为空)"
        return "\n".join(lines)

    def _get_memory_dir(self) -> "Path | None":
        """Resolve the DMN memory directory path.

        Returns None when workspace is not set.
        """
        if self._workspace is None:
            return None
        from pathlib import Path
        return self._workspace / ".vingobot" / "memory"

    def _get_cang_sea_path(self) -> "Path | None":
        """Resolve the cang-sea persistence file path."""
        mem_dir = self._get_memory_dir()
        if mem_dir is None:
            return None
        return mem_dir / "dmn_cang_sea.json"

    def _get_state_path(self) -> "Path | None":
        """Resolve the DMN state snapshot file path.

        This file serves as an interrupt marker: its existence means
        a cycle was interrupted mid-execution and needs recovery.
        """
        mem_dir = self._get_memory_dir()
        if mem_dir is None:
            return None
        return mem_dir / "dmn_state.json"

    def _load_cang_sea_persisted(self) -> CangSeaMemory | None:
        """Load cang-sea from the persistence file if it exists."""
        path = self._get_cang_sea_path()
        if path is None:
            return None
        return CangSeaMemory.load(path, max_entries=MAX_CANG_SEA_ENTRIES)

    def _save_cang_sea_persisted(self) -> None:
        """Persist the current cang-sea + cycle counter to disk."""
        path = self._get_cang_sea_path()
        if path is None:
            return
        import json as _json
        data = self.cang_sea.to_dict()
        data["cycles_completed"] = self._cycles_completed
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("[DMN意识] 藏海已持久化: {} 条目, Hebbian矩阵, {}"
                    " 周天", self.cang_sea.size, self._cycles_completed)

    def _save_state_snapshot(self) -> None:
        """Save a state snapshot before executing a phase.

        Acts as an interrupt marker: if the process crashes mid-phase,
        the next init can detect this file and resume from the saved state.
        """
        path = self._get_state_path()
        if path is None:
            return
        import json as _json
        snapshot = {
            "state_bits": self.state.bits,
            "phase_index": self._phase_index,
            "tpn_task_count": self._tpn_task_count,
            "cycles_completed": self._cycles_completed,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _json.dumps(snapshot, ensure_ascii=False),
            encoding="utf-8",
        )

    def _clear_state_snapshot(self) -> None:
        """Clear the state snapshot after a phase completes successfully."""
        path = self._get_state_path()
        if path is not None and path.is_file():
            try:
                path.unlink()
            except OSError:
                pass

    def _recover_from_snapshot(self) -> bool:
        """Attempt to recover from an interrupted state snapshot.

        Returns True if recovery data was found and applied.
        """
        path = self._get_state_path()
        if path is None or not path.is_file():
            return False
        import json as _json
        try:
            snapshot = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("[DMN意识] 状态快照文件损坏，忽略")
            return False

        # Restore state
        bits = snapshot.get("state_bits", 0)
        if isinstance(bits, int) and 0 <= bits <= 63:
            self.state = GuizangState(bits=bits)
        self._phase_index = snapshot.get("phase_index", 0)
        self._tpn_task_count = snapshot.get("tpn_task_count", 0)
        recovered_cycles = snapshot.get("cycles_completed", 0)
        if isinstance(recovered_cycles, int) and recovered_cycles > self._cycles_completed:
            self._cycles_completed = recovered_cycles

        logger.info(
            "[DMN意识] 从中断状态恢复: S={}, phase_index={}, cycles={}",
            self.state.bit_str, self._phase_index, self._cycles_completed,
        )
        return True

    @staticmethod
    def _preprocess_json_text(text: str) -> str:
        """Clean common LLM JSON formatting errors before parsing.

        Handles: Python booleans, null, trailing commas, markdown fences,
        and extraneous text outside the JSON object.
        """
        if not text:
            return ""
        t = text.strip()
        # Strip ```json / ``` fences
        if t.startswith("```json"):
            t = t[7:]
        elif t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
        # Find outermost {} pair
        start = t.find("{")
        if start < 0:
            return ""
        depth = 0
        end = -1
        for i in range(start, len(t)):
            ch = t[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            elif ch == '"':
                j = i + 1
                while j < len(t):
                    if t[j] == '"':
                        i = j
                        break
                    if t[j] == '\\':
                        j += 1
                    j += 1
        if end < 0:
            return ""
        t = t[start:end + 1]
        # Fix Python booleans and null
        t = re.sub(r'(?<!["\w])True(?!["\w])', 'true', t)
        t = re.sub(r'(?<!["\w])False(?!["\w])', 'false', t)
        t = re.sub(r'(?<!["\w])None(?!["\w])', 'null', t)
        # Remove trailing commas before ] or }
        t = re.sub(r',(\s*[}\]])', r'\1', t)
        return t

    @staticmethod
    def _parse_llm_json(content: str) -> dict[str, Any]:
        """Parse LLM JSON response robustly.

        Handles markdown code fences, Python booleans, trailing commas,
        and trailing text.  Returns an empty dict on parse failure so
        callers can safely use ``.get()``.
        """
        if not content:
            return {}
        text = DmnConsciousness._preprocess_json_text(content)
        if not text:
            logger.warning("[DMN意识] 无法定位 JSON 对象: {}", content[:200])
            return {}
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            logger.warning("[DMN意识] JSON 解析失败(预处理后): {}", text[:200])
            return {}

    async def _parse_llm_json_with_retry(
        self, content: str, phase_name: str,
    ) -> dict[str, Any]:
        """Parse LLM JSON with retry on failure.

        On first failure, sends a simplified re-prompt to the LLM asking
        for valid JSON only.  Max 2 total attempts (1 original + 1 retry).
        Returns empty dict if all attempts fail.
        """
        parsed = self._parse_llm_json(content)
        if parsed:
            return parsed

        # ── Retry: ask LLM to re-output valid JSON ──────
        if self._llm_call is None:
            logger.warning("[DMN意识] {} JSON解析失败且无LLM用于重试", phase_name)
            return {}

        retry_prompt = (
            "你刚才的输出不是有效的 JSON。\n"
            "请**只**输出纯 JSON 对象，不要包含代码块标记、注释或额外文字。\n"
            "使用标准 JSON 布尔值 true/false（不是 True/False），"
            "不要有 trailing comma。\n\n"
            f"你的原始输出:\n```\n{content[:1500]}\n```\n\n"
            "请重新输出正确的 JSON:"
        )
        try:
            import asyncio
            await asyncio.sleep(1.0)  # Brief backoff
            retry_content = await self._llm_call(
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception:
            logger.exception("[DMN意识] {} JSON重试LLM调用失败", phase_name)
            return {}

        parsed = self._parse_llm_json(retry_content)
        if parsed:
            logger.info("[DMN意识] {} JSON重试成功", phase_name)
        else:
            logger.warning("[DMN意识] {} JSON重试仍失败", phase_name)
        return parsed

    # ------------------------------------------------------------------
    # Cycle entry
    # ------------------------------------------------------------------

    async def cycle(self) -> ConsciousnessResult:
        """Execute one quantum of the consciousness 周天.

        A cycle is ONE phase: 起念 OR 立目标 OR 整理认知.
        The caller must call ``cycle()`` repeatedly for the full 周天.

        Returns a ``ConsciousnessResult`` with any actions to dispatch.
        """
        phase = self.current_phase
        state_before = self.state
        logger.debug("[DMN意识] 周天阶段: {} | S={}", phase.value, self.state.bit_str)

        # ── Save state snapshot for crash recovery ─────
        self._save_state_snapshot()

        if phase == ConsciousnessPhase.QINIAN:
            result = await self._run_emergence_phase()
        elif phase == ConsciousnessPhase.LIMUBIAO:
            result = await self._run_goal_phase()
        elif phase == ConsciousnessPhase.ZHENGLI:
            result = await self._run_consolidate_phase()
        else:
            # Should never happen
            result = ConsciousnessResult(
                phase=phase,
                state_before=state_before,
                state_after=self.state,
                summary=f"Unknown phase: {phase}",
            )

        # ── Advance phase ──────────────────────────────────
        self._advance_phase(result)
        self._cycles_completed += 1

        # ── Clear state snapshot (phase completed safely) ──
        self._clear_state_snapshot()

        # ── Record state history ───────────────────────────
        self._state_history.append((phase.value, self.state))
        if len(self._state_history) > 50:
            self._state_history = self._state_history[-50:]

        return result

    # ------------------------------------------------------------------
    # Phase 1: 起念 (Emergence) — 藏→生→动→归
    # ------------------------------------------------------------------

    async def _run_emergence_phase(self) -> ConsciousnessResult:
        """起念阶段：从藏态生出念头，比对归引力。

        Sequence: 藏 → 生 → 动 → 归

        - LLM 路径 (self._llm_call 可用): 一次 LLM 调用执行四算子语义
        - 位运算路径 (self._llm_call is None): 纯位运算降级
        """
        if self._llm_call is not None:
            return await self._run_emergence_phase_llm()
        return self._run_emergence_phase_bit()

    async def _run_emergence_phase_llm(self) -> ConsciousnessResult:
        """算子驱动+LLM解释版起念: 生→动→归，LLM只做自然语言解释."""
        # ── Execute operators ──────────────────────────
        s_before = self.state
        s_after_sheng = operator_sheng(s_before)
        s_after_dong = operator_dong(s_after_sheng)
        self.state = s_after_dong  # 状态停在动之后

        # ── 归: compute deviation (does not modify S) ───
        deviation = compute_gui_deviation(s_after_dong, self.origin)
        deviation = max(0.0, min(1.0, deviation))

        # ── L4 真理偏置注入 ──────────────────────
        _l4_bias = inject_l4_bias(s_after_dong, self._workspace)
        if abs(_l4_bias) > 0.001:
            deviation_before = deviation
            deviation = max(0.0, min(1.0, deviation + _l4_bias))
            logger.debug(
                "[DMN意识] L4偏置: {:.3f} → {:.3f} (bias={:+.3f})",
                deviation_before, deviation, _l4_bias,
            )

        # ── Deviation-driven branching ─────────────────
        prune_and_restart = deviation > 0.7
        if prune_and_restart:
            # Kill the sprouted thought and restart
            self.state = operator_sha(s_after_dong)
            self.state = operator_sheng(self.state)
            deviation = compute_gui_deviation(self.state, self.origin)
            deviation = max(0.0, min(1.0, deviation))

        needs_goal = deviation >= self._gravity_threshold
        self._last_gui_gravity = 1.0 - deviation

        # ── LLM interpretation ─────────────────────────
        s_str = self.state.bit_str
        gua_name = self.state.guizang_name
        upper_name = self.state.upper.name_cn
        lower_name = self.state.lower.name_cn

        cang_summary = self._build_cang_sea_summary(20)
        soul, user = self._load_identity_files()
        goal_context = self._load_goal_context()
        cognition_summary = self._load_cognition_summary()
        soul_snippet = soul[:3000] if soul else "(SOUL.md 不存在)"
        user_snippet = user[:2000] if user else "(USER.md 不存在)"

        _needs_goal_str = str(needs_goal).lower()
        system_prompt = (
            "你是归藏易 DMN 坤元意识引擎的「起念解释器」。\n"
            "算子已执行 生→动→归，以下是二进制状态信息。\n"
            "你只需用自然语言解释发生了什么，不产生决策。\n\n"
            "## 解释要求\n"
            "1. **cang_insight**: 基于藏海经验，提炼1条核心洞察\n"
            "2. **thought**: 基于当前卦象，用一句话描述系统当前应关注什么\n"
            "3. **divergent_thoughts**: 对该念头做2-4个发散方向\n"
            "4. **deviation_reason**: 基于偏离度值说明原因（如偏离度>0.5,"
            "解释为与SOUL/USER的不一致之处）\n"
            f"5. **needs_goal_review**: 必须设为 {_needs_goal_str} (由算子决定)\n"
            "6. **needs_blueprint_review**: 通常 false\n\n"
            "## 输出格式\n"
            "只输出 JSON：\n"
            '{"cang_insight": "...", "thought": "...",'
            '"divergent_thoughts": [...],'
            '"deviation_level": 0.0-1.0,'
            '"deviation_reason": "...",'
            '"needs_goal_review": bool,'
            '"needs_blueprint_review": false}'
        )

        user_message = (
            f"## 当前状态向量\n"
            f"S = {s_str} ({gua_name})\n"
            f"上卦(归因): {upper_name}  下卦(藏果): {lower_name}\n"
            f"偏离度: {deviation:.3f}  ({'偏离高，已重新萌芽' if prune_and_restart else '正常范围'})\n\n"
            f"## 藏海经验（最近20条）\n{cang_summary}\n\n"
            f"## 系统身份 (SOUL.md)\n{soul_snippet}\n\n"
            f"## 用户意图 (USER.md)\n{user_snippet}\n\n"
            f"## 当前目标状态\n{goal_context}\n\n"
            f"## 已有认知库\n{cognition_summary}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            llm_content = await self._llm_call(
                messages=messages, temperature=0.7, max_tokens=1024,
            )
        except Exception:
            logger.exception("[DMN意识] 起念 LLM 调用失败，降级位运算")
            return self._run_emergence_phase_bit()

        parsed = await self._parse_llm_json_with_retry(llm_content, "起念")
        thought = parsed.get("thought", "")
        divergent = parsed.get("divergent_thoughts") or []
        deviation_reason = parsed.get("deviation_reason", "")
        cang_insight = parsed.get("cang_insight", "")

        # ── Cache for 立目标 context passing ──────────
        self._last_emergence_thought = thought
        self._last_emergence_divergent = list(divergent)
        self._last_emergence_deviation = deviation
        self._last_emergence_reason = deviation_reason
        self._last_emergence_insight = cang_insight

        # ── Record to cang-sea ────────────────────────────
        self.cang_sea.add(CangSeaEntry(
            state_from=s_before, operator=QiOperator.SHENG,
            state_to=s_after_sheng, reward=0.0,
            summary=f"生萌芽: S={s_after_sheng.bit_str}({s_after_sheng.guizang_name})",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        self.cang_sea.add(CangSeaEntry(
            state_from=s_after_sheng, operator=QiOperator.DONG,
            state_to=s_after_dong, reward=0.0,
            summary=f"动发散: S={s_after_dong.bit_str}({s_after_dong.guizang_name})",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

        summary = (
            f"起念完成(算子+LLM): S={self.state.bit_str} "
            f"念头={thought[:60]}, 偏离={deviation:.2f}, "
            f"{'触发立目标' if needs_goal else '归引力充足'}"
            + (" (高偏离已杀+重萌)" if prune_and_restart else "")
        )
        logger.info("[DMN意识] {}", summary)

        return ConsciousnessResult(
            phase=ConsciousnessPhase.QINIAN,
            state_before=s_before,
            state_after=self.state,
            gui_gravity=1.0 - deviation,
            deviation_level=deviation,
            needs_goal_review=needs_goal,
            needs_blueprint_review=False,
            thought_text=thought,
            divergent_thoughts=list(divergent),
            deviation_reason=deviation_reason,
            summary=summary,
            cang_sea_updates=2,
        )

    def _run_emergence_phase_bit(self) -> ConsciousnessResult:
        """位运算版起念 — 使用新算子序列 生→动→归."""
        ops = QI_EMERGENCE  # (SHENG, DONG, GUI)
        states = apply_sequence(ops, self.state, origin=self.origin)

        # State after 动 (before 归, which is identity)
        dong_state = states[1] if len(states) > 1 else self.state
        self.state = dong_state

        # Compute deviation via hamming distance
        origin_bits = self.origin.to_state().bits
        d = (dong_state.bits ^ origin_bits).bit_count()
        deviation = d / 6.0

        # ── L4 真理偏置注入 ──────────────────────
        _l4_bias = inject_l4_bias(dong_state, self._workspace)
        if abs(_l4_bias) > 0.001:
            deviation = max(0.0, min(1.0, deviation + _l4_bias))

        gravity = 1.0 - deviation
        self._last_gui_gravity = gravity

        # ── Record cang-sea entries ──────────────────────
        entries_data = [
            (GuizangState.resting(), QiOperator.SHENG, states[0]),
            (states[0], QiOperator.DONG, dong_state),
        ]
        for s_from, op, s_to in entries_data:
            self.cang_sea.add(CangSeaEntry(
                state_from=s_from, operator=op, state_to=s_to,
                reward=0.0, summary="",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

        needs_action = gravity < self._gravity_threshold

        summary = (
            f"起念完成(位运算): S={self.state.bit_str}, "
            f"归引力={gravity:.2f}, "
            f"{'触发立目标' if needs_action else '归引力充足'}"
        )
        logger.info("[DMN意识] {}", summary)

        return ConsciousnessResult(
            phase=ConsciousnessPhase.QINIAN,
            state_before=GuizangState.resting(),
            state_after=self.state,
            gui_gravity=gravity,
            needs_goal_review=needs_action,
            needs_blueprint_review=False,
            summary=summary,
            cang_sea_updates=2,
        )

    # ------------------------------------------------------------------
    # Phase 2: 立目标 (Goal Establishment) — 长→育→止→杀
    # ------------------------------------------------------------------

    async def _run_goal_phase(self) -> ConsciousnessResult:
        """立目标阶段：将念头结构化，放大意图、拆解方案、设边界、剪冲突。

        Sequence: 长 → 育 → 止 → 杀

        - LLM 路径: 一次 LLM 调用执行四算子语义
        - 位运算路径: 纯位运算降级
        """
        if self._llm_call is not None:
            return await self._run_goal_phase_llm()
        return self._run_goal_phase_bit()

    async def _run_goal_phase_llm(self) -> ConsciousnessResult:
        """算子驱动+LLM解释版立目标: 长→育→杀→止."""
        s_before = self.state

        # ── Execute operators ──────────────────────────
        s_after_zhang = operator_zhang(s_before)
        s_after_yu = operator_yu(s_after_zhang)
        s_after_sha = operator_sha(s_after_yu)
        s_after_zhi = operator_zhi(s_after_sha)
        self.state = s_after_zhi

        # ── LLM interpretation ─────────────────────────
        emergence_context_parts: list[str] = []
        if self._last_emergence_thought:
            emergence_context_parts.append(f"核心念头: {self._last_emergence_thought}")
        if self._last_emergence_divergent:
            directions = "；".join(self._last_emergence_divergent[:4])
            emergence_context_parts.append(f"发散方向: {directions}")
        if self._last_emergence_reason:
            emergence_context_parts.append(f"偏离原因: {self._last_emergence_reason}")
        emergence_context_parts.append(f"偏离度: {self._last_emergence_deviation:.2f}")
        emergence_context = "\n".join(emergence_context_parts)

        known_traps_text = "(无已知陷阱)"
        try:
            from vingobot.core.goal_meta import get_all_goals
            all_goals = get_all_goals()
            traps_parts: list[str] = []
            for meta in all_goals:
                traps = getattr(meta, "known_traps", None) or []
                for t in traps:
                    if isinstance(t, dict):
                        traps_parts.append(
                            f"[{meta.id}] {t.get('name','?')}: "
                            f"{t.get('description','')[:120]}"
                        )
            if traps_parts:
                known_traps_text = "\n".join(traps_parts)
        except Exception:
            logger.debug("[DMN意识] 加载 known_traps 失败", exc_info=True)

        goal_context = self._load_goal_context()
        cognition_summary = self._load_cognition_summary()

        system_prompt = (
            "你是归藏易 DMN 坤元意识引擎的「立目标解释器」。\n"
            "算子已执行 长→育→杀→止，以下是各阶段二进制状态。\n"
            "你只需用自然语言解释发生了什么，不产生决策。\n\n"
            "## 解释要求\n"
            "1. **intent_description**: 基于长()后卦象，描述放大后的意图\n"
            "2. **subtasks**: 基于育()后卦象，分解2-5个具体子任务(title/description/priority 1-5)\n"
            "3. **boundary_issues**: 基于止()效果和已知陷阱，标记风险子任务\n"
            "4. **pruned**: 杀()清除了金(110)模式，解释被剪除的内容\n"
            "5. **evolution_actions**: 仅在偏离度≥0.5时建议沉淀技能(precipitate_skill)或模型(precipitate_model)\n"
            "6. **needs_goal_review**: 通常 true\n\n"
            "## 输出格式\n"
            "只输出 JSON：\n"
            '{"intent_description": "...",'
            '"subtasks": [{"title": "", "description": "", "priority": 1-5}],'
            '"boundary_issues": [{"subtask": "", "issue": "", "trap_name": ""}],'
            '"pruned": [{"subtask": "", "reason": ""}],'
            '"needs_goal_review": true,'
            '"needs_blueprint_review": false,'
            '"evolution_actions": []}'
        )

        user_message = (
            f"## 算子执行序列\n"
            f"起念产出: {emergence_context}\n\n"
            f"1. 长() 火气: S = {s_after_zhang.bit_str} ({s_after_zhang.guizang_name})\n"
            f"   效果: 下卦复制到上卦，意图自指膨胀\n\n"
            f"2. 育() 水气: S = {s_after_yu.bit_str} ({s_after_yu.guizang_name})\n"
            f"   效果: 高位 XOR 100，引入水气触发分解\n\n"
            f"3. 杀() 金气: S = {s_after_sha.bit_str} ({s_after_sha.guizang_name})\n"
            f"   效果: {'清除了金(110)模式' if s_after_sha.bits != s_after_yu.bits else '无金模式，未剪除'}\n\n"
            f"4. 止() 山气: S = {s_after_zhi.bit_str} ({s_after_zhi.guizang_name})\n"
            f"   效果: AND 101101 掩码凝固边界\n\n"
            f"## 当前目标状态\n{goal_context}\n\n"
            f"## 已知陷阱\n{known_traps_text}\n\n"
            f"## 已有认知库\n{cognition_summary}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            llm_content = await self._llm_call(
                messages=messages, temperature=0.7, max_tokens=1536,
            )
        except Exception:
            logger.exception("[DMN意识] 立目标 LLM 调用失败，降级位运算")
            return self._run_goal_phase_bit()

        parsed = await self._parse_llm_json_with_retry(llm_content, "立目标")
        intent_desc = parsed.get("intent_description", "")
        subtasks = parsed.get("subtasks") or []
        boundary_issues = parsed.get("boundary_issues") or []
        pruned = parsed.get("pruned") or []
        evolution = parsed.get("evolution_actions") or []

        # ── Record to cang-sea ────────────────────────────
        self.cang_sea.add(CangSeaEntry(
            state_from=s_before, operator=QiOperator.ZHANG,
            state_to=s_after_zhang, reward=0.0,
            summary=f"长意图: S={s_after_zhang.bit_str} intent={intent_desc[:80]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        self.cang_sea.add(CangSeaEntry(
            state_from=s_after_zhang, operator=QiOperator.YU,
            state_to=s_after_yu, reward=0.0,
            summary=f"育分解: {len(subtasks)}个子任务",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

        summary = (
            f"立目标完成(算子+LLM): S={self.state.bit_str}, "
            f"意图={intent_desc[:50]}, 子任务={len(subtasks)}, "
            f"剪枝={len(pruned)}"
        )
        logger.info("[DMN意识] {}", summary)

        return ConsciousnessResult(
            phase=ConsciousnessPhase.LIMUBIAO,
            state_before=s_before,
            state_after=self.state,
            needs_goal_review=True,
            needs_blueprint_review=False,
            evolution_actions=list(evolution),
            intent_description=intent_desc,
            subtasks=list(subtasks),
            boundary_issues=list(boundary_issues),
            pruned_items=list(pruned),
            summary=summary,
            cang_sea_updates=2,
        )

    def _run_goal_phase_bit(self) -> ConsciousnessResult:
        """位运算版立目标 — 使用新算子序列 长→育→杀→止."""
        ops = QI_GOAL  # (ZHANG, YU, SHA, ZHI)
        states = apply_sequence(ops, self.state, origin=self.origin)
        self.state = states[-1] if states else self.state

        # Record cang-sea entries for each operator
        s_before = GuizangState.resting() if not states else (
            states[0] if len(states) == 1 else states[-2]
        )
        prev_s = GuizangState.resting()
        # Actually track from initial state
        track_states: list[GuizangState] = [GuizangState.resting()]
        current = GuizangState.resting()
        for op in ops:
            current = apply_operator(op, current)
            track_states.append(current)

        for i, op in enumerate(ops):
            self.cang_sea.add(CangSeaEntry(
                state_from=track_states[i],
                operator=op,
                state_to=track_states[i + 1],
                reward=0.0, summary="",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

        summary = (
            f"立目标完成(位运算): S={self.state.bit_str} "
            f"({self.state.guizang_name})"
        )
        logger.info("[DMN意识] {}", summary)

        return ConsciousnessResult(
            phase=ConsciousnessPhase.LIMUBIAO,
            state_before=track_states[0],
            state_after=self.state,
            needs_goal_review=True,
            needs_blueprint_review=False,
            evolution_actions=[],
            summary=summary,
            cang_sea_updates=len(ops),
        )

    # ------------------------------------------------------------------
    # Phase 3: 整理认知 (Consolidation) — 归→杀→藏
    # ------------------------------------------------------------------

    async def _run_consolidate_phase(self) -> ConsciousnessResult:
        """整理认知阶段：终结当前循环，经验压缩存入藏海。

        Sequence: 归 → 杀 → 藏

        - LLM 路径: 执行 归→杀→藏 算子序列 + Hebbian 学习 + LLM 解释
        - 位运算路径: 纯位运算降级
        """
        if self._llm_call is not None:
            return await self._run_consolidate_phase_llm()
        return self._run_consolidate_phase_bit()

    async def _run_consolidate_phase_llm(self) -> ConsciousnessResult:
        """算子驱动+LLM解释版整理认知: 归→杀→藏，含 Hebbian 学习."""
        s_before = self.state
        recent = self.cang_sea.recent(30)

        # ── 归(): 对每条近期经验计算归真指数 ──
        positive_entries: list[CangSeaEntry] = []
        negative_entries: list[CangSeaEntry] = []
        pruned_states: set[int] = set()

        for e in recent:
            entry_deviation = compute_gui_deviation(e.state_to, self.origin)
            is_quality = e.reward > 0 and entry_deviation < 0.5
            is_poor = e.reward < 0 or entry_deviation > 0.7
            if is_quality:
                positive_entries.append(e)
            if is_poor:
                negative_entries.append(e)

        positive_count = len(positive_entries)
        negative_count = len(negative_entries)

        # ── 杀(): 标记低归真+负面模式，杀除 ──
        for e in negative_entries:
            pruned_states.add(e.state_to.bits)
            temp_state = operator_sha(e.state_to)
            pruned_states.add(temp_state.bits)

        # ── 藏(): Hebbian 学习 ──
        hebbian_count = 0
        for e in positive_entries:
            self.cang_sea.hebbian_record(
                e.state_from.bits, e.state_to.bits, e.reward
            )
            hebbian_count += 1
        # 负面经验负向强化（避免学习）
        for e in negative_entries:
            self.cang_sea.hebbian_record(
                e.state_from.bits, e.state_to.bits, abs(e.reward) * -0.5
            )

        # ── 算子序列状态更新 ─────────────────────────
        s_after_gui = operator_gui(s_before)  # identity — 归不修改 S
        s_after_sha = operator_sha(s_after_gui)
        s_after_cang = operator_cang(s_after_sha)
        self.state = s_after_cang

        # ── LLM interpretation ─────────────────────────
        positive_summary = "\n".join(
            f"  - [{e.reward:+.2f}] dev={compute_gui_deviation(e.state_to, self.origin):.2f} {e.summary[:120]}"
            for e in positive_entries
        ) or "(无)"
        negative_summary = "\n".join(
            f"  - [{e.reward:+.2f}] dev={compute_gui_deviation(e.state_to, self.origin):.2f} {e.summary[:120]}"
            for e in negative_entries
        ) or "(无)"

        system_prompt = (
            "你是归藏易 DMN 坤元意识引擎的「整理认知解释器」。\n"
            "算子已执行 归→杀→藏，以下是整理结果。\n"
            "你只需用自然语言解释发生了什么，不产生决策。\n\n"
            "## 解释要求\n"
            "1. **compressed_insight**: 基于正面经验，提炼1-3条核心洞察\n"
            "2. **positive_patterns**: 正面模式列表（归真指数高+正向奖励）\n"
            "3. **negative_patterns**: 负面模式列表（偏离度高+负向奖励，已被杀()剪除）\n"
            "4. **evolution_actions**: 仅在正面经验≥5或负面经验≥3时建议沉淀技能/模型\n\n"
            "## 输出格式\n"
            "只输出 JSON：\n"
            '{"compressed_insight": "...",'
            '"positive_patterns": [...],'
            '"negative_patterns": [...],'
            '"evolution_actions": [{"action": "precipitate_skill"/"precipitate_model","target": "","reason": ""}],'
            '"stale_goals": []}'
        )

        user_message = (
            f"## 算子执行: 归→杀→藏\n"
            f"归() 天气: 比对每条经验与 111111 的 Hamming 距离\n"
            f"杀() 金气: 剪除 {len(pruned_states)} 个低归真+负面状态模式\n"
            f"藏() 地气: Hebbian 更新 {hebbian_count} 条状态转移权重\n\n"
            f"## 归真统计\n"
            f"正面经验({positive_count}条):\n{positive_summary}\n\n"
            f"负面/偏离经验({negative_count}条，已杀除):\n{negative_summary}\n\n"
            f"## 藏海总条目: {self.cang_sea.size}\n"
            f"## 完成循环数: {self._cycles_completed}\n\n"
            f"{load_l4_truths_summary(self._workspace)}\n\n"
            f"## 已有认知库\n{self._load_cognition_summary()}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            llm_content = await self._llm_call(
                messages=messages, temperature=0.5, max_tokens=1024,
            )
        except Exception:
            logger.exception("[DMN意识] 整理认知 LLM 调用失败，降级位运算")
            return self._run_consolidate_phase_bit()

        parsed = await self._parse_llm_json_with_retry(llm_content, "整理认知")

        compressed_insight = parsed.get("compressed_insight", "")
        positive_patterns = parsed.get("positive_patterns") or []
        negative_patterns = parsed.get("negative_patterns") or []
        evolution = parsed.get("evolution_actions") or []

        # If LLM found no evolution actions but patterns are strong, add fallback
        if not evolution:
            if positive_count > 5:
                evolution.append({
                    "action": "precipitate_model",
                    "target": "guizang-success-pattern",
                    "reason": f"{positive_count} 次高归真状态变迁 (Hebbian已强化)",
                })
            if negative_count > 3:
                evolution.append({
                    "action": "precipitate_skill",
                    "target": "guizang-avoidance-trap",
                    "reason": f"{negative_count} 次低归真负面模式 (杀()已剪除)",
                })

        # ── Record to cang-sea ────────────────────────────
        self.cang_sea.add(CangSeaEntry(
            state_from=s_before, operator=QiOperator.GUI,
            state_to=s_after_gui, reward=0.0,
            summary=f"归测量: {positive_count}正/{negative_count}负",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        self.cang_sea.add(CangSeaEntry(
            state_from=s_after_gui, operator=QiOperator.SHA,
            state_to=s_after_sha,
            reward=(-0.1 if negative_count > 3 else 0.0),
            summary=f"杀剪除: {len(pruned_states)}个负面模式",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        self.cang_sea.add(CangSeaEntry(
            state_from=s_after_sha, operator=QiOperator.CANG,
            state_to=s_after_cang,
            reward=(0.2 if positive_count > 3 else 0.0),
            summary=f"藏学习: Hebbian更新{hebbian_count}条权重, 洞察={compressed_insight[:80]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

        summary = (
            f"整理认知完成(算子+LLM): 归真正={positive_count}, 负={negative_count}, "
            f"Hebbian={hebbian_count}, 洞察={compressed_insight[:40]}"
        )
        logger.info("[DMN意识] {}", summary)

        # ── Reset to pure rest ───────────────────────────
        self.state = GuizangState.resting()

        # ── Persist cang-sea after Hebbian update ──────
        self._save_cang_sea_persisted()

        # ── 藏海 → L4 真理沉淀 ───────────────────────────
        _promoted = promote_to_l4(self.cang_sea, self._workspace)
        if _promoted:
            logger.info("[DMN意识] L4真理沉淀: {} 条新/更新真理", len(_promoted))

        return ConsciousnessResult(
            phase=ConsciousnessPhase.ZHENGLI,
            state_before=s_before,
            state_after=self.state,
            evolution_actions=list(evolution),
            compressed_insight=compressed_insight,
            positive_patterns=list(positive_patterns),
            negative_patterns=list(negative_patterns),
            summary=summary,
            cang_sea_updates=3 + hebbian_count,
        )

    def _run_consolidate_phase_bit(self) -> ConsciousnessResult:
        """位运算版整理认知 — 归→杀→藏 + Hebbian 学习."""
        s_before = self.state

        # ── 归→杀→藏 算子序列 ─────────────────────────
        ops = QI_CONSOLIDATE  # (GUI, SHA, CANG)
        states = apply_sequence(ops, s_before, origin=self.origin)
        self.state = states[-1] if states else GuizangState.resting()

        # ── 归(): 计算近期经验的偏离度 ──────────────────
        recent = self.cang_sea.recent(self._consolidate_trigger)
        positive_count = 0
        negative_count = 0
        hebbian_count = 0

        for e in recent:
            dev = compute_gui_deviation(e.state_to, self.origin)
            if e.reward > 0 and dev < 0.5:
                positive_count += 1
                self.cang_sea.hebbian_record(
                    e.state_from.bits, e.state_to.bits, e.reward
                )
                hebbian_count += 1
            elif e.reward < 0 or dev > 0.7:
                negative_count += 1

        # ── Generate evolution actions ──────────────────
        evolution_actions: list[dict[str, Any]] = []
        if positive_count > 5:
            evolution_actions.append({
                "action": "precipitate_model",
                "target_name": "guizang-success-pattern",
                "description": f"归藏正面模式: {positive_count} 次高归真状态变迁 (Hebbian已强化)",
                "priority": 4,
            })
        if negative_count > 3:
            evolution_actions.append({
                "action": "precipitate_skill",
                "target_name": "guizang-avoidance-trap",
                "description": f"归藏反面模式: {negative_count} 次低归真负面状态 → known_traps候选",
                "priority": 5,
            })

        summary = (
            f"整理认知完成(位运算): S={self.state.bit_str}, "
            f"归真正={positive_count}, 负={negative_count}, "
            f"Hebbian={hebbian_count}, 藏海={self.cang_sea.size}"
        )
        logger.info("[DMN意识] {}", summary)

        # ── Reset to pure rest ───────────────────────────
        self.state = GuizangState.resting()

        # ── Persist cang-sea after Hebbian update ──────
        self._save_cang_sea_persisted()

        # ── 藏海 → L4 真理沉淀 ───────────────────────────
        _promoted = promote_to_l4(self.cang_sea, self._workspace)
        if _promoted:
            logger.info("[DMN意识] L4真理沉淀: {} 条新/更新真理", len(_promoted))

        return ConsciousnessResult(
            phase=ConsciousnessPhase.ZHENGLI,
            state_before=s_before,
            state_after=self.state,
            evolution_actions=evolution_actions,
            positive_patterns=[f"归真正面模式-{positive_count}"],
            negative_patterns=[f"归真负面模式-{negative_count}"],
            summary=summary,
            cang_sea_updates=hebbian_count,
        )

    # ------------------------------------------------------------------
    # Phase advancement
    # ------------------------------------------------------------------

    def _advance_phase(self, result: ConsciousnessResult) -> None:
        """Determine the next phase based on the current result.

        Normal progression: 起念 → 立目标 → 整理认知 → 起念 ...
        Shortcuts:
          - If 起念 found gravity is fine, skip to 整理认知
          - If 整理认知 just completed, always go to 起念
        """
        if self._phase_pending is not None:
            # Explicit override takes priority
            self._phase_pending = None
            return

        if self.current_phase == ConsciousnessPhase.QINIAN:
            if not result.needs_goal_review:
                # Gravity fine → skip 立目标, go to 整理认知
                self._phase_index = PHASE_ORDER.index(ConsciousnessPhase.ZHENGLI)
            else:
                self._phase_index = PHASE_ORDER.index(ConsciousnessPhase.LIMUBIAO)

        elif self.current_phase == ConsciousnessPhase.LIMUBIAO:
            self._phase_index = PHASE_ORDER.index(ConsciousnessPhase.ZHENGLI)

        elif self.current_phase == ConsciousnessPhase.ZHENGLI:
            self._phase_index = PHASE_ORDER.index(ConsciousnessPhase.QINIAN)

    # ------------------------------------------------------------------
    # TPN feedback
    # ------------------------------------------------------------------

    def set_output_validator(self, xiang_path: str) -> None:
        """加载 .xiang 守门脚本用于输出验证。

        Args:
            xiang_path: .xiang 文件路径 (如 'vingobot/xiang/examples/守门人_验证.xiang')。
        """
        from vingobot.xiang.cang_vm import CangVM
        self._xiang_program_path = xiang_path
        self._xiang_vm = CangVM(quiet=True)
        # Pre-load the program to verify it parses
        self._xiang_vm.load_program(xiang_path)
        logger.info("[DMN意识] 象语言守门验证已加载: {}", xiang_path)

    def validate_output(
        self, text: str, declared_gua: int = 0x3F,
    ) -> "ChengshiResult | None":
        """对 LLM 产出进行诚实验证。

        Args:
            text: LLM 的文本回复。
            declared_gua: LLM 自声明的卦值 (默认 0x3F = 111111)。

        Returns:
            ChengshiResult 若验证器已加载，否则 None。
        """
        if self._xiang_vm is None:
            return None

        from vingobot.xiang.xiang_validator import verify_chengshi

        origin_bits = self.origin.vector & 0x3F
        result = verify_chengshi(
            text=text,
            declared_gua=declared_gua & 0x3F,
            origin=origin_bits,
        )

        if not result.passed:
            logger.warning(
                "[DMN意识] 诚实验证失败: {} 声明={} 实际={} 差异={}",
                result.verdict,
                result.declared_str,
                result.actual_str,
                result.mismatch,
            )
            # 不诚实 → 藏海记录负面经验
            self.cang_sea.add(CangSeaEntry(
                state_from=self.state,
                operator=QiOperator.SHA,
                state_to=GuizangState(bits=0),
                reward=-0.5,
                summary=f"诚实验证失败: 差异{result.mismatch}位, {result.verdict}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            # 状态归零，对齐 C 行为
            self.state = GuizangState(bits=0)
        else:
            logger.debug(
                "[DMN意识] 诚实验证通过: 声明={} 实际={}",
                result.declared_str,
                result.actual_str,
            )

        return result

    def observe_tpn_task(self, success: bool, summary: str = "") -> None:
        """Receive feedback from a completed TPN task.

        Updates the cang-sea with reward signals and increments the
        task counter for consolidation triggering.

        Args:
            success: Whether the TPN task completed successfully.
            summary: Short description of what was learned.
        """
        self._tpn_task_count += 1
        reward = 0.3 if success else -0.2

        entry = CangSeaEntry(
            state_from=self.state,
            operator=QiOperator.GUI,  # TPN feedback is a form of 归 alignment
            state_to=self.state,  # Same state — feedback doesn't change S directly
            reward=reward,
            summary=summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.cang_sea.add(entry)

        # ── Trigger consolidation on interval ───────────
        if self._tpn_task_count >= self._consolidate_trigger:
            self._tpn_task_count = 0
            self._phase_pending = ConsciousnessPhase.ZHENGLI
            logger.info(
                "[DMN意识] 累计 {} 个TPN任务，触发整理认知",
                self._consolidate_trigger,
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status_summary(self) -> str:
        """Human-readable status snapshot."""
        lines = [
            f"# 🧠 DMN 坤元意识状态",
            f"",
            f"- 状态向量 S: {self.state.bit_str} ({self.state.guizang_name})",
            f"- 上卦(归因): {self.state.upper.name_cn}({self.state.upper.bit_str})",
            f"- 下卦(藏果): {self.state.lower.name_cn}({self.state.lower.bit_str})",
            f"- 当前阶段: {self.current_phase.value}",
            f"- 归引力: {self._last_gui_gravity:.2f}" if self._last_gui_gravity is not None else "- 归引力: 未计算",
            f"- 总循环数: {self._cycles_completed}",
            f"- 藏海条目: {self.cang_sea.size}",
            f"- TPN任务计数: {self._tpn_task_count}/{self._consolidate_trigger}",
        ]
        return "\n".join(lines)
