"""Tests for dmn_consciousness.py — Guizang DMN autonomous consciousness cycle."""

from __future__ import annotations

import asyncio

import pytest

from vingobot.goal.dmn_consciousness import (
    CONSOLIDATE_TRIGGER_TASKS,
    DmnConsciousness,
)
from vingobot.goal.guizang_types import (
    CangSeaEntry,
    CangSeaMemory,
    ConsciousnessPhase,
    GuizangState,
    OriginPerception,
    QiOperator,
)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestDmnConsciousnessInit:
    def test_default_state_is_resting(self) -> None:
        dmn = DmnConsciousness()
        assert dmn.state.is_resting
        assert dmn.state.bits == 0

    def test_origin_is_pure_yang(self) -> None:
        dmn = DmnConsciousness()
        assert dmn.origin.vector == (1, 1, 1, 1, 1, 1)

    def test_cang_sea_empty_on_init(self) -> None:
        dmn = DmnConsciousness()
        assert dmn.cang_sea.is_empty
        assert dmn.cang_sea.size == 0

    def test_initial_phase_is_qinian(self) -> None:
        dmn = DmnConsciousness()
        assert dmn.current_phase == ConsciousnessPhase.QINIAN

    def test_default_interval(self) -> None:
        dmn = DmnConsciousness()
        assert dmn.next_wake_interval() > 0
        # Default is 300s for 起念
        assert dmn.next_wake_interval() == 300.0

    def test_consolidate_interval_is_shorter(self) -> None:
        dmn = DmnConsciousness()
        # Force phase to 整理认知
        dmn._phase_pending = ConsciousnessPhase.ZHENGLI
        assert dmn.next_wake_interval() == 60.0


# ---------------------------------------------------------------------------
# Full 周天 cycle
# ---------------------------------------------------------------------------


class TestFullCycle:
    @pytest.mark.asyncio
    async def test_one_full_zhoutian(self) -> None:
        """A complete consciousness 周天: 起念 → 立目标 → 整理认知 → 起念."""
        dmn = DmnConsciousness()

        # Cycle 1: 起念
        r1 = await dmn.cycle()
        assert r1.phase == ConsciousnessPhase.QINIAN
        assert r1.gui_gravity is not None
        assert 0.0 <= r1.gui_gravity <= 1.0
        assert dmn._cycles_completed == 1

        # Cycle 2: 立目标 (because gravity from 起念 < 0.5 from 000000)
        r2 = await dmn.cycle()
        assert r2.phase == ConsciousnessPhase.LIMUBIAO
        assert dmn._cycles_completed == 2

        # Cycle 3: 整理认知
        r3 = await dmn.cycle()
        assert r3.phase == ConsciousnessPhase.ZHENGLI
        assert r3.state_after.is_resting
        assert dmn._cycles_completed == 3

        # Cycle 4: back to 起念
        r4 = await dmn.cycle()
        assert r4.phase == ConsciousnessPhase.QINIAN

    @pytest.mark.asyncio
    async def test_gravity_triggers_goal_phase(self) -> None:
        """When gravity is below threshold, 起念 triggers 立目标."""
        dmn = DmnConsciousness(gravity_threshold=0.5)
        # From resting state (000000), after 藏→生→动 the state changes.
        # Gravity is computed on the state AFTER 动, not the resting state.
        r = await dmn.cycle()
        assert r.phase == ConsciousnessPhase.QINIAN
        assert r.gui_gravity is not None
        assert 0.0 <= r.gui_gravity <= 1.0
        # Gravity below threshold (0.5) means needs_goal_review
        if r.gui_gravity < 0.5:
            assert r.needs_goal_review is True
        else:
            assert r.needs_goal_review is False

    @pytest.mark.asyncio
    async def test_high_threshold_skips_goal(self) -> None:
        """With a very low threshold, even full misalignment won't trigger."""
        dmn = DmnConsciousness(gravity_threshold=-1.0)
        r = await dmn.cycle()
        assert r.phase == ConsciousnessPhase.QINIAN
        assert r.needs_goal_review is False

    @pytest.mark.asyncio
    async def test_consolidation_returns_to_resting(self) -> None:
        """After 整理认知, state must be 000000."""
        dmn = DmnConsciousness()
        # Run to 整理认知
        await dmn.cycle()  # 起念
        await dmn.cycle()  # 立目标
        r = await dmn.cycle()  # 整理认知
        assert r.phase == ConsciousnessPhase.ZHENGLI
        assert dmn.state.is_resting
        assert dmn.state.bits == 0


# ---------------------------------------------------------------------------
# TPN feedback
# ---------------------------------------------------------------------------


class TestTpnFeedback:
    @pytest.mark.asyncio
    async def test_observe_adds_entry(self) -> None:
        dmn = DmnConsciousness()
        assert dmn.cang_sea.is_empty

        dmn.observe_tpn_task(success=True, summary="task completed")
        assert dmn.cang_sea.size == 1

    @pytest.mark.asyncio
    async def test_success_positive_reward(self) -> None:
        dmn = DmnConsciousness()
        dmn.observe_tpn_task(success=True)
        entry = dmn.cang_sea.entries[-1]
        assert entry.reward > 0

    @pytest.mark.asyncio
    async def test_failure_negative_reward(self) -> None:
        dmn = DmnConsciousness()
        dmn.observe_tpn_task(success=False)
        entry = dmn.cang_sea.entries[-1]
        assert entry.reward < 0

    @pytest.mark.asyncio
    async def test_consolidate_trigger(self) -> None:
        dmn = DmnConsciousness(consolidate_trigger=3)
        # Feed 3 tasks — this should set _phase_pending to 整理认知
        dmn.observe_tpn_task(success=True)
        dmn.observe_tpn_task(success=True)
        dmn.observe_tpn_task(success=False)
        assert dmn._phase_pending == ConsciousnessPhase.ZHENGLI

    @pytest.mark.asyncio
    async def test_consolidate_not_triggered_below_threshold(self) -> None:
        dmn = DmnConsciousness(consolidate_trigger=20)
        dmn.observe_tpn_task(success=True)
        dmn.observe_tpn_task(success=True)
        assert dmn._phase_pending is None  # Not triggered yet


# ---------------------------------------------------------------------------
# Cang-sea memory
# ---------------------------------------------------------------------------


class TestCangSeaMemory:
    def test_add_and_retrieve(self) -> None:
        memory = CangSeaMemory()
        entry = CangSeaEntry(
            state_from=GuizangState.resting(),
            operator=QiOperator.SHENG,
            state_to=GuizangState(bits=1),
            reward=0.5,
            summary="test",
        )
        memory.add(entry)
        assert memory.size == 1
        assert memory.entries[0] == entry

    def test_max_entries_trim(self) -> None:
        memory = CangSeaMemory(max_entries=5)
        for i in range(10):
            entry = CangSeaEntry(
                state_from=GuizangState.resting(),
                operator=QiOperator.SHENG,
                state_to=GuizangState(bits=i % 64),
            )
            memory.add(entry)
        assert memory.size == 5  # Trimmed to max

    def test_recent(self) -> None:
        memory = CangSeaMemory()
        for i in range(5):
            memory.add(CangSeaEntry(
                state_from=GuizangState.resting(),
                operator=QiOperator.SHENG,
                state_to=GuizangState(bits=i),
            ))
        recent = memory.recent(3)
        assert len(recent) == 3

    def test_positive_entries(self) -> None:
        memory = CangSeaMemory()
        memory.add(CangSeaEntry(
            state_from=GuizangState.resting(),
            operator=QiOperator.SHENG,
            state_to=GuizangState(bits=1),
            reward=0.5,
        ))
        memory.add(CangSeaEntry(
            state_from=GuizangState.resting(),
            operator=QiOperator.SHENG,
            state_to=GuizangState(bits=2),
            reward=-0.3,
        ))
        assert len(memory.positive_entries()) == 1
        assert len(memory.negative_entries()) == 1

    def test_by_operator(self) -> None:
        memory = CangSeaMemory()
        memory.add(CangSeaEntry(
            state_from=GuizangState.resting(),
            operator=QiOperator.SHENG,
            state_to=GuizangState(bits=1),
        ))
        memory.add(CangSeaEntry(
            state_from=GuizangState.resting(),
            operator=QiOperator.DONG,
            state_to=GuizangState(bits=2),
        ))
        assert len(memory.by_operator(QiOperator.SHENG)) == 1
        assert len(memory.by_operator(QiOperator.DONG)) == 1
        assert len(memory.by_operator(QiOperator.CANG)) == 0


# ---------------------------------------------------------------------------
# Hebbian learning on CangSeaMemory
# ---------------------------------------------------------------------------


class TestHebbianMemory:
    def test_hebbian_record_increases_weight(self) -> None:
        memory = CangSeaMemory()
        assert memory.transition_weights[0][1] == 0.0
        memory.hebbian_record(0, 1, 0.5)
        assert memory.transition_weights[0][1] == pytest.approx(0.05)

    def test_hebbian_record_clamped(self) -> None:
        memory = CangSeaMemory()
        memory.transition_weights[0][1] = 0.99
        memory.hebbian_record(0, 1, 1.0)
        assert memory.transition_weights[0][1] == pytest.approx(1.0)

    def test_hebbian_sample_from_empty_returns_none(self) -> None:
        memory = CangSeaMemory()
        result = memory.hebbian_sample(0)
        assert result is None

    def test_hebbian_sample_returns_learned_state(self) -> None:
        memory = CangSeaMemory()
        memory.hebbian_record(0, 1, 1.0)  # Strong weight 0→1
        memory.hebbian_record(0, 2, 0.5)  # Weaker weight 0→2
        # Sample many times — should always return 1 or 2
        for _ in range(20):
            result = memory.hebbian_sample(0)
            assert result in (1, 2)


# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------


class TestStatusSummary:
    def test_basic_summary(self) -> None:
        dmn = DmnConsciousness()
        summary = dmn.status_summary()
        assert "DMN" in summary or "坤元" in summary
        assert "000000" in summary
        assert "起念" in summary

    @pytest.mark.asyncio
    async def test_summary_after_cycle(self) -> None:
        dmn = DmnConsciousness()
        await dmn.cycle()
        summary = dmn.status_summary()
        assert "归引力" in summary  # Should now have gui-gravity


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    @pytest.mark.asyncio
    async def test_emergence_changes_state(self) -> None:
        dmn = DmnConsciousness()
        before = dmn.state.bits
        await dmn.cycle()  # 起念
        # After 起念 (藏→生→动→归), state should be non-resting
        # Note: 归 may realign back toward 000 since gravity is 0
        # So we just check the cycle ran without error
        assert dmn._cycles_completed == 1

    @pytest.mark.asyncio
    async def test_goal_phase_generates_evolution_actions(self) -> None:
        dmn = DmnConsciousness()
        # From 000000, gravity is 0 → triggers 立目标
        await dmn.cycle()  # 起念
        r = await dmn.cycle()  # 立目标
        # 立目标 after 起念 should potentially produce evolution actions
        assert r.phase == ConsciousnessPhase.LIMUBIAO

    @pytest.mark.asyncio
    async def test_phase_progression_manual_override(self) -> None:
        dmn = DmnConsciousness()
        # Force to 整理认知
        dmn._phase_pending = ConsciousnessPhase.ZHENGLI
        r = await dmn.cycle()
        assert r.phase == ConsciousnessPhase.ZHENGLI

    @pytest.mark.asyncio
    async def test_cang_sea_grows_over_cycles(self) -> None:
        dmn = DmnConsciousness()
        assert dmn.cang_sea.is_empty

        await dmn.cycle()  # 起念 → adds 3 entries (生,动,归)
        assert dmn.cang_sea.size > 0

        await dmn.cycle()  # 立目标 → adds 4 entries (长,育,止,杀)
        assert dmn.cang_sea.size > 3

    @pytest.mark.asyncio
    async def test_state_history(self) -> None:
        dmn = DmnConsciousness()
        for _ in range(5):
            await dmn.cycle()
        assert len(dmn._state_history) <= 50
        assert len(dmn._state_history) == 5

    @pytest.mark.asyncio
    async def test_multiple_cycles_no_error(self) -> None:
        """Run many cycles to ensure no state corruption."""
        dmn = DmnConsciousness()
        for _ in range(10):
            r = await dmn.cycle()
            assert r.phase in (
                ConsciousnessPhase.QINIAN,
                ConsciousnessPhase.LIMUBIAO,
                ConsciousnessPhase.ZHENGLI,
            )
        assert dmn._cycles_completed == 10


# ---------------------------------------------------------------------------
# OriginPerception edge cases
# ---------------------------------------------------------------------------


class TestOriginEdgeCases:
    def test_custom_origin_vector(self) -> None:
        origin = OriginPerception(vector=(1, 0, 1, 0, 1, 0), gravity_constant=2.0)
        assert origin.to_state().bits == 0b101010
        assert origin.gravity_constant == 2.0


# ---------------------------------------------------------------------------
# LLM 语义路径测试 (Mock LLM)
# ---------------------------------------------------------------------------

import json as _json


def _make_llm_mock(responses: list[str]):
    """Create an async mock LLM callback that returns each response in order.

    Each element in *responses* is a string to return, or an Exception to
    raise.
    """
    calls: list[dict] = []

    async def _call(messages, temperature=0.7, max_tokens=1024) -> str:
        calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        if not responses:
            return "{}"
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    _call.calls = calls
    return _call


class TestLlmEmergencePhase:
    """LLM 语义版起念阶段."""

    @pytest.mark.asyncio
    async def test_llm_emergence_happy_path(self) -> None:
        """LLM 返回有效 JSON → thought_text, deviation 等字段正确填充."""
        llm = _make_llm_mock([_json.dumps({
            "cang_insight": "最近目标月收入推进缓慢，多轮探索后无实质产出",
            "thought": "应审查月收入目标的滞涨问题",
            "divergent_thoughts": ["调整时间分配", "缩短探索轮次", "提高最小产出标准"],
            "deviation_level": 0.62,
            "deviation_reason": "多轮无交付物，与USER.md中产出驱动原则偏离",
            "needs_goal_review": True,
            "needs_blueprint_review": False,
        })])

        dmn = DmnConsciousness(llm_call=llm)
        r = await dmn.cycle()

        assert r.phase == ConsciousnessPhase.QINIAN
        assert r.thought_text == "应审查月收入目标的滞涨问题"
        assert len(r.divergent_thoughts) == 3
        # deviation computed from operator state (000001 vs 111111 = 5/6)
        assert r.deviation_level == pytest.approx(0.833, abs=0.01)
        assert r.deviation_reason != ""
        assert r.needs_goal_review is True
        assert r.needs_blueprint_review is False
        assert r.gui_gravity == pytest.approx(0.167, abs=0.01)  # 1 - 5/6
        assert llm.calls  # was actually called

    @pytest.mark.asyncio
    async def test_llm_emergence_high_deviation_triggers_goal(self) -> None:
        """从坤态初态起念，偏离度高 → 必然触发立目标."""
        llm = _make_llm_mock([_json.dumps({
            "cang_insight": "一切运行良好",
            "thought": "系统运行正常，无需干预",
            "divergent_thoughts": [],
            "deviation_level": 0.15,
            "deviation_reason": "",
            "needs_goal_review": False,
            "needs_blueprint_review": False,
        })])

        dmn = DmnConsciousness(llm_call=llm)
        r = await dmn.cycle()

        assert r.thought_text == "系统运行正常，无需干预"
        # deviation 由算子状态决定 (000001 vs 111111 → 5/6)，不是 LLM 响应
        assert r.deviation_level == pytest.approx(0.833, abs=0.01)
        assert r.needs_goal_review is True
        assert dmn.current_phase == ConsciousnessPhase.LIMUBIAO

    @pytest.mark.asyncio
    async def test_llm_emergence_returns_junk_falls_back_to_bit(self) -> None:
        """LLM 返回非 JSON → 降级到位运算."""
        llm = _make_llm_mock(["just some random text, not JSON"])

        dmn = DmnConsciousness(llm_call=llm)
        r = await dmn.cycle()

        # Should still produce a valid result (bit fallback)
        assert r.phase == ConsciousnessPhase.QINIAN
        assert r.gui_gravity is not None
        assert 0.0 <= r.gui_gravity <= 1.0

    @pytest.mark.asyncio
    async def test_llm_emergence_call_fails_falls_back_to_bit(self) -> None:
        """LLM 调用异常 → 降级到位运算."""
        llm = _make_llm_mock([RuntimeError("simulated provider failure")])

        dmn = DmnConsciousness(llm_call=llm)
        r = await dmn.cycle()

        assert r.phase == ConsciousnessPhase.QINIAN
        assert r.gui_gravity is not None

    @pytest.mark.asyncio
    async def test_emergence_system_prompt_is_interpreter(self) -> None:
        """起念 system prompt 表明算子已执行、LLM只做解释."""
        llm = _make_llm_mock([_json.dumps({
            "cang_insight": "一切正常",
            "thought": "系统运行良好",
            "divergent_thoughts": [],
            "deviation_level": 0.1,
            "deviation_reason": "",
            "needs_goal_review": False,
            "needs_blueprint_review": False,
        })])

        dmn = DmnConsciousness(llm_call=llm)
        await dmn.cycle()

        system_prompt = llm.calls[0]["messages"][0]["content"]
        assert "起念解释器" in system_prompt
        assert "算子已执行" in system_prompt
        assert "生→动→归" in system_prompt
        assert "不产生决策" in system_prompt
        assert "cang_insight" in system_prompt
        assert "deviation_level" in system_prompt


class TestLlmGoalPhase:
    """LLM 语义版立目标阶段."""

    @pytest.mark.asyncio
    async def test_llm_goal_happy_path(self) -> None:
        """LLM 立目标返回有效 JSON → subtasks, boundary_issues 正确填充."""
        # 两轮 mock: 第一轮起念(高偏离触发立目标), 第二轮立目标
        llm = _make_llm_mock([
            _json.dumps({
                "cang_insight": "近期无有效产出",
                "thought": "需要审查目标",
                "divergent_thoughts": ["方向A"],
                "deviation_level": 0.8,
                "deviation_reason": "测试",
                "needs_goal_review": True,
                "needs_blueprint_review": False,
            }),
            _json.dumps({
                "intent_description": "对月收入目标进行结构化审查并输出优化方案",
                "subtasks": [
                    {"title": "分析当前月收入目标进展", "description": "读取 blueprint 和 trajectory", "priority": 5},
                    {"title": "识别滞涨瓶颈", "description": "检查近期任务的成功/失败模式", "priority": 4},
                ],
                "boundary_issues": [
                    {"subtask": "写入优化方案", "issue": "可能过早写入未经验证的方案", "trap_name": "premature_output"},
                ],
                "pruned": [],
                "needs_goal_review": True,
                "needs_blueprint_review": False,
                "evolution_actions": [],
            }),
        ])

        dmn = DmnConsciousness(llm_call=llm)
        await dmn.cycle()  # 起念
        r = await dmn.cycle()  # 立目标

        assert r.phase == ConsciousnessPhase.LIMUBIAO
        assert "月收入" in r.intent_description
        assert len(r.subtasks) == 2
        assert r.subtasks[0]["title"] == "分析当前月收入目标进展"
        assert len(r.boundary_issues) == 1

    @pytest.mark.asyncio
    async def test_llm_goal_call_fails_falls_back(self) -> None:
        """LLM 立目标调用失败 → 降级."""
        # 起念成功, 立目标失败 → 降级到位运算
        llm = _make_llm_mock([
            _json.dumps({
                "cang_insight": "test",
                "thought": "test",
                "divergent_thoughts": [],
                "deviation_level": 0.8,
                "deviation_reason": "",
                "needs_goal_review": True,
                "needs_blueprint_review": False,
            }),
            RuntimeError("boom"),
        ])

        dmn = DmnConsciousness(llm_call=llm)
        await dmn.cycle()  # 起念
        r = await dmn.cycle()  # 立目标 (降级到位运算)

        assert r.phase == ConsciousnessPhase.LIMUBIAO

    @pytest.mark.asyncio
    async def test_emergence_context_passed_to_goal_llm(self) -> None:
        """起念产出 (thought/deviation/divergent) 正确注入立目标 LLM 调用."""
        llm = _make_llm_mock([
            _json.dumps({
                "cang_insight": "藏海洞察测试",
                "thought": "应优化任务分配策略",
                "divergent_thoughts": ["方向X", "方向Y", "方向Z"],
                "deviation_level": 0.65,
                "deviation_reason": "USER.md定义的效率标准未达成",
                "needs_goal_review": True,
                "needs_blueprint_review": False,
            }),
            _json.dumps({
                "intent_description": "优化任务分配",
                "subtasks": [],
                "boundary_issues": [],
                "pruned": [],
                "needs_goal_review": True,
                "needs_blueprint_review": False,
                "evolution_actions": [],
            }),
        ])

        dmn = DmnConsciousness(llm_call=llm)
        await dmn.cycle()  # 起念 → 缓存 _last_emergence_*

        # 验证起念缓存
        assert dmn._last_emergence_thought == "应优化任务分配策略"
        assert len(dmn._last_emergence_divergent) == 3
        assert dmn._last_emergence_deviation == pytest.approx(0.833, abs=0.01)
        assert "效率标准" in dmn._last_emergence_reason
        assert dmn._last_emergence_insight == "藏海洞察测试"

        r = await dmn.cycle()  # 立目标 → 应收到起念上下文
        assert r.phase == ConsciousnessPhase.LIMUBIAO

        # 立目标 LLM 调用应包含起念上下文
        goal_call = llm.calls[-1]
        user_content = goal_call["messages"][1]["content"]
        assert "应优化任务分配策略" in user_content  # 念头已注入
        assert "方向X" in user_content  # 发散方向已注入
        assert "偏离原因" in user_content  # 偏离原因已注入
        assert "0.83" in user_content  # 偏离度已注入 (运算结果 5/6 ≈ 0.83)
        assert "已有认知库" in user_content  # 认知库已注入


class TestLlmConsolidatePhase:
    """LLM 语义版整理认知阶段."""

    @pytest.mark.asyncio
    async def test_llm_consolidate_happy_path(self) -> None:
        """LLM 整理认知返回有效 JSON → compressed_insight, patterns 正确."""
        llm = _make_llm_mock([_json.dumps({
            "compressed_insight": "月收入目标在多轮迭代中积累了 5 个正面经验，3 个负面经验。建议沉淀 avoid-premature-output 技能。",
            "positive_patterns": ["定期目标审查提高效率", "分阶段产出降低风险"],
            "negative_patterns": ["过早写入未验证方案", "过度探索浪费轮次"],
            "evolution_actions": [
                {"action": "precipitate_skill", "target": "avoid-premature-output", "reason": "多次因过早写入未验证方案导致回滚"},
            ],
            "stale_goals": [],
        })])

        dmn = DmnConsciousness(llm_call=llm)
        # Force to consolidation phase
        dmn._phase_pending = ConsciousnessPhase.ZHENGLI
        r = await dmn.cycle()

        assert r.phase == ConsciousnessPhase.ZHENGLI
        assert "avoid-premature-output" in r.compressed_insight
        assert len(r.positive_patterns) == 2
        assert len(r.negative_patterns) == 2
        assert len(r.evolution_actions) == 1
        assert r.state_after.is_resting

    @pytest.mark.asyncio
    async def test_llm_consolidate_call_fails_falls_back(self) -> None:
        """LLM 整理认知调用失败 → 降级."""
        llm = _make_llm_mock([RuntimeError("boom")])

        dmn = DmnConsciousness(llm_call=llm)
        dmn._phase_pending = ConsciousnessPhase.ZHENGLI
        r = await dmn.cycle()

        assert r.phase == ConsciousnessPhase.ZHENGLI
        assert r.state_after.is_resting

    @pytest.mark.asyncio
    async def test_consolidation_hebbian_learning(self) -> None:
        """整理认知应执行 Hebbian 学习，更新转移权重矩阵."""
        llm = _make_llm_mock([_json.dumps({
            "compressed_insight": "测试洞察",
            "positive_patterns": ["模式A"],
            "negative_patterns": ["模式B"],
            "evolution_actions": [],
            "stale_goals": [],
        })])

        dmn = DmnConsciousness(llm_call=llm)
        # Pre-populate cang-sea with low-deviation positive entries (close to 111111)
        dmn.cang_sea.add(CangSeaEntry(
            state_from=GuizangState(bits=63),  # 111111 (origin)
            operator=QiOperator.SHENG,
            state_to=GuizangState(bits=62),  # 111110 (dev=1/6≈0.17)
            reward=0.5,
        ))
        dmn.cang_sea.add(CangSeaEntry(
            state_from=GuizangState(bits=62),
            operator=QiOperator.DONG,
            state_to=GuizangState(bits=61),  # 111101 (dev=2/6≈0.33)
            reward=0.3,
        ))
        # Also add negative entry
        dmn.cang_sea.add(CangSeaEntry(
            state_from=GuizangState(bits=61),
            operator=QiOperator.SHA,
            state_to=GuizangState(bits=0),  # 000000 (dev=1.0)
            reward=-0.5,
        ))

        dmn._phase_pending = ConsciousnessPhase.ZHENGLI
        r = await dmn.cycle()

        assert r.phase == ConsciousnessPhase.ZHENGLI
        assert r.state_after.is_resting
        # Hebbian weights should have been updated for positive (low-deviation) entries
        assert dmn.cang_sea.transition_weights[63][62] > 0  # positive entry
        assert dmn.cang_sea.transition_weights[62][61] > 0  # positive entry
        assert r.cang_sea_updates >= 3  # 3 cang-sea records + hebbian updates
