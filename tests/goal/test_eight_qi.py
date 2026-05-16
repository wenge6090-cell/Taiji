"""Tests for eight_qi.py — Guizang binary operator correctness."""

from __future__ import annotations

import pytest

from vingobot.goal.eight_qi import (
    apply_operator,
    apply_sequence,
    compute_gui_deviation,
    compute_gui_gravity,
    compute_gui_pull,
    hamming_distance,
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
    QI_CYCLE_ORDER,
    QI_EMERGENCE,
    QI_GOAL,
    GuizangState,
    OriginPerception,
    QiOperator,
)


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------


class TestHammingDistance:
    def test_identical(self) -> None:
        assert hamming_distance(0b000000, 0b000000) == 0
        assert hamming_distance(0b111111, 0b111111) == 0

    def test_opposite(self) -> None:
        assert hamming_distance(0b000000, 0b111111) == 6

    def test_partial(self) -> None:
        assert hamming_distance(0b111000, 0b111111) == 3
        assert hamming_distance(0b101010, 0b111111) == 3
        assert hamming_distance(0b001001, 0b000000) == 2


# ---------------------------------------------------------------------------
# Gui gravity
# ---------------------------------------------------------------------------


class TestGuiGravity:
    def test_full_alignment(self) -> None:
        state = GuizangState(bits=0b111111)
        origin = OriginPerception.pure_yang()
        assert compute_gui_gravity(state, origin) == 1.0

    def test_full_misalignment(self) -> None:
        state = GuizangState(bits=0b000000)
        origin = OriginPerception.pure_yang()
        assert compute_gui_gravity(state, origin) == 0.0

    def test_partial_alignment(self) -> None:
        state = GuizangState(bits=0b111000)
        origin = OriginPerception.pure_yang()
        assert compute_gui_gravity(state, origin) == 0.5

    def test_gui_pull_full_misalignment(self) -> None:
        state = GuizangState(bits=0b000000)
        origin = OriginPerception.pure_yang()
        assert compute_gui_pull(state, origin) == 1.0

    def test_gui_pull_full_alignment(self) -> None:
        state = GuizangState(bits=0b111111)
        origin = OriginPerception.pure_yang()
        assert compute_gui_pull(state, origin) == 0.0


# ---------------------------------------------------------------------------
# 藏算子 (CANG)
# ---------------------------------------------------------------------------


class TestOperatorCang:
    def test_resting_to_resting(self) -> None:
        state = GuizangState.resting()
        result = operator_cang(state)
        assert result.bits == 0
        assert result.is_resting

    def test_active_to_resting(self) -> None:
        state = GuizangState(bits=0b111000)
        result = operator_cang(state)
        assert result.bits == 0

    def test_any_to_resting(self) -> None:
        for bits in range(64):
            state = GuizangState(bits=bits)
            result = operator_cang(state)
            assert result.bits == 0, f"藏({bits:06b}) should be 000000"


# ---------------------------------------------------------------------------
# 生算子 (SHENG) — flip lowest bit
# ---------------------------------------------------------------------------


class TestOperatorSheng:
    def test_from_resting(self) -> None:
        """藏态 000 → 生 → 001 (木气生)."""
        state = GuizangState.resting()
        result = operator_sheng(state)
        assert result.bits == 0b000001
        assert result.lower == 1  # 木

    def test_from_sprouted(self) -> None:
        """001 → 生 → 001 (OR 001, already has lowest bit set)."""
        state = GuizangState(bits=0b000001)
        result = operator_sheng(state)
        assert result.bits == 0b000001

    def test_preserves_upper(self) -> None:
        """Upper trigram unaffected by lowest-bit flip."""
        state = GuizangState(bits=0b111000)
        result = operator_sheng(state)
        assert result.upper_bits == 0b111
        assert result.lower_bits == (0b000 ^ 1)  # 001


# ---------------------------------------------------------------------------
# 动算子 (DONG) — shift lower + flip lowest
# ---------------------------------------------------------------------------


class TestOperatorDong:
    def test_from_resting(self) -> None:
        """动(000000) = rotate_left(000000,1) XOR 010 = 000010."""
        state = GuizangState.resting()
        result = operator_dong(state)
        assert result.bits == 0b000010
        assert result.lower_bits == 0b010  # 风

    def test_from_sprouted(self) -> None:
        """动(000001) = rotate_left(000001,1)=000010 XOR 010=000000."""
        state = GuizangState(bits=0b000001)
        result = operator_dong(state)
        assert result.bits == 0b000000
        assert result.lower_bits == 0b000  # 地

    def test_preserves_upper(self) -> None:
        """动 rotates entire 6-bit, upper may change."""
        state = GuizangState(bits=0b111001)
        result = operator_dong(state)
        # rotate_left(111001,1)=110011 XOR 010=110001
        assert result.bits == 0b110001
        assert result.upper_bits == 0b110  # 金
        assert result.lower_bits == 0b001  # 木


# ---------------------------------------------------------------------------
# 长算子 (ZHANG) — copy lower to upper
# ---------------------------------------------------------------------------


class TestOperatorZhang:
    def test_from_sprouted(self) -> None:
        """001 → 长 → 001001 (育长苗)."""
        state = GuizangState(bits=0b000001)
        result = operator_zhang(state)
        assert result.bits == 0b001001
        assert result.upper_bits == 0b001
        assert result.lower_bits == 0b001

    def test_from_structured(self) -> None:
        state = GuizangState(bits=0b111000)
        result = operator_zhang(state)
        assert result.bits == 0b000000  # lower=000 → upper=000

    def test_symmetric(self) -> None:
        """长 always makes upper == lower."""
        for bits in [0, 1, 3, 7]:
            state = GuizangState(bits=bits)
            result = operator_zhang(state)
            assert result.upper_bits == result.lower_bits


# ---------------------------------------------------------------------------
# 育算子 (YU) — decompose
# ---------------------------------------------------------------------------


class TestOperatorYu:
    def test_from_complex(self) -> None:
        """育(111000) = 111000 XOR 000100 = 111100."""
        state = GuizangState(bits=0b111000)
        result = operator_yu(state)
        assert result.upper_bits == 0b111  # 天
        assert result.lower_bits == 0b100  # 水

    def test_preserves_masked_bits(self) -> None:
        """育(111111) = 111111 XOR 000100 = 111011."""
        state = GuizangState(bits=0b111111)
        result = operator_yu(state)
        assert result.upper_bits == 0b111  # 天
        assert result.lower_bits == 0b011  # 火
        assert result.bits == 0b111011


# ---------------------------------------------------------------------------
# 止算子 (ZHI) — AND mask
# ---------------------------------------------------------------------------


class TestOperatorZhi:
    def test_default_mask(self) -> None:
        """Default mask 101101 preserves odd-positioned bits."""
        state = GuizangState(bits=0b111111)
        result = operator_zhi(state)
        assert result.bits == 0b101101

    def test_custom_mask(self) -> None:
        state = GuizangState(bits=0b111111)
        result = operator_zhi(state, mask=0b000111)
        assert result.bits == 0b000111

    def test_full_filter(self) -> None:
        state = GuizangState(bits=0b111000)
        result = operator_zhi(state, mask=0b000000)
        assert result.bits == 0b000000


# ---------------------------------------------------------------------------
# 杀算子 (SHA) — XOR / zero lower
# ---------------------------------------------------------------------------


class TestOperatorSha:
    def test_no_conflict_zero_lower(self) -> None:
        """杀(111110) lower=金(110) → 清零下卦 → 111000."""
        state = GuizangState.from_bit_str("111110")
        result = operator_sha(state)
        assert result.bits == 0b111000
        assert result.lower_bits == 0b000

    def test_clear_both_jin(self) -> None:
        """杀(110110) 上下卦皆金 → 全清零 → 000000."""
        state = GuizangState.from_bit_str("110110")
        result = operator_sha(state)
        assert result.bits == 0b000000

    def test_from_sprouted(self) -> None:
        """杀(001110) lower=金(110) → 清零下卦 → 001000."""
        state = GuizangState.from_bit_str("001110")
        result = operator_sha(state)
        assert result.bits == 0b001000


# ---------------------------------------------------------------------------
# 归算子 (GUI) — gravity-based alignment
# ---------------------------------------------------------------------------


class TestOperatorGui:
    def test_aligned_state_stays(self) -> None:
        """归() is identity — aligned state stays aligned."""
        state = GuizangState(bits=0b111111)
        result = operator_gui(state)
        assert result.bits == 0b111111

    def test_misaligned_stays_unchanged(self) -> None:
        """归() is identity — misaligned state unchanged, deviation via compute_gui_deviation()."""
        state = GuizangState(bits=0b000000)
        result = operator_gui(state)
        assert result.bits == 0b000000


# ---------------------------------------------------------------------------
# apply_operator / apply_sequence
# ---------------------------------------------------------------------------


class TestApplyOperator:
    def test_dispatches_correctly(self) -> None:
        state = GuizangState.resting()
        result = apply_operator(QiOperator.SHENG, state)
        assert result.bits == 0b000001

    def test_zhi_with_mask(self) -> None:
        state = GuizangState(bits=0b111111)
        result = apply_operator(QiOperator.ZHI, state, mask=0b000111)
        assert result.bits == 0b000111

    def test_gui_with_origin(self) -> None:
        state = GuizangState(bits=0b111111)
        origin = OriginPerception.pure_yang()
        result = apply_operator(QiOperator.GUI, state, origin=origin)
        assert result.bits == 0b111111


class TestApplySequence:
    def test_emergence_sequence(self) -> None:
        """起念: 生→动→归 (3 operators)."""
        state = GuizangState.resting()
        states = apply_sequence(QI_EMERGENCE, state)
        # 3 operators → 3 results
        assert len(states) == 3
        # First is 生 → 000001
        assert states[0].bits == 0b000001
        # Second is 动 → rotate_left(1) XOR 010 = 000010 XOR 010 = 000000
        assert states[1].bits == 0b000000
        # Third is 归 → identity, stays 000000
        assert states[2].bits == 0b000000

    def test_goal_sequence(self) -> None:
        """立目标: 长→育→杀→止."""
        state = GuizangState(bits=0b000001)  # sprouted
        states = apply_sequence(QI_GOAL, state)
        assert len(states) == 4

    def test_full_cycle_preserves_structure(self) -> None:
        """Full cycle ends at a state where upper can carry intent."""
        state = GuizangState(bits=0b000001)
        states = apply_sequence(QI_CYCLE_ORDER, state)
        assert len(states) == 8
        # After full cycle, the state should be non-random
        assert states[-1].bits >= 0
        assert states[-1].bits <= 63


# ---------------------------------------------------------------------------
# GuizangState methods
# ---------------------------------------------------------------------------


class TestGuizangState:
    def test_from_bit_str(self) -> None:
        s = GuizangState.from_bit_str("111000")
        assert s.bits == 0b111000
        assert s.upper_bits == 0b111
        assert s.lower_bits == 0b000

    def test_from_bit_str_invalid(self) -> None:
        with pytest.raises(ValueError):
            GuizangState.from_bit_str("11100")  # too short
        with pytest.raises(ValueError):
            GuizangState.from_bit_str("11100x")  # non-binary

    def test_from_trigrams(self) -> None:
        from vingobot.goal.guizang_types import EightTrigram

        s = GuizangState.from_trigrams(EightTrigram.TIAN, EightTrigram.DI)
        assert s.bits == 0b111000
        assert s.upper == EightTrigram.TIAN
        assert s.lower == EightTrigram.DI

    def test_is_resting(self) -> None:
        assert GuizangState(bits=0).is_resting
        assert not GuizangState(bits=1).is_resting
        assert not GuizangState(bits=63).is_resting

    def test_is_aligned(self) -> None:
        assert GuizangState(bits=63).is_aligned
        assert not GuizangState(bits=0).is_aligned

    def test_guizang_name(self) -> None:
        s = GuizangState.from_bit_str("111000")
        assert "归藏" in s.guizang_name or "天" in s.guizang_name or "地" in s.guizang_name

    def test_clamp_to_6_bits(self) -> None:
        s = GuizangState(bits=0xFF)  # 255, beyond 6 bits
        assert s.bits == 0b111111  # clamped to 63


# ---------------------------------------------------------------------------
# OriginPerception
# ---------------------------------------------------------------------------


class TestOriginPerception:
    def test_pure_yang(self) -> None:
        origin = OriginPerception.pure_yang()
        assert origin.vector == (1, 1, 1, 1, 1, 1)
        assert origin.to_state().bits == 0b111111

    def test_invalid_dimension(self) -> None:
        with pytest.raises(ValueError):
            OriginPerception(vector=(1, 0, 1))


# ---------------------------------------------------------------------------
# compute_gui_deviation (new: 归 as pure measurement)
# ---------------------------------------------------------------------------


class TestGuiDeviation:
    def test_full_alignment(self) -> None:
        """归(111111) → deviation=0.0."""
        state = GuizangState(bits=0b111111)
        origin = OriginPerception.pure_yang()
        assert compute_gui_deviation(state, origin) == 0.0

    def test_full_misalignment(self) -> None:
        """归(000000) → deviation=1.0."""
        state = GuizangState(bits=0b000000)
        origin = OriginPerception.pure_yang()
        assert compute_gui_deviation(state, origin) == 1.0

    def test_partial(self) -> None:
        """归(111000) → deviation=0.5."""
        state = GuizangState(bits=0b111000)
        origin = OriginPerception.pure_yang()
        assert compute_gui_deviation(state, origin) == 0.5

    def test_identity_operator(self) -> None:
        """归算子本身不修改状态."""
        state = GuizangState(bits=0b101010)
        result = operator_gui(state)
        assert result.bits == state.bits


# ---------------------------------------------------------------------------
# Hebbian learning via eight_qi.hebbian_update
# ---------------------------------------------------------------------------

from vingobot.goal.eight_qi import hebbian_update


class TestHebbianUpdate:
    def test_positive_reward_increases_weight(self) -> None:
        weights = [[0.0] * 64 for _ in range(64)]
        hebbian_update(weights, 0, 1, 0.5)
        assert weights[0][1] == pytest.approx(0.05)  # lr=0.1 * 0.5

    def test_negative_reward_decreases_weight(self) -> None:
        weights = [[0.5] * 64 for _ in range(64)]
        hebbian_update(weights, 0, 1, -0.5)
        assert weights[0][1] == pytest.approx(0.45)  # 0.5 + 0.1*(-0.5)

    def test_weight_clamped_to_one(self) -> None:
        weights = [[0.99] * 64 for _ in range(64)]
        hebbian_update(weights, 0, 1, 1.0)
        assert weights[0][1] == pytest.approx(1.0)

    def test_weight_clamped_to_zero(self) -> None:
        weights = [[0.01] * 64 for _ in range(64)]
        hebbian_update(weights, 0, 1, -1.0)
        assert weights[0][1] == pytest.approx(0.0)
