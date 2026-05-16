"""
八气算子 — Eight Qi operators for the Guizang binary consciousness state machine.

Each qi is a function ``(GuizangState, **ctx) -> GuizangState`` that
transforms the 6-bit consciousness vector.  The operators are pure
bit-manipulation at the low level, with LLM-driven semantic enhancement
at the high level (via ``compute_*`` wrappers that accept optional
LLM-callable arguments).

Reference mapping (counterclockwise):
  天(111) 金(110) 山(101) 水(100) 火(011) 风(010) 木(001) 地(000)
"""

from __future__ import annotations

from typing import Callable

from vingobot.goal.guizang_types import (
    CangSeaEntry,
    CangSeaMemory,
    EightTrigram,
    GuizangState,
    OriginPerception,
    QiOperator,
)

# ---------------------------------------------------------------------------
# 纯位运算核心
# ---------------------------------------------------------------------------


def _bit_flip_lowest(bits: int) -> int:
    """Flip the lowest bit of *bits* (0↔1)."""
    return bits ^ 1


def _rotate_left_6(bits: int) -> int:
    """Circular left shift within 6 bits.

    Example: 001010 → 010100
    """
    return ((bits << 1) | (bits >> 5)) & 0b111111


def _bit_shift_low(bits: int) -> int:
    """Circular shift of the lower 3 bits (下卦)."""
    lower = bits & 0b111
    upper = bits & 0b111000
    # Circular right shift by 1 within lower 3 bits
    shifted = ((lower >> 1) | ((lower & 1) << 2)) & 0b111
    return upper | shifted


def _copy_lower_to_upper(bits: int) -> int:
    """Copy the lower trigram to the upper trigram position."""
    lower = bits & 0b111
    return (lower << 3) | lower


def _split_into_upper_lower(bits: int) -> tuple[int, int, int]:
    """Split 6-bit state into components for 育 (decomposition)."""
    upper = (bits >> 3) & 0b111
    lower = bits & 0b111
    # Decompose by separating upper and lower into individual bit fields
    decomposed = ((upper & 0b011) << 3) | (lower & 0b011)
    return upper, lower, decomposed


def _apply_and_mask(bits: int, mask: int = 0b101101) -> int:
    """Apply AND mask (止 — set boundaries).  Default mask preserves structured bits."""
    return bits & (mask & 0b111111)


def _apply_xor_prune(bits: int, conflict_bits: int = 0) -> int:
    """Prune by XOR-ing away conflicting bits, then zeroing lower."""
    if conflict_bits == 0:
        # No specific conflict: zero the lower trigram (藏果) to reset
        return bits & 0b111000
    # XOR away the conflicting bits
    pruned = bits ^ (conflict_bits & 0b111111)
    return pruned & 0b111111


# ---------------------------------------------------------------------------
# 归引力计算
# ---------------------------------------------------------------------------


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 6-bit values.

    Example::

        hamming_distance(0b000000, 0b111111) → 6
        hamming_distance(0b111000, 0b111111) → 3
        hamming_distance(0b111111, 0b111111) → 0
    """
    return (a ^ b).bit_count()


def compute_gui_gravity(state: GuizangState, origin: OriginPerception) -> float:
    """Calculate the 归引力 (gui-gravity) pull toward the origin.

    Returns a float in [0, 1]:
      - 1.0 = state is fully aligned with origin (111111)
      - 0.0 = state is maximally distant from origin (000000)

    The gravity is ``1 - d/6`` where *d* is the Hamming distance.
    """
    origin_bits = origin.to_state().bits
    d = hamming_distance(state.bits, origin_bits)
    return 1.0 - (d / 6.0)


def compute_gui_pull(state: GuizangState, origin: OriginPerception) -> float:
    """The gravitational pull strength: how much force is pulling state toward origin.

    ``(1 - gravity) * G`` where G is the gravity constant.
    Higher values mean stronger need for realignment.
    """
    gravity = compute_gui_gravity(state, origin)
    return (1.0 - gravity) * origin.gravity_constant


# ---------------------------------------------------------------------------
# 八气算子 (纯位运算层)
# ---------------------------------------------------------------------------


def operator_cang(state: GuizangState) -> GuizangState:
    """藏算子 — 归零复位。

    将状态向量重置为全零 (000000)，模拟"藏海静息"。
    注意：真正的压缩存储由 Consciousness 层的 LLM 完成，
    这里的位运算只是状态归零。
    """
    return GuizangState.resting()


def operator_sheng(state: GuizangState) -> GuizangState:
    """生算子 (木气 001) — 最低位置1，念头萌芽。

    ``S ← S | 0b001``
    从藏态(000000)产生 000001，从已萌态保持生机。
    """
    return GuizangState(bits=state.bits | 0b001)


def operator_dong(state: GuizangState) -> GuizangState:
    """动算子 (风气 010) — 左旋后 XOR 010，念头发散联想。

    ``S ← rotate_left(S, 1) XOR 0b010``
    生→动(000001 → 000011): 左旋得 000010 XOR 010 = 000011
    """
    rotated = _rotate_left_6(state.bits)
    return GuizangState(bits=(rotated ^ 0b010) & 0b111111)


def compute_gui_deviation(state: GuizangState, origin: OriginPerception) -> float:
    """归算子 (天气 111) — 计算偏离度，不修改状态向量。

    Returns deviation ∈ [0, 1] where:
      - 0.0 = fully aligned with origin (111111)
      - 1.0 = maximally distant from origin (000000)
    """
    origin_bits = origin.to_state().bits
    d = hamming_distance(state.bits, origin_bits)
    return d / 6.0


def operator_gui(state: GuizangState) -> GuizangState:
    """归算子 (天气 111) — 身份函数，不修改状态。

    偏离度通过 ``compute_gui_deviation()`` 单独获取。
    这里保留身份函数以兼容 ``apply_sequence`` 管线。
    """
    return state


def operator_zhang(state: GuizangState) -> GuizangState:
    """长算子 — 意念放大。

    将下卦（念头）复制到上卦（意图），形成循环自指结构。
    例如 001 → 001001（育长苗），标志"意图"的诞生。
    """
    return GuizangState(bits=_copy_lower_to_upper(state.bits))


def operator_yu(state: GuizangState) -> GuizangState:
    """育算子 (水气 100) — 高位 XOR 100 引入水气，触发方案分解。

    ``S ← S XOR 0b100``
    长→育(001001 → 001101): 高位 flip bit2，形成上下不对齐的张力。
    """
    return GuizangState(bits=(state.bits ^ 0b100) & 0b111111)


def operator_zhi(state: GuizangState, mask: int = 0b101101) -> GuizangState:
    """止算子 — 设立边界。

    用 AND 掩码过滤状态，保留符合约束的位。
    默认掩码 101101 保留奇位（结构化信息），滤除偶位（发散信息）。
    """
    return GuizangState(bits=_apply_and_mask(state.bits, mask))


def operator_sha(state: GuizangState) -> GuizangState:
    """杀算子 (金气 110) — 清除上下卦中的金模式，剪除干扰。

    ``candidate & ~(110)``
    若下卦=110(金) → 清零下卦；若上卦=110(金) → 清零上卦。
    例: 100110 → 100000 (清除下卦金)。
    """
    bits = state.bits
    lower = bits & 0b111
    upper = (bits >> 3) & 0b111
    if lower == 0b110:   # 下卦为金 → 清零
        bits &= ~0b111
    if upper == 0b110:   # 上卦为金 → 清零
        bits &= ~0b111000
    return GuizangState(bits=bits & 0b111111)


# ---------------------------------------------------------------------------
# Hebbian 学习 — 藏算子辅助
# ---------------------------------------------------------------------------


def hebbian_update(
    weights: list[list[float]],
    state_from: int,
    state_to: int,
    reward: float,
    lr: float = 0.1,
) -> None:
    """Hebbian weight update for the cang-sea transition matrix.

    ``W[from][to] += lr * reward``, clamped to [0, 1].
    """
    w = weights[state_from][state_to] + lr * reward
    weights[state_from][state_to] = max(0.0, min(1.0, w))


# ---------------------------------------------------------------------------
# 算子注册表
# ---------------------------------------------------------------------------

# Type for an operator function: (state, **ctx) -> state
QiOperatorFn = Callable[..., GuizangState]

QI_OPERATORS: dict[QiOperator, QiOperatorFn] = {
    QiOperator.CANG: operator_cang,
    QiOperator.SHENG: operator_sheng,
    QiOperator.DONG: operator_dong,
    QiOperator.ZHANG: operator_zhang,
    QiOperator.YU: operator_yu,
    QiOperator.ZHI: operator_zhi,
    QiOperator.SHA: operator_sha,
    QiOperator.GUI: operator_gui,
}


def apply_operator(
    op: QiOperator,
    state: GuizangState,
    *,
    mask: int = 0b101101,
    origin: OriginPerception | None = None,
) -> GuizangState:
    """Apply a qi operator to a state vector.

    Args:
        op: Which operator to apply.
        state: Current consciousness state.
        mask: AND mask for 止 operator (default 101101).
        origin: Ignored — 归 now uses ``compute_gui_deviation()`` separately.

    Returns:
        New state after applying the operator.
    """
    fn = QI_OPERATORS[op]

    if op == QiOperator.ZHI:
        return fn(state, mask)  # type: ignore[call-arg]
    else:
        return fn(state)  # type: ignore[call-arg]


def apply_sequence(
    ops: tuple[QiOperator, ...],
    state: GuizangState,
    *,
    origin: OriginPerception | None = None,
) -> list[GuizangState]:
    """Apply a sequence of operators and return all intermediate states.

    Returns a list where ``result[i]`` is the state after applying ``ops[i]``.
    """
    results: list[GuizangState] = []
    current = state
    for op in ops:
        current = apply_operator(op, current, origin=origin)
        results.append(current)
    return results


# ---------------------------------------------------------------------------
# 算子命名与元信息
# ---------------------------------------------------------------------------

QI_NAMES_CN: dict[QiOperator, str] = {
    QiOperator.CANG: "藏",
    QiOperator.SHENG: "生",
    QiOperator.DONG: "动",
    QiOperator.ZHANG: "长",
    QiOperator.YU: "育",
    QiOperator.ZHI: "止",
    QiOperator.SHA: "杀",
    QiOperator.GUI: "归",
}

QI_DESCRIPTIONS: dict[QiOperator, str] = {
    QiOperator.CANG: "地气归藏 — Hebbian 学习更新藏海转移矩阵，状态归零",
    QiOperator.SHENG: "木气生发 (001) — S ← S|001，最低位置1，念头萌芽",
    QiOperator.DONG: "风气流动 (010) — S ← rotate_left(S,1) XOR 010，念头发散联想",
    QiOperator.ZHANG: "火气盛长 (011) — S ← (lower<<3)|lower，意图放大为自指结构",
    QiOperator.YU: "水气润育 (100) — S ← S XOR 100，高位引入水气触发方案分解",
    QiOperator.ZHI: "山气止定 (101) — S ← S AND 101101，掩码凝固设立执行边界",
    QiOperator.SHA: "金气肃杀 (110) — 清除上下卦中的金(110)模式，剪除干扰",
    QiOperator.GUI: "天气归元 (111) — 纯测量：偏离度=hamming(S,111111)/6，不修改S",
}
