"""
归藏易类型定义 — Guizang (归藏) binary consciousness state machine types.

《归藏易》以坤为首，以乾为归。八象逆时针排列：
  天(111) → 金(110) → 山(101) → 水(100)
  → 火(011) → 风(010) → 木(001) → 地(000)

核心结构：6-bit 状态向量 S = 上卦(天气/归因) + 下卦(地气/藏果)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 八象 (Eight Trigrams) — 二进制 → 象 → 气
# ---------------------------------------------------------------------------


class EightTrigram(IntEnum):
    """八象，逆时针排列，二进制值与卦象一一对应。"""

    DI = 0  # 000 地 (坤位) — 终极藏态
    MU = 1  # 001 木 (艮位) — 念头萌芽
    FENG = 2  # 010 风 (坎位) — 念头发散
    HUO = 3  # 011 火 (巽位) — 意图放大
    SHUI = 4  # 100 水 (震位) — 方案分解
    SHAN = 5  # 101 山 (离位) — 设立边界
    JIN = 6  # 110 金 (兑位) — 剪枝终结
    TIAN = 7  # 111 天 (乾位) — 元知觉归因

    @property
    def bit_str(self) -> str:
        """3-bit binary string representation."""
        return f"{self.value:03b}"

    @property
    def name_cn(self) -> str:
        """Chinese name."""
        _names = {
            0: "地", 1: "木", 2: "风", 3: "火",
            4: "水", 5: "山", 6: "金", 7: "天",
        }
        return _names[self.value]


TRIGRAM_MAP: dict[int, EightTrigram] = {t.value: t for t in EightTrigram}
"""Value → EightTrigram lookup."""

TRIGRAM_BY_NAME: dict[str, EightTrigram] = {t.name: t for t in EightTrigram}
"""Name → EightTrigram lookup."""


def trigram_from_bits(bits: int) -> EightTrigram:
    """Convert a 3-bit integer (0-7) to its EightTrigram."""
    t = TRIGRAM_MAP.get(bits & 0b111)
    if t is None:
        raise ValueError(f"Invalid trigram bits: {bits}")
    return t


# ---------------------------------------------------------------------------
# 八气 (Eight Qi Operators)
# ---------------------------------------------------------------------------


class QiOperator(str, Enum):
    """八气算子 — 对状态向量的变换函数。

    每个算子对应一个操作语义：
    - 藏: 静息态，经验内化压缩
    - 生: 翻转最低位，念头萌芽
    - 动: 低位位移/翻转，念头发散
    - 长: 复制到高位，意图放大
    - 育: 拆解复杂模式，方案分解
    - 止: AND 掩码，设立边界
    - 杀: XOR / 置零，剪枝终结
    - 归: 汉明距离，元知觉引力
    """

    CANG = "藏"  # 地 — 静息归零 + 经验压缩
    SHENG = "生"  # 木 — 从藏海生念
    DONG = "动"  # 风 — 念头发散流动
    ZHANG = "长"  # 火 — 念头放大为意图
    YU = "育"  # 水 — 意图拆解为方案
    ZHI = "止"  # 山 — 设立边界截止
    SHA = "杀"  # 金 — 剪除冲突终结
    GUI = "归"  # 天 — 元知觉引力比对

    @property
    def trigram(self) -> EightTrigram:
        """The trigram this qi belongs to."""
        _map = {
            QiOperator.CANG: EightTrigram.DI,
            QiOperator.SHENG: EightTrigram.MU,
            QiOperator.DONG: EightTrigram.FENG,
            QiOperator.ZHANG: EightTrigram.HUO,
            QiOperator.YU: EightTrigram.SHUI,
            QiOperator.ZHI: EightTrigram.SHAN,
            QiOperator.SHA: EightTrigram.JIN,
            QiOperator.GUI: EightTrigram.TIAN,
        }
        return _map[self]


# 八气顺序（逆时针）：生→动→长→育→杀→止→归→藏
QI_CYCLE_ORDER: tuple[QiOperator, ...] = (
    QiOperator.SHENG,
    QiOperator.DONG,
    QiOperator.ZHANG,
    QiOperator.YU,
    QiOperator.SHA,
    QiOperator.ZHI,
    QiOperator.GUI,
    QiOperator.CANG,
)

# 起念子序列: 生→动→归
QI_EMERGENCE: tuple[QiOperator, ...] = (
    QiOperator.SHENG,
    QiOperator.DONG,
    QiOperator.GUI,
)

# 立目标子序列: 长→育→杀→止
QI_GOAL: tuple[QiOperator, ...] = (
    QiOperator.ZHANG,
    QiOperator.YU,
    QiOperator.SHA,
    QiOperator.ZHI,
)

# 整理认知子序列: 归→杀→藏
QI_CONSOLIDATE: tuple[QiOperator, ...] = (
    QiOperator.GUI,
    QiOperator.SHA,
    QiOperator.CANG,
)


# ---------------------------------------------------------------------------
# 6-bit 状态向量 S
# ---------------------------------------------------------------------------


@dataclass
class TrigramParts:
    """Parsed upper/lower trigrams from a 6-bit state."""

    upper: EightTrigram
    """上卦 — 天气 / 归因 (bits 5-3)"""
    lower: EightTrigram
    """下卦 — 地气 / 藏果 (bits 2-0)"""
    upper_bits: int = 0
    lower_bits: int = 0


@dataclass
class GuizangState:
    """6-bit binary consciousness state vector.

    布局::

        S = [b₅ b₄ b₃ | b₂ b₁ b₀]
             └── 上卦 ──┘ └── 下卦 ──┘

    上卦为天气（归因），下卦为地气（藏果）。
    """

    bits: int = 0
    """Raw 6-bit integer, range [0, 63]."""

    def __post_init__(self) -> None:
        self.bits = self.bits & 0b111111

    # ── Properties ──────────────────────────────────────────────

    @property
    def bit_str(self) -> str:
        """6-bit binary string."""
        return f"{self.bits:06b}"

    @property
    def upper_bits(self) -> int:
        """Upper trigram bits (5-3)."""
        return (self.bits >> 3) & 0b111

    @property
    def lower_bits(self) -> int:
        """Lower trigram bits (2-0)."""
        return self.bits & 0b111

    @property
    def upper(self) -> EightTrigram:
        """Upper trigram (天气/归因)."""
        return trigram_from_bits(self.upper_bits)

    @property
    def lower(self) -> EightTrigram:
        """Lower trigram (地气/藏果)."""
        return trigram_from_bits(self.lower_bits)

    @property
    def trigrams(self) -> TrigramParts:
        """Parsed trigram parts."""
        return TrigramParts(
            upper=self.upper,
            lower=self.lower,
            upper_bits=self.upper_bits,
            lower_bits=self.lower_bits,
        )

    @property
    def is_resting(self) -> bool:
        """True when state is pure 藏 (000000) — system at rest."""
        return self.bits == 0

    @property
    def is_aligned(self) -> bool:
        """True when state is pure 归 (111111) — fully aligned with origin."""
        return self.bits == 0b111111

    # ── Factory methods ─────────────────────────────────────────

    @classmethod
    def from_trigrams(cls, upper: EightTrigram, lower: EightTrigram) -> GuizangState:
        """Construct from separate upper/lower trigrams."""
        return cls(bits=(upper.value << 3) | lower.value)

    @classmethod
    def from_bit_str(cls, s: str) -> GuizangState:
        """Construct from a 6-character binary string, e.g. '111000'."""
        s = s.strip()
        if len(s) != 6 or not all(c in "01" for c in s):
            raise ValueError(f"Invalid 6-bit string: {s!r}")
        return cls(bits=int(s, 2))

    @classmethod
    def resting(cls) -> GuizangState:
        """The pure resting state: 000000 (地/地)."""
        return cls(bits=0)

    # ── Hexagram names (六十四卦归藏命名) ───────────────────────
    # See user material for full mapping.  Key examples listed.

    @property
    def guizang_name(self) -> str:
        """归藏易卦名 (上卦天气 + 下卦地气)."""
        _names: dict[int, str] = {
            0b000000: "藏藏始基",  # 地地 — 万物之基
            0b111000: "归藏定位",  # 天地 — 天地交泰
            0b000111: "藏归交感",  # 地天 — 藏与归交互
            0b001000: "木藏生萌",  # 木地 — 藏中生有
            0b000001: "藏木萌芽",  # 地木 — 萌而未发
            0b111100: "杀藏墓",  # 金地 — 生机转为终结
            0b001001: "育长苗",  # 木木 — 循环自指
            0b011011: "火火盛长",  # 火火 — 盛长之极
            0b100100: "金气杀",  # 水水 — 诱惑或偏离（兑卦）
        }
        return _names.get(self.bits, f"{self.upper.name_cn}{self.lower.name_cn}")

    def __repr__(self) -> str:
        return (
            f"GuizangState(bits={self.bits}, hex={self.bit_str}, "
            f"upper={self.upper.name_cn}, lower={self.lower.name_cn})"
        )


# ---------------------------------------------------------------------------
# 元知觉向量 U (Origin Perception)
# ---------------------------------------------------------------------------


@dataclass
class OriginPerception:
    """元知觉向量 — 代表用户的恒常意图方向。

    纯白向量 U = [1,1,1,1,1,1]，天之气极。
    """

    vector: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    """6-dimensional binary vector (default: pure yang)."""

    gravity_constant: float = 1.0
    """Gravitational constant G.  Higher = stronger pull toward alignment."""

    def __post_init__(self) -> None:
        if len(self.vector) != 6:
            raise ValueError(f"Origin vector must be 6-dimensional, got {len(self.vector)}")

    @classmethod
    def pure_yang(cls) -> OriginPerception:
        """The default pure-yang origin: 111111."""
        return cls(vector=(1, 1, 1, 1, 1, 1), gravity_constant=1.0)

    def to_state(self) -> GuizangState:
        """Convert to a GuizangState for comparison."""
        bits = sum(b << (5 - i) for i, b in enumerate(self.vector))
        return GuizangState(bits=bits)


# ---------------------------------------------------------------------------
# 藏海 (Cang Sea) — memory matrix
# ---------------------------------------------------------------------------


@dataclass
class CangSeaEntry:
    """A single entry in the cang-sea memory matrix.

    Records a state transition with its reward signal and compressed
    representation (LLM summary or embedding).
    """

    state_from: GuizangState
    """State before the operator was applied."""
    operator: QiOperator
    """The qi operator that was applied."""
    state_to: GuizangState
    """State after the operator was applied."""
    reward: float = 0.0
    """Reward signal ∈ [-1, 1].  Positive = useful transition."""
    summary: str = ""
    """Compressed natural-language summary of what was learned."""
    timestamp: str = ""
    """ISO-8601 timestamp."""

    @property
    def is_positive(self) -> bool:
        return self.reward > 0.0

    @property
    def is_negative(self) -> bool:
        return self.reward < 0.0


@dataclass
class CangSeaMemory:
    """藏海记忆矩阵 — 记录状态变迁序列的关联存储。

    Each entry records ``(S_t, operator, S_{t+1}, reward, summary)``.
    """

    entries: list[CangSeaEntry] = field(default_factory=list)
    max_entries: int = 1000

    # ── Hebbian state-transition matrix (64×64) ──────────
    transition_weights: list[list[float]] = field(
        default_factory=lambda: [[0.0] * 64 for _ in range(64)]
    )
    hebbian_lr: float = 0.1

    def add(self, entry: CangSeaEntry) -> None:
        """Add an entry, evicting low-reward ones if over capacity.

        Eviction strategy:
        - High-reward entries (|reward| > 0.3) are preserved.
        - Low-reward entries are evicted in FIFO order.
        - Ensures entries count never exceeds max_entries.
        """
        self.entries.append(entry)
        if len(self.entries) <= self.max_entries:
            return
        # ── Evict: preserve high-reward entries ─────────
        excess = len(self.entries) - self.max_entries
        # Partition: low-reward first (FIFO eviction candidates), high-reward last
        low_reward = [e for e in self.entries if abs(e.reward) <= 0.3]
        high_reward = [e for e in self.entries if abs(e.reward) > 0.3]
        # Evict from low-reward pool (oldest first)
        to_remove = min(excess, len(low_reward))
        if to_remove > 0:
            low_reward = low_reward[to_remove:]
            excess -= to_remove
        # If still over capacity, evict from high-reward pool too
        if excess > 0:
            high_reward = high_reward[excess:]
        # Reconstruct in original insertion order
        self.entries = low_reward + high_reward
        # Sort back to insertion order preserved by timestamp
        self.entries.sort(key=lambda e: e.timestamp or "")

    def recent(self, n: int = 10) -> list[CangSeaEntry]:
        """Return the most recent n entries."""
        return self.entries[-n:]

    def positive_entries(self) -> list[CangSeaEntry]:
        """Entries with positive reward."""
        return [e for e in self.entries if e.is_positive]

    def negative_entries(self) -> list[CangSeaEntry]:
        """Entries with negative reward."""
        return [e for e in self.entries if e.is_negative]

    def by_operator(self, op: QiOperator) -> list[CangSeaEntry]:
        """Filter entries by the qi operator used."""
        return [e for e in self.entries if e.operator == op]

    # ── Hebbian learning ─────────────────────────────

    def hebbian_record(
        self, state_from: int, state_to: int, reward: float
    ) -> None:
        """Update the transition weight matrix via Hebbian learning.

        ``W[from][to] += lr * reward``
        """
        w = self.transition_weights[state_from][state_to] + self.hebbian_lr * reward
        self.transition_weights[state_from][state_to] = max(0.0, min(1.0, w))

    def hebbian_sample(self, state_from: int) -> int | None:
        """Sample a next state based on learned transition weights.

        Returns None if no experience has been recorded from this state.
        """
        row = self.transition_weights[state_from]
        total = sum(row)
        if total <= 0:
            return None
        r = random.uniform(0, total)
        acc = 0.0
        for i, w in enumerate(row):
            acc += w
            if r <= acc:
                return i
        return max(range(64), key=lambda i: row[i])

    # ── Persistence ───────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the full cang-sea memory to a JSON-safe dict."""
        entry_dicts: list[dict] = []
        for e in self.entries[-200:]:  # Keep last 200 entries max
            entry_dicts.append({
                "state_from": e.state_from.bits,
                "operator": e.operator.value,
                "state_to": e.state_to.bits,
                "reward": e.reward,
                "summary": e.summary[:200],
                "timestamp": e.timestamp,
            })
        return {
            "version": 1,
            "entries": entry_dicts,
            "transition_weights": self.transition_weights,
        }

    @classmethod
    def from_dict(cls, data: dict, max_entries: int = 1000) -> "CangSeaMemory":
        """Restore cang-sea memory from a serialized dict."""
        mem = cls(max_entries=max_entries)
        for ed in data.get("entries", []):
            try:
                mem.entries.append(CangSeaEntry(
                    state_from=GuizangState(bits=ed["state_from"]),
                    operator=QiOperator(ed["operator"]),
                    state_to=GuizangState(bits=ed["state_to"]),
                    reward=float(ed.get("reward", 0.0)),
                    summary=ed.get("summary", ""),
                    timestamp=ed.get("timestamp", ""),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        # Restore transition weights
        weights = data.get("transition_weights")
        if weights and isinstance(weights, list) and len(weights) == 64:
            for i, row in enumerate(weights):
                if isinstance(row, list) and len(row) == 64:
                    mem.transition_weights[i] = [float(v) for v in row]
        return mem

    def save(self, path: "Path") -> None:
        """Persist cang-sea memory to a JSON file."""
        import json as _json
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        path.write_text(
            _json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: "Path", max_entries: int = 1000) -> "CangSeaMemory | None":
        """Load cang-sea memory from a JSON file.

        Returns None if the file does not exist or is corrupted.
        """
        import json as _json
        if not path.is_file():
            return None
        try:
            data = _json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data, max_entries=max_entries)
        except Exception:
            from loguru import logger
            logger.warning("[藏海] 加载持久化文件失败: {}", path)
            return None

    # ── Properties ───────────────────────────────────
    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def is_empty(self) -> bool:
        return len(self.entries) == 0


# ---------------------------------------------------------------------------
# 意识阶段 (Consciousness Phase)
# ---------------------------------------------------------------------------


class ConsciousnessPhase(str, Enum):
    """归藏意识周天的三个阶段。

    - 起念 (QINIAN): 藏中生有 — 从静息态生成念头，与元知觉比对
    - 立目标 (LIMUBIAO): 念头的结构化 — 放大意图、拆解方案、设立边界、剪除冲突
    - 整理认知 (ZHENGLI): 万物归藏 — 将经验压缩存入藏海，回到静息态
    """

    QINIAN = "起念"  # Emergence: 生→动→归
    LIMUBIAO = "立目标"  # Goal-setting: 长→育→杀→止
    ZHENGLI = "整理认知"  # Consolidation: 归→杀→藏


# 意识周天完整子序列映射
PHASE_OPERATORS: dict[ConsciousnessPhase, tuple[QiOperator, ...]] = {
    ConsciousnessPhase.QINIAN: QI_EMERGENCE,
    ConsciousnessPhase.LIMUBIAO: QI_GOAL,
    ConsciousnessPhase.ZHENGLI: QI_CONSOLIDATE,
}

# 常规周天顺序（起念 → 立目标 → 整理认知）
PHASE_ORDER: tuple[ConsciousnessPhase, ...] = (
    ConsciousnessPhase.QINIAN,
    ConsciousnessPhase.LIMUBIAO,
    ConsciousnessPhase.ZHENGLI,
)


# ---------------------------------------------------------------------------
# 意识周天产出 (Consciousness Result)
# ---------------------------------------------------------------------------


@dataclass
class ConsciousnessResult:
    """Output of a single DMN consciousness quantum (one phase step).

    The consumer loop reads this after each ``cycle()`` call to decide
    what actions to dispatch to TPN.
    """

    phase: ConsciousnessPhase
    """Which phase just completed."""

    state_before: GuizangState
    """State vector before this phase."""

    state_after: GuizangState
    """State vector after this phase."""

    gui_gravity: float | None = None
    """归引力 in [0, 1], if computed during this phase."""

    # ── 语义产出 (LLM 路径填充，位运算路径留空) ─────
    thought_text: str = ""
    """生算子产出：当前应关注的念头."""

    divergent_thoughts: list[str] = field(default_factory=list)
    """动算子产出：发散的关联方向."""

    deviation_level: float = 0.0
    """归算子产出：真实偏离度 0-1 (LLM路径替代位运算归引力)."""

    deviation_reason: str = ""
    """归算子产出：偏离原因描述."""

    intent_description: str = ""
    """长算子产出：意图描述."""

    subtasks: list[dict[str, Any]] = field(default_factory=list)
    """育算子产出：分解的子任务."""

    boundary_issues: list[dict[str, Any]] = field(default_factory=list)
    """止算子产出：边界问题."""

    pruned_items: list[dict[str, Any]] = field(default_factory=list)
    """杀算子产出：被剪枝的条目."""

    compressed_insight: str = ""
    """藏算子产出：压缩的认知洞察."""

    positive_patterns: list[str] = field(default_factory=list)
    """正面模式."""

    negative_patterns: list[str] = field(default_factory=list)
    """负面模式."""

    needs_goal_review: bool = False
    """DMN detected goals that need attention (stagnation, conflict, drift)."""

    needs_blueprint_review: bool = False
    """DMN recommends blueprint re-evaluation."""

    evolution_actions: list[dict[str, Any]] = field(default_factory=list)
    """Pending cognitive evolution actions to enqueue."""

    summary: str = ""
    """Human-readable summary of what this phase produced."""

    cang_sea_updates: int = 0
    """Number of new entries written to the cang-sea memory."""

    @property
    def is_resting(self) -> bool:
        """True when the consciousness is fully at rest (no action needed)."""
        return (
            not self.needs_goal_review
            and not self.needs_blueprint_review
            and not self.evolution_actions
            and not self.subtasks
            and not self.compressed_insight
        )


# ---------------------------------------------------------------------------
# 默认阈值
# ---------------------------------------------------------------------------

# 归引力低于此阈值时，从起念阶段进入立目标阶段
GUI_GRAVITY_THRESHOLD: float = 0.5
"""When gui-gravity falls below this, the DMN takes action."""

# 默认意识周天休眠间隔（秒）
DEFAULT_CYCLE_INTERVAL: float = 300.0  # 5 minutes
