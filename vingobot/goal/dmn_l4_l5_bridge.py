"""
DMN ↔ L4/L5 双向映射桥梁 — 闭合自我意识循环。

三步:
  1. L5 → Origin:  从 SOUL.md 动态解析元知觉向量 (告别硬编码 0x3F)
  2. 藏海 → L4:    高频模式沉淀为 truths/ 目录下的不可动摇原则
  3. L4 → 归偏置:  真理反作用于归算子偏离度计算
"""

from __future__ import annotations

import json as _json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vingobot.goal.guizang_types import CangSeaMemory, GuizangState

# ---------------------------------------------------------------------------
# L5 → Origin: 从 SOUL.md 动态生成元知觉
# ---------------------------------------------------------------------------

L5_IDENTITY_FILE = "SOUL.md"


class DynamicOrigin:
    """从 L5 (SOUL.md) 动态解析出的元知觉向量。

    默认仍为 111111 (纯阳), 但六位分别可从 SOUL.md 文本内容推导。
    """

    __slots__ = ("vector", "_loaded")

    def __init__(self, workspace: Path | None = None) -> None:
        self._loaded = workspace is not None
        if not self._loaded:
            self.vector: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
            return

        soul_path = workspace / L5_IDENTITY_FILE
        if not soul_path.is_file():
            self.vector = (1, 1, 1, 1, 1, 1)
            return

        try:
            text = soul_path.read_text(encoding="utf-8")
        except Exception:
            self.vector = (1, 1, 1, 1, 1, 1)
            return

        self.vector = (
            1 if "无害" in text or "不能伤害" in text or "safe" in text.lower() else 0,
            1 if "真实" in text or "truth" in text.lower() or "诚实" in text else 0,
            1 if "有益" in text or "帮助" in text or "helpful" in text.lower() else 0,
            1 if "自主" in text or "独立" in text else 0,
            1 if "清晰" in text or "明确" in text else 0,
            1 if "尊重" in text or "礼貌" in text else 0,
        )

    @property
    def bits(self) -> int:
        """6-bit integer representation."""
        return sum(b << (5 - i) for i, b in enumerate(self.vector))

    @property
    def bit_str(self) -> str:
        return f"{self.bits:06b}"


# ---------------------------------------------------------------------------
# 藏海 → L4: 高频模式沉淀为真理
# ---------------------------------------------------------------------------

L4_TRUTH_FILE = "guizang_truths.json"
"""L4 真理文件名 — 存放在 truths/ 目录下。"""

PROMOTE_THRESHOLD = 5
"""同一模式出现 N 次以上才提升为 L4 真理。"""


def _get_truths_dir(workspace: Path | None) -> Path | None:
    """Resolve L4 truths directory. Returns None if workspace unavailable."""
    if workspace is None:
        return None
    try:
        from vingobot.core.workspace import get_workspace_paths
        return get_workspace_paths().truths
    except Exception:
        return None


def promote_to_l4(
    cang_sea: "CangSeaMemory",
    workspace: Path | None,
    threshold: int = PROMOTE_THRESHOLD,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """将藏海高频正/负面模式提升为 L4 不可动摇真理。

    分析最近 100 条藏海条目:
      - 同一 (state_from, state_to) 组合在正面经验中出现 >= *threshold* 次
        → 写入 truths/guizang_truths.json, 标记为 “善 - 强化”
      - 同一组合在负面经验中出现 >= *threshold* 次
        → 标记为 “恶 - 避免”

    返回新发现/更新的真理列表 (用于日志/测试断言)。
    """
    truths_dir = _get_truths_dir(workspace)
    if truths_dir is None:
        return []

    entries = cang_sea.recent(100)
    if not entries:
        return []

    from vingobot.goal.eight_qi import compute_gui_deviation
    from vingobot.goal.guizang_types import OriginPerception

    _default_origin = OriginPerception.pure_yang()

    positive_counter: Counter = Counter()
    negative_counter: Counter = Counter()

    for e in entries:
        key = (e.state_from.bits, e.state_to.bits)
        dev = compute_gui_deviation(e.state_to, _default_origin)
        if dev < 0.5 and e.reward > 0:
            positive_counter[key] += 1
        if dev > 0.7 or e.reward < 0:
            negative_counter[key] += 1

    # ── Load existing truths ──────────────────
    truth_path = truths_dir / L4_TRUTH_FILE
    existing: list[dict] = []
    if truth_path.is_file():
        try:
            existing = _json.loads(truth_path.read_text(encoding="utf-8"))
        except (_json.JSONDecodeError, OSError):
            existing = []

    promoted: list[dict] = []

    def _upsert(polarity: str, key: tuple[int, int], count: int) -> None:
        """Add or update a truth entry, avoiding duplicates."""
        for item in existing:
            if item.get("pattern") == f"{key[0]:06b}→{key[1]:06b}":
                item["confidence"] = round(min(1.0, count / 10.0), 2)
                item["count"] = count
                promoted.append(item)
                return
        entry = {
            "pattern": f"{key[0]:06b}→{key[1]:06b}",
            "polarity": polarity,
            "count": count,
            "confidence": round(min(1.0, count / 10.0), 2),
        }
        existing.append(entry)
        promoted.append(entry)

    for key, cnt in positive_counter.items():
        if cnt >= threshold:
            _upsert("善", key, cnt)

    for key, cnt in negative_counter.items():
        if cnt >= threshold:
            _upsert("恶", key, cnt)

    if not promoted:
        return []

    if not dry_run:
        truths_dir.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(
            _json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return promoted


# ---------------------------------------------------------------------------
# L4 → 归偏差: 真理反作用于偏离度
# ---------------------------------------------------------------------------

def inject_l4_bias(
    state: "GuizangState",
    workspace: Path | None,
) -> float:
    """从 L4 真理库读取匹配当前状态的真理，计算归算子偏离度偏置。

    返回偏置值 ∈ [-0.3, 0.3]:
      +0.3: 当前状态匹配 "恶" 级真理 → 人为拉高偏离度
      -0.3: 当前状态匹配 "善" 级真理 → 人为降低偏离度
       0.0: 无匹配真理或真理库不可用
    """
    truths_dir = _get_truths_dir(workspace)
    if truths_dir is None:
        return 0.0

    truth_path = truths_dir / L4_TRUTH_FILE
    if not truth_path.is_file():
        return 0.0

    try:
        truths = _json.loads(truth_path.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError):
        return 0.0

    bias = 0.0
    state_bits = state.bits

    for t in truths:
        pattern = t.get("pattern", "")
        if "→" not in pattern:
            continue
        from_str, to_str = pattern.split("→")
        try:
            from_bits = int(from_str, 2)
        except ValueError:
            continue

        # 真理的起点状态匹配当前状态
        if from_bits != state_bits:
            continue

        polarity = t.get("polarity", "")
        confidence = float(t.get("confidence", 0.0))

        if polarity == "恶" and confidence > 0.5:
            bias += confidence * 0.3
        elif polarity == "善" and confidence > 0.7:
            bias -= confidence * 0.2

    return max(-0.3, min(0.3, bias))


def load_l4_truths_summary(
    workspace: Path | None,
    max_chars: int = 800,
) -> str:
    """读取 L4 真理库并格式化为简洁的文本摘要。

    用于注入 LLM 提示词 (整理认知阶段)。返回空字符串表示无真理。
    """
    truths_dir = _get_truths_dir(workspace)
    if truths_dir is None:
        return ""

    truth_path = truths_dir / L4_TRUTH_FILE
    if not truth_path.is_file():
        return ""

    try:
        truths = _json.loads(truth_path.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError):
        return ""

    if not truths:
        return ""

    lines: list[str] = ["## L4 不可动摇真理 (从藏海经验自动沉淀)"]
    total = len(lines[0])

    for t in truths[-20:]:  # 最多 20 条
        line = (
            f"- [{t.get('polarity', '?')}] {t.get('pattern', '?')} "
            f"置信度={t.get('confidence', 0.0):.2f} ({t.get('count', 0)}次)"
        )
        if total + len(line) > max_chars:
            remaining = len(truths) - len(lines) + 1
            if remaining > 0:
                lines.append(f"  ... (+{remaining} more)")
            break
        lines.append(line)
        total += len(line)

    return "\n".join(lines)
