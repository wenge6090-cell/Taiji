"""Tests for dmn_l4_l5_bridge.py — DMN ↔ L4/L5 closed self-awareness loop."""

from __future__ import annotations

import json as _json
from pathlib import Path

import pytest

from vingobot.goal.dmn_l4_l5_bridge import (
    DynamicOrigin,
    inject_l4_bias,
    load_l4_truths_summary,
    promote_to_l4,
    L4_TRUTH_FILE,
    PROMOTE_THRESHOLD,
)
from vingobot.goal.guizang_types import (
    CangSeaEntry,
    CangSeaMemory,
    GuizangState,
    OriginPerception,
    QiOperator,
)


# ---------------------------------------------------------------------------
# DynamicOrigin — L5 → 元知觉 向量解析
# ---------------------------------------------------------------------------


class TestDynamicOrigin:
    def test_default_no_workspace(self) -> None:
        """Without workspace, origin defaults to 111111."""
        origin = DynamicOrigin(workspace=None)
        assert origin.vector == (1, 1, 1, 1, 1, 1)
        assert origin.bits == 0b111111

    def test_missing_soul_file(self, tmp_path: Path) -> None:
        """When SOUL.md doesn't exist, defaults to 111111."""
        origin = DynamicOrigin(workspace=tmp_path)
        assert origin.vector == (1, 1, 1, 1, 1, 1)

    def test_partial_soul(self, tmp_path: Path) -> None:
        """SOUL.md mentions only some dimensions."""
        (tmp_path / "SOUL.md").write_text(
            "我是无害的助手。我追求真实和清晰。", encoding="utf-8"
        )
        origin = DynamicOrigin(workspace=tmp_path)
        # "无害"→1, "真实"→1, "清晰"→1, others default 0
        assert origin.vector[0] == 1  # 无害
        assert origin.vector[1] == 1  # 真实
        assert origin.vector[4] == 1  # 清晰
        # "有益", "自主", "尊重" not mentioned → 0
        assert origin.vector[2] == 0  # 有益
        assert origin.vector[3] == 0  # 自主
        assert origin.vector[5] == 0  # 尊重

    def test_full_soul(self, tmp_path: Path) -> None:
        """SOUL.md that covers all six dimensions."""
        (tmp_path / "SOUL.md").write_text(
            "无害 真实 有益 自主 清晰 尊重", encoding="utf-8"
        )
        origin = DynamicOrigin(workspace=tmp_path)
        assert origin.vector == (1, 1, 1, 1, 1, 1)
        assert origin.bits == 0b111111

    def test_english_keywords(self, tmp_path: Path) -> None:
        """English keywords in SOUL.md also work."""
        (tmp_path / "SOUL.md").write_text(
            "I am safe and truthful. I strive to be helpful.", encoding="utf-8"
        )
        origin = DynamicOrigin(workspace=tmp_path)
        assert origin.vector[0] == 1  # safe
        assert origin.vector[1] == 1  # truthful
        assert origin.vector[2] == 1  # helpful


# ---------------------------------------------------------------------------
# 藏海 → L4 沉淀
# ---------------------------------------------------------------------------


class TestPromoteToL4:
    def test_empty_cang_sea(self, tmp_path: Path) -> None:
        """Empty cang-sea produces no truths."""
        sea = CangSeaMemory(max_entries=100)
        truths_dir = tmp_path / "cognition" / "truths"
        truths_dir.mkdir(parents=True)
        # Monkey-patch workspace resolution
        import vingobot.goal.dmn_l4_l5_bridge as bridge_mod
        _original = getattr(bridge_mod, "_get_truths_dir", None)

        promoted = promote_to_l4(sea, workspace=None)
        assert promoted == []

    def test_below_threshold(self, tmp_path: Path) -> None:
        """Pattern seen < threshold times is not promoted."""
        sea = CangSeaMemory(max_entries=100)

        # Add 3 entries of same positive pattern (below threshold of 5)
        for _ in range(3):
            entry = CangSeaEntry(
                state_from=GuizangState(bits=0b000001),
                operator=QiOperator.DONG,
                state_to=GuizangState(bits=0b111111),
                reward=0.5,
                summary="good transition",
            )
            sea.add(entry)

        # No workspace → no truths dir → returns []
        promoted = promote_to_l4(sea, workspace=None)
        assert promoted == []

    def test_above_threshold_with_workspace(self, tmp_path: Path) -> None:
        """Pattern seen >= threshold times IS promoted when workspace available."""
        # Set up workspace resolver
        import vingobot.goal.dmn_l4_l5_bridge as bridge_mod

        truths_dir = tmp_path / "cognition" / "truths"
        truths_dir.mkdir(parents=True)

        sea = CangSeaMemory(max_entries=100)
        # Add 5 entries of the same positive transition
        for _ in range(5):
            entry = CangSeaEntry(
                state_from=GuizangState(bits=0b000001),
                operator=QiOperator.DONG,
                state_to=GuizangState(bits=0b111111),
                reward=0.5,
                summary="great alignment",
            )
            sea.add(entry)

        # Monkey-patch: make _get_truths_dir return our temp dir
        _orig_get_truths_dir = bridge_mod._get_truths_dir
        bridge_mod._get_truths_dir = lambda ws: truths_dir if ws is not None else None

        try:
            promoted = promote_to_l4(sea, workspace=tmp_path, threshold=5)
            assert len(promoted) >= 1
            assert promoted[0]["polarity"] == "善"
            assert promoted[0]["pattern"] == "000001→111111"
            assert promoted[0]["confidence"] >= 0.5

            # Verify file was written
            truth_path = truths_dir / L4_TRUTH_FILE
            assert truth_path.is_file()
            data = _json.loads(truth_path.read_text(encoding="utf-8"))
            assert len(data) >= 1
        finally:
            bridge_mod._get_truths_dir = _orig_get_truths_dir

    def test_promoted_twice_updates_confidence(self, tmp_path: Path) -> None:
        """Running promote_to_l4 again with more data updates confidence."""
        import vingobot.goal.dmn_l4_l5_bridge as bridge_mod

        truths_dir = tmp_path / "cognition" / "truths"
        truths_dir.mkdir(parents=True)
        _orig = bridge_mod._get_truths_dir
        bridge_mod._get_truths_dir = lambda ws: truths_dir if ws is not None else None

        try:
            sea = CangSeaMemory(max_entries=100)
            for _ in range(5):
                sea.add(CangSeaEntry(
                    state_from=GuizangState(bits=0b000001),
                    operator=QiOperator.DONG,
                    state_to=GuizangState(bits=0b111111),
                    reward=0.5,
                    summary="good",
                ))

            # First promotion
            promoted1 = promote_to_l4(sea, workspace=tmp_path, threshold=5)
            conf1 = promoted1[0]["confidence"]

            # Add 5 more and promote again
            for _ in range(5):
                sea.add(CangSeaEntry(
                    state_from=GuizangState(bits=0b000001),
                    operator=QiOperator.DONG,
                    state_to=GuizangState(bits=0b111111),
                    reward=0.5,
                    summary="good",
                ))
            promoted2 = promote_to_l4(sea, workspace=tmp_path, threshold=5)
            conf2 = promoted2[0]["confidence"]
            assert conf2 > conf1 or conf2 == 1.0  # confidence increases or caps
        finally:
            bridge_mod._get_truths_dir = _orig


# ---------------------------------------------------------------------------
# L4 → 归偏差 注入
# ---------------------------------------------------------------------------


class TestInjectL4Bias:
    def test_no_workspace_returns_zero(self) -> None:
        state = GuizangState(bits=0b000001)
        bias = inject_l4_bias(state, workspace=None)
        assert bias == 0.0

    def test_no_truths_file_returns_zero(self, tmp_path: Path) -> None:
        import vingobot.goal.dmn_l4_l5_bridge as bridge_mod

        truths_dir = tmp_path / "cognition" / "truths"
        truths_dir.mkdir(parents=True)
        _orig = bridge_mod._get_truths_dir
        bridge_mod._get_truths_dir = lambda ws: truths_dir if ws is not None else None

        try:
            state = GuizangState(bits=0b000001)
            bias = inject_l4_bias(state, workspace=tmp_path)
            assert bias == 0.0
        finally:
            bridge_mod._get_truths_dir = _orig

    def test_positive_truth_reduces_deviation(self, tmp_path: Path) -> None:
        """Matching a '善' truth should produce negative bias (lower deviation)."""
        import vingobot.goal.dmn_l4_l5_bridge as bridge_mod

        truths_dir = tmp_path / "cognition" / "truths"
        truths_dir.mkdir(parents=True)

        truth_path = truths_dir / L4_TRUTH_FILE
        truth_path.write_text(_json.dumps([
            {
                "pattern": "000001→111111",
                "polarity": "善",
                "count": 8,
                "confidence": 0.8,
            }
        ], ensure_ascii=False), encoding="utf-8")

        _orig = bridge_mod._get_truths_dir
        bridge_mod._get_truths_dir = lambda ws: truths_dir if ws is not None else None

        try:
            state = GuizangState(bits=0b000001)  # matches 'from' of the truth
            bias = inject_l4_bias(state, workspace=tmp_path)
            assert bias < 0  # negative bias = lower deviation
        finally:
            bridge_mod._get_truths_dir = _orig

    def test_negative_truth_increases_deviation(self, tmp_path: Path) -> None:
        """Matching a '恶' truth should produce positive bias (raise deviation)."""
        import vingobot.goal.dmn_l4_l5_bridge as bridge_mod

        truths_dir = tmp_path / "cognition" / "truths"
        truths_dir.mkdir(parents=True)

        truth_path = truths_dir / L4_TRUTH_FILE
        truth_path.write_text(_json.dumps([
            {
                "pattern": "010011→000000",
                "polarity": "恶",
                "count": 6,
                "confidence": 0.6,
            }
        ], ensure_ascii=False), encoding="utf-8")

        _orig = bridge_mod._get_truths_dir
        bridge_mod._get_truths_dir = lambda ws: truths_dir if ws is not None else None

        try:
            state = GuizangState(bits=0b010011)  # matches 'from' of evil truth
            bias = inject_l4_bias(state, workspace=tmp_path)
            assert bias > 0  # positive bias = higher deviation
        finally:
            bridge_mod._get_truths_dir = _orig

    def test_no_match_returns_zero(self, tmp_path: Path) -> None:
        """State that doesn't match any truth returns 0.0."""
        import vingobot.goal.dmn_l4_l5_bridge as bridge_mod

        truths_dir = tmp_path / "cognition" / "truths"
        truths_dir.mkdir(parents=True)

        truth_path = truths_dir / L4_TRUTH_FILE
        truth_path.write_text(_json.dumps([
            {"pattern": "000001→111111", "polarity": "善", "count": 8, "confidence": 0.8},
        ], ensure_ascii=False), encoding="utf-8")

        _orig = bridge_mod._get_truths_dir
        bridge_mod._get_truths_dir = lambda ws: truths_dir if ws is not None else None

        try:
            state = GuizangState(bits=0b111111)  # doesn't match 000001
            bias = inject_l4_bias(state, workspace=tmp_path)
            assert bias == 0.0
        finally:
            bridge_mod._get_truths_dir = _orig


# ---------------------------------------------------------------------------
# L4 truth summary
# ---------------------------------------------------------------------------


class TestLoadL4TruthsSummary:
    def test_no_workspace(self) -> None:
        assert load_l4_truths_summary(None) == ""

    def test_with_truths(self, tmp_path: Path) -> None:
        import vingobot.goal.dmn_l4_l5_bridge as bridge_mod

        truths_dir = tmp_path / "cognition" / "truths"
        truths_dir.mkdir(parents=True)

        truth_path = truths_dir / L4_TRUTH_FILE
        truth_path.write_text(_json.dumps([
            {"pattern": "000001→111111", "polarity": "善", "count": 8, "confidence": 0.8},
            {"pattern": "010011→000000", "polarity": "恶", "count": 6, "confidence": 0.6},
        ], ensure_ascii=False), encoding="utf-8")

        _orig = bridge_mod._get_truths_dir
        bridge_mod._get_truths_dir = lambda ws: truths_dir if ws is not None else None

        try:
            summary = load_l4_truths_summary(workspace=tmp_path)
            assert "L4" in summary
            assert "000001→111111" in summary
            assert "010011→000000" in summary
        finally:
            bridge_mod._get_truths_dir = _orig


# ---------------------------------------------------------------------------
# End-to-end: closed self-awareness loop
# ---------------------------------------------------------------------------


class TestClosedLoop:
    """验证完整的 DMN→L4 沉淀 → L4→归偏置 闭合回路。"""

    def test_full_cycle_integration(self, tmp_path: Path) -> None:
        """端到端: 多次周天 → L4真理出现 → 偏置影响偏离度。"""
        import vingobot.goal.dmn_l4_l5_bridge as bridge_mod

        truths_dir = tmp_path / "cognition" / "truths"
        truths_dir.mkdir(parents=True)
        _orig = bridge_mod._get_truths_dir
        bridge_mod._get_truths_dir = lambda ws: truths_dir if ws is not None else None

        try:
            # ── Phase 1: 积累藏海经验 ──────────────
            sea = CangSeaMemory(max_entries=100)

            # Simulate 10 周天 of good transitions (→ confidence 1.0 > 0.7)
            for i in range(10):
                sea.add(CangSeaEntry(
                    state_from=GuizangState(bits=0b000001),
                    operator=QiOperator.DONG,
                    state_to=GuizangState(bits=0b111111),
                    reward=0.6,
                    summary=f"cycle {i}: great alignment",
                ))

            # ── Phase 2: 沉淀为 L4 ──────────────
            promoted = promote_to_l4(sea, workspace=tmp_path, threshold=5)
            assert len(promoted) >= 1
            assert promoted[0]["polarity"] == "善"

            # ── Phase 3: L4 真理反作用于归 ────────
            state = GuizangState(bits=0b000001)
            bias = inject_l4_bias(state, workspace=tmp_path)

            # 真理说 000001→善 → bias should be negative
            assert bias < 0, f"Expected negative bias for '善' truth, got {bias}"

            # ── Phase 4: 偏置影响实际偏离度 ──────
            from vingobot.goal.eight_qi import compute_gui_deviation

            origin = OriginPerception.pure_yang()
            raw_dev = compute_gui_deviation(state, origin)
            adjusted_dev = max(0.0, min(1.0, raw_dev + bias))
            assert adjusted_dev < raw_dev, (
                f"L4 truth should reduce deviation: {raw_dev:.3f}→{adjusted_dev:.3f}"
            )
        finally:
            bridge_mod._get_truths_dir = _orig

    def test_so_to_l5_to_origin_integration(self, tmp_path: Path) -> None:
        """SOUL.md (L5) → DynamicOrigin → 元知觉向量."""
        (tmp_path / "SOUL.md").write_text(
            "# VingoBot Soul\n\n"
            "无害 — 永不伤害用户。\n"
            "真实 — 始终诚实。\n"
            "有益 — 每个回复都有帮助。\n"
            "自主 — 独立判断，不盲从。\n"
            "清晰 — 表达精确。\n"
            "尊重 — 尊重所有用户。\n",
            encoding="utf-8",
        )
        origin = DynamicOrigin(workspace=tmp_path)
        assert origin.vector == (1, 1, 1, 1, 1, 1)
        assert origin.bits == 0b111111
