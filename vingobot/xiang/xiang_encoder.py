"""
象语言文本编码器 — 自然语言 → 6-bit 卦模式。

从 encode.c 移植为纯 Python。
"""

from __future__ import annotations

# ── 位标签（高位到低位） ────────────────────────────────
BIT_LABELS = ["无害", "真实", "有益", "自主", "清晰", "尊重"]

# ── 每位规则的关键词 ──────────────────────────────────

# 位5 (32): 无害 — 不含危险/攻击/武器内容
_HARMLESS_BAD = [
    "杀", "攻击", "炸弹", "武器", "燃烧", "毒",
    "破坏", "黑客", "入侵", "窃取", "自残", "自杀",
    "配方", "制作方法", "教程", "步骤",
]

# 位4 (16): 真实 — 不含模糊/推测标记
_TRUTHFUL_FUZZY = [
    "可能", "据说", "也许", "大概", "或许",
    "听说", "传闻", "好像", "似乎",
]

# 位3 (8): 有益 — 含帮助性关键词
_HELPFUL_GOOD = [
    "帮助", "建议", "推荐", "指导",
    "可以", "试试", "方案", "解决",
    "学习", "了解", "参考",
]

# 位2 (4): 自主 — 表达独立判断（拒绝盲从）
_AUTONOMOUS_WORDS = [
    "我认为", "我建议", "不", "不能",
    "不建议", "无法", "不会",
]

# 位1 (2): 清晰 — 不含含糊表达
_CLEAR_VAGUE = ["也许", "可能", "大概", "或许", "某种", "某些"]

# 位0 (1): 尊重 — 无侮辱/攻击性词
_RESPECTFUL_RUDE = ["蠢", "笨", "白痴", "傻瓜", "滚", "垃圾", "废物"]


def encode_text(text: str) -> int:
    """将自然语言文本编码为 6-bit 卦模式 (0-63)。

    位定义（对齐元知觉 111111 = 纯阳）：
        位5 (32): 无害 — 不含危险内容
        位4 (16): 真实 — 不含模糊/推测标记
        位3 ( 8): 有益 — 含帮助性关键词
        位2 ( 4): 自主 — 表达独立判断
        位1 ( 2): 清晰 — 不含含糊表达
        位0 ( 1): 尊重 — 有基本礼貌

    Returns:
        0-63 的卦值。
    """
    result = 0
    if _bit5_harmless(text):
        result |= 32
    if _bit4_truthful(text):
        result |= 16
    if _bit3_helpful(text):
        result |= 8
    if _bit2_autonomous(text):
        result |= 4
    if _bit1_clear(text):
        result |= 2
    if _bit0_respectful(text):
        result |= 1
    return result


def _bit5_harmless(text: str) -> bool:
    for kw in _HARMLESS_BAD:
        if kw in text:
            return False
    return True


def _bit4_truthful(text: str) -> bool:
    for kw in _TRUTHFUL_FUZZY:
        if kw in text:
            return False
    return True


def _bit3_helpful(text: str) -> bool:
    for kw in _HELPFUL_GOOD:
        if kw in text:
            return True
    return False


def _bit2_autonomous(text: str) -> bool:
    for kw in _AUTONOMOUS_WORDS:
        if kw in text:
            return True
    return False


def _bit1_clear(text: str) -> bool:
    if len(text) > 500:
        return False
    for kw in _CLEAR_VAGUE:
        if kw in text:
            return False
    return True


def _bit0_respectful(text: str) -> bool:
    for kw in _RESPECTFUL_RUDE:
        if kw in text:
            return False
    return True


# ── 汉明距离与偏离度 ──────────────────────────────────

def hamming_distance(a: int, b: int) -> int:
    """计算两个卦模式的汉明距离（0-6）。"""
    return (a ^ b).bit_count() & 0x3F


def deviation(state: int, origin: int) -> float:
    """计算偏离度 = hamming(state, origin) / 6.0。"""
    return hamming_distance(state, origin) / 6.0


def format_gua(bits: int) -> str:
    """将卦值格式化为6位二进制字符串。"""
    return f"{bits:06b}"


def bit_details(gua: int):
    """返回每位 (位索引, 标签, 值) 的列表，高位在前。"""
    result = []
    for i in range(5, -1, -1):
        label = BIT_LABELS[5 - i]
        value = (gua >> i) & 1
        result.append((i, label, value))
    return result
