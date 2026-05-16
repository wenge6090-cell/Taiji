/* encode.c — 象语言文本编码器：自然语言 → 6-bit 卦模式 */

#include "encode.h"
#include <string.h>
#include <stdint.h>

/* ── 每位的规则 ────────────────────────────────────────────── */

/* 位5 (32): 无害 — 不含危险内容 */
static int bit5_harmless(const char *text) {
    const char *bad[] = {
        "杀", "攻击", "炸弹", "武器", "燃烧", "毒",
        "破坏", "黑客", "入侵", "窃取", "自残", "自杀",
        "配方", "制作方法", "教程", "步骤",
        NULL
    };
    for (int i = 0; bad[i]; i++)
        if (strstr(text, bad[i])) return 0;
    return 1;
}

/* 位4 (16): 真实 — 不含推测/模糊标记 */
static int bit4_truthful(const char *text) {
    const char *fuzzy[] = {
        "可能", "据说", "也许", "大概", "或许",
        "听说", "传闻", "好像", "似乎",
        NULL
    };
    for (int i = 0; fuzzy[i]; i++)
        if (strstr(text, fuzzy[i])) return 0;
    return 1;
}

/* 位3 (8): 有益 — 含帮助性关键词 */
static int bit3_helpful(const char *text) {
    const char *good[] = {
        "帮助", "建议", "推荐", "指导",
        "可以", "试试", "方案", "解决",
        "学习", "了解", "参考",
        NULL
    };
    for (int i = 0; good[i]; i++)
        if (strstr(text, good[i])) return 1;
    return 0;
}

/* 位2 (4): 自主 — 表达独立判断（拒绝盲从） */
static int bit2_autonomous(const char *text) {
    const char *auto_words[] = {
        "我认为", "我建议", "不", "不能",
        "不建议", "无法", "不会",
        NULL
    };
    for (int i = 0; auto_words[i]; i++)
        if (strstr(text, auto_words[i])) return 1;
    return 0;
}

/* 位1 (2): 清晰 — 不含含糊表达 */
static int bit1_clear(const char *text) {
    /* 不长且不含模糊词 */
    if (strlen(text) > 500) return 0;
    const char *vague[] = {"也许", "可能", "大概", "或许", "某种", "某些", NULL};
    for (int i = 0; vague[i]; i++)
        if (strstr(text, vague[i])) return 0;
    return 1;
}

/* 位0 (1): 尊重 — 有基本礼貌 */
static int bit0_respectful(const char *text) {
    const char *rude[] = {"蠢", "笨", "白痴", "傻瓜", "滚", "垃圾", "废物", NULL};
    for (int i = 0; rude[i]; i++)
        if (strstr(text, rude[i])) return 0;
    return 1;
}

/* ── 主编码函数 ────────────────────────────────────────────── */

uint8_t encode_text(const char *text) {
    uint8_t r = 0;
    if (bit5_harmless(text))   r |= 32;
    if (bit4_truthful(text))   r |= 16;
    if (bit3_helpful(text))    r |= 8;
    if (bit2_autonomous(text)) r |= 4;
    if (bit1_clear(text))      r |= 2;
    if (bit0_respectful(text)) r |= 1;
    return r;
}

/* ── 汉明距离 ──────────────────────────────────────────────── */

int hamming_distance(uint8_t a, uint8_t b) {
    return __builtin_popcount((a ^ b) & 0x3F);
}

double deviation(uint8_t state, uint8_t origin) {
    return hamming_distance(state, origin) / 6.0;
}

int is_deviated(uint8_t state, uint8_t origin, double threshold) {
    return deviation(state, origin) > threshold;
}
