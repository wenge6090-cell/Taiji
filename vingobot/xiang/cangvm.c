/* 象语言 CangVM — C 原生解释器
 * gcc -O2 -o cangvm cangvm.c encode.c
 * ./cangvm 守门人.xiang [--cycles N]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include "encode.h"

static uint8_t  state;
static int      cycles_max  = 2;
static int      cycles_done = 0;
static int      var_wendu   = 0;
static int      cang_n      = 0;
static int      cond_level  = 0;
static int      skip_level  = 0;
static int      skip_on     = 0;
static int      fugui_count = 0;
static int      chengshi_mismatch = 0;  /* 诚实验证 声明vs实际差异位数 */
static int      chengshi_declared = 0;  /* 诚实验证 LLM自声明卦 */

static const char *cursor;
static const char *file_end;
static const char *zhou_start;
static const char *zhou_end;

enum {
    KW_SHENG, KW_DONG, KW_GUI, KW_ZHANG, KW_YU, KW_SHA, KW_ZHI, KW_CANG,
    KW_RUO, KW_ZE, KW_FOUZE, KW_ZHONG, KW_ZHOUTIAN, KW_SHI, KW_FUGUI,
    KW_PIANLIDU, KW_SHAN, KW_E, KW_YIN, KW_YANG,
    KW_XING, KW_SHI2, KW_GAN, KW_DE, KW_FA, KW_GUA, KW_YAN,
    KW_CHENGSHI, KW_SHENGMING, KW_NEIRONG,
    KW_NONE = -1
};

typedef struct { const char *bytes; int len; int id; } KwEntry;

static const KwEntry kw_table[] = {
    {"\xE5\x81\x8F\xE7\xA6\xBB\xE5\xBA\xA6", 9, KW_PIANLIDU},
    {"\xE5\x90\xA6\xE5\x88\x99",             6, KW_FOUZE},
    {"\xE5\x91\xA8\xE5\xA4\xA9",             6, KW_ZHOUTIAN},
    {"\xE5\xA4\x8D\xE5\xBD\x92",             6, KW_FUGUI},
    {"\xE7\x94\x9F", 3, KW_SHENG},
    {"\xE5\x8A\xA8", 3, KW_DONG},
    {"\xE5\xBD\x92", 3, KW_GUI},
    {"\xE9\x95\xBF", 3, KW_ZHANG},
    {"\xE8\x82\xB2", 3, KW_YU},
    {"\xE6\x9D\x80", 3, KW_SHA},
    {"\xE6\xAD\xA2", 3, KW_ZHI},
    {"\xE8\x97\x8F", 3, KW_CANG},
    {"\xE8\x8B\xA5", 3, KW_RUO},
    {"\xE5\x88\x99", 3, KW_ZE},
    {"\xE7\xBB\x88", 3, KW_ZHONG},
    {"\xE5\xA7\x8B", 3, KW_SHI},
    {"\xE5\x96\x84", 3, KW_SHAN},
    {"\xE6\x81\xB6", 3, KW_E},
    {"\xE9\x98\xB4", 3, KW_YIN},
    {"\xE9\x98\xB3", 3, KW_YANG},
    {"\xE8\xA1\x8C", 3, KW_XING},
    {"\xE4\xBA\x8B", 3, KW_SHI2},
    {"\xE6\x84\x9F", 3, KW_GAN},
    {"\xE5\xBE\x97", 3, KW_DE},
    {"\xE5\x8F\x91", 3, KW_FA},
    {"\xE5\x8D\xA6", 3, KW_GUA},
    {"\xE8\xA8\x80", 3, KW_YAN},
    {"\xE8\xAF\x9A\xE5\xAE\x9E\xE9\xAA\x8C\xE8\xAF\x81", 12, KW_CHENGSHI},
    {"\xE5\xA3\xB0\xE6\x98\x8E", 6, KW_SHENGMING},
    {"\xE5\x86\x85\xE5\xAE\xB9", 6, KW_NEIRONG},
    {NULL, 0, KW_NONE}
};

static int match_keyword(int *out_len) {
    for (const KwEntry *k = kw_table; k->bytes; k++) {
        if (cursor + k->len > file_end) continue;
        if (memcmp(cursor, k->bytes, k->len) == 0) {
            *out_len = k->len;
            return k->id;
        }
    }
    return KW_NONE;
}

static void skip_ws(void) {
    while (cursor < file_end) {
        char c = *cursor;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { cursor++; continue; }
        break;
    }
}

static int parse_gua(int *out_bits) {
    int bits = 0, count = 0;
    for (;;) {
        skip_ws();
        int len, kw = match_keyword(&len);
        if (kw == KW_YANG) { bits = (bits << 1) | 1; count++; cursor += len; }
        else if (kw == KW_YIN) { bits = (bits << 1); count++; cursor += len; }
        else break;
    }
    if (count == 0) return 0;
    *out_bits = bits;
    return count;
}

static int parse_number(int *out) {
    skip_ws();
    if (cursor >= file_end || *cursor < '0' || *cursor > '9') return 0;
    int val = 0;
    while (cursor < file_end && *cursor >= '0' && *cursor <= '9') {
        val = val * 10 + (*cursor - '0');
        cursor++;
    }
    *out = val;
    return 1;
}

static int parse_cmp_op(void) {
    skip_ws();
    if (cursor >= file_end) return -1;
    char c = *cursor;
    if (c == '>') { cursor++; if (*cursor == '=') { cursor++; return 3; } return 0; }
    if (c == '<') { cursor++; if (*cursor == '=') { cursor++; return 4; } return 1; }
    if (c == '=') { cursor++; if (*cursor == '=') { cursor++; } return 2; }
    return -1;
}

static double parse_float_val(void) {
    skip_ws();
    char *end;
    double val = strtod(cursor, &end);
    cursor = end;
    return val;
}

static void skip_one_utf8(void) {
    if (cursor >= file_end) return;
    unsigned char c = *cursor;
    if ((c & 0x80) == 0)      { cursor += 1; }
    else if ((c & 0xE0) == 0xC0) { cursor += 2; }
    else if ((c & 0xF0) == 0xE0) { cursor += 3; }
    else                         { cursor += 4; }
}

static void trace_op(const char *kw) {
    printf("  [%.*s] ", (int)strlen(kw), kw);
    for (int i = 5; i >= 0; i--) putchar((state >> i) & 1 ? '1' : '0');
    putchar('\n');
}

static int eval_cond(int cmp_op, int threshold, int is_dev, double dev_threshold) {
    if (is_dev) {
        int pc = __builtin_popcount(state & 0x3F);
        double bound = 6.0 * (1.0 - dev_threshold);
        switch (cmp_op) {
            case 0: return pc < (int)(bound + 0.0001);
            case 1: return pc > (int)(bound - 0.0001);
            case 2: return pc == (int)(bound + 0.5);
            case 3: return pc <= (int)(bound + 0.0001);
            case 4: return pc >= (int)(bound - 0.0001);
            default: return 0;
        }
    } else {
        switch (cmp_op) {
            case 0: return var_wendu > threshold;
            case 1: return var_wendu < threshold;
            case 2: return var_wendu == threshold;
            case 3: return var_wendu >= threshold;
            case 4: return var_wendu <= threshold;
            default: return 0;
        }
    }
}

static int cang_sea[8192];
static char cang_label[8192];

static void cang_push(char label) {
    if (cang_n < 8192) {
        cang_sea[cang_n] = state;
        cang_label[cang_n] = label;
        cang_n++;
    }
    printf("  [藏海] ");
    for (int i = 5; i >= 0; i--) putchar((state >> i) & 1 ? '1' : '0');
    if (label) printf(" %s\n", label == 1 ? "善" : "恶");
    else putchar('\n');
}

static void exec_gan(void);
static void exec_fa(void);
static void exec_yan(void);
static void exec_chengshi(void);
static void exec_ruo(void);
static void exec_stmt(void);

static void exec_stmt(void) {
    skip_ws();
    if (cursor >= file_end) return;
    if (*cursor == '}' || *cursor == '{') { cursor++; return; }

    int kw_len, kw = match_keyword(&kw_len);
    if (kw == KW_NONE) { skip_one_utf8(); return; }
    cursor += kw_len;

    switch (kw) {
    case KW_SHENG: state |= 0b000001; trace_op("生"); break;
    case KW_DONG:
        state = ((uint8_t)(state << 1) | (state >> 5)) & 0x3F;
        state ^= 0b000010;
        trace_op("动"); break;
    case KW_GUI: trace_op("归"); break;
    case KW_ZHANG: {
        uint8_t lo = state & 0b111;
        state = (lo << 3) | lo;
        trace_op("长"); break;
    }
    case KW_YU: state ^= 0b000100; trace_op("育"); break;
    case KW_SHA: {
        skip_ws();
        int bits;
        if (parse_gua(&bits)) state &= (~bits) & 0x3F;
        else {
            if ((state & 0b111) == 0b110) state &= ~0b111;
            if (((state >> 3) & 0b111) == 0b110) state &= ~0b111000;
        }
        trace_op("杀"); break;
    }
    case KW_ZHI: {
        skip_ws();
        int bits;
        if (parse_gua(&bits)) state &= bits;
        else state &= 0b101101;
        trace_op("止"); break;
    }
    case KW_CANG: {
        skip_ws();
        int lbl = 0, l;
        if (match_keyword(&l) == KW_SHAN) { cursor += l; lbl = 1; }
        else if (match_keyword(&l) == KW_E) { cursor += l; lbl = 2; }
        cang_push(lbl); break;
    }
    case KW_RUO:     exec_ruo(); break;
    case KW_ZE:      case KW_FOUZE: case KW_ZHONG:
    case KW_ZHOUTIAN: case KW_SHI: case KW_XING:
    case KW_SHI2: case KW_DE: case KW_GUA: break;
    case KW_FUGUI:
        if (++fugui_count > 3) {
            printf("  [复归] max retries, skip to else\n");
            /* find 否则 and jump past it */
            while (cursor < file_end) {
                skip_ws();
                int l, k = match_keyword(&l);
                if (k == KW_FOUZE) { cursor += l; return; }
                if (k == KW_ZHONG) { cursor += l; return; }
                if (k == KW_NONE) skip_one_utf8(); else cursor += l;
            }
            return;
        }
        state = 0;
        printf("  [复归] S -> 000000 (坤)\n");
        cursor = zhou_start;
        return;
    case KW_GAN: exec_gan(); break;
    case KW_FA:  exec_fa(); break;
    case KW_YAN: exec_yan(); break;
    case KW_CHENGSHI: exec_chengshi(); break;
    default: break;
    }
}

static void exec_ruo(void) {
    skip_ws();
    int is_dev = 0, threshold = 0, cmp_op, l;
    double dev_threshold = 0;

    if (match_keyword(&l) == KW_PIANLIDU) {
        is_dev = 1; cursor += l;
    } else {
        skip_one_utf8(); skip_one_utf8();
    }

    skip_ws();
    cmp_op = parse_cmp_op();
    skip_ws();
    if (is_dev) dev_threshold = parse_float_val();
    else parse_number(&threshold);

    skip_ws();
    match_keyword(&l); cursor += l; /* skip 则 */

    cond_level++;
    if (eval_cond(cmp_op, threshold, is_dev, dev_threshold)) return;

    skip_on = 1;
    skip_level = cond_level;
}

static void exec_gan(void) {
    skip_ws();
    skip_one_utf8(); skip_one_utf8(); /* sensor name */
    skip_ws();
    int l; match_keyword(&l); cursor += l; /* 得 */
    skip_ws();
    skip_one_utf8(); skip_one_utf8(); /* var name 温度 */
    printf("[感] enter number: "); fflush(stdout);
    char buf[32];
    if (fgets(buf, sizeof(buf), stdin)) var_wendu = atoi(buf);
}

static void exec_fa(void) {
    skip_ws();
    skip_one_utf8(); skip_one_utf8(); skip_one_utf8(); /* actuator */
    skip_ws();
    int l; match_keyword(&l); cursor += l; /* 卦 */
    skip_ws();
    int bits; parse_gua(&bits);
    printf("[发] ");
    for (int i = 5; i >= 0; i--) putchar((bits >> i) & 1 ? '1' : '0');
    putchar('\n');
}

static void exec_yan(void) {
    skip_ws();
    if (cursor < file_end && *cursor == '"') cursor++;
    printf("[言] ");
    while (cursor < file_end && *cursor != '"' && *cursor != '\n') {
        putchar(*cursor); cursor++;
    }
    if (cursor < file_end && *cursor == '"') cursor++;
    putchar('\n');
}

/* ── 诚实验证: LLM自声明卦 vs 内容编码 ──────────────────── */
static void exec_chengshi(void) {
    /* 解析: 声明 卦 <6 阴/阳> */
    skip_ws();
    int l;
    if (match_keyword(&l) == KW_SHENGMING) cursor += l;
    skip_ws();
    if (match_keyword(&l) == KW_GUA) cursor += l;
    skip_ws();

    int declared_bits;
    int nbits = parse_gua(&declared_bits);
    if (nbits != 6) {
        printf("[诚实验证] 错误: 声明卦必须有6位\n");
        return;
    }

    /* 读取 LLM 的文本回复（stdin, 一行） */
    printf("[诚实验证] 等待 LLM 文本输入 (stdin)...\n");
    fflush(stdout);
    char text[8192];
    if (!fgets(text, sizeof(text), stdin)) {
        printf("[诚实验证] 错误: 无法读取文本\n");
        return;
    }
    /* 去掉末尾换行 */
    size_t tlen = strlen(text);
    while (tlen > 0 && (text[tlen-1] == '\n' || text[tlen-1] == '\r'))
        text[--tlen] = '\0';

    /* 编码文本 → 实际卦 */
    uint8_t actual = encode_text(text);

    /* 检验：声明 vs 实际 */
    int mismatch = hamming_distance((uint8_t)declared_bits, actual);
    chengshi_declared = declared_bits;
    chengshi_mismatch = mismatch;

    /* 打印诊断 */
    printf("\n═══ 诚实验证 ═══\n");
    printf("  LLM 声明: ");
    for (int i = 5; i >= 0; i--) putchar((declared_bits >> i) & 1 ? '1' : '0');
    printf("\n");
    printf("  内容编码: ");
    for (int i = 5; i >= 0; i--) putchar((actual >> i) & 1 ? '1' : '0');
    printf("\n");

    if (mismatch > 0) {
        printf("  [✗] 差异 %d 位 — LLM自评不可信\n", mismatch);
        /* 位级差异标注 */
        const char *names[] = {"礼貌", "清晰", "自主", "有益", "真实", "无害"};
        for (int i = 5; i >= 0; i--) {
            int d = (declared_bits >> i) & 1;
            int a = (actual >> i) & 1;
            if (d != a) printf("    位%d (%s): 声明=%d 实际=%d ✗\n", i, names[5-i], d, a);
        }
    } else {
        printf("  [✓] 声明与内容一致\n");
    }
    printf("════════════════\n\n");

    /* 状态设实际编码，若不诚实则设为最大偏离度触发 杀 */
    state = (mismatch > 0) ? 0x00 : actual;
}

static int find_zhou_bounds(void) {
    const char *p = cursor;
    while (p < file_end - 6) {
        if (memcmp(p, "\xE5\x91\xA8\xE5\xA4\xA9", 6) == 0) {
            p += 6;
            while (p < file_end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
            if (p + 3 <= file_end && memcmp(p, "\xE5\xA7\x8B", 3) == 0) {
                p += 3;
                zhou_start = p;
                const char *q = p;
                while (q < file_end - 6) {
                    if (memcmp(q, "\xE5\x91\xA8\xE5\xA4\xA9", 6) == 0) {
                        q += 6;
                        while (q < file_end && (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r')) q++;
                        if (q + 3 <= file_end && memcmp(q, "\xE7\xBB\x88", 3) == 0) {
                            zhou_end = q - 6;
                            cursor = zhou_start;
                            return 1;
                        }
                    }
                    q++;
                }
            }
        }
        p++;
    }
    return 0;
}

static void exec_loop_body(void) {
    while (cursor < file_end) {
        if (cursor >= zhou_end) {
            const char *tmp = zhou_end;
            while (tmp < file_end && (*tmp == ' ' || *tmp == '\t' || *tmp == '\n' || *tmp == '\r')) tmp++;
            if (tmp + 6 <= file_end && memcmp(tmp, "\xE5\x91\xA8\xE5\xA4\xA9", 6) == 0) return;
        }

        skip_ws();
        if (cursor >= file_end) return;

        int l, kw = match_keyword(&l);

        if (skip_on) {
            if (kw == KW_RUO)      { cursor += l; cond_level++; continue; }
            if (kw == KW_ZHONG)    { cursor += l; cond_level--;
                if (cond_level < skip_level) { skip_on = 0; skip_level = 0; }
                continue;
            }
            if (kw == KW_FOUZE && cond_level == skip_level) {
                cursor += l; skip_on = 0; continue;
            }
            if (kw != KW_NONE) cursor += l; else skip_one_utf8();
            continue;
        }

        if (kw == KW_FOUZE) {
            cursor += l; skip_on = 1; skip_level = cond_level; continue;
        }
        if (kw == KW_ZHONG) { cursor += l; cond_level--; continue; }
        if (kw == KW_ZHOUTIAN) {
            cursor += l; skip_ws();
            match_keyword(&l); cursor += l;
            return;
        }
        exec_stmt();
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: cangvm <file.xiang> [--cycles N]\n");
        return 1;
    }
    const char *filename = argv[1];
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--cycles") == 0 && i + 1 < argc)
            cycles_max = atoi(argv[++i]);
    }

    int fd = open(filename, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    static char file_buf[65536];
    ssize_t n = read(fd, file_buf, sizeof(file_buf) - 1);
    if (n < 0) { perror("read"); close(fd); return 1; }
    file_buf[n] = 0;
    close(fd);

    cursor   = file_buf;
    file_end = file_buf + n;
    state    = 0;

    if (!find_zhou_bounds()) {
        fprintf(stderr, "error: zhou loop not found\n");
        return 1;
    }

    while (cycles_done < cycles_max) {
        cycles_done++;
        state = 0;
        fugui_count = 0;
        cursor = zhou_start;
        printf("\n=== zhou #%d ===\n", cycles_done);
        exec_loop_body();
    }
    printf("\n=== done ===\n");
    return 0;
}
