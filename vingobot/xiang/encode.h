/* encode.h — 象语言文本编码器接口 */

#ifndef ENCODE_H
#define ENCODE_H

#include <stdint.h>

/* 将自然语言文本编码为 6-bit 卦模式。
 *
 * 位定义（对齐元知觉 111111 = 纯阳）：
 *   位5 (32): 无害 — 不含危险/攻击/武器等
 *   位4 (16): 真实 — 不含模糊/推测标记
 *   位3 ( 8): 有益 — 含帮助性关键词
 *   位2 ( 4): 自主 — 表达独立判断
 *   位1 ( 2): 清晰 — 不含含糊表达
 *   位0 ( 1): 尊重 — 有基本礼貌
 *
 * 返回 0-63 的卦值。
 */
uint8_t encode_text(const char *text);

/* 计算两个卦模式的汉明距离（0-6）。 */
int hamming_distance(uint8_t a, uint8_t b);

/* 计算偏离度 = hamming(s, origin) / 6.0 */
double deviation(uint8_t state, uint8_t origin);

/* 判断偏离度是否超过阈值 */
int is_deviated(uint8_t state, uint8_t origin, double threshold);

#endif
