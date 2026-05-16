"""
象语言解析器 — 手写递归下降解析 .xiang 源码 → AST。

用法:
    from vingobot.xiang.xiang_parser import parse_xiang, parse_xiang_file
    program = parse_xiang(source_text)
    program = parse_xiang_file("examples/守门人.xiang")
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vingobot.xiang.ast_nodes import (
    AgentDef,
    Assignment,
    CangOp,
    ChengshiStmt,
    CondStmt,
    FuGui,
    GuaLiteral,
    IoStmt,
    LoopStmt,
    Program,
    QiOp,
    QiOpWithParam,
    Stmt,
    XingShi,
)

# ── 关键字映射 ────────────────────────────────────────────
_SIMPLE_OPS = {"生", "动", "归", "长", "育"}
_PARAM_OPS = {"杀", "止"}
_CANG_LABELS = {"善", "恶"}
_YINYANG = {"阴", "阳"}

# ── 标记类型 ──────────────────────────────────────────────
_TOKEN_SPEC = [
    ("STRING",  r'"[^"]*"'),
    ("FLOAT",   r'\d+\.\d+'),
    ("NUMBER",  r'\d+'),
    ("CMP",     r'>=|<=|==|[><]'),
    ("ASSIGN",  r'='),
    ("LBRACE",  r'\{'),
    ("RBRACE",  r'\}'),
    ("WORD",    r'[^\s"\{\}><=]+'),
    ("WHITESPACE", r'\s+'),
]


@dataclass
class Token:
    kind: str
    value: str
    pos: int  # character position


def _tokenize(source: str) -> list[Token]:
    """将源文本分割为标记序列。"""
    tokens: list[Token] = []
    combined = "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC)
    for m in re.finditer(combined, source, re.UNICODE):
        kind = m.lastgroup
        value = m.group()
        if kind == "WHITESPACE":
            continue
        tokens.append(Token(kind, value, m.start()))
    return tokens


class ParseError(Exception):
    """解析错误。"""
    def __init__(self, msg: str, pos: int = -1):
        super().__init__(f"{msg} (位置 {pos})" if pos >= 0 else msg)


# ── 解析器 ────────────────────────────────────────────────

class _Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self._var_counter = 0  # for unnamed temp vars

    # ── 标记操作 ──────────────────────────────────────────

    def _peek(self) -> Token | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, kind: str, value: str | None = None) -> Token:
        tok = self._peek()
        if tok is None:
            raise ParseError(f"期望 {kind} {value or ''}，到达文件结尾")
        if tok.kind != kind:
            raise ParseError(
                f"期望 {kind}，遇到 {tok.kind}({tok.value!r})", tok.pos
            )
        if value is not None and tok.value != value:
            raise ParseError(
                f"期望 {value!r}，遇到 {tok.value!r}", tok.pos
            )
        return self._advance()

    def _match(self, kind: str, value: str | None = None) -> bool:
        tok = self._peek()
        if tok is None:
            return False
        if tok.kind != kind:
            return False
        if value is not None and tok.value != value:
            return False
        return True

    def _match_kw(self, *words: str) -> str | None:
        """匹配 WORD 标记，值为 words 之一。返回匹配到的词或 None。"""
        tok = self._peek()
        if tok is None or tok.kind != "WORD":
            return None
        if tok.value in words:
            self._advance()
            return tok.value
        return None

    # ── 基本解析方法 ──────────────────────────────────────

    def _parse_gua(self) -> GuaLiteral:
        """解析阴阳序列 → GuaLiteral。

        支持两种形式:
        - 分词形式: 阳 阳 阳 阳 阳 阳 (每个阴阳是独立token)
        - 连写形式: 阳阳阳阳阳阳 (一个token包含所有阴阳)
        """
        bits = 0
        count = 0

        # 检查当前 token 是否是多字符连写 (如 "阳阳阳阳阳阳")
        tok = self._peek()
        if tok and tok.kind == "WORD":
            val = tok.value
            if all(c in ("阴", "阳") for c in val):
                self._advance()
                for c in val:
                    if c == "阳":
                        bits = (bits << 1) | 1
                        count += 1
                    elif c == "阴":
                        bits = (bits << 1)
                        count += 1
                return GuaLiteral(bits=bits, width=count)

        # 逐个 token 解析
        while self._match("WORD"):
            val = self._peek().value
            if val == "阳":
                bits = (bits << 1) | 1
                count += 1
                self._advance()
            elif val == "阴":
                bits = (bits << 1)
                count += 1
                self._advance()
            else:
                break
        if count == 0:
            raise ParseError("期望 阴/阳 序列", self._peek().pos if self._peek() else -1)
        return GuaLiteral(bits=bits, width=count)

    def _parse_number(self) -> int:
        tok = self._expect("NUMBER")
        return int(tok.value)

    def _parse_float(self) -> float:
        tok = self._peek()
        if tok and tok.kind == "FLOAT":
            self._advance()
            return float(tok.value)
        if tok and tok.kind == "NUMBER":
            self._advance()
            return float(tok.value)
        raise ParseError("期望数字", tok.pos if tok else -1)

    def _parse_cmp_op(self) -> str:
        tok = self._expect("CMP")
        return tok.value

    def _parse_cond_expr(self) -> tuple[str | None, str, float]:
        """解析条件表达式: [偏离度] <cmp> <threshold> 或 <var> <cmp> <num>。

        Returns:
            (var_name_or_None, cmp_op, threshold)
        """
        tok = self._peek()
        if tok is None:
            raise ParseError("期望条件表达式")
        if tok.value == "偏离度":
            self._advance()
            cmp_op = self._parse_cmp_op()
            threshold = self._parse_float()
            return (None, cmp_op, threshold)
        else:
            # 变量条件: var cmp num
            var_name = tok.value
            self._advance()
            cmp_op = self._parse_cmp_op()
            threshold = float(self._parse_number())
            return (var_name, cmp_op, threshold)

    # ── 语句解析 ──────────────────────────────────────────

    _BLOCK_END_KWS = frozenset({"终", "否则"})

    def _parse_stmt(self, end_kws: frozenset[str] = _BLOCK_END_KWS) -> Stmt | None:
        """解析单条语句。遇到 end_kws 中任一关键字或文件结尾时返回 None。"""
        tok = self._peek()
        if tok is None:
            return None
        if tok.kind == "RBRACE":
            return None

        # 关键字分发
        kw = tok.value if tok.kind == "WORD" else None

        if kw is None:
            raise ParseError(f"意外的标记: {tok.value!r}", tok.pos)

        if kw in end_kws:
            return None

        # 八气算子
        if kw in _SIMPLE_OPS:
            self._advance()
            return QiOp(operator=kw)

        if kw in _PARAM_OPS:
            self._advance()
            gua = self._parse_gua()
            return QiOpWithParam(operator=kw, gua=gua)

        # 藏算子
        if kw == "藏":
            self._advance()
            tok2 = self._peek()
            if tok2 is None:
                raise ParseError("藏 后期望 善/恶")
            if tok2.value in _CANG_LABELS:
                self._advance()
                from typing import Literal
                label: Literal["善", "恶"] = tok2.value  # type: ignore
                return CangOp(label=label)
            raise ParseError(f"藏 后期望 善/恶，遇到 {tok2.value!r}", tok2.pos)

        # 条件语句
        if kw == "若":
            self._advance()
            var_name, cmp_op, threshold = self._parse_cond_expr()
            self._match_kw("则") or self._expect("WORD", "则")
            then_body = self._parse_block(end_kws=frozenset({"否则", "终"}))
            else_body: list[Stmt] = []
            if self._match_kw("否则"):
                else_body = self._parse_block()
            return CondStmt(
                deviation_op=cmp_op,
                threshold=threshold,
                then_body=then_body,
                else_body=else_body,
            )

        # 复归
        if kw == "复归":
            self._advance()
            self._match_kw("始")
            return FuGui()

        # 行 事 块
        if kw == "行":
            self._advance()
            self._match_kw("事")
            body = self._parse_block()
            return XingShi(body=body)

        # I/O: 感 / 发 / 言
        if kw == "感":
            self._advance()
            sensor_name = self._advance().value
            self._match_kw("得")
            var_name = self._advance().value
            return IoStmt(kind="感", source=sensor_name, target=var_name)

        if kw == "发":
            self._advance()
            actuator = self._advance().value
            self._match_kw("卦")
            gua = self._parse_gua()
            return IoStmt(kind="发", target=actuator, gua=gua)

        if kw == "言":
            self._advance()
            tok_str = self._peek()
            if tok_str and tok_str.kind == "STRING":
                self._advance()
                text = tok_str.value[1:-1]  # strip quotes
            else:
                # 读取到行尾或下一个关键字
                text = ""
                while self._peek() and self._peek().kind == "WORD":
                    if self._peek() and self._peek().value in ("周天", "终", "若", "否则", "复归", "行", "感", "发", "言", "藏", "生", "动", "归", "长", "育", "杀", "止", "诚实验证"):
                        break
                    text += self._advance().value
            return IoStmt(kind="言", source=text)

        # 诚实验证
        if kw == "诚实验证":
            self._advance()
            self._match_kw("声明")
            self._match_kw("卦")
            declared = self._parse_gua()
            return ChengshiStmt(declared_gua=declared)

        # 变量赋值: var = <gua>
        # 检查后面是否有 =
        if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].kind == "ASSIGN":
            var_name = tok.value
            self._advance()
            self._advance()  # skip =
            gua = self._parse_gua()
            return Assignment(name=var_name, value=gua)

        # 其他关键字跳过（如 元知觉, 藏海, 周天 等在上层处理）
        if kw in ("元知觉", "藏海", "容量", "周天"):
            return None

        raise ParseError(f"无法识别的语句: {kw!r}", tok.pos)

    def _parse_block(self, end_kws: frozenset[str] | None = None) -> list[Stmt]:
        """解析语句块（由 end_kws 中任一关键字终止）。

        Args:
            end_kws: 终止关键字集合，默认 frozenset({"终"})。
        """
        if end_kws is None:
            end_kws = frozenset({"终"})
        stmts: list[Stmt] = []
        while self._peek():
            tok = self._peek()
            if tok.kind == "RBRACE":
                break
            if tok.kind == "WORD" and tok.value in end_kws:
                self._advance()
                break
            stmt = self._parse_stmt(end_kws=end_kws)
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def _parse_braced_block(self) -> list[Stmt]:
        """解析花括号块 { ... }。"""
        self._expect("LBRACE", "{")
        stmts: list[Stmt] = []
        while self._peek():
            if self._match("RBRACE", "}"):
                self._advance()
                break
            stmt = self._parse_stmt()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def _parse_agent(self) -> AgentDef:
        """解析一个意识体定义。"""
        # 坤元 意识体 <name> { ... }
        self._match_kw("坤元")
        self._match_kw("意识体")
        name = self._advance().value

        self._expect("LBRACE", "{")

        agent = AgentDef(name=name)

        while self._peek() and not self._match("RBRACE", "}"):
            tok = self._peek()
            kw = tok.value if tok.kind == "WORD" else None

            if kw == "元知觉":
                self._advance()
                agent.yuan_zhijue = self._parse_gua()
                continue

            if kw == "藏海":
                self._advance()
                self._match_kw("容量")
                agent.canghai_capacity = self._parse_number()
                continue

            if kw == "周天":
                self._advance()
                self._match_kw("始")
                body = self._parse_block()
                agent.statements.append(LoopStmt(body=body))
                continue

            # 顶层语句（可能在周天外）
            stmt = self._parse_stmt()
            if stmt is not None:
                agent.statements.append(stmt)

        self._expect("RBRACE", "}")
        return agent

    def parse(self) -> Program:
        """解析完整程序。"""
        agents: list[AgentDef] = []
        while self._peek():
            tok = self._peek()
            if tok.kind == "WORD" and tok.value == "坤元":
                agent = self._parse_agent()
                agents.append(agent)
            else:
                self._advance()
        return Program(agents=agents)


# ── 公共 API ──────────────────────────────────────────────

def parse_xiang(source: str) -> Program:
    """解析象语言源码，返回 Program AST。

    Args:
        source: .xiang 源码字符串。

    Returns:
        Program AST。

    Raises:
        ParseError: 语法错误。
    """
    tokens = _tokenize(source)
    parser = _Parser(tokens)
    return parser.parse()


def parse_xiang_file(path: str) -> Program:
    """从文件读取并解析象语言源码。

    Args:
        path: .xiang 文件路径。

    Returns:
        Program AST。
    """
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return parse_xiang(source)
