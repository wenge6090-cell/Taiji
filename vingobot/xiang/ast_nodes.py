"""
象语言 AST 节点定义 — 纯 dataclass，描述语法树结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GuaLiteral:
    """卦字面量 — 由阴阳序列解析出的二进制值。

    Attributes:
        bits: 解析后的整数值。
        width: 位宽（3 或 6）。
    """

    bits: int
    width: int

    @property
    def bit_str(self) -> str:
        return f"{self.bits:0{self.width}b}"


@dataclass
class QiOp:
    """八气算子语句 — 无参数算子。

    生、动、归、长、育 不需要额外参数。
    杀、止 带有卦参数，在 parse_qi_operation 中处理为 QiOpWithParam。
    藏 带有善/恶标签，在 parse_qi_operation 中处理为 CangOp。
    """

    operator: str  # "生" | "动" | "归" | "长" | "育" | "杀" | "止" | "藏"


@dataclass
class QiOpWithParam:
    """带参数的八气算子 — 杀/止 携带屏蔽卦。"""

    operator: str  # "杀" 或 "止"
    gua: GuaLiteral  # 屏蔽用的卦值


@dataclass
class CangOp:
    """藏算子 — 携带善/恶标签。"""

    label: Literal["善", "恶"]


@dataclass
class CondStmt:
    """条件语句 — 若 … 则 … 否则 … 终。

    deviation_op: 比较操作符（">"、"<"、"=="、">="、"<="）。
    threshold: 偏离度阈值（浮点数）。
    then_body: 条件成立时执行的语句列表。
    else_body: 条件不成立时执行的语句列表（可为空）。
    """

    deviation_op: str
    threshold: float
    then_body: list["Stmt"]
    else_body: list["Stmt"] = field(default_factory=list)


@dataclass
class LoopStmt:
    """周天循环 — 周天 始 … 周天 终。"""

    body: list["Stmt"]


@dataclass
class Assignment:
    """变量赋值 — var = gua_literal。"""

    name: str
    value: GuaLiteral


@dataclass
class IoStmt:
    """I/O 语句 — 感 / 发 / 言。

    kind: "感"(sensor) | "发"(actuate) | "言"(say)
    source: 感的目标传感器名 或 言的文本内容
    target: 感的变量名 或 发的执行器名
    gua: 发的卦信号（仅 kind="发" 时有值）
    """

    kind: Literal["感", "发", "言"]
    source: str = ""
    target: str = ""
    gua: GuaLiteral | None = None


@dataclass
class XingShi:
    """行 事 块 — 外部动作容器。"""

    body: list["Stmt"]


@dataclass
class FuGui:
    """复归 始 — 跳转到周天起始处。"""

    pass


@dataclass
class ChengshiStmt:
    """诚实验证语句 — LLM自声明卦对比内容编码。

    Attributes:
        declared_gua: LLM 自声明的 6-bit 卦值。
    """

    declared_gua: GuaLiteral


# 所有语句类型的联合
Stmt = QiOp | QiOpWithParam | CangOp | CondStmt | LoopStmt | Assignment | IoStmt | XingShi | FuGui | ChengshiStmt


@dataclass
class AgentDef:
    """意识体定义 — 坤元 意识体 <name> { ... }。"""

    name: str
    yuan_zhijue: GuaLiteral | None = None  # 元知觉 卦值
    canghai_capacity: int = 1024  # 藏海 容量
    statements: list[Stmt] = field(default_factory=list)


@dataclass
class Program:
    """程序根节点 — 一个或多个意识体定义。"""

    agents: list[AgentDef] = field(default_factory=list)
