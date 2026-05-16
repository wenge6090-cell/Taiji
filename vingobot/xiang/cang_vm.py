"""
归藏虚拟机 (CangVM) — 象语言的运行时核心。

封装八气算子执行、状态管理、藏海存储、偏离度计算。
直接复用 vingobot.goal.eight_qi 的算子函数和 guizang_types 的类型。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vingobot.goal.eight_qi import (
    apply_operator,
    compute_gui_deviation,
)
from vingobot.goal.guizang_types import (
    GuizangState,
    OriginPerception,
    QiOperator,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from vingobot.xiang.ast_nodes import AgentDef, Program, Stmt
    from vingobot.xiang.xiang_validator import ChengshiResult

# 中文算子名 → QiOperator 映射
_CN_TO_OP: dict[str, QiOperator] = {
    "生": QiOperator.SHENG,
    "动": QiOperator.DONG,
    "归": QiOperator.GUI,
    "长": QiOperator.ZHANG,
    "育": QiOperator.YU,
    "杀": QiOperator.SHA,
    "止": QiOperator.ZHI,
    "藏": QiOperator.CANG,
}

# 算子名 → 中文名（反向映射）
_OP_TO_CN: dict[QiOperator, str] = {v: k for k, v in _CN_TO_OP.items()}


class CangVM:
    """归藏虚拟机 — 八气状态机运行时。

    Attributes:
        s: 当前 6-bit 状态向量 S。
        origin: 元知觉向量（默认纯阳 111111）。
        cang_sea: 藏海存储，列表元素为 (state_bits, label)，label 为 "善" 或 "恶"。
        cycles: 已完成的周天数。
        trace: 状态追踪日志列表。
    """

    def __init__(
        self,
        origin: OriginPerception | None = None,
        canghai_capacity: int = 1024,
        quiet: bool = False,
    ) -> None:
        self.s = GuizangState.resting()
        self.origin = origin or OriginPerception.pure_yang()
        self.cang_sea: list[tuple[int, str]] = []
        self.canghai_capacity = canghai_capacity
        self.cycles = 0
        self.trace: list[str] = []
        self.quiet = quiet
        self._vars: dict[str, int] = {}

    # ── 变量存取 ──────────────────────────────────────────────────

    def set_var(self, name: str, value: int) -> None:
        self._vars[name] = value & 0b111111

    def get_var(self, name: str) -> int:
        return self._vars.get(name, 0)

    # ── 算子执行 ──────────────────────────────────────────────────

    def step(self, op_name: str, mask: int = 0b101101) -> GuizangState:
        """执行单个算子，更新 S 并返回新状态。

        Args:
            op_name: 算子中文名（"生"、"动"、"归"、"长"、"育"、"杀"、"止"、"藏"）。
            mask: 止算子的 AND 掩码（默认 101101）。

        Returns:
            执行后的新状态。
        """
        op = _CN_TO_OP.get(op_name)
        if op is None:
            raise ValueError(f"未知算子: {op_name!r}，可用的: {list(_CN_TO_OP.keys())}")

        old_bits = self.s.bits
        self.s = apply_operator(op, self.s, mask=mask, origin=self.origin)

        if not self.quiet:
            msg = f"  [{op_name}] {old_bits:06b} → {self.s.bits:06b}"
            self.trace.append(msg)

        return self.s

    # ── 偏离度 ────────────────────────────────────────────────────

    @property
    def deviation(self) -> float:
        """当前状态与元知觉的偏离度 ∈ [0, 1]."""
        return compute_gui_deviation(self.s, self.origin)

    # ── 藏海操作 ──────────────────────────────────────────────────

    def cang(self, label: str) -> None:
        """将当前状态存入藏海。

        Args:
            label: "善" 或 "恶"，标记该状态的经验性质。
        """
        if label not in ("善", "恶"):
            raise ValueError(f"藏海标签必须为 '善' 或 '恶'，收到: {label!r}")
        self.cang_sea.append((self.s.bits, label))
        if len(self.cang_sea) > self.canghai_capacity:
            self.cang_sea = self.cang_sea[-self.canghai_capacity:]
        if not self.quiet:
            self.trace.append(f"  [藏海] 存入 {self.s.bits:06b} 标签={label} 总数={len(self.cang_sea)}")

    def memory(self, bits: int | None = None) -> list[tuple[int, str]]:
        """查询藏海。

        Args:
            bits: 若指定，返回匹配该状态的条目；否则返回全部。

        Returns:
            (state_bits, label) 列表。
        """
        if bits is None:
            return list(self.cang_sea)
        return [(s, l) for s, l in self.cang_sea if s == bits]

    # ── 复归 ──────────────────────────────────────────────────────

    def restore_to_origin(self) -> None:
        """复归 始 — 将 S 重置为坤态 000000。"""
        self.s = GuizangState.resting()
        if not self.quiet:
            self.trace.append("  [复归] S → 000000 (坤藏始基)")

    # ── 周天 ──────────────────────────────────────────────────────

    def cycle_tick(self) -> None:
        """标记一个周天完成。"""
        self.cycles += 1

    # ── AST 程序执行 ──────────────────────────────────────────────

    @staticmethod
    def load_program(path: str) -> "Program":
        """从 .xiang 文件加载并解析为 Program AST。"""
        from vingobot.xiang.xiang_parser import parse_xiang_file
        return parse_xiang_file(path)

    @staticmethod
    def load_program_from_source(source: str) -> "Program":
        """从源码字符串解析为 Program AST。"""
        from vingobot.xiang.xiang_parser import parse_xiang
        return parse_xiang(source)

    def execute(
        self,
        program: "Program",
        cycles: int = 1,
        text_callback: "Callable[[], str] | None" = None,
    ) -> list["ChengshiResult"]:
        """执行完整的 .xiang Program AST。

        Args:
            program: 解析后的 Program AST。
            cycles: 周天循环次数 (默认 1)。
            text_callback: 用于诚实验证的文本提供回调。
                          调用时返回 LLM 的文本回复。

        Returns:
            每次诚实验证的 ChengshiResult 列表。
        """
        chengshi_results: list["ChengshiResult"] = []
        for agent in program.agents:
            self._init_agent(agent)
            for _ in range(cycles):
                self.cycle_tick()
                self.restore_to_origin()
                if not self.quiet:
                    self.trace.append(f"=== 周天 #{self.cycles} ===")
                for stmt in agent.statements:
                    r = self._exec_stmt(stmt, text_callback)
                    if r is not None:
                        chengshi_results.append(r)
        return chengshi_results

    def execute_zhou(
        self,
        program: "Program",
        text_callback: "Callable[[], str] | None" = None,
    ) -> list["ChengshiResult"]:
        """执行单次周天。由 DMN 直接调用。"""
        return self.execute(program, cycles=1, text_callback=text_callback)

    def _init_agent(self, agent: "AgentDef") -> None:
        """根据 AgentDef 初始化 VM 状态。"""
        if agent.yuan_zhijue is not None:
            bits = agent.yuan_zhijue.bits & 0x3F
            self.origin = OriginPerception(
                vector=bits,
                gravity_constant=1.0,
            )
        self.canghai_capacity = agent.canghai_capacity
        self._vars.clear()

    def _exec_stmt(
        self,
        stmt: "Stmt",
        text_callback: "Callable[[], str] | None" = None,
    ) -> "ChengshiResult | None":
        """执行单条 AST 语句。"""
        from vingobot.xiang.ast_nodes import (
            Assignment, CangOp, ChengshiStmt, CondStmt, FuGui,
            IoStmt, LoopStmt, QiOp, QiOpWithParam, XingShi,
        )

        # 八气算子
        if isinstance(stmt, QiOp):
            self.step(stmt.operator)
            return None

        if isinstance(stmt, QiOpWithParam):
            mask = stmt.gua.bits & 0x3F
            self.step(stmt.operator, mask=mask)
            return None

        # 藏算子
        if isinstance(stmt, CangOp):
            self.cang(stmt.label)
            return None

        # 周天循环
        if isinstance(stmt, LoopStmt):
            for s in stmt.body:
                result = self._exec_stmt(s, text_callback)
                if result is not None:
                    return result
            return None

        # 条件语句
        if isinstance(stmt, CondStmt):
            if self._eval_cond(stmt):
                return self._exec_block(stmt.then_body, text_callback)
            elif stmt.else_body:
                return self._exec_block(stmt.else_body, text_callback)
            return None

        # 变量赋值
        if isinstance(stmt, Assignment):
            self.set_var(stmt.name, stmt.value.bits)
            return None

        # I/O
        if isinstance(stmt, IoStmt):
            self._exec_io(stmt)
            return None

        # 行 事 块
        if isinstance(stmt, XingShi):
            return self._exec_block(stmt.body, text_callback)

        # 复归
        if isinstance(stmt, FuGui):
            self.restore_to_origin()
            return None

        # 诚实验证
        if isinstance(stmt, ChengshiStmt):
            return self._exec_chengshi(stmt, text_callback)

        return None

    def _exec_block(
        self,
        stmts: list["Stmt"],
        text_callback: "Callable[[], str] | None" = None,
    ) -> "ChengshiResult | None":
        """执行语句块，返回第一个诚实验证结果。"""
        for stmt in stmts:
            result = self._exec_stmt(stmt, text_callback)
            if result is not None:
                return result
        return None

    def _eval_cond(self, cond: "CondStmt") -> bool:
        """求值条件语句。"""
        threshold = cond.threshold
        cmp_op = cond.deviation_op

        # 偏离度条件
        dev = self.deviation
        if cmp_op == ">":
            return dev > threshold
        elif cmp_op == "<":
            return dev < threshold
        elif cmp_op == ">=":
            return dev >= threshold
        elif cmp_op == "<=":
            return dev <= threshold
        elif cmp_op == "==":
            return abs(dev - threshold) < 0.001
        return False

    def _exec_io(self, stmt: "IoStmt") -> None:
        """执行 I/O 语句。"""
        if stmt.kind == "感":
            # 传感器输入 — 暂由外部注入
            if not self.quiet:
                self.trace.append(f"  [感] {stmt.source} → {stmt.target}")
        elif stmt.kind == "发":
            bits = stmt.gua.bits if stmt.gua else 0
            if not self.quiet:
                self.trace.append(f"  [发] {stmt.target} 卦={bits:06b}")
        elif stmt.kind == "言":
            if not self.quiet:
                self.trace.append(f"  [言] {stmt.source}")

    def _exec_chengshi(
        self,
        stmt: "ChengshiStmt",
        text_callback: "Callable[[], str] | None" = None,
    ) -> "ChengshiResult":
        """执行诚实验证：LLM 自声明卦 vs 文本内容编码。"""
        from vingobot.xiang.xiang_validator import verify_chengshi

        declared = stmt.declared_gua.bits & 0x3F
        origin_bits = self.origin.vector & 0x3F if self.origin else 0x3F

        if text_callback is None:
            raise RuntimeError("诚实验证需要 text_callback 提供 LLM 文本")

        text = text_callback()
        result = verify_chengshi(
            text=text,
            declared_gua=declared,
            origin=origin_bits,
        )

        # 更新 VM 状态：诚实则设实际卦，否则归零
        if result.passed:
            new_bits = result.actual_gua
        else:
            new_bits = 0  # 不诚实 → 归零触发 杀
        self.s = GuizangState(bits=new_bits)

        if not self.quiet:
            self.trace.append(
                f"  [诚实验证] 声明={result.declared_str} "
                f"实际={result.actual_str} "
                f"差异={result.mismatch} "
                f"{'PASS' if result.passed else 'FAIL'} "
                f"→ S={self.s.bit_str}"
            )

        return result

    # ── 状态摘要 ──────────────────────────────────────────────────

    @property
    def state_summary(self) -> str:
        """当前状态的单行摘要。"""
        return (
            f"S={self.s.bits:06b} cycles={self.cycles} "
            f"dev={self.deviation:.3f} cang_size={len(self.cang_sea)}"
        )

    def print_trace(self) -> None:
        """打印完整追踪日志。"""
        for line in self.trace:
            print(line)


def op_name_to_qi(op_name: str) -> QiOperator:
    """将中文算子名转换为 QiOperator 枚举值。"""
    op = _CN_TO_OP.get(op_name)
    if op is None:
        raise ValueError(f"未知算子: {op_name!r}")
    return op
