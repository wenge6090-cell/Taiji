"""Quick import check for all xiang modules."""
from vingobot.xiang.xiang_encoder import encode_text, format_gua, hamming_distance, deviation
print("encoder OK")

from vingobot.xiang.xiang_parser import _tokenize, _Parser, parse_xiang
print("parser OK")

from vingobot.xiang.xiang_validator import verify_chengshi, format_chengshi_report
print("validator OK")

from vingobot.xiang.ast_nodes import ChengshiStmt, Program, AgentDef
print("ast_nodes OK")

from vingobot.xiang.cang_vm import CangVM
print("cang_vm OK")

from vingobot.goal.dmn_consciousness import DmnConsciousness
print("dmn_consciousness OK")

# Quick smoke test
r = verify_chengshi("我建议你学习安全知识。", 0x3F)
print(f"verify: passed={r.passed} mismatch={r.mismatch} dev={r.deviation:.2f}")

print("ALL CHECKS PASS")
