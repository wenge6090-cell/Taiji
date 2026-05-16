"""Full debug: tokenize and parse both .xiang files."""
import sys, traceback
from vingobot.xiang.xiang_parser import _tokenize, _Parser

for fname in ["守门人.xiang", "守门人_验证.xiang"]:
    path = f"vingobot/xiang/examples/{fname}"
    src = open(path, encoding="utf-8").read()
    print(f"=== {fname} ({len(src)} chars) ===")
    
    # Show tokens
    toks = _tokenize(src)
    print("TOKENS:")
    for t in toks:
        print(f"  {t.pos:4d} {t.kind:10s} {t.value!r}")
    
    # Try parse
    print("\nPARSE:")
    try:
        p = _Parser(toks)
        prog = p.parse()
        a = prog.agents[0]
        print(f"  Agent={a.name} YuanZhiJue={a.yuan_zhijue} CangHai={a.canghai_capacity} N={len(a.statements)}")
        for s in a.statements:
            print(f"    [{type(s).__name__}]")
            if hasattr(s, 'body'):
                for st in s.body:
                    print(f"      {type(st).__name__}")
    except Exception as e:
        traceback.print_exc()
    print()
print("DONE")
