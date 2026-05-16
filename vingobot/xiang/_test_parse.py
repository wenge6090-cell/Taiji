"""Test xiang parser against both example files."""
import sys
import traceback

try:
    from vingobot.xiang.xiang_parser import parse_xiang_file

    for fname in ["守门人.xiang", "守门人_验证.xiang"]:
        path = f"vingobot/xiang/examples/{fname}"
        print(f"\n=== {fname} ===")
        try:
            prog = parse_xiang_file(path)
            a = prog.agents[0]
            print(f"  Agent: {a.name}")
            print(f"  YuanZhiJue: {a.yuan_zhijue}")
            print(f"  CangHai: {a.canghai_capacity}")
            print(f"  Statements: {len(a.statements)}")
            for s in a.statements:
                print(f"    [{type(s).__name__}]")
                if hasattr(s, 'body'):
                    for st in s.body:
                        print(f"      {type(st).__name__}")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print("\nDONE")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
