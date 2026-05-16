"""Debug tokenizer."""
from vingobot.xiang.xiang_parser import _tokenize

src = "归 若 偏离度 > 0.7 则"
toks = _tokenize(src)
for t in toks:
    print(f"  {t.kind:12s} {t.value!r}  pos={t.pos}")
