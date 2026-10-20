"""Format HF Trainer log dicts read from stdin (one per line)."""

import ast
import sys

for line in sys.stdin:
    d = ast.literal_eval(line.replace("nan", "None"))
    print(f"  ep {d['epoch']:.3f}  lr {d['learning_rate']:.2e}  "
          f"loss {d['loss']:.4f}  ce {d.get('ce') or 0:.4f}  "
          f"kd {d.get('kd') or 0:.4f}  "
          f"qcos {d.get('query_attn_cosine') or 0:.3f}")
