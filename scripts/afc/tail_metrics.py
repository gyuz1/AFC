"""Follow a run's metrics in the terminal, one row per logged step (the values
W&B charts show: train/loss, ce, kd, learning rate, query diagnostics, and the
eval/* rows every eval_steps).

    python scripts/afc/tail_metrics.py global_all            # last 20 rows, then follow
    python scripts/afc/tail_metrics.py block_all --last 100
    python scripts/afc/tail_metrics.py global_all --no-follow

Reads checkpoints/llava-v1.5-7b-afc-<run>/train.log (unbuffered). Ctrl+C
stops the view only, never the training.
"""

import argparse
import ast
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DICT = re.compile(r"\{'(?:loss|eval_loss)'[^}]*\}")
PROG = re.compile(r"(\d+)/(\d+) \[([^\]<]+)<([^\],]+),\s*([\d.]+)s/it\]")
HEADER = (f"{'kind':5s} {'step':>11s} {'epoch':>6s} {'lr':>9s} {'loss':>7s} "
          f"{'ce':>7s} {'kd':>7s} {'attn_H':>7s} {'q_cos':>6s}")


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def row(d, step, total):
    st = f"{step}/{total}" if step else "-"
    if "eval_loss" in d:
        return (f"\033[1;36m{'EVAL':5s} {st:>11s} {d['epoch']:6.3f} {'':>9s} "
                f"{fmt(d['eval_loss'], '7.4f')} {fmt(d.get('eval_ce'), '7.4f')} "
                f"{fmt(d.get('eval_kd'), '7.4f')}\033[0m")
    return (f"{'train':5s} {st:>11s} {d['epoch']:6.3f} "
            f"{fmt(d.get('learning_rate'), '9.2e')} {fmt(d['loss'], '7.4f')} "
            f"{fmt(d.get('ce'), '7.4f')} {fmt(d.get('kd'), '7.4f')} "
            f"{fmt(d.get('attn_entropy'), '7.3f')} "
            f"{fmt(d.get('query_attn_cosine'), '6.3f')}")


def parse(text, state):
    """Rows from new log text; state carries the latest progress-bar match."""
    out = []
    for tok in re.finditer(f"{DICT.pattern}|{PROG.pattern}", text):
        s = tok.group(0)
        if s.startswith("{"):
            d = ast.literal_eval(s.replace("nan", "None"))
            out.append(row(d, state.get("step"), state.get("total")))
        else:
            m = PROG.match(s)
            state.update(step=int(m.group(1)), total=int(m.group(2)),
                         elapsed=m.group(3), eta=m.group(4), rate=m.group(5))
    return out


def status(state):
    if "step" not in state:
        return ""
    pct = 100 * state["step"] / state["total"]
    return (f"\r\033[2K\033[33m  step {state['step']}/{state['total']} "
            f"({pct:.1f}%)  {state['rate']} s/it  elapsed {state['elapsed']}  "
            f"ETA {state['eta']}\033[0m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--no-follow", action="store_true")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()
    log = REPO / f"checkpoints/llava-v1.5-7b-afc-{args.run}/train.log"
    if not log.exists():
        sys.exit(f"no log: {log}")
    state = {}
    with open(log, errors="replace") as f:
        rows = parse(f.read(), state)
        pos = f.tell()
    print(f"== {args.run}  ({log})")
    print(HEADER)
    print("\n".join(rows[-args.last:]))
    if args.no_follow:
        print(status(state).lstrip("\r"))
        return
    try:
        while True:
            sys.stdout.write(status(state))
            sys.stdout.flush()
            time.sleep(args.interval)
            with open(log, errors="replace") as f:
                f.seek(pos)
                new = f.read()
                pos = f.tell()
            # keep an incomplete trailing dict for the next read
            cut = new.rfind("{")
            if cut != -1 and "}" not in new[cut:]:
                pos -= len(new[cut:].encode())
                new = new[:cut]
            for r in parse(new, state):
                sys.stdout.write("\r\033[2K" + r + "\n")
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
