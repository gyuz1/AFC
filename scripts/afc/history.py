"""Training / validation log history of a run as a table.

    python scripts/afc/history.py block75k            # every logged step
    python scripts/afc/history.py block75k --every 5  # every 5th train row
    python scripts/afc/history.py block75k global75k --eval   # validation only

Reads the run's W&B console capture (live) or, without W&B, its train.log
(whose newest rows can lag by a few minutes: stdout is buffered by tee).
"""

import argparse
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def source(run: str) -> Path:
    """W&B console capture (no stdout-buffer lag) if present, else train.log."""
    log = REPO / f"checkpoints/llava-v1.5-7b-afc-{run}/train.log"
    m = re.search(r"wandb\.ai/\S+/runs/([a-z0-9]+)", log.read_text())
    if m:
        hits = sorted(REPO.glob(f"wandb/run-*-{m.group(1)}/files/output.log"))
        if hits:
            return hits[-1]
    return log


def rows(run: str):
    text = source(run).read_text()
    for m in re.findall(r"\{'(?:loss|eval_loss)'[^}]*\}", text):
        yield ast.literal_eval(m.replace("nan", "None"))


def fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--eval", action="store_true", help="validation rows only")
    args = ap.parse_args()
    for run in args.runs:
        print(f"== {run}")
        print(f"{'kind':5s} {'epoch':>6s} {'lr':>9s} {'loss':>7s} {'ce':>7s} "
              f"{'kd':>7s} {'qcos':>6s} {'entropy':>7s}")
        n_train = 0
        for r in rows(run):
            if "eval_loss" in r:
                print(f"{'EVAL':5s} {r['epoch']:6.3f} {'':>9s} "
                      f"{fmt(r['eval_loss'])} {fmt(r.get('eval_ce'))} "
                      f"{fmt(r.get('eval_kd'))}")
                continue
            n_train += 1
            if args.eval or (n_train - 1) % args.every:
                continue
            print(f"{'train':5s} {r['epoch']:6.3f} {r['learning_rate']:9.2e} "
                  f"{fmt(r['loss'])} {fmt(r.get('ce'))} {fmt(r.get('kd'))} "
                  f"{fmt(r.get('query_attn_cosine'), 3):>6s} "
                  f"{fmt(r.get('attn_entropy'), 3):>7s}")


if __name__ == "__main__":
    main()
