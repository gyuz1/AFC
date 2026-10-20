"""Collect the scores written by scripts/afc/run_eval.sh into one table.

    python -m llava.eval.afc.table outputs/eval/*            # terminal table
    python -m llava.eval.afc.table outputs/eval/* --format markdown|latex|csv
    python -m llava.eval.afc.table outputs/eval/<run> --wandb   # also log to W&B

One row per run directory. Columns follow the LLaVA-1.5 / Fourier-VLM tables:
    TextVQA  VQA accuracy (official LLaVA eval, OCR-token prompt)
    GQA      testdev-balanced exact match (lmms-eval)
    POPE     F1, mean of the adversarial / popular / random splits
             (as reported for LLaVA-1.5; pooled F1 is used only if the
             per-sample log is missing, marked with *)
    SQA-IMG  accuracy (lmms-eval scienceqa_img)
    MMB      dev, CircularEval accuracy (local scorer)
    MMMU     val accuracy (lmms-eval)
    Avg      mean of the six, only when all six are present
Runs evaluated with LIMIT (directory suffix ``_limitN``) are partial and
flagged; never put them in a final table.
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, Optional

BENCHMARKS = ("TextVQA", "GQA", "POPE", "SQA-IMG", "MMB", "MMMU")
TUNING = ("TextVQA-tune", "GQA-tune")
REFERENCE = {  # LLaVA-1.5-7B as reported (LLaVA-1.5 paper / Fourier-VLM)
    "TextVQA": 58.2, "GQA": 62.0, "POPE": 85.9, "SQA-IMG": 66.8,
    "MMB": 64.3, "MMMU": 35.3}
# Fourier-VLM (arXiv 2508.06038) Table 3, LLaVA-1.5-7B, as reported there
# (retrained models; MQT-LLaVA rows quoted by Fourier-VLM). Order:
# TextVQA, GQA, POPE, SQA-IMG, MMB, MMMU.
PAPER_ROWS = [
    ("Fourier-LLaVA (reported)", 256, (56.0, 62.7, 85.3, 69.9, 66.4, 33.1)),
    ("MQT-LLaVA (reported)", 256, (53.2, 61.6, 84.4, 67.6, 64.3, 34.8)),
    ("Fourier-LLaVA (reported)", 144, (54.7, 61.7, 85.0, 69.0, 65.6, 35.3)),
    ("MQT-LLaVA (reported)", 144, (52.6, 61.4, 83.9, 67.5, 64.4, 34.4)),
    ("Fourier-LLaVA (reported)", 64, (52.6, 60.4, 85.3, 69.3, 64.7, 34.4)),
    ("MQT-LLaVA (reported)", 64, (51.7, 60.0, 83.6, 67.0, 63.5, 34.4)),
    ("Fourier-LLaVA (reported)", 36, (51.0, 59.5, 84.0, 69.0, 64.3, 32.8)),
    ("MQT-LLaVA (reported)", 36, (50.4, 58.8, 81.9, 66.8, 63.4, 34.4)),
]


def _lmms(run: str, bench: str, task: str) -> Optional[dict]:
    files = sorted(glob.glob(os.path.join(run, f"lmms_{bench}", "**",
                                          "results.json"), recursive=True))
    if not files:
        return None
    return json.load(open(files[-1]))["results"].get(task)


def _pope_split_f1(run: str) -> Optional[float]:
    files = sorted(glob.glob(os.path.join(run, "lmms_pope", "**",
                                          "pope.json"), recursive=True))
    if not files:
        return None
    counts: Dict[str, list] = {}
    for log in json.load(open(files[-1]))["logs"]:
        rec = log["pope_f1_score"]
        pred = rec["prediction"].strip().lower().startswith("yes")
        gold = rec["ground_truth"].strip().lower() == "yes"
        c = counts.setdefault(log["doc"]["category"], [0, 0, 0])  # tp fp fn
        c[0] += pred and gold
        c[1] += pred and not gold
        c[2] += gold and not pred
    f1s = []
    for tp, fp, fn in counts.values():
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return 100 * sum(f1s) / len(f1s)


def collect(run: str) -> dict:
    row: Dict[str, object] = {"name": os.path.basename(run.rstrip("/"))}
    cfg_file = os.path.join(run, "model_config.json")
    vc = {}
    if os.path.exists(cfg_file):
        vc = json.load(open(cfg_file)).get("visual_compressor") or {}
    mode = vc.get("compression_mode", "none")
    row["mode"] = mode
    row["tokens"] = vc.get("target_visual_tokens", 576) if mode != "none" \
        else 576
    m = re.search(r"_limit(\d+)$", row["name"])
    row["limit"] = int(m.group(1)) if m else None

    f = os.path.join(run, "textvqa_score.txt")
    if os.path.exists(f):
        acc = re.findall(r"Accuracy:\s*([\d.]+)%", open(f).read())
        if acc:
            row["TextVQA"] = float(acc[-1])
    r = _lmms(run, "gqa", "gqa")
    if r:
        row["GQA"] = 100 * r["exact_match,none"]
    split_f1 = _pope_split_f1(run)
    if split_f1 is not None:
        row["POPE"] = split_f1
    else:
        r = _lmms(run, "pope", "pope")
        if r:
            row["POPE"] = 100 * r["pope_f1_score,none"]
            row["POPE_pooled"] = True
    r = _lmms(run, "scienceqa", "scienceqa_img")
    if r:
        row["SQA-IMG"] = 100 * r["exact_match,none"]
    f = os.path.join(run, "mmbench_score.json")
    if os.path.exists(f):
        row["MMB"] = json.load(open(f))["circular_acc"]
    r = _lmms(run, "mmmu", "mmmu_val")
    if r:
        row["MMMU"] = 100 * r["mmmu_acc,none"]
    for key, fname in (("TextVQA-tune", "textvqa_tune_score.json"),
                       ("GQA-tune", "gqa_tune_score.json")):
        f = os.path.join(run, fname)
        if os.path.exists(f):
            row[key] = json.load(open(f))["accuracy"]
    if all(b in row for b in BENCHMARKS):
        row["Avg"] = sum(row[b] for b in BENCHMARKS) / len(BENCHMARKS)
    return row


def _cell(row: dict, key: str) -> str:
    if key not in row:
        return "-"
    star = "*" if key == "POPE" and row.get("POPE_pooled") else ""
    return f"{row[key]:.1f}{star}"


def paper_rows():
    rows = []
    for name, tokens, scores in PAPER_ROWS:
        r = {"name": name, "tokens": tokens, "limit": None, "mode": "paper",
             **dict(zip(BENCHMARKS, scores))}
        r["Avg"] = sum(scores) / len(scores)
        rows.append(r)
    return rows


def render(rows, fmt: str, reference: bool) -> str:
    cols = ["Method", "Tokens", *BENCHMARKS, "Avg"]
    if any(any(t in r for t in TUNING) for r in rows):
        cols += list(TUNING)
    body = []
    if reference:
        body.append(["LLaVA-1.5-7B (reported)", "576",
                     *[f"{REFERENCE[b]:.1f}" for b in BENCHMARKS],
                     f"{sum(REFERENCE.values()) / 6:.1f}"]
                    + ["-"] * (len(cols) - 9))
    for r in rows:
        name = r["name"] + (f" [LIMIT {r['limit']}]" if r["limit"] else "")
        if r.get("mode") != "paper" and "Avg" in r:      # retention vs 576
            name += f" ({100 * r['Avg'] / (sum(REFERENCE.values()) / 6):.0f}%)"
        body.append([name, str(r["tokens"]),
                     *[_cell(r, c) for c in cols[2:]]])
    if fmt == "csv":
        return "\n".join(",".join(x) for x in [cols] + body)
    if fmt == "latex":
        lines = ["\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}",
                 "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
        lines += [" & ".join(x).replace("_", "\\_") + " \\\\" for x in body]
        lines += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(lines)
    widths = [max(len(x[i]) for x in [cols] + body) for i in range(len(cols))]

    def line(x):
        cells = [x[0].ljust(widths[0])] + [x[i].rjust(widths[i])
                                           for i in range(1, len(x))]
        sep = " | " if fmt == "markdown" else "  "
        out = sep.join(cells)
        return f"| {out} |" if fmt == "markdown" else out

    out = [line(cols)]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|"
               if fmt == "markdown" else "-" * len(line(cols)))
    out += [line(x) for x in body]
    return "\n".join(out)


def log_wandb(rows):
    import wandb

    keys = BENCHMARKS + ("Avg",) + TUNING
    for r in rows:
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "afc"),
                         name=f"eval-{r['name']}", job_type="eval",
                         config={"model": r["name"], "mode": r["mode"],
                                 "visual_tokens": r["tokens"],
                                 "limit": r["limit"]}, reinit=True)
        scores = {f"eval/{k}": r[k] for k in keys if k in r}
        run.summary.update(scores)
        run.log(scores)
        table = wandb.Table(columns=["benchmark", "score"],
                            data=[[k, r[k]] for k in keys if k in r])
        run.log({"eval/table": table})
        run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run_eval.sh output dirs")
    parser.add_argument("--format", default="text",
                        choices=["text", "markdown", "latex", "csv"])
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--paper", action="store_true",
                        help="add Fourier-LLaVA / MQT-LLaVA rows reported in "
                             "Fourier-VLM Table 3")
    parser.add_argument("--names", nargs="*", default=None,
                        help="display names for the runs (same order)")
    parser.add_argument("--wandb", action="store_true",
                        help="log each run's scores to W&B (WANDB_PROJECT, "
                             "default afc; one eval run per directory)")
    parser.add_argument("--sort", default="tokens",
                        choices=["tokens", "name", "none"])
    args = parser.parse_args()
    runs = [r for r in args.runs if os.path.isdir(r)
            and not os.path.basename(r.rstrip("/")) == "logs"]
    rows = [collect(r) for r in runs]
    if args.names:
        for r, n in zip(rows, args.names):
            r["name"] = n
    if args.paper:
        rows += paper_rows()
    if args.sort == "tokens":
        rows.sort(key=lambda r: (-r["tokens"], r["name"]))
    elif args.sort == "name":
        rows.sort(key=lambda r: r["name"])
    print(render(rows, args.format, not args.no_reference))
    if args.wandb:
        log_wandb(rows)
    if any(r["limit"] for r in rows):
        print("\n[LIMIT N] partial runs: not comparable to full-split numbers.")
    if any(r.get("POPE_pooled") for r in rows):
        print("* POPE pooled F1 (per-split log missing).")


if __name__ == "__main__":
    main()
