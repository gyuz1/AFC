"""Benchmark accuracy split by how much of the square-padded input is padding.

LLaVA-1.5 pads every image to a square with the CLIP mean colour before the
336x336 resize, so a W x H image (W >= H) spends a fraction 1 - H/W of the
24x24 visual tokens on padding. Per-sample results of the existing
evaluations are grouped into padding bins; retention = model / vanilla-576
accuracy inside the same bin (controls for per-bin difficulty).

    python scripts/afc/padding_analysis.py            # default runs
    python scripts/afc/padding_analysis.py --runs llava-v1.5-7b-afc-global_all

Benchmarks: TextVQA (VQA acc), GQA (EM), POPE (accuracy; the paper metric is
F1, accuracy is used here because F1 is not defined per sample), ScienceQA-IMG
(EM). MMBench / MMMU are left out (circular / multi-image).
"""

import argparse
import ast
import glob
import io
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from llava.eval.eval_textvqa import prompt_processor
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator

REPO = Path(__file__).resolve().parents[2]
DATA = (REPO / "../afc_common/data").resolve()
HUB = DATA / "hf_cache/hub"
BINS = [("square (<5%)", 0.0, 0.05), ("4:3-like (5-30%)", 0.05, 0.30),
        ("3:2 and wider (>=30%)", 0.30, 1.01)]
RUNS = ["llava-v1.5-7b", "llava-v1.5-7b-afc-fourier_vlm64",
        "llava-v1.5-7b-afc-avgpool64", "llava-v1.5-7b-afc-block75k_fp",
        "llava-v1.5-7b-afc-global75k_fp"]


def pad_frac(w, h):
    return 1.0 - min(w, h) / max(w, h)


def size_of(blob):
    if isinstance(blob, dict):
        blob = blob["bytes"]
    return Image.open(io.BytesIO(blob)).size


def parquet_sizes(pattern, key):
    out = {}
    for f in sorted(glob.glob(str(HUB / pattern))):
        t = pq.read_table(f, columns=[key, "image"]).to_pydict()
        for k, im in zip(t[key], t["image"]):
            if k not in out and im is not None:
                out[k] = size_of(im)
    return out


def sqa_sizes():
    f = sorted(glob.glob(str(HUB / "datasets--lmms-lab--ScienceQA/snapshots/*/"
                                   "ScienceQA-IMG/test-*.parquet")))[0]
    return [size_of(im) for im in pq.read_table(f, columns=["image"])
            .column("image").to_pylist()]


def as_obj(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def lmms_logs(run, bench):
    f = [x for x in sorted(glob.glob(str(REPO / f"outputs/eval/{run}/lmms_{bench}/*/*.json")))
         if not x.endswith("results.json")]
    return json.load(open(f[-1]))["logs"] if f else None


def collect(run, cache):
    """(benchmark, pad_frac, score) per sample."""
    rows = []
    tv = REPO / f"outputs/eval/{run}/textvqa_answers.jsonl"
    if tv.exists():
        ann = {(a["image_id"], a["question"].lower()): a for a in
               json.load(open(DATA / "benchmarks/textvqa/TextVQA_0.5.1_val.json"))["data"]}
        ev = TextVQAAccuracyEvaluator()
        for line in open(tv):
            r = json.loads(line)
            a = ann[(r["question_id"], prompt_processor(r["prompt"]))]
            if a["image_id"] not in cache["textvqa"]:
                cache["textvqa"][a["image_id"]] = Image.open(
                    DATA / f"benchmarks/textvqa/train_images/{a['image_id']}.jpg").size
            s = ev._compute_answer_scores(a["answers"]).get(
                ev.answer_processor(r["text"]), 0.0)
            rows.append(("TextVQA", pad_frac(*cache["textvqa"][a["image_id"]]), s))
    logs = lmms_logs(run, "gqa")
    if logs:
        for r in logs:
            d = as_obj(r["doc"])
            rows.append(("GQA", pad_frac(*cache["gqa"][d["imageId"]]),
                         float(r["exact_match"])))
    logs = lmms_logs(run, "pope")
    if logs:
        for r in logs:
            d = as_obj(r["doc"])
            s = as_obj(r["pope_accuracy"])["score"]
            rows.append(("POPE", pad_frac(*cache["pope"][d["id"]]), float(s)))
    logs = lmms_logs(run, "scienceqa")
    if logs:
        for r in logs:
            rows.append(("SQA-IMG", pad_frac(*cache["sqa"][int(r["doc_id"])]),
                         float(r["exact_match"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--out", default=str(REPO / "outputs/report/padding_analysis.md"))
    args = ap.parse_args()
    cache = {"textvqa": {},
             "gqa": parquet_sizes("datasets--lmms-lab--GQA/snapshots/*/"
                                  "testdev_balanced_images/*.parquet", "id"),
             "pope": parquet_sizes("datasets--lmms-lab--POPE/snapshots/*/data/*.parquet", "id"),
             "sqa": sqa_sizes()}
    acc = {}                       # run -> bench -> bin -> [sum, n]
    for run in args.runs:
        agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        for bench, p, s in collect(run, cache):
            for name, lo, hi in BINS:
                if lo <= p < hi:
                    agg[bench][name][0] += s
                    agg[bench][name][1] += 1
        acc[run] = agg
    base = args.runs[0]
    lines = ["# Accuracy by padding fraction (per-sample results of existing evals)", "",
             f"retention = model / `{base}` in the same bin", ""]
    benches = [b for b in ("TextVQA", "GQA", "POPE", "SQA-IMG") if b in acc[base]]
    lines.append("| run | bench | " + " | ".join(n for n, _, _ in BINS) + " |")
    lines.append("|---|---|" + "---|" * len(BINS))
    for b in benches:
        lines.append(f"| (samples) | {b} | " + " | ".join(
            str(acc[base][b][n][1]) for n, _, _ in BINS) + " |")
    for run in args.runs:
        means = []
        for b in benches:
            cells = []
            for n, _, _ in BINS:
                s, c = acc[run][b][n]
                bs, bc = acc[base][b][n]
                if not c:
                    cells.append("-")
                    continue
                a = 100 * s / c
                r = (s / c) / (bs / bc) * 100 if bs else float("nan")
                cells.append(f"{a:.1f}" + ("" if run == base else f" ({r:.0f}%)"))
            lines.append(f"| {run.replace('llava-v1.5-7b-afc-', '')} | {b} | "
                         + " | ".join(cells) + " |")
        # mean retention over benchmarks per bin
        if run != base:
            cells = []
            for n, _, _ in BINS:
                rs = [(acc[run][b][n][0] / acc[run][b][n][1])
                      / (acc[base][b][n][0] / acc[base][b][n][1])
                      for b in benches if acc[run][b][n][1] and acc[base][b][n][0]]
                cells.append(f"**{100 * sum(rs) / len(rs):.1f}%**" if rs else "-")
            lines.append(f"| {run.replace('llava-v1.5-7b-afc-', '')} | mean retention | "
                         + " | ".join(cells) + " |")
    text = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
