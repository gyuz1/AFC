"""Check benchmark / tuning data: files present, counts, missing images.

Usage (conda env ``afc``):
    python scripts/check_data.py
"""

import json
from pathlib import Path

import pyarrow.parquet as pq

from llava.eval.afc import AFC_DATA, BENCH, MODEL_PATH, SPLITS, TUNING

ROOT = AFC_DATA

# LLaVA-format question files: (name, jsonl, image dir, expected count)
JSONL_TARGETS = [
    ("textvqa val (ocr)", BENCH / "textvqa/llava_textvqa_val_v051_ocr.jsonl",
     BENCH / "textvqa/train_images", 5000),
    ("textvqa tune (ocr)", SPLITS / "textvqa_train_2000_ocr.jsonl",
     TUNING / "textvqa/train_images", 2000),
    ("gqa tune", SPLITS / "gqa_train_2000.jsonl",
     TUNING / "gqa/images", 2000),
]

# lmms-eval parquet data: (name, dir, glob, expected rows)
PARQUET_TARGETS = [
    ("gqa testdev-bal", BENCH / "gqa/hf",
     "testdev_balanced_instructions/*.parquet", 12578),
    ("pope", BENCH / "pope/hf", "data/*.parquet", 9000),
    ("scienceqa-img test", BENCH / "scienceqa/hf",
     "ScienceQA-IMG/test-*.parquet", 2017),
    ("mmmu val", BENCH / "mmmu/hf", "data/validation-*.parquet", 900),
]


def status(ok):
    return "OK  " if ok else "FAIL"


def check_jsonl(name, qfile, image_dir, expected):
    if not qfile.exists():
        print(f"[MISS] {name}: {qfile}")
        return
    rows = [json.loads(line) for line in open(qfile)]
    missing = [r["image"] for r in rows
               if not (image_dir / r["image"]).exists()]
    ok = len(rows) == expected and not missing
    print(f"[{status(ok)}] {name}: {len(rows)}/{expected} questions, "
          f"{len(missing)} missing images")


def check_parquet(name, root, pattern, expected):
    files = sorted(root.glob(pattern))
    if not files:
        print(f"[MISS] {name}: {root.relative_to(ROOT)}/{pattern}")
        return
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"[{status(n == expected)}] {name}: {n}/{expected} rows")


def check_mmbench():
    tsv = BENCH / "mmbench/mmbench_dev_20230712.tsv"
    if not tsv.exists():
        print("[MISS] mmbench dev tsv")
        return
    import pandas as pd  # questions contain newlines; count real rows

    n = len(pd.read_csv(tsv, sep="\t", usecols=["index"]))
    print(f"[{status(n == 4377)}] mmbench dev: {n}/4377 rows")


def main():
    for target in JSONL_TARGETS:
        check_jsonl(*target)
    for target in PARQUET_TARGETS:
        check_parquet(*target)
    check_mmbench()
    for path in [BENCH / "textvqa/TextVQA_0.5.1_val.json",
                 MODEL_PATH / "config.json"]:
        ok = path.exists()
        print(f"[{'OK  ' if ok else 'MISS'}] {path}")


if __name__ == "__main__":
    main()
