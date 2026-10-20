"""Build fixed tuning subsets from TextVQA train and GQA train.

Writes LLaVA ``model_vqa_loader`` question jsonl + ground-truth jsonl to
``splits/`` (versioned). At most one question per image; image ids that
appear in the evaluation split (TextVQA val, GQA testdev) are rejected.
Sampled GQA images are extracted to ``data/tuning/gqa/images``.

Usage (conda env ``afc``):
    python scripts/make_tuning_split.py --n 2000 --seed 0
"""

import argparse
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from llava.eval.afc import BENCH, SPLITS, TUNING
SHORT_ANSWER = "Answer the question using a single word or phrase."


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_one_per_image(items, key, n, seed):
    rng = random.Random(seed)
    items = sorted(items, key=lambda d: str(d["question_id"]))
    rng.shuffle(items)
    seen, picked = set(), []
    for item in items:
        if item[key] in seen:
            continue
        seen.add(item[key])
        picked.append(item)
        if len(picked) == n:
            break
    return picked


def textvqa_split(n, seed):
    train = json.load(
        open(TUNING / "textvqa" / "TextVQA_0.5.1_train.json"))["data"]
    val = json.load(open(BENCH / "textvqa" / "TextVQA_0.5.1_val.json"))["data"]
    val_images = {d["image_id"] for d in val}
    # OCR tokens live in a separate Rosetta file; LLaVA's val jsonl uses
    # Rosetta v0.2 verbatim (5000/5000 match), incl. an empty line.
    ocr = {d["image_id"]: d["ocr_tokens"] for d in json.load(open(
        TUNING / "textvqa" / "TextVQA_Rosetta_OCR_v0.2_train.json"))["data"]}

    train = [d for d in train if d["image_id"] not in val_images]
    picked = sample_one_per_image(train, "image_id", n, seed)
    n_ocr = sum(bool(ocr.get(d["image_id"])) for d in picked)
    print(f"textvqa: {len(picked)} picked, {n_ocr} with ocr_tokens")

    for use_ocr in (True, False):
        questions = []
        for d in picked:
            text = d["question"]
            # Same layout as llava_textvqa_val_v051_ocr.jsonl.
            if use_ocr:
                tokens = ocr.get(d["image_id"], [])
                text += "\nReference OCR token: " + ", ".join(tokens)
            questions.append({
                "question_id": d["question_id"],
                "image": f"{d['image_id']}.jpg",
                "text": text + "\n" + SHORT_ANSWER,
                "category": "default",
            })
        tag = "ocr" if use_ocr else "noocr"
        write_jsonl(SPLITS / f"textvqa_train_{n}_{tag}.jsonl", questions)

    gts = [{"question_id": d["question_id"], "answers": d["answers"]}
           for d in picked]
    write_jsonl(SPLITS / f"textvqa_train_{n}_gt.jsonl", gts)


def read_parquets(pattern, columns):
    rows = []
    for path in sorted(pattern[0].glob(pattern[1])):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows


def gqa_split(n, seed):
    hf_train = TUNING / "gqa" / "hf"
    hf_testdev = BENCH / "gqa" / "hf"
    testdev_images = {
        r["imageId"] for r in read_parquets(
            (hf_testdev, "testdev_balanced_instructions/*.parquet"),
            ["imageId"])
    }
    train = read_parquets(
        (hf_train, "train_balanced_instructions/*.parquet"),
        ["id", "imageId", "question", "answer"])
    items = [{"question_id": r["id"], **r} for r in train
             if r["imageId"] not in testdev_images]
    picked = sample_one_per_image(items, "imageId", n, seed)

    questions = [{
        "question_id": q["question_id"],
        "image": f"{q['imageId']}.jpg",
        "text": q["question"] + "\n" + SHORT_ANSWER,
        "category": "default",
    } for q in picked]
    gts = [{"question_id": q["question_id"], "answer": q["answer"]}
           for q in picked]
    write_jsonl(SPLITS / f"gqa_train_{n}.jsonl", questions)
    write_jsonl(SPLITS / f"gqa_train_{n}_gt.jsonl", gts)

    extract_gqa_images({q["imageId"] for q in picked}, hf_train)
    print(f"gqa: {len(questions)} questions")


def extract_gqa_images(wanted, hf_train):
    out_dir = TUNING / "gqa" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = {i for i in wanted if not (out_dir / f"{i}.jpg").exists()}
    for path in sorted(hf_train.glob("train_balanced_images/*.parquet")):
        if not todo:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                if row["id"] not in todo:
                    continue
                raw = row["image"]["bytes"]
                out = out_dir / f"{row['id']}.jpg"
                if raw[:3] == b"\xff\xd8\xff":  # keep original JPEG bytes
                    out.write_bytes(raw)
                else:
                    Image.open(io.BytesIO(raw)).convert("RGB").save(
                        out, quality=95)
                todo.discard(row["id"])
    if todo:
        print(f"[warn] {len(todo)} GQA images not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    SPLITS.mkdir(parents=True, exist_ok=True)
    textvqa_split(args.n, args.seed)
    gqa_split(args.n, args.seed)


if __name__ == "__main__":
    main()
