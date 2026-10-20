"""Local scorers: MMBench dev CircularEval, tuning splits (TextVQA / GQA).

Usage:
    python -m llava.eval.afc.metrics mmbench --annotation <tsv> --result <jsonl>
    python -m llava.eval.afc.metrics textvqa_tune --gt <gt.jsonl> --result <jsonl>
    python -m llava.eval.afc.metrics gqa_tune --gt <gt.jsonl> --result <jsonl>
"""

import argparse
import json
import re

import pandas as pd

CIRCULAR_OFFSET = 1_000_000
OPTIONS = "ABCD"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_choice(text: str, row) -> str:
    """Option letter from a free-form answer; '' if none can be matched."""
    text = text.strip()
    m = re.match(r"^\(?([A-D])(?:[\).:\s]|$)", text)
    if m:
        return m.group(1)
    for letter in OPTIONS:  # answer given as option text
        opt = row.get(letter)
        if isinstance(opt, str) and text.lower() == opt.strip().lower():
            return letter
    return ""


def mmbench_circular(annotation: str, result: str) -> dict:
    """CircularEval: a base question counts only if all rotations are right."""
    ann = pd.read_csv(annotation, sep="\t")
    ann = ann.set_index("index")
    preds = {r["question_id"]: r["text"] for r in read_jsonl(result)}
    rows = []
    for idx, row in ann.iterrows():
        if idx not in preds:
            continue
        choice = extract_choice(preds[idx], row)
        rows.append({"base": idx % CIRCULAR_OFFSET, "ok": choice == row["answer"],
                     "category": row["category"], "l2": row["l2-category"],
                     "parsed": choice != ""})
    df = pd.DataFrame(rows)
    per_q = df.groupby("base").agg(ok=("ok", "all"), l2=("l2", "first"))
    return {
        "n_rows": len(df), "n_missing": len(ann) - len(df),
        "n_questions": len(per_q), "unparsed": int((~df["parsed"]).sum()),
        "circular_acc": 100 * per_q["ok"].mean(),
        "single_pass_acc": 100 * df["ok"].mean(),
        "l2_acc": (100 * per_q.groupby("l2")["ok"].mean()).round(2).to_dict(),
    }


def textvqa_accuracy(gt: str, result: str) -> dict:
    """Standard VQA accuracy with the TextVQA (m4c) answer processor."""
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator

    answers = {r["question_id"]: r["answers"] for r in read_jsonl(gt)}
    preds = read_jsonl(result)
    items = [{"pred_answer": p["text"], "gt_answers": answers[p["question_id"]]}
             for p in preds if p["question_id"] in answers]
    acc = TextVQAAccuracyEvaluator().eval_pred_list(items)
    return {"n": len(items), "accuracy": 100 * acc}


def _gqa_norm(text: str) -> str:
    text = text.strip().lower().rstrip(".")
    return re.sub(r"\s+", " ", text)


def gqa_accuracy(gt: str, result: str) -> dict:
    answers = {r["question_id"]: r["answer"] for r in read_jsonl(gt)}
    preds = read_jsonl(result)
    ok = [_gqa_norm(p["text"]) == _gqa_norm(answers[p["question_id"]])
          for p in preds if p["question_id"] in answers]
    return {"n": len(ok), "accuracy": 100 * sum(ok) / max(len(ok), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["mmbench", "textvqa_tune",
                                         "gqa_tune"])
    parser.add_argument("--annotation")
    parser.add_argument("--gt")
    parser.add_argument("--result", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.task == "mmbench":
        res = mmbench_circular(args.annotation, args.result)
    elif args.task == "textvqa_tune":
        res = textvqa_accuracy(args.gt, args.result)
    else:
        res = gqa_accuracy(args.gt, args.result)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
