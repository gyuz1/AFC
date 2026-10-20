"""Latency / memory / FLOPs of one LLaVA model directory on one GPU.

Measured per image (batch size 1, fp16, TextVQA val images, fixed order):
    visual_tokens  tokens inserted for the image (576 vanilla)
    encode_ms      vision tower + compressor + projector (``encode_images``)
    compressor_ms  compressor alone on precomputed CLIP features (0 vanilla)
    prefill_ms     full prefill forward = time to first token (TTFT)
    generate_ms    greedy generation of exactly ``--new-tokens`` tokens
    peak_mem_mb    peak allocated memory during generate
FLOPs of the prefill come from calflops (reported as-is, or the error).
All timings use torch.cuda.synchronize() after ``--warmup`` untimed images.
Compare configurations only when measured on the same GPU.

    CUDA_VISIBLE_DEVICES=0 python -m llava.eval.afc.efficiency \
        --model-path <model dir> --n 50 --out outputs/efficiency
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from llava.eval.afc import BENCH, MODEL_PATH
from llava.eval.afc.llava_io import build_inputs, load_model

QUESTIONS = BENCH / "textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGES = BENCH / "textvqa/train_images"


def _timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3


def count_flops(model, ids, pixels):
    try:
        from calflops import calculate_flops

        flops, macs, _ = calculate_flops(
            model=model, kwargs={"input_ids": ids, "images": pixels},
            print_results=False, output_as_string=False)
        return {"prefill_flops": flops, "prefill_macs": macs}
    except Exception as exc:  # report, never fabricate
        return {"flops_error": repr(exc)}


@torch.inference_mode()
def measure(model, tok, proc, rows, warmup, new_tokens):
    compressor = model.get_model().get_visual_compressor()
    vision_tower = model.get_vision_tower()
    rec = []
    for i, row in enumerate(rows):
        ids, pixels = build_inputs(tok, model, proc, IMAGES / row["image"],
                                   row["text"])
        emb, enc = _timed(lambda: model.encode_images(pixels))
        comp = 0.0
        if compressor is not None:
            feats = vision_tower(pixels)
            _, comp = _timed(lambda: compressor(feats))
        _, pre = _timed(lambda: model(input_ids=ids, images=pixels,
                                      use_cache=True))
        torch.cuda.reset_peak_memory_stats()
        _, gen = _timed(lambda: model.generate(
            ids, images=pixels, do_sample=False, max_new_tokens=new_tokens,
            min_new_tokens=new_tokens, use_cache=True))
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        if i >= warmup:
            rec.append((emb.shape[1], enc, comp, pre, gen, peak))
    return np.asarray(rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--out", default="outputs/efficiency")
    args = parser.parse_args()

    tok, model, proc = load_model(args.model_path)
    with open(QUESTIONS) as f:
        rows = [json.loads(next(f)) for _ in range(args.n + args.warmup)]
    rec = measure(model, tok, proc, rows, args.warmup, args.new_tokens)

    compressor = model.get_model().get_visual_compressor()
    ids, pixels = build_inputs(tok, model, proc, IMAGES / rows[0]["image"],
                               rows[0]["text"])
    name = Path(args.model_path).name
    summary = {
        "model": name, "gpu": torch.cuda.get_device_name(0),
        "compression_mode": "none" if compressor is None
        else compressor.cfg.compression_mode,
        "n_images": len(rec), "new_tokens": args.new_tokens,
        "visual_tokens": float(rec[:, 0].mean()),
        "compressor_params": 0 if compressor is None
        else sum(p.numel() for p in compressor.parameters()),
        "encode_ms": float(rec[:, 1].mean()),
        "compressor_ms": float(rec[:, 2].mean()),
        "prefill_ms_ttft": float(rec[:, 3].mean()),
        "generate_ms": float(rec[:, 4].mean()),
        "peak_mem_mb": float(rec[:, 5].max()),
        **count_flops(model, ids, pixels),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
