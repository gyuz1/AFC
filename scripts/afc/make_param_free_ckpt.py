"""Model directory for a parameter-free compressor (no training needed).

    python scripts/afc/make_param_free_ckpt.py --mode fourier_vlm --tokens 64
    python scripts/afc/make_param_free_ckpt.py --mode avgpool --tokens 64

Writes checkpoints/llava-v1.5-7b-afc-<mode><tokens>/ = base config + the
compressor entry + an empty visual_compressor.bin, other files linked to the
base model; evaluate it with scripts/afc/run_eval.sh like any checkpoint.
"""

import argparse
import json
import os
from pathlib import Path

import torch

from llava.eval.afc import MODEL_PATH
from llava.model.visual_compressor import VisualCompressorConfig

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fourier_vlm", "avgpool", "prumerge"])
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--no-fourier-scale", action="store_true")
    ap.add_argument("--base", default=str(MODEL_PATH))
    args = ap.parse_args()
    base = Path(args.base).resolve()
    tag = f"{args.mode}{args.tokens}" + ("_noscale" if args.no_fourier_scale else "")
    out = REPO / f"checkpoints/llava-v1.5-7b-afc-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = VisualCompressorConfig(compression_mode=args.mode,
                                 target_visual_tokens=args.tokens,
                                 fourier_scale=not args.no_fourier_scale)
    cfg.resolve(24, 1024)
    conf = json.load(open(base / "config.json"))
    conf["visual_compressor"] = cfg.to_dict()
    json.dump(conf, open(out / "config.json", "w"), indent=2)
    torch.save({}, out / "visual_compressor.bin")
    for name in os.listdir(base):
        dst = out / name
        if name not in ("config.json", "README.md") and not dst.exists():
            dst.symlink_to(os.path.relpath(base / name, out))
    print(out)


if __name__ == "__main__":
    main()
