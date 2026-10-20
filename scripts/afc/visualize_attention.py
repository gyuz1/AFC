"""Query-frequency affinity maps of a trained block-DCT compressor.

Attention weights show how the pooling queries use the frequency tokens
(utilisation patterns); they are not causal importance.

    python scripts/afc/visualize_attention.py \
        --model-path checkpoints/llava-v1.5-7b-afc-main --n 32 \
        --out outputs/figures/attention_main

Writes:
    per_query.png        mean 6x6 frequency map of each query (blocks+images)
    block_grid_<i>.png   4x4 block grid of one image, each cell a 6x6 map
                         (head- and query-averaged)
    summary.json         diagnostics + low-frequency / horizontal / vertical
                         attention mass
"""

import argparse
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llava.eval.afc import BENCH  # noqa: E402
from llava.eval.afc.llava_io import build_inputs, load_model  # noqa: E402
from llava.model.visual_compressor.diagnostics import summarize  # noqa: E402

QUESTIONS = BENCH / "textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGES = BENCH / "textvqa/train_images"


@torch.inference_mode()
def collect(model, tok, proc, rows):
    comp = model.get_model().get_visual_compressor()
    if comp is None or comp.pool is None:
        raise SystemExit("model has no learned-query compressor")
    vt = model.get_vision_tower()
    attn, pixels_kept = [], []
    for row in rows:
        _, pixels = build_inputs(tok, model, proc, IMAGES / row["image"],
                                 row["text"])
        comp(vt(pixels), return_attention=True)
        attn.append(comp.last_attention.float().cpu())   # (1, J, H, k, N)
        pixels_kept.append(pixels[0].float().cpu())
    return comp, torch.cat(attn), pixels_kept


def frequency_stats(a: torch.Tensor, b: int) -> dict:
    """``a`` (..., b*b) attention -> mass split by frequency region."""
    grid = a.reshape(*a.shape[:-1], b, b)
    u = torch.arange(b)[:, None].expand(b, b)
    v = torch.arange(b)[None, :].expand(b, b)
    low = (u + v) < b // 2
    return {
        "low_freq_mass(u+v<b/2)": grid[..., low].sum(-1).mean().item(),
        "dc_mass": grid[..., 0, 0].mean().item(),
        "vertical_only_mass(u>0,v=0)": grid[..., (u > 0) & (v == 0)].sum(
            -1).mean().item(),
        "horizontal_only_mass(u=0,v>0)": grid[..., (u == 0) & (v > 0)].sum(
            -1).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--grids", type=int, default=4,
                        help="images to draw as block grids")
    parser.add_argument("--out", default="outputs/figures/attention")
    args = parser.parse_args()

    tok, model, proc = load_model(args.model_path)
    with open(QUESTIONS) as f:
        rows = [json.loads(next(f)) for _ in range(args.n)]
    comp, attn, pixels = collect(model, tok, proc, rows)
    cfg = comp.cfg
    if cfg.compression_mode not in ("block_dct", "global_dct"):
        print(f"[warn] {cfg.compression_mode}: inner axes are spatial, "
              "not frequency")
    b, k, m = cfg.effective_block_size, cfg.queries_per_block, \
        cfg.blocks_per_side
    a = attn.mean(dim=2)                                  # (I, J, k, N)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_query = a.mean(dim=(0, 1)).reshape(k, b, b)       # (k, b, b)
    cols = min(k, 8)
    rows_n = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.4 * cols,
                                                    2.4 * rows_n),
                             squeeze=False)
    for q in range(rows_n * cols):
        ax = axes[q // cols][q % cols]
        ax.axis("off")
        if q < k:
            ax.imshow(per_query[q], cmap="viridis")
            ax.set_title(f"query {q}")
    fig.suptitle("mean query-frequency affinity (u down, v right)")
    fig.tight_layout()
    fig.savefig(out / "per_query.png", dpi=120)
    plt.close(fig)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    for i in range(min(args.grids, len(pixels))):
        fig, axes = plt.subplots(m, m + 1, figsize=(2 * (m + 1), 2 * m),
                                 squeeze=False)
        img = (pixels[i] * std + mean).clamp(0, 1).permute(1, 2, 0)
        for r in range(m):
            axes[r][0].axis("off")
            for c in range(m):
                ax = axes[r][c + 1]
                ax.imshow(a[i, r * m + c].mean(0).reshape(b, b),
                          cmap="viridis")
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0][0].imshow(img)
        fig.suptitle(f"image {i}: per-block frequency affinity "
                     "(query mean)")
        fig.tight_layout()
        fig.savefig(out / f"block_grid_{i}.png", dpi=110)
        plt.close(fig)

    summary = {"model": Path(args.model_path).name, "images": len(rows),
               **summarize(attn), **frequency_stats(a, b),
               "per_query": {f"q{q}": frequency_stats(a[:, :, q], b)
                             for q in range(k)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k_: v for k_, v in summary.items()
                      if k_ != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
