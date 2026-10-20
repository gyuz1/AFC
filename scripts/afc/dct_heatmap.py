"""DCT energy heatmaps of one image's CLIP features (what the compressor sees).

    python scripts/afc/dct_heatmap.py outputs/figures/cat_fruit_336.png

CLIP ViT-L/14-336 (CPU), hidden_states[-2] without CLS -> 24x24x1024 grid.
  global: one 24x24 DCT-II over the whole grid (AFC global / Fourier-VLM)
  block : 16 blocks of 6x6, a 6x6 DCT-II in each (AFC block)
Energy per coefficient = mean over the 1024 channels of coef^2, shown on a
log10 scale (DC at the top-left of every map). Writes <stem>_dct_*.png.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str((REPO / "../afc_common/data/hf_cache").resolve()))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import CLIPImageProcessor, CLIPVisionModel  # noqa: E402

from llava.model.visual_compressor.dct import dct2, dct_matrix  # noqa: E402

CLIP = "openai/clip-vit-large-patch14-336"
CMAP = "Reds"
MARK = "#1565C0"   # annotations: blue, distinct from the red ramp


@torch.no_grad()
def clip_grid(path):
    proc = CLIPImageProcessor.from_pretrained(CLIP)
    model = CLIPVisionModel.from_pretrained(CLIP).eval()
    img = Image.open(path).convert("RGB")      # already square-padded 336
    px = proc(img, return_tensors="pt")["pixel_values"]
    hs = model(px, output_hidden_states=True).hidden_states[-2][0, 1:]
    return hs.reshape(24, 24, -1).float(), img


def energy(coef):                               # (..., u, v, D) -> (..., u, v)
    return coef.pow(2).mean(-1)


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else REPO / "outputs/figures/cat_fruit_336.png")
    grid, img = clip_grid(path)

    g = np.log10(energy(dct2(grid, dct_matrix(24))).numpy())
    blocks = grid.reshape(4, 6, 4, 6, -1).permute(0, 2, 1, 3, 4)  # (4,4,6,6,D)
    b = np.log10(energy(dct2(blocks, dct_matrix(6))).numpy())     # (4,4,6,6)
    vmin, vmax = min(g.min(), b.min()), max(g.max(), b.max())

    def draw_global(ax):
        im = ax.imshow(g, cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_title("Global DCT (24x24)", fontsize=11)
        ax.set_xlabel("horizontal frequency v")
        ax.set_ylabel("vertical frequency u")
        return im

    def draw_block(ax):
        mosaic = np.full((4 * 7 - 1, 4 * 7 - 1), np.nan)
        for r in range(4):
            for c in range(4):
                mosaic[r * 7:r * 7 + 6, c * 7:c * 7 + 6] = b[r, c]
        cm = plt.get_cmap(CMAP).copy()
        cm.set_bad("white")
        im = ax.imshow(np.ma.masked_invalid(mosaic), cmap=cm, vmin=vmin,
                       vmax=vmax)
        ax.set_title("Block DCT (16 blocks x 6x6)", fontsize=11)
        ax.set_xticks([2.5 + 7 * i for i in range(4)], [f"col {i}" for i in range(4)])
        ax.set_yticks([2.5 + 7 * i for i in range(4)], [f"row {i}" for i in range(4)])
        ax.tick_params(length=0)
        return im

    def draw_image(ax, grid_step):
        ax.imshow(img)
        for k in range(0, 337, grid_step):
            ax.axhline(k - 0.5, color="white", lw=0.4 if grid_step == 14 else 1.2)
            ax.axvline(k - 0.5, color="white", lw=0.4 if grid_step == 14 else 1.2)
        ax.set_axis_off()

    stem = path.with_suffix("")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2),
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    draw_image(axes[0], 14)
    for k in range(0, 337, 84):
        axes[0].axhline(k - 0.5, color=MARK, lw=1.4)
        axes[0].axvline(k - 0.5, color=MARK, lw=1.4)
    axes[0].set_title("Input 336x336: 24x24 CLIP patches\n(blue = 6x6-patch blocks)",
                      fontsize=11)
    draw_global(axes[1])
    im = draw_block(axes[2])
    cb = fig.colorbar(im, ax=axes[1:], shrink=0.85, pad=0.02)
    cb.set_label("log10 energy (mean over 1024 channels)")
    fig.savefig(f"{stem}_dct_both.png", dpi=200, bbox_inches="tight", facecolor="white")

    for name, fn in (("global", draw_global), ("block", draw_block)):
        f, ax = plt.subplots(figsize=(5.6, 5))
        im = fn(ax)
        c = f.colorbar(im, ax=ax, shrink=0.85)
        c.set_label("log10 energy")
        f.savefig(f"{stem}_dct_{name}.png", dpi=200, bbox_inches="tight",
                  facecolor="white")

    tot = 10 ** g
    print(f"global: DC share {tot[0, 0] / tot.sum():.1%}, "
          f"8x8 low-freq share {tot[:8, :8].sum() / tot.sum():.1%}")
    bt = 10 ** b
    print(f"block : DC share (mean over blocks) "
          f"{(bt[..., 0, 0] / bt.sum((-1, -2))).mean():.1%}")
    print(stem)


if __name__ == "__main__":
    main()
