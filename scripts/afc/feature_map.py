"""Pre-DCT CLIP feature maps of one image (the input of the compressor's DCT).

    python scripts/afc/feature_map.py outputs/figures/cat_fruit_336.png

CLIP ViT-L/14-336 (CPU), hidden_states[-2] without CLS -> 24x24x1024 grid.
Each cell = L2 norm of a patch's 1024-d feature.
  global: the whole 24x24 grid (what the global DCT transforms)
  block : the same grid split into 16 blocks of 6x6 (what each block DCT sees)
Writes <stem>_feat_{global,block,both}.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dct_heatmap import clip_grid  # noqa: E402

CMAP = "Blues"


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else Path(__file__).resolve().parents[2]
                / "outputs/figures/cat_fruit_336.png")
    grid, img = clip_grid(path)
    norm = grid.norm(dim=-1).numpy()                      # (24, 24)
    vmin, vmax = norm.min(), np.percentile(norm, 99.5)
    top = np.argsort(norm.ravel())[::-1][:5]
    print("norm range", norm.min().round(1), norm.max().round(1),
          "| top cells (row, col):", [divmod(int(i), 24) for i in top])

    def draw_global(ax):
        im = ax.imshow(norm, cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_title("Global: 24x24 grid", fontsize=11)
        ax.set_xlabel("patch column")
        ax.set_ylabel("patch row")
        return im

    def draw_block(ax):
        mosaic = np.full((4 * 7 - 1, 4 * 7 - 1), np.nan)
        for r in range(4):
            for c in range(4):
                mosaic[r * 7:r * 7 + 6, c * 7:c * 7 + 6] = \
                    norm[r * 6:(r + 1) * 6, c * 6:(c + 1) * 6]
        cm = plt.get_cmap(CMAP).copy()
        cm.set_bad("white")
        im = ax.imshow(np.ma.masked_invalid(mosaic), cmap=cm, vmin=vmin,
                       vmax=vmax)
        ax.set_title("Block: 16 blocks of 6x6", fontsize=11)
        ax.set_xticks([2.5 + 7 * i for i in range(4)], [f"col {i}" for i in range(4)])
        ax.set_yticks([2.5 + 7 * i for i in range(4)], [f"row {i}" for i in range(4)])
        ax.tick_params(length=0)
        return im

    stem = path.with_suffix("")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle("CLIP features right before the DCT (patch feature L2 norm)", fontsize=12)
    axes[0].imshow(img)
    axes[0].set_title("Input 336x336", fontsize=11)
    axes[0].set_axis_off()
    draw_global(axes[1])
    im = draw_block(axes[2])
    cb = fig.colorbar(im, ax=axes[1:], shrink=0.85, pad=0.02)
    cb.set_label("feature L2 norm (1024-d)")
    fig.savefig(f"{stem}_feat_both.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    for name, fn in (("global", draw_global), ("block", draw_block)):
        f, ax = plt.subplots(figsize=(5.6, 5))
        c = f.colorbar(fn(ax), ax=ax, shrink=0.85)
        c.set_label("feature L2 norm")
        f.savefig(f"{stem}_feat_{name}.png", dpi=200, bbox_inches="tight",
                  facecolor="white")
    print(stem)


if __name__ == "__main__":
    main()
