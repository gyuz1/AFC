"""Paper figure: what the two ablations change (schematic, side by side).

Generic shapes: N x N CLIP grid (N=24), feature dim D (1024), K output tokens
(64/144/256). 'w/o padding' applies the DCT to the image area only (fewer
than N^2 tokens, e.g. 16 x 24 patches for a 3:2 image).

    python scripts/afc/plot_ablation_diagram.py   # outputs/report/ablation_diagram.{pdf,png}

Left pair : w/ KD vs. w/o KD    (training objective; inference path identical)
Right pair: w/ padding vs. w/o padding  (which CLIP tokens enter the DCT)
Style follows a two-panel 'w/ X | w/o X' schematic with tensor shapes in grey.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/report"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})

FROZEN, FROZEN_E = "#9fb4e3", "#6f86c4"         # blue boxes (frozen)
TRAIN, TRAIN_E = "#f6c48f", "#d9893a"           # orange (trainable AFC)
LOSS, LOSS_E = "#fde9c8", "#d9893a"
GHOST = "#d9d9d9"
SHAPE = "#8c8c8c"
INK = "#1a1a1a"
FS, FS_S = 6.3, 5.4


def box(ax, x, y, w, h, text, fc=FROZEN, ec=FROZEN_E, ls="-", color=INK, fs=FS, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.0,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.7, ls=ls, alpha=alpha))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-", lw=0.7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=6))


def shape(ax, x, y, s, ha="left"):
    ax.text(x, y, s, fontsize=FS_S, color=SHAPE, style="italic", ha=ha, va="center")


def frame(ax, x0, x1, y0, y1, label):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.0,rounding_size=0.12",
                                fc="none", ec="#4a5f9c", lw=0.8))
    ax.text(x0 - 0.06, (y0 + y1) / 2, label, rotation=90, ha="right", va="center",
            fontsize=FS, color=INK)


def title(ax, x, y, s):
    ax.text(x, y, s, ha="center", va="center", fontsize=FS + 0.6,
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec=INK, lw=0.7))


# ------------------------------------------------------------------ KD pair
def kd_panel(ax, cx, with_kd):
    title(ax, cx + 0.35, 6.55, "w/ KD" if with_kd else "w/o KD")
    xs = cx                                   # student column
    ax.text(xs, 0.0, r"$\mathbf{x}$", ha="center", va="center", fontsize=FS + 1)
    shape(ax, xs + 0.12, 0.0, "336×336")
    arrow(ax, xs, 0.18, xs, 0.62)
    box(ax, xs, 0.85, 1.25, 0.42, "CLIP ViT")
    arrow(ax, xs, 1.06, xs, 1.5)
    shape(ax, xs - 0.08, 1.3, "576×1024", ha="right")
    box(ax, xs, 1.75, 1.25, 0.42, "AFC", fc=TRAIN, ec=TRAIN_E)
    arrow(ax, xs, 1.96, xs, 2.4)
    shape(ax, xs - 0.08, 2.18, "64×1024", ha="right")
    box(ax, xs, 2.65, 1.25, 0.42, "Projector")
    arrow(ax, xs, 2.86, xs, 3.3)
    box(ax, xs, 3.55, 1.25, 0.42, "LLaVA-1.5")
    arrow(ax, xs, 3.76, xs, 4.2)
    ax.text(xs, 4.38, r"$p_S$", ha="center", va="center", fontsize=FS + 1)
    shape(ax, xs - 0.2, 4.38, "student", ha="right")

    xt = cx + 1.45                             # teacher column
    if with_kd:
        ax.plot([xs, xs + 0.3, xt], [1.16, 1.16, 1.16], color=INK, lw=0.7)
        arrow(ax, xt, 1.16, xt, 2.4)
        box(ax, xt, 2.65, 1.0, 0.42, "Projector")
        arrow(ax, xt, 2.86, xt, 3.3)
        box(ax, xt, 3.55, 1.0, 0.42, "LLaVA-1.5")
        arrow(ax, xt, 3.76, xt, 4.2)
        ax.text(xt, 4.38, r"$p_T$", ha="center", va="center", fontsize=FS + 1)
        shape(ax, xt + 0.2, 4.38, "teacher", ha="left")
        shape(ax, xt + 0.06, 1.8, "576×1024\n(no AFC)", ha="left")
        arrow(ax, xs, 4.55, cx + 0.72, 5.1)
        arrow(ax, xt, 4.55, cx + 0.72, 5.1)
        box(ax, cx + 0.72, 5.42, 2.6, 0.55,
            r"$\mathcal{L}_{CE} + \lambda\,\mathcal{L}_{KD}$",
            fc=LOSS, ec=LOSS_E)
    else:
        box(ax, xt, 2.65, 1.0, 0.42, "", fc="none", ec=GHOST, ls="--")
        box(ax, xt, 3.55, 1.0, 0.42, "", fc="none", ec=GHOST, ls="--")
        ax.text(xt, 3.1, "no teacher", ha="center", va="center", fontsize=FS_S,
                color=SHAPE, style="italic")
        arrow(ax, xs, 4.55, cx + 0.72, 5.1)
        box(ax, cx + 0.72, 5.42, 2.6, 0.55, r"$\mathcal{L}_{CE}$ only",
            fc=LOSS, ec=LOSS_E)
    frame(ax, cx - 0.8, cx + 2.2, 0.45, 5.85, "")


# ------------------------------------------------------------- padding pair
def grid(ax, x0, y0, cell, rows, cols, pad_rows=(), cut=False):
    for r in range(rows):
        for c in range(cols):
            is_pad = r in pad_rows
            if cut and is_pad:
                continue
            ax.add_patch(Rectangle((x0 + c * cell, y0 + (rows - 1 - r) * cell), cell, cell,
                                   fc="#c9c9c9" if is_pad else "#7e98d8",
                                   ec="white", lw=0.15))


def pad_panel(ax, cx, with_pad):
    title(ax, cx, 6.55, "w/ padding" if with_pad else "w/o padding")
    cell = 0.055                               # 24 cells -> 1.32 wide
    n = 24
    x0 = cx - n * cell / 2
    pads = list(range(4)) + list(range(20, 24))
    grid(ax, x0, 0.25, cell, n, n, pad_rows=pads, cut=not with_pad)
    if not with_pad:                          # outline of the removed padding rows
        for yy in (0.25, 0.25 + 20 * cell):
            ax.add_patch(Rectangle((x0, yy), n * cell, 4 * cell, fc="none", ec=GHOST,
                                   lw=0.6, ls="--"))
    ax.text(cx, 0.08, "CLIP features (24×24)", ha="center", va="center", fontsize=FS_S,
            color=SHAPE, style="italic")
    top = 0.25 + n * cell
    ax.text(x0 + n * cell + 0.06, 0.25 + 22 * cell, "pad.", fontsize=FS_S, color=SHAPE,
            style="italic", va="center")
    ax.text(x0 + n * cell + 0.06, 0.25 + 2 * cell, "pad.", fontsize=FS_S, color=SHAPE,
            style="italic", va="center")
    arrow(ax, cx, top + 0.03, cx, top + 0.4)
    if with_pad:
        box(ax, cx, top + 0.7, 1.75, 0.6, "2D DCT\n(full 24×24)")
        shape(ax, cx + 0.08, top + 1.13, "576 freq. tokens")
    else:
        box(ax, cx, top + 0.7, 1.75, 0.6, "2D DCT\n(image area only)")
        shape(ax, cx + 0.08, top + 1.13, "< 576 freq. tokens")
    arrow(ax, cx, top + 0.96, cx, top + 1.35)
    box(ax, cx, top + 1.6, 1.7, 0.42, "Query pooling", fc=TRAIN, ec=TRAIN_E)
    arrow(ax, cx, top + 1.81, cx, top + 2.25)
    box(ax, cx, top + 2.5, 1.7, 0.42, "Adapter + AvgPool", fc=TRAIN, ec=TRAIN_E)
    arrow(ax, cx, top + 2.71, cx, top + 3.2)
    ax.text(cx, top + 3.38, "64 visual tokens", ha="center", va="center", fontsize=FS)
    shape(ax, cx + 0.08, top + 2.95, "64×1024")
    frame(ax, cx - 1.02, cx + 1.02, top + 0.25, top + 2.85, "")
    ax.text(cx - 1.02, top + 2.9, "AFC", fontsize=FS_S, color="#4a5f9c", va="bottom")


def main():
    fig, ax = plt.subplots(figsize=(7.16, 2.95))
    ax.set_xlim(-0.2, 13.6)
    ax.set_ylim(-0.2, 6.85)
    ax.axis("off")
    kd_panel(ax, 1.15, True)
    kd_panel(ax, 4.45, False)
    ax.plot([7.55, 7.55], [0.0, 6.8], color="#4a5f9c", lw=0.8)
    pad_panel(ax, 9.25, True)
    pad_panel(ax, 12.15, False)
    for i, (fc, ec, lab) in enumerate([(FROZEN, FROZEN_E, "frozen"), (TRAIN, TRAIN_E, "trainable")]):
        ax.add_patch(FancyBboxPatch((5.75, 6.62 - i * 0.38), 0.32, 0.2,
                                    boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, lw=0.6))
        ax.text(6.13, 6.72 - i * 0.38, lab, fontsize=FS_S, va="center")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"ablation_diagram.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
    print(OUT / "ablation_diagram.png")


if __name__ == "__main__":
    main()
