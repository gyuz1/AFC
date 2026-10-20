"""Paper figure: the padding ablation, drawn exactly as implemented.

Both settings feed the SAME padded 336x336 image to the frozen CLIP, so the
24x24 CLIP feature map contains the padding rows in both. 'w/o padding' cuts
those rows inside AFC (from the original image size) before the DCT; the
average-pooling branch also uses the cut region. 64 output tokens in both.

    python scripts/afc/plot_padding_diagram.py   # outputs/report/ablation_padding.{pdf,png}
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/report"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
FROZEN, FROZEN_E = "#9fb4e3", "#6f86c4"
TRAIN, TRAIN_E = "#f6c48f", "#d9893a"
FREE, FREE_E = "#e8ecf6", "#8fa0c8"            # parameter-free (DCT, AvgPool, crop)
PAD, IMG = "#c9c9c9", "#7e98d8"
SHAPE, INK, FRAME = "#7f7f7f", "#1a1a1a", "#4a5f9c"
FS, FS_S = 6.6, 5.6


def box(ax, x, y, w, h, text, fc, ec, fs=FS):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, p, q, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7,
                                                    ls=ls, shrinkA=0, shrinkB=0,
                                                    mutation_scale=6))


def line(ax, pts):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=INK, lw=0.7)


def shape(ax, x, y, s, ha="left"):
    ax.text(x, y, s, fontsize=FS_S, color=SHAPE, style="italic", ha=ha, va="center")


def grid(ax, cx, y0, cell, rows, cols, pad_rows=()):
    x0 = cx - cols * cell / 2
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x0 + c * cell, y0 + (rows - 1 - r) * cell), cell, cell,
                                   fc=PAD if r in pad_rows else IMG, ec="white", lw=0.12))
    return x0, y0 + rows * cell                        # left, top


def panel(ax, cx, crop):
    ax.text(cx, 8.85, "w/o padding" if crop else "w/ padding", ha="center", va="center",
            fontsize=FS + 1, bbox=dict(boxstyle="square,pad=0.35", fc="white", ec=INK, lw=0.7))
    # padded input image -> frozen CLIP
    iw, ih = 0.9, 0.9
    ax.add_patch(Rectangle((cx - iw / 2, 0.05), iw, ih, fc=PAD, ec="none"))
    ax.add_patch(Rectangle((cx - iw / 2, 0.05 + ih * 0.17), iw, ih * 0.66, fc="#c7d3ee", ec="none"))
    shape(ax, cx + iw / 2 + 0.06, 0.5, "padded\n336×336")
    arrow(ax, (cx, 0.98), (cx, 1.23))
    box(ax, cx, 1.45, 1.5, 0.42, "CLIP ViT", FROZEN, FROZEN_E)
    arrow(ax, (cx, 1.66), (cx, 1.9))
    # CLIP feature map, padding rows present in BOTH settings
    cell = 0.042
    pads = list(range(4)) + list(range(20, 24))
    gx0, gtop = grid(ax, cx, 1.93, cell, 24, 24, pads)
    shape(ax, gx0 + 24 * cell + 0.05, 1.93 + 22 * cell, "pad.")
    shape(ax, gx0 + 24 * cell + 0.05, 1.93 + 2 * cell, "pad.")
    shape(ax, gx0 + 24 * cell + 0.05, 1.93 + 12 * cell, "24×24")
    y = gtop + 0.08
    # ---- AFC ----
    if crop:
        arrow(ax, (cx, y), (cx, y + 0.25))
        box(ax, cx, y + 0.45, 1.75, 0.38, "Remove padding rows", FREE, FREE_E)
        arrow(ax, (cx, y + 0.64), (cx, y + 0.86))
        _, ctop = grid(ax, cx, y + 0.88, cell, 16, 24)
        shape(ax, cx + 24 * cell / 2 + 0.05, y + 0.88 + 8 * cell, "image area\n(< 24×24)")
        base_y = y + 0.88 + 8 * cell                 # AvgPool branch starts at the cut map
        y = ctop + 0.05
        dct = "2D DCT\n(image area)"
        ntok = "< 576 freq. tokens"
    else:
        arrow(ax, (cx, y), (cx, y + 0.86 + 16 * cell + 0.05))   # same height as the cut path
        shape(ax, cx + 0.08, y + 0.6, "full map\n(padding kept)")
        base_y = 1.93 + 12 * cell
        y = y + 0.88 + 16 * cell + 0.05
        dct = "2D DCT\n(full 24×24)"
        ntok = "576 freq. tokens"
    if crop:
        arrow(ax, (cx, y), (cx, y + 0.22))
    box(ax, cx, y + 0.4, 1.75, 0.5, dct, FREE, FREE_E)
    arrow(ax, (cx, y + 0.65), (cx, y + 0.86))
    shape(ax, cx + 0.06, y + 0.77, ntok)
    box(ax, cx, y + 1.05, 1.75, 0.38, "Freq. encoder", TRAIN, TRAIN_E)
    arrow(ax, (cx, y + 1.24), (cx, y + 1.46))
    box(ax, cx, y + 1.65, 1.75, 0.38, "Query pooling", TRAIN, TRAIN_E)
    arrow(ax, (cx, y + 1.84), (cx, y + 2.06))
    box(ax, cx, y + 2.25, 1.75, 0.38, "Adapter", TRAIN, TRAIN_E)
    plus_y = y + 2.72
    arrow(ax, (cx, y + 2.44), (cx, plus_y - 0.1))
    ax.add_patch(Circle((cx, plus_y), 0.1, fc="white", ec=INK, lw=0.7))
    ax.text(cx, plus_y, "+", ha="center", va="center", fontsize=FS + 1)
    # average-pooling branch (left side), from the map used for the DCT
    bx = cx - 1.2
    line(ax, [(cx - 24 * cell / 2, base_y), (bx, base_y), (bx, y + 1.62)])
    box(ax, bx, y + 1.85, 0.62, 0.42, "Avg\nPool", FREE, FREE_E, fs=FS - 0.4)
    line(ax, [(bx, y + 2.06), (bx, plus_y), (cx - 0.1, plus_y)])
    arrow(ax, (cx - 0.25, plus_y), (cx - 0.1, plus_y))
    arrow(ax, (cx, plus_y + 0.1), (cx, plus_y + 0.45))
    ax.text(cx, plus_y + 0.6, "64 visual tokens", ha="center", va="center", fontsize=FS)
    shape(ax, cx + 0.06, plus_y + 0.3, "64×1024")
    # AFC frame
    f0 = gtop + 0.0
    ax.add_patch(FancyBboxPatch((cx - 1.6, f0), 2.6, plus_y + 0.18 - f0,
                                boxstyle="round,pad=0,rounding_size=0.1",
                                fc="none", ec=FRAME, lw=0.8))
    ax.text(cx - 1.55, plus_y + 0.08, "AFC", fontsize=FS_S, color=FRAME, va="top")


def main():
    fig, ax = plt.subplots(figsize=(3.45, 4.25))
    ax.set_xlim(-0.2, 6.6)
    ax.set_ylim(-0.1, 9.15)
    ax.axis("off")
    panel(ax, 1.75, False)
    ax.plot([3.25, 3.25], [0.2, 9.1], color=FRAME, lw=0.8)
    panel(ax, 5.05, True)
    for i, (fc, ec, lab) in enumerate([(FROZEN, FROZEN_E, "frozen"),
                                       (FREE, FREE_E, "parameter-free"),
                                       (TRAIN, TRAIN_E, "trainable")]):
        x = 0.0 + i * 2.15
        ax.add_patch(FancyBboxPatch((x, -0.08), 0.22, 0.16,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=fc, ec=ec, lw=0.6))
        ax.text(x + 0.3, 0.0, lab, fontsize=FS_S, va="center")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"ablation_padding.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
    print(OUT / "ablation_padding.png")


if __name__ == "__main__":
    main()
