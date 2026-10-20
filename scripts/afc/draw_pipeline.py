"""AFC pipeline diagram (global variant) -> outputs/figures/afc_pipeline.{png,pdf}.

    python scripts/afc/draw_pipeline.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/figures"
FROZEN, TRAIN, FREE, LOSS = "#E3E7ED", "#FBD9B5", "#FFFFFF", "#D9ECF7"
EDGE_F, EDGE_T = "#6B7785", "#D9822B"


def box(ax, x, y, w, h, title, sub="", fc=FROZEN, ec=EDGE_F, bold=True,
        ls="-", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4, ls=ls))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=fs, weight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=fs - 2, color="#333333")
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, weight="bold" if bold else "normal")


def arrow(ax, p, q, label="", color="#333333", ls="-", lx=0, ly=0.12, rad=0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.3, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label,
                ha="center", va="bottom", fontsize=7.5, color="#444444",
                family="monospace")


def main():
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # ---- top row: LLaVA-1.5 with the plug-in compressor -------------------
    y, h = 6.3, 1.1
    box(ax, 0.2, y, 1.6, h, "Image", "336x336 (pad)", fc=FREE)
    box(ax, 2.4, y, 2.0, h, "CLIP ViT-L/14", "frozen  |  layer -2, no CLS")
    box(ax, 5.2, y, 2.6, h, "AFC compressor", "trainable, 1.2M params",
        fc=TRAIN, ec=EDGE_T)
    box(ax, 8.6, y, 2.0, h, "mm_projector", "frozen  |  MLP 1024→4096")
    box(ax, 11.4, y, 2.0, h, "Vicuna-7B", "frozen LLM  |  32 layers")
    box(ax, 13.8, y + 0.25, 1.0, 0.6, "Answer", fc=FREE, fs=9)
    arrow(ax, (1.8, y + h / 2), (2.4, y + h / 2))
    arrow(ax, (4.4, y + h / 2), (5.2, y + h / 2), "576x1024", ly=0.1)
    arrow(ax, (7.8, y + h / 2), (8.6, y + h / 2), "64x1024", ly=0.1)
    arrow(ax, (10.6, y + h / 2), (11.4, y + h / 2), "64x4096", ly=0.1)
    arrow(ax, (13.4, y + h / 2), (13.8, y + h / 2))
    ax.text(11.4 + 1.0, y + h + 0.18, "+ text tokens (64 image tokens at <image>)",
            ha="center", fontsize=7.5, color="#444444")

    # ---- expansion of the compressor --------------------------------------
    ax.add_patch(FancyBboxPatch((0.2, 1.75), 14.6, 3.75,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#FFF8F0", ec=EDGE_T, lw=1.2, ls="--"))
    ax.plot([5.2, 0.25], [y, 5.5], color=EDGE_T, lw=1, ls=":")
    ax.plot([7.8, 14.75], [y, 5.5], color=EDGE_T, lw=1, ls=":")
    ax.text(0.45, 5.2, "AFC compressor (global variant)", fontsize=10,
            weight="bold", color="#8A4B10")

    y2, h2 = 3.5, 1.15
    xs = [0.45, 2.55, 4.85, 7.15, 9.45, 11.95]
    w = 1.85
    box(ax, xs[0], y2, w, h2, "Reshape + 2D DCT-II",
        "24x24 grid, spatial axes\northonormal, no params", fc=FREE)
    box(ax, xs[1], y2, w + 0.2, h2, "Freq. token encoder",
        "LN + Linear 1024→128\n+ pos. emb e_u + e_v", fc=TRAIN, ec=EDGE_T)
    box(ax, xs[2], y2, w + 0.2, h2, "Transformer x2",
        "Pre-LN, d=128\n4 heads, FFN 512", fc=TRAIN, ec=EDGE_T)
    box(ax, xs[3], y2, w + 0.2, h2, "Query pooling",
        "64 learned queries\ncross-attn (Pre-LN)", fc=TRAIN, ec=EDGE_T)
    box(ax, xs[4], y2, w + 0.4, h2, "Adapter",
        "LN → 128→512 → GELU\n→ 512→1024 (zero-init)", fc=TRAIN, ec=EDGE_T)
    # residual sum
    cx, cy = 12.85, y2 + h2 / 2
    ax.add_patch(plt.Circle((cx, cy), 0.22, fc="white", ec="#333333", lw=1.3))
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=14)
    box(ax, 13.45, y2 + 0.2, 1.25, h2 - 0.4, "64 tokens", "64x1024",
        fc=FREE, fs=9)

    m = y2 + h2 / 2
    arrow(ax, (xs[0] + w, m), (xs[1], m), "576x1024", ly=0.08)
    arrow(ax, (xs[1] + w + 0.2, m), (xs[2], m), "576x128", ly=0.08)
    arrow(ax, (xs[2] + w + 0.2, m), (xs[3], m), "576x128", ly=0.08)
    arrow(ax, (xs[3] + w + 0.2, m), (xs[4], m), "64x128", ly=0.08)
    arrow(ax, (xs[4] + w + 0.4, m), (cx - 0.22, m))
    arrow(ax, (cx + 0.22, m), (13.45, m))

    # parameter-free residual base
    box(ax, 7.15, 2.0, 3.4, 0.85, "Average pooling 8x8 (3x3 patch cells)",
        "parameter-free base: output = avgpool + adapter", fc=FREE, fs=9)
    arrow(ax, (1.4, y2), (7.15, 2.42), "576x1024 (CLIP features)", ly=-0.32,
          rad=0.08)
    arrow(ax, (10.55, 2.42), (cx, cy - 0.22), "64x1024", lx=0.5, ly=-0.3,
          rad=0.25)

    # ---- objective -----------------------------------------------------------
    box(ax, 0.2, 0.15, 9.2, 1.25, "Loss = CE(answer tokens) + 1.0 · KD(T=2)",
        "KD teacher: the same frozen LLaVA without the compressor (576 tokens, no grad);\n"
        "teacher/student answer tokens aligned by position, full-vocabulary KL",
        fc=LOSS, ec="#4A86A8", fs=10)
    # legend
    lx = 9.9
    for i, (fc, ec, t) in enumerate([(TRAIN, EDGE_T, "trainable"),
                                      (FROZEN, EDGE_F, "frozen (original LLaVA-1.5-7B)"),
                                      (FREE, EDGE_F, "parameter-free")]):
        yy = 1.05 - i * 0.38
        ax.add_patch(FancyBboxPatch((lx, yy), 0.4, 0.24,
                                    boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(lx + 0.55, yy + 0.12, t, va="center", fontsize=9)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"afc_pipeline.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(OUT / "afc_pipeline.png")


if __name__ == "__main__":
    main()
