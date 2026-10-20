"""Paper figure: the two ablations of AFC (KD and square padding).

Frozen LLaVA-1.5-7B, 64 visual tokens, whole training data (544k).

    python scripts/afc/plot_ablation_kd_pad.py          # outputs/report/ablation_kd_pad.{pdf,png}
    python scripts/afc/plot_ablation_kd_pad.py --mock   # preview with stand-in w/o-KD values
                                                        # (watermarked, *_example.*)

(a) Knowledge distillation: per-benchmark score with KD (default) and without
    KD (CE only), AFC-global; arrows point from the default to the ablation.
(b) Square padding: per-benchmark score with padding (default) and without
    padding (valid-region DCT), for AFC-global and AFC-block.
(c) Where padding hurts: retention vs. LLaVA-1.5 grouped by the padding
    fraction of the input image, AFC-block with / without padding.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from llava.eval.afc.table import collect  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
EVAL = REPO / "outputs/eval"
OUT = REPO / "outputs/report"
BENCH = ["GQA", "SQA-IMG", "TextVQA", "POPE", "MMB", "MMMU", "Avg"]
BLABEL = ["GQA", "SciQA", r"VQA$^\mathrm{T}$", "POPE", "MMB", "MMMU", "Avg."]

C_DEF, C_ABL, C_GAIN, C_LOSS = "#4d4d4d", "#D55E00", "#0072B2", "#D55E00"
GRID, MUTED = "#e6e6e6", "#8a8a8a"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})
MOCK = {}


def res(name):
    r = collect(str(EVAL / f"llava-v1.5-7b-afc-{name}"))
    return r if r.get("Avg") is not None else MOCK.get(name)


def delta_panel(ax, pairs, title, legend_loc="upper right"):
    """pairs: [(label, default_run, ablated_run, marker)] -> per-benchmark change
    (ablated - default) as grouped bars; default value printed under the axis."""
    n = len(pairs)
    w = 0.8 / n
    x = np.arange(len(BENCH))
    ax.axhline(0, color="#333333", lw=0.6)
    for i, (label, d_run, a_run, color, label_values) in enumerate(pairs):
        d, a = res(d_run), res(a_run)
        xs = x - 0.4 + w / 2 + i * w
        if d is None or a is None:
            ax.bar(xs, [0.6] * len(BENCH), bottom=-0.3, width=w * 0.9, facecolor="white",
                   edgecolor=MUTED, hatch="////", lw=0.5, label=f"{label} (pending)")
            continue
        diff = np.array([a[b] - d[b] for b in BENCH])
        ax.bar(xs, diff, width=w * 0.9, color=color, edgecolor="white", lw=0.4, label=label)
        if label_values:
            for xi, v in zip(xs, diff):
                ax.text(xi, v + (0.06 if v >= 0 else -0.06),
                        f"{v:+.1f}" if abs(v) >= 0.05 else "0", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=5, color=color)
    ax.axvline(len(BENCH) - 1.5, color=GRID, lw=0.8)
    ax.set_xticks(x, BLABEL, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel("score change vs. default")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -1.6) - 0.3, max(hi, 1.6) + 0.4)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc=legend_loc, handlelength=1.2,
              borderpad=0.3)


def panel_padding_bins(ax):
    import padding_analysis as pa
    cache = {"textvqa": {},
             "gqa": pa.parquet_sizes("datasets--lmms-lab--GQA/snapshots/*/"
                                     "testdev_balanced_images/*.parquet", "id"),
             "pope": pa.parquet_sizes("datasets--lmms-lab--POPE/snapshots/*/data/*.parquet", "id"),
             "sqa": pa.sqa_sizes()}

    def per_bin(run):
        agg = {}
        for bench, p, s in pa.collect(run, cache):
            for name, lo, hi in pa.BINS:
                if lo <= p < hi:
                    a = agg.setdefault((bench, name), [0.0, 0])
                    a[0] += s
                    a[1] += 1
        return agg

    base = per_bin("llava-v1.5-7b")
    bins = [b[0] for b in pa.BINS]
    lines = [("AFC-block, with padding", "block_all", dict(color=C_DEF, marker="o", ls="-")),
             ("AFC-block, w/o padding", "block_valid_all", dict(color=C_ABL, marker="o", ls="-"))]
    for label, run, st in lines:
        agg = per_bin(f"llava-v1.5-7b-afc-{run}")
        if not agg:
            ax.plot([], [], label=f"{label} (pending)", **st)
            continue
        ys = []
        for n in bins:
            rs = [(agg[k][0] / agg[k][1]) / (base[k][0] / base[k][1])
                  for k in agg if k[1] == n and base.get(k, (0, 0))[0]]
            ys.append(100 * np.mean(rs) if rs else np.nan)
        ax.plot(range(3), ys, lw=1.4, ms=4, label=label, **st)
        for xi, y in enumerate(ys):
            ax.text(xi, y + (0.12 if st["color"] == C_ABL else -0.2), f"{y:.1f}",
                    ha="center", fontsize=5.5, color=st["color"])
    ax.set_xticks(range(3), ["<5%\n(square)", "5–30%\n(4:3)", "≥30%\n(3:2+)"])
    ax.set_xlim(-0.35, 2.35)
    ax.set_xlabel("padding fraction of the input image")
    ax.set_ylabel("retention vs. LLaVA-1.5 (%)")
    ax.set_title("(c) Where padding hurts (block)", loc="left")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.25, hi + 1.4)
    ax.legend(frameon=False, loc="upper center", handlelength=1.6, ncol=1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    mock = "--mock" in sys.argv
    if mock:   # stand-in values ONLY for the missing w/o-KD run
        ref = collect(str(EVAL / "llava-v1.5-7b-afc-global_all"))
        r = {b: ref[b] + d for b, d in zip(BENCH[:-1], [-1.8, -0.6, -2.2, -2.5, -3.5, -0.5])}
        r["Avg"] = sum(r.values()) / 6
        MOCK["global_nokd_all"] = r

    fig = plt.figure(figsize=(7.16, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.95], wspace=0.38)
    delta_panel(fig.add_subplot(gs[0, 0]),
                [("AFC-global w/o KD", "global_all", "global_nokd_all", C_LOSS, True)],
                "(a) Effect of knowledge distillation")
    delta_panel(fig.add_subplot(gs[0, 1]),
                [("AFC-global w/o padding", "global_all", "global_valid_all", "#9e9e9e", False),
                 ("AFC-block w/o padding", "block_all", "block_valid_all", C_GAIN, True)],
                "(b) Effect of removing square padding", legend_loc="lower left")
    panel_padding_bins(fig.add_subplot(gs[0, 2]))

    stem = "ablation_kd_pad_example" if mock else "ablation_kd_pad"
    if mock:
        fig.text(0.2, 0.5, "EXAMPLE\nmock w/o-KD values", ha="center", va="center",
                 fontsize=12, color="#cc0000", alpha=0.25, rotation=12, weight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
    print(OUT / f"{stem}.png")


if __name__ == "__main__":
    main()
