"""Paper figure: ablations of AFC (English, vector PDF + PNG).

All runs: frozen LLaVA-1.5-7B, 64 visual tokens, 75k training samples unless
stated; reference = AFC-global 75k (global75k_fp).

    python scripts/afc/plot_ablation.py            # outputs/report/ablation.{pdf,png}
    python scripts/afc/plot_ablation.py --mock     # layout preview, stand-in values
                                                   # for missing runs (watermarked,
                                                   # ablation_example.{pdf,png})

(a) Where does the gain come from?   cumulative build-up from the training-free
    start: average pooling -> + learnable query pooling (spatial tokens)
    -> + DCT (= AFC) -> + whole data (544k); separate bar: AFC without KD.
(b) Which abilities change?          per-benchmark difference to the reference.
(c) Effect of square padding.        retention vs. padding fraction of the image,
    block DCT with / without the valid-region DCT (and Fourier-VLM).
Missing results are drawn as hatched "pending" marks.
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

REF = "global75k_fp"
BENCH = ["GQA", "SQA-IMG", "TextVQA", "POPE", "MMB", "MMMU"]
BLABEL = ["GQA", "SciQA", r"VQA$^\mathrm{T}$", "POPE", "MMB", "MMMU"]
BUILD = [("Avg\npool", "avgpool64"),
         ("+Query\npool", "spatial_global75k"),
         ("+DCT\n(AFC)", REF),
         ("+Data\n544k", "global_all")]
NO_KD = ("w/o\nKD", "global_nokd75k")
HEAT = [("w/o DCT", "spatial_global75k"), ("w/o KD", "global_nokd75k"),
        ("Block DCT", "block75k_fp"), ("Valid (global)", "global_valid75k"),
        ("Valid (block)", "block_valid75k"), ("Data 544k", "global_all")]
PAD = [("Block DCT", "block75k_fp"), ("Block + valid", "block_valid75k"),
       ("Fourier-VLM", "fourier_vlm64")]

INK, MUTED, GRID = "#1a1a1a", "#7a7a7a", "#e5e5e5"
C_BASE, C_GAIN, C_LOSS = "#9e9e9e", "#E69F00", "#0072B2"      # Okabe-Ito
PAD_STYLE = {"Block DCT": dict(color="#9e9e9e", marker="o", ls="-"),
             "Block + valid": dict(color="#D55E00", marker="o", ls="-"),
             "Fourier-VLM": dict(color="#0072B2", marker="s", ls="--")}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MOCK = {}


def res(name):
    r = collect(str(EVAL / f"llava-v1.5-7b-afc-{name}"))
    if r.get("Avg") is not None:
        return r
    return MOCK.get(name)


def hatched_bar(ax, x, bottom, height, width=0.62):
    ax.bar(x, height, bottom=bottom, width=width, facecolor="white",
           edgecolor=MUTED, hatch="////", lw=0.6)
    ax.text(x, bottom + height + 0.25, "pending", ha="center", fontsize=6.5,
            color=MUTED, style="italic")


# --------------------------------------------------------------- panel (a)
def panel_build(ax):
    prev = None
    for i, (label, run) in enumerate(BUILD):
        r = res(run)
        if r is None:
            hatched_bar(ax, i, prev if prev is not None else 50, 2.5)
            continue
        v = r["Avg"]
        if prev is None:
            ax.bar(i, v - 50, bottom=50, width=0.62, color=C_BASE)
            ax.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=6.5)
        else:
            d = v - prev
            ax.bar(i, d, bottom=prev, width=0.62, color=C_GAIN if d >= 0 else C_LOSS)
            ax.plot([i - 1 + 0.31, i - 0.31], [prev, prev], color=MUTED, lw=0.6, ls=":")
            ax.text(i, max(v, prev) + 0.25, f"{d:+.1f}", ha="center", fontsize=7,
                    weight="bold")
            ax.text(i, 50.35, f"{v:.1f}", ha="center", fontsize=6, color=INK)
        prev = v
    # AFC without KD, compared with AFC
    x = len(BUILD) + 0.35
    afc, nokd = res(REF), res(NO_KD[1])
    if afc is None or nokd is None:
        hatched_bar(ax, x, (afc["Avg"] if afc else 58) - 2.5, 2.5)
    else:
        d = nokd["Avg"] - afc["Avg"]
        ax.bar(x, d, bottom=afc["Avg"], width=0.62, color=C_LOSS if d < 0 else C_GAIN)
        ax.text(x, min(nokd["Avg"], afc["Avg"]) - 0.75, f"{d:+.1f}", ha="center",
                fontsize=6.5, weight="bold")
    ax.axvline(len(BUILD) - 0.18, color=GRID, lw=0.8)
    ax.set_xticks(list(range(len(BUILD))) + [x], [b[0] for b in BUILD] + [NO_KD[0]],
                  fontsize=6)
    ax.set_ylim(50, 61)
    ax.set_xlim(-0.55, x + 0.55)
    ax.set_ylabel("Avg. score (6 benchmarks)")
    ax.set_title("(a) Gain decomposition", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# --------------------------------------------------------------- panel (b)
def panel_heat(ax):
    ref = res(REF)
    M = np.full((len(HEAT), len(BENCH) + 1), np.nan)
    for i, (_, run) in enumerate(HEAT):
        r = res(run)
        if r is not None and ref is not None:
            M[i] = [r[b] - ref[b] for b in BENCH] + [r["Avg"] - ref["Avg"]]
    lim = 3.0 if np.all(np.isnan(M)) else max(3.0, float(np.nanmax(np.abs(M))))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="white",
                                           edgecolor="#cfcfcf", hatch="////", lw=0))
                continue
            ax.text(j, i, "0.0" if abs(v) < 0.05 else f"{v:+.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 * lim else INK,
                    weight="bold" if j == M.shape[1] - 1 else "normal")
    ax.axvline(len(BENCH) - 0.5, color="white", lw=2.5)
    ax.set_xticks(range(len(BENCH) + 1), BLABEL + ["Avg."], rotation=45, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(range(len(HEAT)), [h[0] for h in HEAT])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(b) Per-benchmark change", loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.set_title("Δ", fontsize=7, pad=3)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)


# --------------------------------------------------------------- panel (c)
def panel_padding(ax):
    import padding_analysis as pa
    cache = {"textvqa": {},
             "gqa": pa.parquet_sizes("datasets--lmms-lab--GQA/snapshots/*/"
                                     "testdev_balanced_images/*.parquet", "id"),
             "pope": pa.parquet_sizes("datasets--lmms-lab--POPE/snapshots/*/data/*.parquet", "id"),
             "sqa": pa.sqa_sizes()}

    def per_bin(run):
        agg = {}
        for bench, p, s in pa.collect(f"llava-v1.5-7b-afc-{run}" if run else "llava-v1.5-7b",
                                      cache):
            for name, lo, hi in pa.BINS:
                if lo <= p < hi:
                    a = agg.setdefault((bench, name), [0.0, 0])
                    a[0] += s
                    a[1] += 1
        return agg

    base = per_bin(None)
    bins = [b[0] for b in pa.BINS]
    for label, run in PAD:
        agg = per_bin(run)
        if not agg:
            continue
        ys = []
        for n in bins:
            rs = [(agg[k][0] / agg[k][1]) / (base[k][0] / base[k][1])
                  for k in agg if k[1] == n and base.get(k, (0, 0))[0]]
            ys.append(100 * np.mean(rs) if rs else np.nan)
        st = PAD_STYLE[label]
        ax.plot(range(3), ys, lw=1.4, ms=4, label=label, **st)
        ax.text(2.1, ys[-1], f"{ys[-1]:.1f}", va="center", fontsize=6.5, color=st["color"])
    ax.set_xticks(range(3), ["<5%\n(square)", "5–30%\n(4:3)", "≥30%\n(3:2+)"])
    ax.set_xlim(-0.25, 2.5)
    ax.set_xlabel("padding fraction")
    ax.set_ylabel("Retention (%)")
    ax.set_title("(c) Effect of padding", loc="left")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42), handlelength=1.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    mock = "--mock" in sys.argv
    if mock:   # illustrative stand-ins ONLY for runs that do not exist yet
        ref = collect(str(EVAL / f"llava-v1.5-7b-afc-{REF}"))
        for name, shift in (("spatial_global75k", [-2.5, -1.0, -4.0, -3.5, -6.0, -0.8]),
                            ("global_nokd75k", [-1.8, -0.6, -2.2, -2.5, -3.5, -0.5])):
            r = {b: ref[b] + d for b, d in zip(BENCH, shift)}
            r["Avg"] = sum(r.values()) / len(BENCH)
            MOCK[name] = r

    fig = plt.figure(figsize=(7.16, 2.3))            # two-column width (IEEE: 7.16 in)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0], wspace=0.5)
    panel_build(fig.add_subplot(gs[0, 0]))
    panel_heat(fig.add_subplot(gs[0, 1]))
    panel_padding(fig.add_subplot(gs[0, 2]))

    stem = "ablation_example" if mock else "ablation"
    if mock:
        fig.text(0.5, 0.5, "EXAMPLE - mock values for w/o DCT and w/o KD",
                 ha="center", va="center", fontsize=13, color="#cc0000", alpha=0.22,
                 rotation=10, weight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
    print(OUT / f"{stem}.pdf")


if __name__ == "__main__":
    main()
