"""Paper figure: ablation scores, 'w/' vs. 'w/o' side by side.

Frozen LLaVA-1.5-7B, 64 visual tokens, whole training data (544k).
Rows: KD (AFC-global), padding (AFC-global), padding (AFC-block).
Columns: the six benchmarks and their average; every cell shows the two
scores as bars with their values (y-axis zoomed per cell).

    python scripts/afc/plot_ablation_wwo.py   # outputs/report/ablation_wwo.{pdf,png}
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llava.eval.afc.table import collect  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "outputs/eval"
OUT = REPO / "outputs/report"
BENCH = ["GQA", "SQA-IMG", "TextVQA", "POPE", "MMB", "MMMU", "Avg"]
BLABEL = ["GQA", "SciQA", r"VQA$^\mathrm{T}$", "POPE", "MMB", "MMMU", "Avg."]
ROWS = [  # (row label, w/ label, w/ run, w/o label, w/o run)
    ("KD\n(AFC-global)", "w/ KD", "global_all", "w/o KD", "global_nokd_all"),
    ("Padding\n(AFC-global)", "w/ pad.", "global_all", "w/o pad.", "global_valid_all"),
    ("Padding\n(AFC-block)", "w/ pad.", "block_all", "w/o pad.", "block_valid_all"),
]
C_W, C_WO, MUTED = "#7f96cf", "#e8964a", "#8a8a8a"

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 6.5,
                     "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42})


def res(name):
    r = collect(str(EVAL / f"llava-v1.5-7b-afc-{name}"))
    return r if r.get("Avg") is not None else None


def main():
    fig, axes = plt.subplots(len(ROWS), len(BENCH), figsize=(7.16, 3.3),
                             gridspec_kw=dict(wspace=0.12, hspace=0.32))
    for i, (row, lw, rw, lwo, rwo) in enumerate(ROWS):
        a, b = res(rw), res(rwo)
        for j, bench in enumerate(BENCH):
            ax = axes[i, j]
            va = a[bench] if a else None
            vb = b[bench] if b else None
            vals = [v for v in (va, vb) if v is not None]
            lo = min(vals) - 2.0 if vals else 0
            hi = max(vals) + 1.2 if vals else 1
            ax.set_ylim(lo, hi)
            for k, (v, c) in enumerate(((va, C_W), (vb, C_WO))):
                if v is None:
                    ax.bar(k, hi - lo, bottom=lo, width=0.7, facecolor="white",
                           edgecolor=MUTED, hatch="////", lw=0.5)
                    ax.text(k, lo + (hi - lo) / 2, "pending", rotation=90, ha="center",
                            va="center", fontsize=5, color=MUTED, style="italic")
                    continue
                ax.bar(k, v - lo, bottom=lo, width=0.7, color=c,
                       edgecolor="black" if bench == "Avg" else "none", lw=0.6)
                ax.text(k, v + 0.12, f"{v:.1f}", ha="center", va="bottom", fontsize=5.6,
                        weight="bold" if bench == "Avg" else "normal")
            if va is not None and vb is not None:
                d = vb - va
                ax.text(0.5, hi - 0.05 * (hi - lo), f"{d:+.1f}" if abs(d) >= 0.05 else "0.0",
                        ha="center", va="top", fontsize=5.6,
                        color="#b23a1e" if d < 0 else "#1f5fa8")
            ax.set_xticks([0, 1], [lw, lwo], fontsize=5.4)
            ax.tick_params(axis="x", length=0, pad=1.5)
            ax.set_yticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(BLABEL[j], fontsize=7, weight="bold" if bench == "Avg" else "normal")
            if j == 0:
                ax.set_ylabel(row, fontsize=6.5, rotation=0, ha="right", va="center",
                              labelpad=4)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"ablation_wwo.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
    print(OUT / "ablation_wwo.png")


if __name__ == "__main__":
    main()
