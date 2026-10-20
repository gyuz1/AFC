"""Token-budget comparison (64 / 144 / 256 visual tokens): table + figure.

    python scripts/afc/plot_budget.py                    # variants from the queue
    python scripts/afc/plot_budget.py --variant full --block-variant valid

Rows (all with the frozen LLaVA-1.5-7B):
    AFC global      64: global75k_fp (full) / global_valid75k (valid)
                    144 / 256: global_<variant><t>_75k
    AFC block       64: block75k_fp / block_valid75k
                    144 / 256: block_<variant><t>_75k
    Fourier-VLM FFC fourier_vlm<t>   (parameter-free, frozen LLM)
    Avg pooling     avgpool<t>
    LLaVA-1.5 576   llava-v1.5-7b    (upper bound, dashed line)
Only our own measurements (no reported numbers from other papers).
Writes outputs/report/budget.md and budget.png / budget.pdf; missing runs
are skipped.
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llava.eval.afc.table import BENCHMARKS, PAPER_ROWS, collect  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "outputs/eval"
TOKENS = (64, 144, 256)
PENDING = "⏳"
DOC = REPO / "docs/AFC_RESULTS_TABLE.md"
# Okabe-Ito (colour-blind safe); identity also carried by marker + label
STYLE = {
    "AFC global (ours)": dict(color="#D55E00", marker="o"),
    "AFC block (ours)": dict(color="#CC79A7", marker="D"),
    "Fourier-VLM FFC": dict(color="#0072B2", marker="s"),
    "PruMerge": dict(color="#56B4E9", marker="v"),
    "Avg pooling": dict(color="#009E73", marker="^"),
}


def runs(variant, block_variant, recipe="all"):
    """Run names per method and token count. ``recipe`` "all" = whole data
    (64: global_all / block_all), "75k" = the 75k subset."""
    if recipe == "all":
        g = {64: "global_all", 144: f"global_{variant}144_all",
             256: f"global_{variant}256_all"}
        b = {64: "block_all", 144: f"block_{block_variant}144_all",
             256: f"block_{block_variant}256_all"}
    else:
        g = {64: "global_valid75k" if variant == "valid" else "global75k_fp",
             144: f"global_{variant}144_75k", 256: f"global_{variant}256_75k"}
        b = {64: "block_valid75k" if block_variant == "valid" else "block75k_fp",
             144: f"block_{block_variant}144_75k",
             256: f"block_{block_variant}256_75k"}
    return {
        "AFC global (ours)": g,
        "AFC block (ours)": b,
        "Fourier-VLM FFC": {t: f"fourier_vlm{t}" for t in TOKENS},
        "PruMerge": {t: f"prumerge{t}" for t in TOKENS},
        "Avg pooling": {t: f"avgpool{t}" for t in TOKENS},
    }


ABLATION = [  # (label, run, reference run for the difference) -- whole data (544k)
    ("AFC-global (ours)", "global_all", None),
    ("  w/o KD", "global_nokd_all", "global_all"),
    ("  w/o padding", "global_valid_all", "global_all"),
    ("AFC-block (ours)", "block_all", None),
    ("  w/o padding", "block_valid_all", "block_all"),
]


def ablation_table(order, head, out):
    """Per-benchmark ablation table (75k recipe, 64 tokens) -> md lines; also
    writes outputs/report/ablation_table.{md,tex}."""
    md = ["| Setting | " + " | ".join(head) + " | Avg. |",
          "|---|" + "---|" * (len(head) + 1)]
    tex = [r"\begin{tabular}{l|cccccc|c}", r"\toprule",
           "Setting & " + " & ".join(h.replace("^T", "$^\\text{T}$") for h in head)
           + r" & Avg. \\", r"\midrule"]
    for label, run, ref_run in ABLATION:
        r = load(run)
        ref = load(ref_run) if ref_run else None
        note = "reference" if ref_run is None else ""
        cells_md, cells_tex = [], []
        for b in order + ["Avg"]:
            if r is None or r.get(b) is None:
                cells_md.append(PENDING)
                cells_tex.append("--")
                continue
            v = r[b]
            if note == "reference" or ref is None:
                cells_md.append(f"{v:.1f}")
                cells_tex.append(f"{v:.1f}")
            else:
                dlt = v - ref[b]
                ds = "0.0" if abs(dlt) < 0.05 else f"{dlt:+.1f}"
                cells_md.append(f"{v:.1f} ({ds})")
                cells_tex.append(rf"{v:.1f} {{\scriptsize ({ds})}}")
        name = f"**{label}**" if note else label.strip()
        md.append(f"| {name if note else '└ ' + name} | " + " | ".join(cells_md) + " |")
        tex.append((label.strip() if note else r"\quad " + label.strip()) + " & "
                   + " & ".join(cells_tex) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "ablation_table.md").write_text("\n".join(md) + "\n")
    (out / "ablation_table.tex").write_text(
        "% frozen LLaVA-v1.5-7B, 64 tokens, whole data (544k); (diff vs the same variant with KD and padding)\n"
        + "\n".join(tex) + "\n")
    return md


def load(name):
    r = collect(str(EVAL / f"llava-v1.5-7b-afc-{name}"))
    return r if r.get("Avg") is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["full", "valid"], default=None,
                    help="AFC global variant (default: the queue's choice)")
    ap.add_argument("--block-variant", choices=["full", "valid"], default=None)
    ap.add_argument("--recipe", choices=["all", "75k"], default="all",
                    help="AFC training data: whole data (default) or 75k")
    args = ap.parse_args()

    def chosen(arg, name):
        f = REPO / f"outputs/next_stage/{name}"
        return arg or (f.read_text().strip() if f.exists() else "full")
    args.variant = chosen(args.variant, "budget_variant")
    args.block_variant = chosen(args.block_variant, "budget_block_variant")
    vanilla = collect(str(EVAL / "llava-v1.5-7b"))
    data = {m: {t: load(n) for t, n in d.items()}
            for m, d in runs(args.variant, args.block_variant, args.recipe).items()}
    paper = {t: s for name, t, s in PAPER_ROWS if name.startswith("Fourier-LLaVA")}

    # ---- table (layout of the Fourier-VLM paper, Table 3) ----------------
    order = ["GQA", "SQA-IMG", "TextVQA", "POPE", "MMB", "MMMU"]
    head = ["GQA", "SciQA", "VQA^T", "POPE", "MMB", "MMMU"]
    names = {"Avg pooling": "Avg pooling", "Fourier-VLM FFC": "Fourier-VLM (frozen LLM)",
             "PruMerge": "PruMerge (frozen LLM)",
             "AFC block (ours)": "AFC-block (ours)", "AFC global (ours)": "AFC-global (ours)"}

    def cells(r):
        return [f"{r[b]:.1f}" if r and r.get(b) is not None else PENDING
                for b in order]

    md = ["| Model | #I | " + " | ".join(head) + " | Avg. |",
          "|---|---|" + "---|" * (len(head) + 1)]
    tex = [r"\begin{tabular}{lc|cccccc|c}", r"\toprule",
           "Model & \\#I & " + " & ".join(h.replace("^T", "$^\\text{T}$") for h in head)
           + r" & Avg. \\", r"\midrule",
           "LLaVA-v1.5-7B & 576 & " + " & ".join(cells(vanilla))
           + f" & {vanilla['Avg']:.1f} " + r"\\"]
    md.append("| LLaVA-v1.5-7B | 576 | " + " | ".join(cells(vanilla))
              + f" | {vanilla['Avg']:.1f} |")
    for t in sorted(TOKENS, reverse=True):
        ratio = 100 * (1 - t / 576)
        md.append(f"| *Compression ratio: {ratio:.1f}%* | | " + " | " * len(head) + " |")
        tex += [r"\midrule", rf"\multicolumn{{9}}{{c}}{{\textit{{Compression ratio: {ratio:.1f}\%}}}} \\",
                r"\midrule"]
        group = [(names[m], data[m][t]) for m in
                 ("Avg pooling", "Fourier-VLM FFC", "PruMerge", "AFC block (ours)",
                  "AFC global (ours)")]
        best = {b: max((round(r[b], 1) for _, r in group if r and r.get(b) is not None), default=None)
                for b in order + ["Avg"]}
        for name, r in group:
            c = cells(r)
            avg = f"{r['Avg']:.1f}" if r and r.get("Avg") is not None else PENDING
            bold_md = [f"**{x}**" if r and x != PENDING and float(x) == best[b] else x
                       for x, b in zip(c + [avg], order + ["Avg"])]
            bold_tex = [rf"\textbf{{{x}}}" if r and x != PENDING and float(x) == best[b]
                        else (x if x != PENDING else "--")
                        for x, b in zip(c + [avg], order + ["Avg"])]
            md.append(f"| {name} | {t} | " + " | ".join(bold_md) + " |")
            tex.append(f"{name} & {t} & " + " & ".join(bold_tex) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    out = REPO / "outputs/report"
    out.mkdir(parents=True, exist_ok=True)
    note = (f"All rows use the frozen LLaVA-v1.5-7B; AFC trained on "
            f"{'the whole data (544k)' if args.recipe == 'all' else '75k samples'}; "
            "Avg. over the six benchmarks shown.")
    (out / "budget.md").write_text("# Token budget\n\n" + note + "\n\n" + "\n".join(md) + "\n")
    rows = [l for l in md[2:] if "Compression" not in l]
    done = sum(PENDING not in l for l in rows)
    abl_md = ablation_table(order, head, out)
    DOC.write_text(
        "# AFC 결과 표 (자동 갱신)\n\n"
        f"마지막 갱신: {datetime.now():%Y-%m-%d %H:%M} · 완료 {done}/{len(rows)}행 · "
        f"{PENDING} = 실험 중 또는 대기\n\n"
        "모든 행은 원본 LLaVA-v1.5-7B(비전 인코더, 프로젝터, LLM 고정)로 직접 측정한 값이다. "
        "AFC는 압축기만 LLaVA-1.5 instruction 데이터 544k로 학습했다. "
        "Avg.는 표의 6개 벤치마크 평균이고, 압축률 구간마다 최고값을 굵게 표시했다.\n\n"
        + "\n".join(md) + "\n\n"
        "## 표 2. 소거 실험: KD와 패딩 (전체 데이터 544k, 64토큰)\n\n"
        "모든 행은 LLM 고정 LLaVA-v1.5-7B, 64토큰, 학습 데이터 544k(전체). "
        "괄호는 같은 방식의 기본 설정(KD 사용, 패딩 포함) 대비 차이.\n\n" + "\n".join(abl_md) + "\n\n"
        "- 생성 스크립트: `scripts/afc/plot_budget.py` · LaTeX: `outputs/report/budget.tex` · "
        "그래프: `outputs/report/budget.png`\n")
    (out / "budget.tex").write_text("% " + note + "\n" + "\n".join(tex) + "\n")
    print("\n".join(md))

    # ---- figure: average + per-benchmark small multiples ---------------
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.2, 1, 1, 1, 0.05], wspace=0.35,
                          hspace=0.45)
    panels = [("Avg", fig.add_subplot(gs[:, 0]))] + [
        (b, fig.add_subplot(gs[i // 3, 1 + i % 3])) for i, b in enumerate(BENCHMARKS)]
    for key, ax in panels:
        big = key == "Avg"
        ax.axhline(vanilla[key], color="#333333", ls="--", lw=1.2, zorder=1)
        if big:
            ax.text(TOKENS[0], vanilla[key] + 0.15, "LLaVA-1.5, 576 tokens",
                    va="bottom", fontsize=9, color="#333333")
        for m, d in data.items():
            pts = [(t, d[t][key]) for t in TOKENS if d[t] is not None
                   and d[t].get(key) is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, lw=2, ms=8 if big else 5, label=m, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.2, **STYLE[m])
            if big:
                ax.text(xs[-1] + 6, ys[-1], m, va="center", fontsize=9,
                        color=STYLE[m]["color"], weight="bold")
                for x, y in pts:
                    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=8, color="#333333")
        ax.set_xticks(TOKENS)
        ax.set_xlim(40, 330 if big else 280)
        ax.set_title("6-benchmark average" if big else key, fontsize=12 if big else 10)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8 if not big else 9)
        if big:
            ax.set_xlabel("visual tokens")
            ax.set_ylabel("score")
    panels[0][1].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle("Visual token budget (frozen LLaVA-v1.5-7B, measured)", fontsize=13, y=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"budget.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(out / "budget.png")


if __name__ == "__main__":
    main()
