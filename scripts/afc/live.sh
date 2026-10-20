#!/usr/bin/env bash
# Live view of running trainings (no stdout-buffer lag): progress bar and the
# latest logged losses, read from W&B's console capture (wandb/run-*/files/
# output.log), refreshed every INTERVAL seconds. Ctrl+C stops the view only.
#   bash scripts/afc/live.sh [run ...]      (default: block75k_fp global75k_fp)
#   INTERVAL=5 bash scripts/afc/live.sh block75k
#   ONCE=1 bash scripts/afc/live.sh                (print once and exit)
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO}"
RUNS=("$@"); [[ ${#RUNS[@]} -gt 0 ]] || RUNS=(block75k_fp global75k_fp)
PY="${PY:-python}"
src() {  # train.log: unbuffered since PYTHONUNBUFFERED (older runs: W&B capture)
    local L="checkpoints/llava-v1.5-7b-afc-$1/train.log" id f
    if grep -q "'loss'" "${L}" 2>/dev/null || [[ ! -f "${L}" ]]; then
        echo "${L}"; return
    fi
    id="$(grep -oE 'wandb.ai/[^ ]*/runs/[a-z0-9]+' "${L}" | head -1 | grep -oE '[a-z0-9]+$')"
    f="$(ls wandb/run-*-"${id}"/files/output.log 2>/dev/null | tail -1)"
    echo "${f:-${L}}"
}
while true; do
    [[ -n "${ONCE:-}" ]] || clear
    date '+%F %T'
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    for r in "${RUNS[@]}"; do
        f="$(src "${r}")"
        echo
        echo "== ${r}   ($(tail -1 "outputs/pipeline/${r}/status" 2>/dev/null | cut -c1-60))"
        L="checkpoints/llava-v1.5-7b-afc-${r}/train.log"   # progress bar: stderr -> train.log
        echo "  iter : $(tr '\r' '\n' < "${L}" 2>/dev/null | grep -oE ' [0-9]+/[0-9]+ \[[^]]*\]' | tail -1)"
        grep -oE "\{'loss'[^}]*\}" "${f}" | tail -5 \
            | ${PY} scripts/afc/fmt_log_lines.py
    done
    [[ -n "${ONCE:-}" ]] && break
    sleep "${INTERVAL:-10}"
done
