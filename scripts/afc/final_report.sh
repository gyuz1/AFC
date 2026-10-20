#!/usr/bin/env bash
# Wait for pipeline runs to finish, then write one combined results table.
#   setsid nohup bash scripts/afc/final_report.sh block75k global75k &
# Output: outputs/report/final_<date>.{txt,md,tex} (text / markdown / LaTeX)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO}"
PY="${PY:-python}"
RUNS=("$@")
OUT="outputs/report"
mkdir -p "${OUT}"
for r in "${RUNS[@]}"; do
    until grep -qE "PIPELINE DONE|FAIL" "outputs/pipeline/${r}/status" 2>/dev/null; do
        sleep 300
    done
done
# every finished evaluation (full splits) plus quick (LIMIT) runs, flagged
DIRS=()
for d in outputs/eval/llava-v1.5-7b*; do
    [[ -d "${d}" && -f "${d}/model_config.json" ]] && DIRS+=("${d}")
done
STAMP="$(date +%m%d_%H%M)"
for fmt in text markdown latex; do
    ext=$([[ ${fmt} == text ]] && echo txt || ([[ ${fmt} == markdown ]] && echo md || echo tex))
    ${PY} -m llava.eval.afc.table "${DIRS[@]}" --paper --format "${fmt}" \
        > "${OUT}/final_${STAMP}.${ext}"
done
{ echo "status:"; for r in "${RUNS[@]}"; do echo "  ${r}: $(tail -1 outputs/pipeline/${r}/status)"; done; } \
    >> "${OUT}/final_${STAMP}.txt"
ln -sfn "final_${STAMP}.txt" "${OUT}/latest.txt"
cat "${OUT}/final_${STAMP}.txt"
