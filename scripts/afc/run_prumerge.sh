#!/usr/bin/env bash
# PruMerge baseline (fixed token budget):  GPU=0 bash scripts/afc/run_prumerge.sh [64 144 256]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
for t in "${@:-64 144 256}"; do
    "${PY}" scripts/afc/make_param_free_ckpt.py --mode prumerge --tokens "${t}"
    GPU="${GPU:-0}" bash scripts/afc/run_eval.sh "checkpoints/llava-v1.5-7b-afc-prumerge${t}" outputs/eval
done
