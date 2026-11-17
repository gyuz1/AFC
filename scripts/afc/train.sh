#!/usr/bin/env bash
# Train one compressor configuration.
#   GPU=0 bash scripts/afc/train.sh <run name> [config yaml] [extra flags...]
# GPU="0,1" runs DDP; output: checkpoints/llava-v1.5-7b-afc-<run name>/
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO}"
NAME="${1:?run name}"; shift
EXTRA_CFG=()
if [[ $# -gt 0 && "$1" == *.yaml ]]; then EXTRA_CFG=(--config "$1"); shift; fi
AFC_DATA="$(realpath -m "${AFC_DATA:-${REPO}/../afc_common/data}")"
export HF_HOME="${HF_HOME:-${AFC_DATA}/hf_cache}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PY:-python}"
MODEL="${MODEL:-${AFC_DATA}/models/llava-v1.5-7b}"
OUT="checkpoints/llava-v1.5-7b-afc-${NAME}"
mkdir -p "${OUT}"
NPROC="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)"
if [[ "${NPROC}" -gt 1 ]]; then
    LAUNCH=(${PY} -m torch.distributed.run --nproc_per_node "${NPROC}"
            --master_port "${MASTER_PORT:-29517}")
else
    LAUNCH=(${PY})
fi
"${LAUNCH[@]}" -m llava.train.train_compressor \
    --config configs/afc/block_dct_64.yaml "${EXTRA_CFG[@]}" \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH:?DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER:?IMAGE_FOLDER}" \
    ${MAX_TRAIN_SAMPLES:+--max_train_samples ${MAX_TRAIN_SAMPLES}} \
    --output_dir "${OUT}" "$@" 2>&1 | tee -a "${OUT}/train.log"
