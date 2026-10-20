#!/usr/bin/env bash
# Train one compressor configuration on one GPU.
#
#   GPU=0 DATA_PATH=<llava json> IMAGE_FOLDER=<image root> \
#       bash scripts/afc/train.sh <run name> [ablation yaml] [extra flags...]
#
# Base config: configs/afc/block_dct_64.yaml; the optional ablation yaml
# overrides single keys. Output: checkpoints/llava-v1.5-7b-afc-<run name>
# (a complete LLaVA model dir: config + visual_compressor.bin + links).
# Env: MODEL (default $AFC_DATA/models/llava-v1.5-7b), MAX_TRAIN_SAMPLES,
#      PY, AFC_DATA, GPU (one id, or a list like "0,1" -> DDP via torchrun).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO}"
NAME="${1:?run name}"; shift
EXTRA_CFG=()
if [[ $# -gt 0 && "$1" == *.yaml ]]; then EXTRA_CFG=(--config "$1"); shift; fi
AFC_DATA="$(realpath -m "${AFC_DATA:-${REPO}/../afc_common/data}")"
export HF_HOME="${HF_HOME:-${AFC_DATA}/hf_cache}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1   # log lines reach train.log immediately (tee)
# Long multi-turn samples fragment the allocator; expandable segments help.
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
