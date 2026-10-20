#!/usr/bin/env bash
# Unattended pipeline for the main AFC model (block DCT, 64 tokens):
#   wait_data -> smoke -> profile -> train -> eval_quick -> eval_full
#
# Launch detached (survives logout):
#   setsid nohup bash scripts/afc/pipeline_main.sh > /dev/null 2>&1 &
# Resume from a stage after fixing a failure:
#   START_STAGE=train setsid nohup bash scripts/afc/pipeline_main.sh ... &
#
# Progress: outputs/pipeline/main/status (one line per stage event) and
#           outputs/pipeline/main/<stage>.log ; pipeline.log is the driver.
# Env: MAX_TRAIN_SAMPLES (10000; "all" = whole dataset), TRAIN_GPU (0, used
#      for profiling), TRAIN_GPUS (training: "0" or "0,1" for DDP),
#      EXTRA_CONFIG (yaml applied after the base config), RESUME (checkpoint
#      dir to resume the train stage from), PROFILE_BS ("1 2 4 8"),
#      EVAL_GPUS ("0 1"),
#      MEM_LIMIT_GIB (22), TARGET_EFFECTIVE_BATCH (16), QUICK_LIMIT (500),
#      WANDB_PROJECT (afc; credentials from `wandb login`, never stored here),
#      USE_WANDB (1).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO}"

export PY="${PY:-python}"
export AFC_DATA="${AFC_DATA:-$(realpath -m "${REPO}/../afc_common/data")}"
export HF_HOME="${HF_HOME:-${AFC_DATA}/hf_cache}"
export DATA_PATH="${DATA_PATH:-${AFC_DATA}/llava_train/llava_v1_5_available.json}"
export IMAGE_FOLDER="${IMAGE_FOLDER:-${AFC_DATA}/llava_train}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-10000}"
[[ "${MAX_TRAIN_SAMPLES}" == all ]] && MAX_TRAIN_SAMPLES=""
TRAIN_GPU="${TRAIN_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-${TRAIN_GPU}}"
N_TRAIN_GPUS="$(echo "${TRAIN_GPUS}" | tr ',' '\n' | grep -c .)"
CFG=()
[[ -n "${EXTRA_CONFIG:-}" ]] && CFG=("${EXTRA_CONFIG}")
read -r -a PROFILE_BS <<< "${PROFILE_BS:-1 2 4 8}"
read -r -a EVAL_GPUS <<< "${EVAL_GPUS:-0 1}"
MEM_LIMIT_GIB="${MEM_LIMIT_GIB:-22}"
TARGET_EFFECTIVE_BATCH="${TARGET_EFFECTIVE_BATCH:-16}"
QUICK_LIMIT="${QUICK_LIMIT:-500}"
RUN_NAME="${RUN_NAME:-main}"
export WANDB_PROJECT="${WANDB_PROJECT:-afc}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_ARGS=()
[[ "${USE_WANDB}" == 1 ]] && WANDB_ARGS=(--report_to wandb \
    --run_name "afc-${RUN_NAME}-$(date +%m%d_%H%M)")
CKPT="checkpoints/llava-v1.5-7b-afc-${RUN_NAME}"
DIR="outputs/pipeline/${RUN_NAME}"
STAGES=(wait_data smoke profile train eval_quick eval_full)
mkdir -p "${DIR}"
echo $$ > "${DIR}/pid"
exec >> "${DIR}/pipeline.log" 2>&1
echo "===== pipeline start $(date '+%F %T') pid $$ (START_STAGE=${START_STAGE:-wait_data})"

status() { echo "$(date '+%F %T') $*" | tee -a "${DIR}/status"; }

run_stage() {  # run_stage <name> <function>
    status "STAGE $1 START"
    if "$2" > "${DIR}/$1.log" 2>&1; then
        status "STAGE $1 OK"
    else
        status "STAGE $1 FAIL (see ${DIR}/$1.log)"
        exit 1
    fi
}

# ---------------------------------------------------------------- stages
wait_data() {
    while pgrep -f download_llava_train.sh > /dev/null \
            || [[ ! -s "${DATA_PATH}" ]]; do
        echo "$(date '+%T') waiting for ${DATA_PATH}"
        sleep 120
    done
    ${PY} - <<'EOF'
import json, os
d = json.load(open(os.environ["DATA_PATH"]))
missing = [x["image"] for x in d[:2000]
           if not os.path.exists(os.path.join(os.environ["IMAGE_FOLDER"], x["image"]))]
print(f"{len(d)} samples; missing images in first 2000: {len(missing)}")
assert not missing, missing[:5]
EOF
}

smoke() {
    rm -rf checkpoints/llava-v1.5-7b-afc-pipeline_smoke
    GPU="${TRAIN_GPU}" MAX_TRAIN_SAMPLES=64 bash scripts/afc/train.sh \
        pipeline_smoke "${CFG[@]}" --max_steps 8 --logging_steps 2 \
        --save_strategy no --report_to none \
        --per_device_train_batch_size 2 --gradient_accumulation_steps 1
    grep -q "saved compressor checkpoint" \
        checkpoints/llava-v1.5-7b-afc-pipeline_smoke/train.log
    rm -rf checkpoints/llava-v1.5-7b-afc-pipeline_smoke
}

profile() {
    local best=0
    for bs in "${PROFILE_BS[@]}"; do
        rm -rf "checkpoints/llava-v1.5-7b-afc-memprofile_bs${bs}"
        if GPU="${TRAIN_GPU}" MAX_TRAIN_SAMPLES=$((bs * 6)) \
                bash scripts/afc/train.sh "memprofile_bs${bs}" "${CFG[@]}" \
                --save_strategy no --report_to none \
                --per_device_train_batch_size "${bs}" \
                --gradient_accumulation_steps 1 --max_steps 6 \
                --logging_steps 1 --dataloader_num_workers 2 \
                --subset_strategy longest; then
            peak="$(grep -oP 'peak GPU memory: \K[0-9.]+' \
                "checkpoints/llava-v1.5-7b-afc-memprofile_bs${bs}/train.log")"
            echo "bs=${bs} peak=${peak} GiB"
            if awk "BEGIN{exit !(${peak} <= ${MEM_LIMIT_GIB})}"; then
                best="${bs}"
            fi
        else
            echo "bs=${bs} failed (OOM?)"
        fi
        rm -rf "checkpoints/llava-v1.5-7b-afc-memprofile_bs${bs}"
        [[ "${best}" == "${bs}" ]] || break
    done
    [[ "${best}" -ge 1 ]] || { echo "no batch size fits"; return 1; }
    echo "${best}" > "${DIR}/batch_size"
    echo "chosen per-device batch ${best} on ${N_TRAIN_GPUS} GPU(s)," \
         "accumulation $(accum_for "${best}")"
}

accum_for() {  # gradient accumulation for per-device batch $1
    local a=$((TARGET_EFFECTIVE_BATCH / ($1 * N_TRAIN_GPUS)))
    echo $((a > 0 ? a : 1))
}

train() {
    local bs resume=()
    bs="$(cat "${DIR}/batch_size" 2>/dev/null || echo 4)"
    [[ -n "${RESUME:-}" ]] && resume=(--resume_from_checkpoint "${RESUME}")
    GPU="${TRAIN_GPUS}" MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES}" \
        bash scripts/afc/train.sh "${RUN_NAME}" "${CFG[@]}" \
        --per_device_train_batch_size "${bs}" \
        --gradient_accumulation_steps "$(accum_for "${bs}")" \
        --return_compressor_attention true "${WANDB_ARGS[@]}" "${resume[@]}"
    test -f "${CKPT}/visual_compressor.bin"
}

eval_quick() {
    GPU="${EVAL_GPUS[0]}" LIMIT="${QUICK_LIMIT}" \
        bash scripts/afc/run_eval.sh "${CKPT}" outputs/eval
}

eval_full() {
    local a=(textvqa gqa) b=(mmbench pope scienceqa mmmu textvqa_tune gqa_tune)  # ~equal GPU time
    if [[ ${#EVAL_GPUS[@]} -ge 2 ]]; then
        GPU="${EVAL_GPUS[0]}" BENCH="${a[*]}" bash scripts/afc/run_eval.sh \
            "${CKPT}" outputs/eval > "${DIR}/eval_full_a.log" 2>&1 &
        local pa=$!
        GPU="${EVAL_GPUS[1]}" BENCH="${b[*]}" bash scripts/afc/run_eval.sh \
            "${CKPT}" outputs/eval > "${DIR}/eval_full_b.log" 2>&1 &
        local pb=$!
        wait "${pa}"; local ra=$?
        wait "${pb}"; local rb=$?
        [[ ${ra} -eq 0 && ${rb} -eq 0 ]] || return 1
    else
        GPU="${EVAL_GPUS[0]}" BENCH="${a[*]} ${b[*]}" \
            bash scripts/afc/run_eval.sh "${CKPT}" outputs/eval
    fi
    local wb=()
    [[ "${USE_WANDB}" == 1 ]] && wb=(--wandb)
    ${PY} -m llava.eval.afc.table "outputs/eval/$(basename "${CKPT}")" \
        "${wb[@]}"
}

# ---------------------------------------------------------------- driver
started=0
for s in "${STAGES[@]}"; do
    [[ "${s}" == "${START_STAGE:-wait_data}" ]] && started=1
    [[ ${started} -eq 1 ]] && run_stage "${s}" "${s}"
done
status "PIPELINE DONE"
