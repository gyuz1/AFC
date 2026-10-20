#!/usr/bin/env bash
# Evaluate ONE LLaVA model directory on ONE GPU (batch size 1).
#   vanilla:    the original llava-v1.5-7b directory
#   compressor: a train_compressor checkpoint directory (config.json carries
#               the compressor settings; weights in visual_compressor.bin)
#
# Usage:
#   GPU=0 LIMIT=500 bash scripts/afc/run_eval.sh <model dir> [out root]
#
# Env:
#   GPU      CUDA device index (default 0)
#   LIMIT    dev runs: questions per benchmark (MMBench: base questions);
#            empty = full split (final tables)
#   BENCH    subset of: textvqa mmbench gqa pope scienceqa mmmu
#            textvqa_tune gqa_tune   (default: the 6 evaluation benchmarks)
#   TUNE_N   questions of the tuning splits (default 1000)
#   PY       python of the `afc` env
# Outputs: <out root>/<model dir name>[_limitN]/   (out root default:
#          ./outputs/eval)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AFC_DATA="$(realpath -m "${AFC_DATA:-${REPO}/../afc_common/data}")"
MODEL="$(realpath "${1:?model dir}")"
OUT_ROOT="$(realpath -m "${2:-outputs/eval}")"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HOME="${HF_HOME:-${AFC_DATA}/hf_cache}"
[[ -f "${HOME}/.cache/huggingface/token" ]] \
    && export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
# machines without an HF login: token in <workspace>/.hf_token (mode 600)
_HFT="$(dirname "${REPO:-$(pwd)}")/.hf_token"
[[ -z "${HF_TOKEN:-}" && -z "${HF_TOKEN_PATH:-}" && -f "${_HFT}" ]] && export HF_TOKEN="$(cat "${_HFT}")"
PY="${PY:-python}"
BENCH="${BENCH:-textvqa mmbench gqa pope scienceqa mmmu}"
LIMIT="${LIMIT:-}"
TUNE_N="${TUNE_N:-1000}"
EVAL="${AFC_DATA}/benchmarks"
SPLITS="${REPO}/splits/afc"

TAG="$(basename "${MODEL}")"
RUN="${OUT_ROOT}/${TAG}${LIMIT:+_limit${LIMIT}}"
mkdir -p "${RUN}"
echo "[run_eval] ${TAG} gpu=${CUDA_VISIBLE_DEVICES} limit=${LIMIT:-full}" \
     "-> ${RUN}"
cp -f "${MODEL}/config.json" "${RUN}/model_config.json" 2>/dev/null || true  # parallel halves race on this copy

subset_jsonl() {  # subset_jsonl <in> <n> <out>
    if [[ -n "$2" ]]; then head -n "$2" "$1" > "$3"; else cp "$1" "$3"; fi
}

llava_vqa() {  # llava_vqa <name> <question jsonl> <image dir>
    local name="$1" qfile="$2" imgdir="$3"
    ${PY} -m llava.eval.model_vqa_loader --model-path "${MODEL}" \
        --question-file "${qfile}" --image-folder "${imgdir}" \
        --answers-file "${RUN}/${name}_answers.jsonl" \
        --temperature 0 --conv-mode vicuna_v1
}

declare -A TASK=([gqa]=gqa [pope]=pope [scienceqa]=scienceqa_img
                 [mmmu]=mmmu_val)

for bench in ${BENCH}; do
    echo "===== ${bench} ====="
    case "${bench}" in
    textvqa)
        subset_jsonl "${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
            "${LIMIT}" "${RUN}/textvqa_questions.jsonl"
        llava_vqa textvqa "${RUN}/textvqa_questions.jsonl" \
            "${EVAL}/textvqa/train_images"
        ${PY} -m llava.eval.eval_textvqa \
            --annotation-file "${EVAL}/textvqa/TextVQA_0.5.1_val.json" \
            --result-file "${RUN}/textvqa_answers.jsonl" \
            | tee "${RUN}/textvqa_score.txt"
        ;;
    mmbench)
        TSV="${EVAL}/mmbench/mmbench_dev_20230712.tsv"
        if [[ -n "${LIMIT}" ]]; then  # keep whole circular groups
            ${PY} - "${TSV}" "${LIMIT}" "${RUN}/mmbench.tsv" <<'EOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1], sep="\t")
base = df["index"] % 1_000_000
keep = sorted(base.unique())[: int(sys.argv[2])]
df[base.isin(keep)].to_csv(sys.argv[3], sep="\t", index=False)
EOF
            TSV="${RUN}/mmbench.tsv"
        fi
        ${PY} -m llava.eval.model_vqa_mmbench --model-path "${MODEL}" \
            --question-file "${TSV}" \
            --answers-file "${RUN}/mmbench_answers.jsonl" \
            --single-pred-prompt --temperature 0 --conv-mode vicuna_v1
        ${PY} -m llava.eval.afc.metrics mmbench --annotation "${TSV}" \
            --result "${RUN}/mmbench_answers.jsonl" \
            --out "${RUN}/mmbench_score.json"
        ;;
    gqa|pope|scienceqa|mmmu)
        ${PY} -m lmms_eval --model llava \
            --model_args "pretrained=${MODEL},conv_template=vicuna_v1" \
            --tasks "${TASK[${bench}]}" --batch_size 1 --log_samples \
            ${LIMIT:+--limit ${LIMIT}} \
            --output_path "${RUN}/lmms_${bench}"
        ;;
    textvqa_tune)
        subset_jsonl "${SPLITS}/textvqa_train_2000_ocr.jsonl" \
            "${TUNE_N}" "${RUN}/textvqa_tune_questions.jsonl"
        llava_vqa textvqa_tune "${RUN}/textvqa_tune_questions.jsonl" \
            "${AFC_DATA}/tuning/textvqa/train_images"
        ${PY} -m llava.eval.afc.metrics textvqa_tune \
            --gt "${SPLITS}/textvqa_train_2000_gt.jsonl" \
            --result "${RUN}/textvqa_tune_answers.jsonl" \
            --out "${RUN}/textvqa_tune_score.json"
        ;;
    gqa_tune)
        subset_jsonl "${SPLITS}/gqa_train_2000.jsonl" \
            "${TUNE_N}" "${RUN}/gqa_tune_questions.jsonl"
        llava_vqa gqa_tune "${RUN}/gqa_tune_questions.jsonl" \
            "${AFC_DATA}/tuning/gqa/images"
        ${PY} -m llava.eval.afc.metrics gqa_tune \
            --gt "${SPLITS}/gqa_train_2000_gt.jsonl" \
            --result "${RUN}/gqa_tune_answers.jsonl" \
            --out "${RUN}/gqa_tune_score.json"
        ;;
    *) echo "unknown benchmark ${bench}" >&2; exit 1 ;;
    esac
done
echo "[run_eval] done ${TAG}"
${PY} -m llava.eval.afc.table "${RUN}"
