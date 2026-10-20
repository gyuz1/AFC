# Source in every new shell / terminal:   source scripts/afc/env.sh
# Activates the `afc` env, cds to the repo and exports the AFC paths.
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-afc}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AFC_DATA="${AFC_DATA:-$(realpath -m "${PWD}/../afc_common/data")}"
export HF_HOME="${HF_HOME:-${AFC_DATA}/hf_cache}"
export MODEL="${MODEL:-${AFC_DATA}/models/llava-v1.5-7b}"
export DATA_PATH="${DATA_PATH:-${AFC_DATA}/llava_train/llava_v1_5_available.json}"
export IMAGE_FOLDER="${IMAGE_FOLDER:-${AFC_DATA}/llava_train}"
echo "[afc] repo=${PWD}"
echo "[afc] AFC_DATA=${AFC_DATA}"
echo "[afc] DATA_PATH=${DATA_PATH} ($( [[ -f ${DATA_PATH} ]] && echo ready || echo 'not downloaded yet'))"
