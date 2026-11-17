#!/usr/bin/env bash
# Create the `afc` conda env.   bash scripts/afc/setup_env.sh
# transformers 4.37.2 (pinned by LLaVA) requires lmms-eval v0.2.0.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WS="$(realpath -m "${AFC_WORKSPACE:-${REPO}/../afc_common}")"
ENV_NAME="${ENV_NAME:-afc}"

eval "$(conda shell.bash hook)"
conda env list | grep -q "^${ENV_NAME} " \
    || conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

mkdir -p "${WS}/third_party" "${WS}/data"
cd "${WS}/third_party"
[[ -d lmms-eval ]] || git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
[[ -d LLaVA ]] || git clone https://github.com/haotian-liu/LLaVA.git
git -C lmms-eval checkout -q v0.2.0
git -C LLaVA checkout -q c121f04

pip install torch==2.1.2 torchvision==0.16.2
pip install -e "${REPO}"
pip install -e lmms-eval
pip install calflops gdown pyarrow pandas pytest hf_transfer ipykernel
pip install "numpy==1.26.4" "accelerate==0.29.3" "peft==0.10.0" \
    "httpx==0.23.3"
# Optional reference implementation of Fourier-VLM, if checked out:
[[ -d "${REPO}/../Fourier-Compressor-baseline" ]] \
    && pip install --no-deps -e "${REPO}/../Fourier-Compressor-baseline"

python -c "import torch, transformers, llava, lmms_eval; \
print(torch.__version__, transformers.__version__, llava.__file__, \
torch.cuda.device_count())"
