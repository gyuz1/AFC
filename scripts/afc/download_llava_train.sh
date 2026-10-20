#!/usr/bin/env bash
# Download the LLaVA-1.5 instruction-tuning data into $AFC_DATA/llava_train
# and write llava_v1_5_available.json (only samples whose image exists).
#
#   bash scripts/afc/download_llava_train.sh       (~57 GB of zips, resumable)
#
# Image folders expected by llava_v1_5_mix665k.json:
#   coco/train2017  gqa/images  vg/VG_100K  vg/VG_100K_2  textvqa/train_images
#   ocr_vqa/images  (NOT downloaded: per-image crawling; those samples are
#                    dropped from llava_v1_5_available.json)
# KEEP_ZIPS=1 keeps the archives after extraction.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AFC_DATA="$(realpath -m "${AFC_DATA:-${REPO}/../afc_common/data}")"
T="${AFC_DATA}/llava_train"
mkdir -p "${T}"/{coco,gqa,vg,textvqa,zips}
cd "${T}"

get() {  # get <url> <zip name> <extract dir> <marker dir>
    if [[ -d "$4" ]]; then echo "[skip] $4 exists"; return; fi
    wget -c -q --show-progress -O "zips/$2" "$1"
    echo "[unzip] $2"
    unzip -q "zips/$2" -d "$3"
    [[ "${KEEP_ZIPS:-0}" == 1 ]] || rm -f "zips/$2"
}

if [[ ! -f llava_v1_5_mix665k.json ]]; then
    huggingface-cli download liuhaotian/LLaVA-Instruct-150K \
        llava_v1_5_mix665k.json --repo-type dataset --local-dir .
fi
get http://images.cocodataset.org/zips/train2017.zip coco_train2017.zip \
    coco coco/train2017
get https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip gqa_images.zip \
    gqa gqa/images
get https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip vg_part1.zip \
    vg vg/VG_100K
get https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip vg_part2.zip \
    vg vg/VG_100K_2
ln -sfn "${AFC_DATA}/benchmarks/textvqa/train_images" textvqa/train_images

python - <<'PY'
import collections, json, os
d = json.load(open("llava_v1_5_mix665k.json"))
keep = [x for x in d if "image" in x and os.path.exists(x["image"])]
by_src = collections.Counter(x["image"].split("/")[0] for x in keep)
drop = collections.Counter(x["image"].split("/")[0] if "image" in x else
                           "text-only" for x in d
                           if not ("image" in x and os.path.exists(x["image"])))
json.dump(keep, open("llava_v1_5_available.json", "w"))
print(f"kept {len(keep)} / {len(d)}  by source {dict(by_src)}")
print(f"dropped {dict(drop)}")
PY
