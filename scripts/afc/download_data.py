"""Download benchmarks, tuning sources, and model weights into $AFC_DATA.

Layout of $AFC_DATA (default <repo>/../afc_common/data):
    benchmarks/<name>/   evaluation data (LLaVA official or lmms-eval)
    tuning/<name>/       hyperparameter tuning sources (train splits)
    models/              model weights
    hf_cache/            HF_HOME; lmms-eval reads datasets from here
    downloads/           raw archives

Usage (conda env ``afc``):
    python scripts/afc/download_data.py                 # everything
    python scripts/afc/download_data.py --only textvqa gqa
"""

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

from llava.eval.afc import AFC_DATA as DATA  # sets HF_HOME / HF_TOKEN_PATH
from llava.eval.afc import BENCH, DOWNLOADS, TUNING

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import snapshot_download  # noqa: E402  (after HF_HOME)

# LLaVA eval bundle (LLaVA docs/Evaluation.md).
LLAVA_EVAL_GDRIVE_ID = "1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy"
TEXTVQA_URL = "https://dl.fbaipublicfiles.com/textvqa"
MMBENCH_URL = ("https://download.openmmlab.com/mmclassification/datasets/"
               "mmbench/mmbench_dev_20230712.tsv")

# lmms-eval datasets: (benchmark dir, repo id, allow_patterns)
HF_BENCHMARKS = {
    "gqa": ("lmms-lab/GQA", ["testdev_balanced_*/*", "README.md"]),
    "pope": ("lmms-lab/POPE", ["data/*", "README.md"]),
    "scienceqa": ("lmms-lab/ScienceQA", ["ScienceQA-IMG/test-*", "README.md"]),
    "mmmu": ("lmms-lab/MMMU", ["data/validation-*", "data/dev-*",
                               "README.md"]),
}


def wget(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest}")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["wget", "-q", "-c", "-O", str(part),
                    url], check=True)
    part.rename(dest)


def link(target, name):
    """Create a readable symlink ``name`` -> ``target``."""
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(target)


def hf_dataset(name):
    repo_id, patterns = HF_BENCHMARKS[name]
    print(f"[hf] {repo_id} {patterns}")
    path = snapshot_download(repo_id, repo_type="dataset",
                             allow_patterns=patterns)
    link(Path(path), BENCH / name / "hf")


def llava_eval_bundle():
    """Extract LLaVA question files for our benchmarks (skip answers)."""
    import gdown

    archive = DOWNLOADS / "llava_eval.zip"
    if not archive.exists():
        gdown.download(id=LLAVA_EVAL_GDRIVE_ID, output=str(archive))
    keep = {"textvqa", "mmbench", "gqa", "pope", "scienceqa"}
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            parts = Path(member.filename).parts
            if member.is_dir() or parts[0] not in keep:
                continue
            if any(p.startswith("answers") for p in parts):
                continue
            out = BENCH / Path(*parts)
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(member))


def textvqa():
    llava_eval_bundle()
    tv = BENCH / "textvqa"
    wget(f"{TEXTVQA_URL}/data/TextVQA_0.5.1_val.json",
         tv / "TextVQA_0.5.1_val.json")
    archive = DOWNLOADS / "textvqa_train_val_images.zip"
    wget(f"{TEXTVQA_URL}/images/train_val_images.zip", archive)
    if not (tv / "train_images").exists():
        print("[unzip] TextVQA images")
        subprocess.run(["unzip", "-q", str(archive), "-d", str(tv)],
                       check=True)

    # Tuning: train questions share the same image folder (disjoint ids).
    tt = TUNING / "textvqa"
    wget(f"{TEXTVQA_URL}/data/TextVQA_0.5.1_train.json",
         tt / "TextVQA_0.5.1_train.json")
    # OCR tokens (LLaVA's val prompt uses Rosetta v0.2).
    for split in ("train", "val"):
        name = f"TextVQA_Rosetta_OCR_v0.2_{split}.json"
        wget(f"{TEXTVQA_URL}/data/{name}", tt / name)
    link(Path("../../benchmarks/textvqa/train_images"), tt / "train_images")


def mmbench():
    llava_eval_bundle()
    wget(MMBENCH_URL, BENCH / "mmbench" / "mmbench_dev_20230712.tsv")


def gqa():
    hf_dataset("gqa")


def gqa_tuning():
    path = snapshot_download(
        "lmms-lab/GQA", repo_type="dataset",
        allow_patterns=["train_balanced_*/*", "README.md"])
    link(Path(path), TUNING / "gqa" / "hf")


def models():
    snapshot_download("liuhaotian/llava-v1.5-7b",
                      local_dir=DATA / "models" / "llava-v1.5-7b")
    # LLaVA resolves the vision tower by repo id, so keep it in HF_HOME.
    snapshot_download("openai/clip-vit-large-patch14-336",
                      allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])


STAGES = {
    "textvqa": textvqa,
    "mmbench": mmbench,
    "gqa": gqa,
    "pope": lambda: hf_dataset("pope"),
    "scienceqa": lambda: hf_dataset("scienceqa"),
    "mmmu": lambda: hf_dataset("mmmu"),
    "gqa_tuning": gqa_tuning,
    "models": models,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(STAGES),
                        default=list(STAGES))
    args = parser.parse_args()

    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in args.only:
        print(f"===== {name} =====", flush=True)
        try:
            STAGES[name]()
        except Exception as exc:  # keep going; report at the end
            print(f"[FAIL] {name}: {exc!r}", flush=True)
            failed.append(name)
    print("failed:", failed or "none")


if __name__ == "__main__":
    main()
