"""AFC evaluation / data tooling: paths shared by scripts and tools.

Large files (benchmarks, checkpoints, HF cache) live outside the repository
in ``AFC_DATA`` (default: ``<repo>/../afc_common/data``); the versioned
tuning splits live in ``<repo>/splits/afc``.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AFC_DATA = Path(os.environ.get(
    "AFC_DATA", REPO_ROOT.parent / "afc_common" / "data")).resolve()
BENCH = AFC_DATA / "benchmarks"
TUNING = AFC_DATA / "tuning"
DOWNLOADS = AFC_DATA / "downloads"
MODEL_PATH = AFC_DATA / "models" / "llava-v1.5-7b"
SPLITS = REPO_ROOT / "splits" / "afc"

os.environ.setdefault("HF_HOME", str(AFC_DATA / "hf_cache"))
_TOKEN = Path.home() / ".cache" / "huggingface" / "token"
if _TOKEN.exists():
    os.environ.setdefault("HF_TOKEN_PATH", str(_TOKEN))
