"""Build the compressor from a LLaVA model config.

The compressor settings live in ``config.visual_compressor`` (a dict saved in
the checkpoint's config.json). Missing or ``compression_mode=none`` -> None,
so vanilla checkpoints build no extra module.
"""

import contextlib
import os
from typing import Optional

import torch

from .compressor import VisualCompressor
from .config import PARAMETER_FREE_MODES, VisualCompressorConfig
from .fixed_compressors import ParameterFreeCompressor

WEIGHTS_NAME = "visual_compressor.bin"


def compressor_config_from(model_config,
                           vision_tower=None) -> VisualCompressorConfig:
    """Read ``config.visual_compressor``; fill grid / width from the tower."""
    cfg = VisualCompressorConfig.from_dict(
        getattr(model_config, "visual_compressor", None))
    if cfg.enabled and (cfg.grid_size is None or cfg.input_dim is None):
        if vision_tower is None:
            raise ValueError("grid_size / input_dim missing and no vision "
                             "tower to read them from")
        cfg.resolve(vision_tower.num_patches_per_side,
                    vision_tower.hidden_size)
    return cfg.validate()


def create_compressor(cfg: VisualCompressorConfig):
    """Learned (VisualCompressor) or parameter-free baseline, by mode."""
    if cfg.compression_mode in PARAMETER_FREE_MODES:
        return ParameterFreeCompressor(cfg)
    return VisualCompressor(cfg)


def build_visual_compressor(model_config,
                            vision_tower=None) -> Optional[VisualCompressor]:
    cfg = compressor_config_from(model_config, vision_tower)
    return create_compressor(cfg) if cfg.enabled else None


def load_compressor_weights(model, model_path: str) -> bool:
    """Load ``visual_compressor.bin`` from a checkpoint directory.

    Returns False when the model has no compressor (vanilla). Raises when a
    compressor is configured but its weights are missing: an untrained
    compressor must never be evaluated silently.
    """
    compressor = model.get_model().get_visual_compressor()
    if compressor is None:
        return False
    path = os.path.join(model_path, WEIGHTS_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"config.json enables visual_compressor "
            f"({compressor.cfg.compression_mode}) but {path} is missing")
    state = torch.load(path, map_location="cpu")
    if any(k.startswith(("compressor.", "mm_projector.")) for k in state):
        comp = {k[len("compressor."):]: v for k, v in state.items()
                if k.startswith("compressor.")}
        proj = {k[len("mm_projector."):]: v for k, v in state.items()
                if k.startswith("mm_projector.")}
    else:                                   # flat format (compressor only)
        comp, proj = state, {}
    compressor.load_state_dict(comp, strict=True)   # {} for parameter-free
    if proj:  # projector trained together with the compressor
        model.get_model().mm_projector.load_state_dict(proj, strict=True)
    # HF warns that visual_compressor.* were "newly initialized" because the
    # LLM shards do not contain them; they are replaced here.
    print(f"[afc] loaded {len(comp)} compressor"
          f"{f' + {len(proj)} mm_projector' if proj else ''} tensors from {path} "
          f"({compressor.cfg.compression_mode}, "
          f"{compressor.cfg.target_visual_tokens} visual tokens)")
    return True


@contextlib.contextmanager
def compressor_disabled(model):
    """Temporarily take the literal vanilla path (e.g. for a KD teacher)."""
    inner = model.get_model()
    prev = getattr(inner, "_visual_compressor_enabled", True)
    inner._visual_compressor_enabled = False
    try:
        yield model
    finally:
        inner._visual_compressor_enabled = prev
