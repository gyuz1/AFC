"""Frequency-to-visual adapter: d -> 1024-D CLIP-like features.

The frozen LLaVA ``mm_projector`` expects CLIP patch features, so pooled
d-dimensional tokens are mapped back to the CLIP width.
"""

from torch import nn


def build_adapter(adapter_type: str, dim: int, out_dim: int,
                  hidden: int) -> nn.Module:
    if adapter_type == "linear":
        return nn.Linear(dim, out_dim)
    if adapter_type == "mlp":
        return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                             nn.GELU(), nn.Linear(hidden, out_dim))
    raise ValueError(f"unknown adapter_type {adapter_type!r}")
