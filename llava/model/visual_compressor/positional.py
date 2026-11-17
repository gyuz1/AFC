"""Learned factorized 2-D position embeddings (frequency or block position)."""

import torch
from torch import nn


class Factorized2dEmbedding(nn.Module):
    """``e[a, b] = e_a[a] + e_b[b]`` for an ``n_a x n_b`` grid."""

    def __init__(self, n_a: int, n_b: int, dim: int, std: float = 0.02):
        super().__init__()
        self.n_a, self.n_b = n_a, n_b
        self.emb_a = nn.Parameter(torch.randn(n_a, dim) * std)
        self.emb_b = nn.Parameter(torch.randn(n_b, dim) * std)

    def forward(self) -> torch.Tensor:
        """``(n_a * n_b, dim)`` in raster order (a outer, b inner)."""
        grid = self.emb_a[:, None, :] + self.emb_b[None, :, :]
        return grid.reshape(self.n_a * self.n_b, -1)
