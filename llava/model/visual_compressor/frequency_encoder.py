"""Token encoder shared by all compressor modes.

    X_{j,n} = Linear_{D->d}(LayerNorm_D(F_{j,n})) [+ e_inner(n)] [+ e_block(j)]
    H_j     = TransformerEncoder(X_j)     (or H_j = X_j when disabled)

``LayerNorm_D`` normalises the 1024-D input coefficients (a coefficient's
magnitude is therefore not passed through as an implicit energy prior);
the Transformer's own Pre-LN norms are a separate, internal mechanism.
The D -> d projection is a channel bottleneck: the token count is unchanged.
"""

import torch
from torch import nn

from .positional import Factorized2dEmbedding


class FrequencyTokenEncoder(nn.Module):
    def __init__(self, input_dim: int, dim: int, inner_side: int,
                 blocks_side: int, use_inner_pos: bool, use_block_pos: bool,
                 use_transformer: bool, layers: int, heads: int,
                 ffn_dim: int, dropout: float, pos_std: float = 0.02):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, dim)
        self.inner_pos = (Factorized2dEmbedding(inner_side, inner_side, dim,
                                                pos_std)
                          if use_inner_pos else None)
        self.block_pos = (Factorized2dEmbedding(blocks_side, blocks_side, dim,
                                                pos_std)
                          if use_block_pos else None)
        if use_transformer and layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=ffn_dim,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True)
            # enable_nested_tensor only matters with padding masks (none here)
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=layers, enable_nested_tensor=False)
        else:
            self.transformer = None

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, J, N, D)`` -> ``(B, J, N, d)`` (projection + positions)."""
        x = self.input_proj(self.input_norm(tokens))
        if self.inner_pos is not None:
            x = x + self.inner_pos()[None, None]          # (1, 1, N, d)
        if self.block_pos is not None:
            x = x + self.block_pos()[None, :, None]       # (1, J, 1, d)
        return x

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, J, N, D)`` -> ``H`` ``(B, J, N, d)``."""
        x = self.embed(tokens)
        if self.transformer is None:
            return x
        bsz, j, n, d = x.shape
        h = self.transformer(x.reshape(bsz * j, n, d))    # blocks independent
        return h.reshape(bsz, j, n, d)
