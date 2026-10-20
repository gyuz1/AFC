"""Learnable query pooling: k queries cross-attend to all N tokens of a block.

    Q = Q_pool W_Q,  K_j = H_j W_K,  V_j = H_j W_V
    A_j = softmax(Q K_j^T / sqrt(d_h))         (per head, k x N, rows sum to 1)
    O_j = concat_heads(A_j V_j) W_O            (k x d)

With ``prenorm`` (default) the queries and the keys/values are LayerNorm-ed
first (Pre-LN cross-attention, as in Perceiver / Q-Former). Without it the
small query init (std 0.02) gives near-zero scores: every query attends
uniformly and all k outputs of a block are identical at initialisation.

Every query sees every input token; outputs are soft combinations, nothing is
selected discretely. One query set is shared by all blocks by default
(``block_specific`` gives each block its own queries, for ablation).
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn


class QueryPooling(nn.Module):
    def __init__(self, dim: int, num_queries: int, heads: int,
                 num_blocks: int = 1, block_specific: bool = False,
                 dropout: float = 0.0, init_std: float = 0.02,
                 prenorm: bool = True, learnable_scale: bool = False):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.dim, self.k, self.heads = dim, num_queries, heads
        self.head_dim = dim // heads
        n_sets = num_blocks if block_specific else 1
        self.block_specific = block_specific
        # (S, k, d) with S = 1 (shared) or J (block-specific)
        self.queries = nn.Parameter(torch.randn(n_sets, num_queries, dim)
                                    * init_std)
        self.norm_q = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.norm_kv = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        # log-scale per head, exp(0) = 1 at init (no change to attention)
        self.log_scale = (nn.Parameter(torch.zeros(heads, 1, 1))
                          if learnable_scale else None)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # (..., L, d) -> (..., heads, L, head_dim)
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)

    def forward(self, h: torch.Tensor
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``h`` ``(B, J, N, d)`` -> ``(O (B, J, k, d), A (B, J, heads, k, N))``."""
        bsz, j, n, d = h.shape
        if self.block_specific and self.queries.shape[0] != j:
            raise ValueError(f"block-specific queries for "
                             f"{self.queries.shape[0]} blocks, got {j}")
        q = self._split(self.q_proj(self.norm_q(self.queries)))  # (S,H,k,hd)
        q = q.unsqueeze(0)                                # (1, S, H, k, hd)
        h = self.norm_kv(h)
        k = self._split(self.k_proj(h))                   # (B, J, H, N, hd)
        v = self._split(self.v_proj(h))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if self.log_scale is not None:
            scores = scores * self.log_scale.exp()        # (H, 1, 1)
        attn = scores.softmax(dim=-1)                     # (B, J, H, k, N)
        out = self.attn_drop(attn) @ v                    # (B, J, H, k, hd)
        out = out.transpose(-3, -2).flatten(-2)           # (B, J, k, d)
        return self.out_proj(out), attn
