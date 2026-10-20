"""Spatial grid <-> non-overlapping blocks (deterministic raster order).

Block j covers rows ``r_j * b ... r_j * b + b - 1`` and columns
``c_j * b ...`` of the grid, with ``j = r_j * (N / b) + c_j``
(block 0 = top-left, block 1 = its right neighbour, ...).
"""

import math

import torch


def seq_to_grid(x: torch.Tensor) -> torch.Tensor:
    """``(B, N*N, D)`` -> ``(B, N, N, D)``; token i is (i // N, i % N)."""
    bsz, n_tok, dim = x.shape
    n = math.isqrt(n_tok)
    if n * n != n_tok:
        raise ValueError(f"token count {n_tok} is not a square grid")
    return x.reshape(bsz, n, n, dim)


def blocks_per_side(grid: int, block: int) -> int:
    if block < 1 or grid % block != 0:
        raise ValueError(f"block size {block} does not tile a {grid}x{grid} "
                         "grid")
    return grid // block


def partition_blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    """``(B, N, N, D)`` -> ``(B, J, b, b, D)`` with ``J = (N / b)^2``."""
    bsz, n, n2, dim = x.shape
    if n != n2:
        raise ValueError(f"expected a square grid, got {tuple(x.shape)}")
    m = blocks_per_side(n, block)
    x = x.reshape(bsz, m, block, m, block, dim).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(bsz, m * m, block, block, dim)


def merge_blocks(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`partition_blocks`."""
    bsz, j, block, _, dim = x.shape
    m = math.isqrt(j)
    if m * m != j:
        raise ValueError(f"block count {j} is not a square")
    x = x.reshape(bsz, m, m, block, block, dim).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(bsz, m * block, m * block, dim)


def block_coords(grid: int, block: int) -> torch.Tensor:
    """``(J, 2)`` long tensor of (block row r_j, block col c_j)."""
    m = blocks_per_side(grid, block)
    rr, cc = torch.meshgrid(torch.arange(m), torch.arange(m), indexing="ij")
    return torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=1)
