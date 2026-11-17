"""Orthonormal 2-D DCT-II over the spatial axes: F = D X D^T per channel."""

import math

import torch
from torch import nn


def dct_matrix(n: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``(n, n)``."""
    if n < 1:
        raise ValueError(f"DCT size must be positive, got {n}")
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    mat = torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    mat[0] *= math.sqrt(1.0 / n)
    mat[1:] *= math.sqrt(2.0 / n)
    return mat.to(dtype=dtype, device=device)


def dct2(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """``(..., n, n, D)`` -> DCT coefficients ``(..., n, n, D)``."""
    return torch.einsum("uh,...hwd,vw->...uvd", mat, x, mat)


def idct2(f: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`dct2`."""
    return torch.einsum("uh,...uvd,vw->...hwd", mat, f, mat)


class SpatialDCT2d(nn.Module):
    """2-D DCT of ``(..., n, n, D)`` (computed in float32)."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.register_buffer("mat", dct_matrix(n), persistent=False)

    def _check(self, x: torch.Tensor):
        if x.shape[-3] != self.n or x.shape[-2] != self.n:
            raise ValueError(f"expected (..., {self.n}, {self.n}, D), "
                             f"got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = dct2(x.float(), self.mat.float())
        return out.to(x.dtype)

    def inverse(self, f: torch.Tensor) -> torch.Tensor:
        self._check(f)
        with torch.autocast(device_type=f.device.type, enabled=False):
            out = idct2(f.float(), self.mat.float())
        return out.to(f.dtype)
