"""Image area of LLaVA's square-padded input on the CLIP patch grid.

LLaVA-1.5 (``image_aspect_ratio=pad``) pastes a W x H image at offset
((L - W) // 2, (L - H) // 2) on an L x L canvas (L = max(W, H)) filled with
the CLIP mean colour and resizes it to the vision tower's square input. On an
N x N patch grid the image occupies [off * N / L, (off + size) * N / L) along
each axis. A patch row / column is kept when more than half of it is image
(its centre lies inside), so the cut is exact when the border falls on a patch
boundary (4:3 and 3:2 photos) and off by at most half a patch otherwise.
"""

import math
from typing import Sequence, Tuple

import torch


def valid_box(width: int, height: int, n: int, min_side: int = 1
              ) -> Tuple[int, int, int, int]:
    """``(r0, r1, c0, c1)``: rows ``r0:r1`` and cols ``c0:c1`` hold the image.

    Each side keeps at least ``min_side`` patches (grown symmetrically into
    the padding for extreme aspect ratios).
    """
    side = max(width, height)

    def span(size: int) -> Tuple[int, int]:
        off = (side - size) // 2
        start, end = off * n / side, (off + size) * n / side
        lo = max(0, math.floor(start + 0.5))
        hi = min(n, math.floor(end + 0.5))
        if hi - lo < min_side:                          # extreme aspect ratio
            centre = (start + end) / 2
            lo = min(max(0, round(centre - min_side / 2)), n - min_side)
            hi = lo + min_side
        return lo, hi

    r0, r1 = span(height)
    c0, c1 = span(width)
    return r0, r1, c0, c1


def split_sizes(length: int, parts: int) -> Sequence[int]:
    """Sizes of ``torch.tensor_split`` into ``parts`` near-equal chunks."""
    return [len(c) for c in torch.arange(length).tensor_split(parts)]


def interp_rows(table: torch.Tensor, length: int) -> torch.Tensor:
    """Rows of a learned ``(S, d)`` table at normalised positions.

    Index ``u`` of a length-``length`` axis maps to ``u * S / length``
    (same normalised frequency / position); linear interpolation between the
    two neighbouring rows. ``length == S`` returns ``table`` unchanged.
    """
    size = table.shape[0]
    if length == size:
        return table
    pos = torch.arange(length, device=table.device, dtype=torch.float32) \
        * (size / length)
    lo = pos.floor().long().clamp(max=size - 1)
    hi = (lo + 1).clamp(max=size - 1)
    w = (pos - lo.float()).unsqueeze(1).to(table.dtype)
    return table[lo] * (1 - w) + table[hi] * w
