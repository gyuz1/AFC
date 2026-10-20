"""Parameter-free compression baselines (same interface as VisualCompressor).

    fourier_vlm  Fourier-VLM's FFC (arXiv 2508.06038): 2-D DCT over the whole
                 N x N grid, keep the top-left C x C coefficients, inverse DCT
                 with the C-point basis -> C x C tokens (raster), C^2 = target.
                 ``fourier_scale`` multiplies by C / N so a constant feature
                 map is preserved; the original code has no such factor (it
                 absorbs the N / C amplification by retraining the LLM).
    avgpool      spatial average pooling of the N x N grid to the most square
                 r x s grid with r * s = target (8x8 for 64 tokens: each token
                 averages a 3x3 patch cell). Equals the training-free base of
                 the residual compressors.

    prumerge     LLaVA-PruMerge (Shang et al., 2024), training free: port of the
                 official `token_prune_merge_advanced` (github.com/42Shawn/
                 LLaVA-PruMerge, Apache-2.0). [CLS] attention from the last CLIP
                 layer's q / k, top tokens kept in attention order, each merged
                 with its 32 key-nearest tokens weighted by [CLS] attention, plus
                 one token summarising all pruned tokens. Only change: instead of
                 the adaptive IQR count, K - 1 tokens are kept so that exactly K
                 (= target_visual_tokens) tokens are output.

All are used with the frozen LLaVA (no training).
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import seq_to_grid
from .config import VisualCompressorConfig
from .dct import SpatialDCT2d



def complement_idx(idx, dim):
    """From the official LLaVA-PruMerge code (Apache-2.0)."""
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


def factor_grid(k: int):
    """Most square (rows, cols) with rows * cols = k (rows <= cols)."""
    r = int(k ** 0.5)
    while k % r:
        r -= 1
    return (r, k // r)


class ParameterFreeCompressor(nn.Module):
    def __init__(self, cfg: VisualCompressorConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.mode = cfg.compression_mode
        n = cfg.grid_size
        if self.mode == "prumerge":
            pass
        elif self.mode == "fourier_vlm":
            self.cutoff = int(round(cfg.target_visual_tokens ** 0.5))
            self.dct_full = SpatialDCT2d(n)
            self.dct_cut = SpatialDCT2d(self.cutoff)
        else:
            self.grid = factor_grid(cfg.target_visual_tokens)
        # Same diagnostics surface as VisualCompressor.
        self.keep_attention = False
        self.last_attention: Optional[torch.Tensor] = None
        self.last_shapes: Dict[str, tuple] = {}
        self.output_dtype: Optional[torch.dtype] = None
        self.pool = None

    def _prumerge(self, image_features, q, k):
        """Official token_prune_merge_advanced, fixed budget (see module doc)."""
        if q is None or k is None:
            raise ValueError("prumerge needs the vision tower's q / k projections")
        in_dtype = image_features.dtype
        image_features = image_features.float()
        desired_layer_q, desired_layer_k = q.float(), k.float()
        B, N, C = image_features.shape
        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)
        cls_attn = attn[:, 0, 1:]
        left = self.cfg.target_visual_tokens - 1          # + 1 summary token
        _, idx = torch.topk(cls_attn, left, dim=1, largest=True)
        index = idx.unsqueeze(-1).expand(-1, -1, C)
        Key_wo_cls = desired_layer_k[:, 1:]
        x_others = torch.gather(image_features, dim=1, index=index)
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)
        compl = complement_idx(idx, N)
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)
        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)
        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)
        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b, i, :].unsqueeze(0).unsqueeze(0)
                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)
                after_i_Key = Key_others_norm[b, i + 1:, :].unsqueeze(0)
                before_i_x_others = x_others[b, :i, :].unsqueeze(0)
                after_i_x_others = x_others[b, i + 1:, :].unsqueeze(0)
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others,
                                           non_topk[b, :, :].unsqueeze(0)], dim=1)
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)
                after_i_x_others_attn = x_others_attn[b, i + 1:].unsqueeze(0)
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn,
                                                non_topk_attn[b, :].unsqueeze(0)], dim=1)
                rest_Keys = torch.cat([before_i_Key, after_i_Key,
                                       non_topk_Key_norm[b, :, :].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))
                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)
                cluster_tokens = rest_x_others[:, cluster_indices.squeeze(), :]
                weights = rest_x_others_attn[:, cluster_indices.squeeze()].unsqueeze(-1)
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1)
                updated_x_others[b, i, :] = weighted_avg + x_others[b, i, :]
        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
        out = torch.cat([updated_x_others, extra_one_token], dim=1)
        self.last_attention = None
        self.last_shapes = {"clip": (B, N, C), "output": tuple(out.shape)}
        return out.to(self.output_dtype or in_dtype)

    def describe(self) -> str:
        return f"{self.cfg.describe()}\nParameters: 0 (parameter-free)"

    @property
    def num_output_tokens(self) -> int:
        return self.cfg.target_visual_tokens

    def forward(self, features: torch.Tensor,
                return_attention: Optional[bool] = None,
                q: Optional[torch.Tensor] = None,
                k: Optional[torch.Tensor] = None) -> torch.Tensor:
        cfg = self.cfg
        if self.mode == "prumerge":
            return self._prumerge(features, q, k)
        if features.dim() != 3 or features.shape[1] != cfg.grid_size ** 2:
            raise ValueError(f"expected (B, {cfg.grid_size ** 2}, D), got "
                             f"{tuple(features.shape)}")
        grid = seq_to_grid(features.float())               # (B, N, N, D)
        bsz, n, _, dim = grid.shape
        if self.mode == "fourier_vlm":
            c = self.cutoff
            coeffs = self.dct_full(grid)[:, :c, :c]        # (B, C, C, D)
            out = self.dct_cut.inverse(coeffs)             # (B, C, C, D)
            if cfg.fourier_scale:
                out = out * (c / n)
            out = out.reshape(bsz, c * c, dim)
        else:
            x = F.adaptive_avg_pool2d(grid.permute(0, 3, 1, 2), self.grid)
            out = x.flatten(2).transpose(1, 2)              # (B, r*s, D)
        self.last_attention = None
        self.last_shapes = {"clip": tuple(features.shape),
                            "output": tuple(out.shape)}
        return out.to(self.output_dtype or features.dtype)
