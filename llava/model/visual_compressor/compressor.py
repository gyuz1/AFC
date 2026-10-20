"""Visual token compressor inserted between CLIP and the LLaVA mm_projector.

Main mode ``block_dct`` (defaults: 24x24 grid, b=6, 64 tokens):

    CLIP patches             (B, 576, 1024)
    -> grid                  (B, 24, 24, 1024)
    -> blocks                (B, 16, 6, 6, 1024)
    -> block DCT             (B, 16, 6, 6, 1024)   spatial axes only
    -> frequency tokens      (B, 16, 36, 1024)
    -> LN + 1024->128 + pos  (B, 16, 36, 128)
    -> frequency Transformer (B, 16, 36, 128)
    -> query pooling (k=4)   (B, 16, 4, 128)       attention (B, 16, H, 4, 36)
    -> adapter 128->1024     (B, 16, 4, 1024)
    -> flatten (raster)      (B, 64, 1024)         -> frozen mm_projector

Ablation modes differ ONLY in how block tokens are formed; projection,
positions, Transformer, pooling and adapter are the same modules:

    global_dct      one 24x24 DCT, 576 frequency tokens, T queries, no block pos
    spatial_block   6x6 blocks WITHOUT DCT (inner pos = spatial row/col)
    fixed_lowpass   keep the top-left c x c DCT coefficients per block
                    (no Transformer, no pooling): projection + pos + adapter
    spatial_average c x c average pooling of each spatial block, same head
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .adapter import build_adapter
from .blocks import partition_blocks, seq_to_grid
from .config import (DCT_MODES, FIXED_MODES, PARAMETER_FREE_MODES,
                     VisualCompressorConfig)
from .dct import SpatialDCT2d, dct_matrix
from .frequency_encoder import FrequencyTokenEncoder
from .query_pool import QueryPooling
from .valid_region import interp_rows, valid_box


def _factor_grid(k: int):
    """Most square (rows, cols) with rows * cols = k (rows <= cols)."""
    r = int(k ** 0.5)
    while k % r:
        r -= 1
    return (r, k // r)


class VisualCompressor(nn.Module):
    def __init__(self, cfg: VisualCompressorConfig):
        super().__init__()
        cfg.validate()
        if not cfg.enabled:
            raise ValueError("compression_mode=none builds no compressor")
        if cfg.compression_mode in PARAMETER_FREE_MODES:
            raise ValueError(f"{cfg.compression_mode} is parameter-free: use "
                             "builder.create_compressor")
        self.cfg = cfg
        self.mode = cfg.compression_mode
        self.fixed = self.mode in FIXED_MODES
        b = cfg.effective_block_size
        self.dct = SpatialDCT2d(b) if self.mode in DCT_MODES else None

        inner_side = cfg.fixed_side if self.fixed else b
        use_inner = cfg.pos_mode in ("freq_only", "freq_and_block")
        use_block = (cfg.pos_mode == "freq_and_block"
                     and self.mode != "global_dct")
        self.encoder = FrequencyTokenEncoder(
            input_dim=cfg.input_dim, dim=cfg.compressor_hidden_dim,
            inner_side=inner_side, blocks_side=cfg.blocks_per_side,
            use_inner_pos=use_inner, use_block_pos=use_block,
            use_transformer=cfg.use_frequency_transformer and not self.fixed,
            layers=cfg.compressor_layers, heads=cfg.compressor_heads,
            ffn_dim=cfg.compressor_ffn_dim, dropout=cfg.compressor_dropout,
            pos_std=cfg.pos_init_std)
        self.pool = None if self.fixed else QueryPooling(
            dim=cfg.compressor_hidden_dim, num_queries=cfg.queries_per_block,
            heads=cfg.pool_heads, num_blocks=cfg.num_blocks,
            block_specific=cfg.query_sharing == "block_specific",
            dropout=cfg.compressor_dropout, init_std=cfg.pos_init_std,
            prenorm=cfg.pool_prenorm,
            learnable_scale=cfg.pool_learnable_scale)
        self.adapter = build_adapter(cfg.adapter_type,
                                     cfg.compressor_hidden_dim,
                                     cfg.input_dim, cfg.adapter_hidden_dim)
        self.base_grid = None
        if cfg.residual_base == "avgpool":
            self.base_grid = _factor_grid(cfg.queries_per_block)
            last = [m for m in self.adapter.modules()
                    if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)                   # start at the base
            nn.init.zeros_(last.bias)
        # Diagnostics (not part of the forward result). ``keep_attention``
        # stores the pooling attention (with its graph, e.g. for an optional
        # diversity loss) on every forward call made by LLaVA.
        self.keep_attention = False
        # dtype of the returned tokens (None = input dtype); set when the
        # consumer (mm_projector) runs in a different precision.
        self.output_dtype: Optional[torch.dtype] = None
        self.last_attention: Optional[torch.Tensor] = None
        self.last_shapes: Dict[str, tuple] = {}

    # ------------------------------------------------------------------
    def describe(self) -> str:
        n_param = sum(p.numel() for p in self.parameters())
        return f"{self.cfg.describe()}\nParameters: {n_param:,}"

    @property
    def num_output_tokens(self) -> int:
        return self.cfg.target_visual_tokens

    def block_tokens(self, grid: torch.Tensor) -> torch.Tensor:
        """``(B, N, N, D)`` -> input tokens ``(B, J, n_in, D)``."""
        b = self.cfg.effective_block_size
        blocks = partition_blocks(grid, b)                # (B, J, b, b, D)
        if self.dct is not None:
            blocks = self.dct(blocks)                     # same shape, freq
        if self.mode == "fixed_lowpass":
            c = self.cfg.fixed_side
            blocks = blocks[:, :, :c, :c]                 # (B, J, c, c, D)
        elif self.mode == "spatial_average":
            c = self.cfg.fixed_side
            bsz, j, _, _, dim = blocks.shape
            x = blocks.reshape(bsz * j, b, b, dim).permute(0, 3, 1, 2)
            x = F.adaptive_avg_pool2d(x, c)               # (B*J, D, c, c)
            blocks = x.permute(0, 2, 3, 1).reshape(bsz, j, c, c, dim)
        bsz, j, s1, s2, dim = blocks.shape
        return blocks.reshape(bsz, j, s1 * s2, dim)

    def residual_base(self, grid: torch.Tensor) -> torch.Tensor:
        """Per-block spatial average pooling to the k output positions.

        ``(B, N, N, D)`` -> ``(B, J, k, D)`` (row-major inside the block):
        training-free tokens that stay in the CLIP feature distribution.
        """
        b = self.cfg.effective_block_size
        blocks = partition_blocks(grid, b)                # (B, J, b, b, D)
        bsz, j, _, _, dim = blocks.shape
        x = blocks.reshape(bsz * j, b, b, dim).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, self.base_grid)      # (B*J, D, r, s)
        return x.flatten(2).transpose(1, 2).reshape(bsz, j, -1, dim)

    def forward(self, features: torch.Tensor,
                return_attention: Optional[bool] = None,
                image_sizes: Optional[Sequence] = None) -> torch.Tensor:
        """``(B, P, D)`` CLIP patch features -> ``(B, T, D)``, T = target.

        ``image_sizes`` (original ``(width, height)`` per image) is used, and
        required, only with ``valid_region``.
        """
        cfg = self.cfg
        if features.dim() != 3 or features.shape[1] != cfg.grid_size ** 2 \
                or features.shape[2] != cfg.input_dim:
            raise ValueError(
                f"expected (B, {cfg.grid_size ** 2}, {cfg.input_dim}) "
                f"features, got {tuple(features.shape)}")
        in_dtype = features.dtype
        p_dtype = self.adapter_dtype()
        grid = seq_to_grid(features.to(p_dtype))           # (B, N, N, D)
        if cfg.valid_region:
            if image_sizes is None or len(image_sizes) != grid.shape[0]:
                raise ValueError("valid_region needs one (width, height) per "
                                 "image in image_sizes")
            out, attn = self._forward_valid(grid, image_sizes)
            self._remember(return_attention, attn, features, out)
            return out.to(self.output_dtype or in_dtype)
        tokens = self.block_tokens(grid)                   # (B, J, n_in, D)
        h = self.encoder(tokens)                           # (B, J, n_in, d)
        if self.pool is not None:
            pooled, attn = self.pool(h)                    # (B, J, k, d)
        else:
            pooled, attn = h, None                         # fixed: n_in = k
        out = self.adapter(pooled)                         # (B, J, k, D)
        if self.base_grid is not None:
            out = out + self.residual_base(grid)           # base + residual
        bsz, j, k, dim = out.shape
        assert j * k == cfg.target_visual_tokens, (j, k)
        out = out.reshape(bsz, j * k, dim)                 # raster blocks

        keep = self._remember(return_attention, attn)
        if keep or not self.last_shapes:
            self.last_shapes = {
                "clip": tuple(features.shape), "block_tokens": tuple(
                    tokens.shape), "encoder_out": tuple(h.shape),
                "pooled": tuple(pooled.shape), "attention": None if attn
                is None else tuple(attn.shape), "adapter_out": (bsz, j, k,
                                                                dim),
                "output": tuple(out.shape)}
        return out.to(self.output_dtype or in_dtype)

    def _remember(self, return_attention, attn, features=None, out=None):
        keep = self.keep_attention if return_attention is None \
            else return_attention
        self.last_attention = attn if keep else None
        if features is not None and (keep or not self.last_shapes):
            self.last_shapes = {"clip": tuple(features.shape),
                                "output": tuple(out.shape)}
        return keep

    # ----- valid_region ---------------------------------------------------
    def _dct_rect(self, x: torch.Tensor) -> torch.Tensor:
        """``(G, h, w, D)`` -> orthonormal 2-D DCT-II (float32 inside)."""
        h, w = x.shape[1], x.shape[2]
        with torch.autocast(device_type=x.device.type, enabled=False):
            mh = dct_matrix(h, device=x.device)
            mw = dct_matrix(w, device=x.device)
            out = torch.einsum("uh,ghwd,vw->guvd", mh, x.float(), mw)
        return out.to(x.dtype)

    def _encode_group(self, blocks: torch.Tensor, coords) -> torch.Tensor:
        """Same-shape blocks ``(G, h, w, D)`` -> encoder states ``(G, h*w, d)``.

        Inner positions are the learned b x b tables read at normalised
        coordinates (``interp_rows``), so a full b x b block gets exactly the
        positions of the square path.
        """
        enc = self.encoder
        g, h, w, dim = blocks.shape
        x = enc.input_proj(enc.input_norm(blocks.reshape(g, h * w, dim)))
        if enc.inner_pos is not None:
            ea = interp_rows(enc.inner_pos.emb_a, h)
            eb = interp_rows(enc.inner_pos.emb_b, w)
            x = x + (ea[:, None] + eb[None]).reshape(1, h * w, -1)
        if enc.block_pos is not None:
            r = torch.tensor([c[0] for c in coords], device=x.device)
            c = torch.tensor([c[1] for c in coords], device=x.device)
            x = x + (enc.block_pos.emb_a[r] + enc.block_pos.emb_b[c])[:, None]
        if enc.transformer is not None:
            x = enc.transformer(x)
        return x

    def _forward_valid(self, grid: torch.Tensor, image_sizes):
        """Per image: cut the padding, split the image area into the same
        m x m block layout (near-equal sizes), DCT each block, encode, pool.
        Blocks of equal shape are processed together.
        """
        cfg = self.cfg
        n, m = cfg.grid_size, cfg.blocks_per_side
        k = cfg.queries_per_block
        outs, attns = [], []
        for i, (w_img, h_img) in enumerate(image_sizes):
            r0, r1, c0, c1 = valid_box(int(w_img), int(h_img), n, min_side=m)
            region = grid[i, r0:r1, c0:c1]                 # (h, w, D)
            rows = region.tensor_split(m, dim=0)
            blocks = [(rb, cb, cell) for rb, part in enumerate(rows)
                      for cb, cell in enumerate(part.tensor_split(m, dim=1))]
            groups: Dict[tuple, list] = {}
            for j, (rb, cb, cell) in enumerate(blocks):
                groups.setdefault(tuple(cell.shape[:2]), []).append(j)
            out_i = [None] * len(blocks)
            attn_i = [None] * len(blocks)
            for idx in groups.values():
                cells = torch.stack([blocks[j][2] for j in idx])   # (G,h,w,D)
                tokens = self._dct_rect(cells) if self.dct is not None \
                    else cells
                hstate = self._encode_group(
                    tokens, [(blocks[j][0], blocks[j][1]) for j in idx])
                pooled, attn = self.pool(hstate.unsqueeze(0))     # (1,G,k,d)
                res = self.adapter(pooled[0])                    # (G,k,D)
                if self.base_grid is not None:
                    x = cells.permute(0, 3, 1, 2)
                    base = F.adaptive_avg_pool2d(x, self.base_grid)
                    res = res + base.flatten(2).transpose(1, 2)
                for t, j in enumerate(idx):
                    out_i[j] = res[t]
                    attn_i[j] = attn[0, t]                       # (H,k,n_j)
            outs.append(torch.cat(out_i, dim=0))                 # (m*m*k, D)
            n_max = max(a.shape[-1] for a in attn_i)
            attns.append(torch.stack([F.pad(a, (0, n_max - a.shape[-1]))
                                      for a in attn_i]))         # (J,H,k,n)
        out = torch.stack(outs)
        assert out.shape[1] == cfg.target_visual_tokens, out.shape
        n_max = max(a.shape[-1] for a in attns)
        attn = torch.stack([F.pad(a, (0, n_max - a.shape[-1])) for a in attns])
        return out, attn

    def adapter_dtype(self) -> torch.dtype:
        return next(self.adapter.parameters()).dtype
