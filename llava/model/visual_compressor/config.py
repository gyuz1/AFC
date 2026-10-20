"""Configuration of the visual token compressor.

The user-facing budget is ``target_visual_tokens``; the number of outputs per
block (``queries_per_block``, "k") is derived and must divide exactly.
``k`` counts OUTPUT tokens per block, not input candidates (a b x b block
always has b*b input tokens).
"""

import dataclasses
import math
from typing import Optional

COMPRESSION_MODES = ("none", "block_dct", "global_dct", "spatial_block",
                     "fourier_vlm", "avgpool", "prumerge",
                     "fixed_lowpass", "spatial_average")
LEARNED_POOL_MODES = ("block_dct", "global_dct", "spatial_block")
FIXED_MODES = ("fixed_lowpass", "spatial_average")
PARAMETER_FREE_MODES = ("fourier_vlm", "avgpool", "prumerge")
DCT_MODES = ("block_dct", "global_dct", "fixed_lowpass")
POS_MODES = ("none", "freq_only", "freq_and_block")
ADAPTER_TYPES = ("linear", "mlp")
RESIDUAL_BASES = ("none", "avgpool")
QUERY_SHARING = ("shared", "block_specific")


@dataclasses.dataclass
class VisualCompressorConfig:
    compression_mode: str = "none"
    block_size: int = 6
    target_visual_tokens: int = 64
    compressor_hidden_dim: int = 128
    compressor_layers: int = 2
    compressor_heads: int = 4
    compressor_ffn_dim: int = 512
    compressor_dropout: float = 0.0
    pool_heads: int = 4
    pool_prenorm: bool = True
    # Learnable per-head logit scale of the pooling attention (init 1.0).
    pool_learnable_scale: bool = False
    # Residual base: output = base + adapter(pooled). "avgpool" = per-block
    # spatial average pooling to the k output positions (training-free,
    # CLIP-distributed); the adapter's last layer starts at zero, so training
    # starts from the base. "none" = adapter output only.
    residual_base: str = "none"
    # fourier_vlm only: multiply by C / N after the C-point inverse DCT
    fourier_scale: bool = True
    pos_mode: str = "freq_and_block"
    use_frequency_transformer: bool = True
    adapter_type: str = "mlp"
    adapter_hidden_dim: int = 512
    query_sharing: str = "shared"
    pos_init_std: float = 0.02
    # Compress only the image area of LLaVA's square-padded input: the
    # padding rows/cols (located from the original image size) are cut from
    # the CLIP grid before the DCT; frequency positions are interpolated on a
    # normalised axis. Requires ``image_sizes`` at forward time.
    valid_region: bool = False
    # Filled from the vision tower / LLaVA config by ``resolve``.
    grid_size: Optional[int] = None
    input_dim: Optional[int] = None

    # ----- derived quantities -------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.compression_mode != "none"

    @property
    def effective_block_size(self) -> int:
        """Global modes treat the whole grid as one block."""
        return self.grid_size if self.compression_mode in (
            "global_dct",) + PARAMETER_FREE_MODES else self.block_size

    @property
    def blocks_per_side(self) -> int:
        return self.grid_size // self.effective_block_size

    @property
    def num_blocks(self) -> int:
        return self.blocks_per_side ** 2

    @property
    def tokens_per_block_in(self) -> int:
        return self.effective_block_size ** 2

    @property
    def queries_per_block(self) -> int:
        return self.target_visual_tokens // self.num_blocks

    @property
    def fixed_side(self) -> int:
        """Side of the c x c grid kept per block by the fixed baselines."""
        return math.isqrt(self.queries_per_block)

    # ----- validation ------------------------------------------------------
    def resolve(self, grid_size: int, input_dim: int):
        """Set backbone-dependent sizes and validate everything."""
        self.grid_size = int(grid_size)
        self.input_dim = int(input_dim)
        return self.validate()

    def validate(self):
        def need(cond, msg):
            if not cond:
                raise ValueError(f"[visual_compressor] {msg}")

        need(self.compression_mode in COMPRESSION_MODES,
             f"compression_mode must be one of {COMPRESSION_MODES}")
        if not self.enabled:
            return self
        need(self.grid_size and self.input_dim,
             "grid_size / input_dim unknown: call resolve()")
        need(self.pos_mode in POS_MODES, f"pos_mode must be in {POS_MODES}")
        need(self.adapter_type in ADAPTER_TYPES,
             f"adapter_type must be in {ADAPTER_TYPES}")
        need(self.query_sharing in QUERY_SHARING,
             f"query_sharing must be in {QUERY_SHARING}")
        need(self.residual_base in RESIDUAL_BASES,
             f"residual_base must be in {RESIDUAL_BASES}")
        b = self.effective_block_size
        need(self.grid_size % b == 0,
             f"block_size {b} does not tile the {self.grid_size}x"
             f"{self.grid_size} grid")
        need(self.target_visual_tokens % self.num_blocks == 0,
             f"target_visual_tokens={self.target_visual_tokens} is not "
             f"divisible by num_blocks={self.num_blocks} (block_size={b}); "
             "the budget is never changed silently")
        k = self.queries_per_block
        need(k >= 1, "need at least one output token per block")
        if self.compression_mode == "fourier_vlm":
            c = math.isqrt(self.target_visual_tokens)
            need(c * c == self.target_visual_tokens and c <= self.grid_size,
                 "fourier_vlm keeps a C x C low-pass grid: target_visual_tokens"
                 f" must be a square <= {self.grid_size ** 2}")
        elif self.compression_mode in ("avgpool", "prumerge"):
            need(self.target_visual_tokens <= self.grid_size ** 2,
                 "avgpool cannot produce more tokens than the grid has")
        elif self.compression_mode in FIXED_MODES:
            need(self.fixed_side ** 2 == k,
                 f"{self.compression_mode} keeps a c x c grid per block, so "
                 f"tokens per block must be a square, got {k}")
            need(self.fixed_side <= b, f"c={self.fixed_side} exceeds b={b}")
        else:
            need(k <= self.tokens_per_block_in,
                 f"{k} queries per block exceed the {self.tokens_per_block_in}"
                 " input tokens (compression expected)")
        d, h = self.compressor_hidden_dim, self.compressor_heads
        need(d % h == 0, f"hidden {d} not divisible by {h} encoder heads")
        need(d % self.pool_heads == 0,
             f"hidden {d} not divisible by {self.pool_heads} pool heads")
        if self.compression_mode == "global_dct":
            need(self.query_sharing == "shared",
                 "global_dct has a single block; use query_sharing=shared")
        if self.valid_region:
            need(self.compression_mode in LEARNED_POOL_MODES,
                 f"valid_region supports {LEARNED_POOL_MODES}")
            need(self.query_sharing == "shared",
                 "valid_region needs query_sharing=shared")
        return self

    # ----- (de)serialisation --------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[dict]):
        values = dict(values or {})
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"unknown visual_compressor keys {sorted(unknown)}")
        return cls(**values)

    def describe(self) -> str:
        if not self.enabled:
            return "Compression mode: none (vanilla LLaVA path)"
        lines = [
            f"Compression mode: {self.compression_mode}",
            f"CLIP grid: {self.grid_size}x{self.grid_size} | dim "
            f"{self.input_dim}",
            f"Block size: {self.effective_block_size} | blocks: "
            f"{self.num_blocks} | input tokens/block: "
            f"{self.tokens_per_block_in}",
            f"Output tokens/block: {self.queries_per_block} | final visual "
            f"tokens: {self.target_visual_tokens}",
            f"Encoder: dim {self.compressor_hidden_dim}, layers "
            f"{self.compressor_layers if self.use_frequency_transformer else 0}"
            f", heads {self.compressor_heads}, ffn {self.compressor_ffn_dim}",
            f"Pooling: {'learned queries' if self.compression_mode in LEARNED_POOL_MODES else 'fixed'}"
            f" ({self.query_sharing}, {self.pool_heads} heads)",
            f"Positional mode: {self.pos_mode} | Adapter: {self.adapter_type}"
            f" | residual base: {self.residual_base}",
        ]
        return "\n".join(lines)
