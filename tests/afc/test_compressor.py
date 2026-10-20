import pytest
import torch

from llava.model.visual_compressor import (VisualCompressor,
                                           VisualCompressorConfig)
from llava.model.visual_compressor.diagnostics import (diversity_penalty,
                                                       summarize)

torch.manual_seed(0)
X = torch.randn(2, 576, 1024)


def make(**kw):
    from llava.model.visual_compressor.builder import create_compressor
    return create_compressor(VisualCompressorConfig(**kw).resolve(24, 1024))


def test_main_block_dct_shapes():
    m = make(compression_mode="block_dct")
    y = m(X, return_attention=True)
    s = m.last_shapes
    assert s["block_tokens"] == (2, 16, 36, 1024)
    assert s["encoder_out"] == (2, 16, 36, 128)
    assert s["pooled"] == (2, 16, 4, 128)
    assert s["attention"] == (2, 16, 4, 4, 36)
    assert s["adapter_out"] == (2, 16, 4, 1024)
    assert y.shape == (2, 64, 1024)
    assert torch.allclose(m.last_attention.sum(-1),
                          torch.ones(2, 16, 4, 4), atol=1e-5)


@pytest.mark.parametrize("kw,tokens", [
    (dict(compression_mode="global_dct"), 64),
    (dict(compression_mode="spatial_block"), 64),
    (dict(compression_mode="fixed_lowpass"), 64),
    (dict(compression_mode="spatial_average"), 64),
    (dict(compression_mode="block_dct", target_visual_tokens=32), 32),
    (dict(compression_mode="block_dct", target_visual_tokens=128), 128),
    (dict(compression_mode="block_dct", target_visual_tokens=144), 144),
    (dict(compression_mode="block_dct", block_size=4,
          target_visual_tokens=144), 144),
    (dict(compression_mode="block_dct", block_size=8,
          target_visual_tokens=144), 144),
    (dict(compression_mode="block_dct", use_frequency_transformer=False), 64),
    (dict(compression_mode="block_dct", pos_mode="none"), 64),
    (dict(compression_mode="block_dct", pos_mode="freq_only"), 64),
    (dict(compression_mode="block_dct", query_sharing="block_specific"), 64),
    (dict(compression_mode="block_dct", adapter_type="linear"), 64),
])
def test_ablation_shapes(kw, tokens):
    assert make(**kw)(X).shape == (2, tokens, 1024)


def test_spatial_and_block_dct_have_equal_capacity():
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(make(compression_mode="spatial_block")) == \
        n(make(compression_mode="block_dct"))


def test_global_has_no_block_pos_and_576_inputs():
    m = make(compression_mode="global_dct")
    assert m.encoder.block_pos is None
    m(X, return_attention=True)
    assert m.last_shapes["attention"] == (2, 1, 4, 64, 576)


def test_fixed_lowpass_takes_top_left_coefficients():
    m = make(compression_mode="fixed_lowpass")
    grid = X.reshape(2, 24, 24, 1024)
    tok = m.block_tokens(grid)
    full = m.dct(grid.reshape(2, 4, 6, 4, 6, 1024).permute(0, 1, 3, 2, 4, 5)
                 .reshape(2, 16, 6, 6, 1024))
    assert torch.allclose(tok, full[:, :, :2, :2].reshape(2, 16, 4, 1024))


def test_budget_errors():
    for kw in (dict(target_visual_tokens=60),
               dict(block_size=5),
               dict(compression_mode="fixed_lowpass", target_visual_tokens=32),
               dict(target_visual_tokens=16 * 37)):
        kw.setdefault("compression_mode", "block_dct")
        with pytest.raises(ValueError):
            VisualCompressorConfig(**kw).resolve(24, 1024)


def test_dtype_preserved_and_deterministic():
    m = make(compression_mode="block_dct").eval()
    x = X.half()
    with torch.no_grad():
        a, b = m(x), m(x)
    assert a.dtype == torch.float16 and torch.equal(a, b)


def test_synthetic_gradients_reach_every_trainable_part():
    m = make(compression_mode="block_dct")
    m(X).pow(2).mean().backward()
    for name in ("encoder.input_proj.weight", "encoder.inner_pos.emb_a",
                 "encoder.block_pos.emb_a", "encoder.transformer.layers.0."
                 "self_attn.in_proj_weight", "pool.queries",
                 "pool.q_proj.weight", "pool.k_proj.weight",
                 "pool.v_proj.weight", "pool.out_proj.weight",
                 "adapter.1.weight", "adapter.3.weight"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name


def test_diagnostics():
    m = make(compression_mode="block_dct")
    m(X, return_attention=True)
    s = summarize(m.last_attention)
    assert 0 < s["attn_entropy"] <= s["attn_entropy_max"] + 1e-6
    assert diversity_penalty(m.last_attention).item() >= 0


@pytest.mark.parametrize("k", [32, 64, 128, 144])
def test_residual_base_starts_at_average_pooling(k):
    m = make(compression_mode="block_dct", target_visual_tokens=k,
             residual_base="avgpool", pool_learnable_scale=True)
    base = m.residual_base(X.reshape(2, 24, 24, 1024)).reshape(2, -1, 1024)
    assert torch.allclose(m(X), base)
    # block 0, position 0 of a 2x2 grid = mean of the top-left 3x3 patches
    if k == 64:
        ref = X.reshape(2, 24, 24, 1024)[:, :3, :3].mean(dim=(1, 2))
        assert torch.allclose(base[:, 0], ref, atol=1e-5)


def test_residual_base_trains_adapter_after_first_step():
    m = make(compression_mode="block_dct", residual_base="avgpool")
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m(X).pow(2).mean().backward()
    assert m.adapter[3].weight.grad.abs().sum() > 0
    opt.step()
    assert m.adapter[3].weight.abs().sum() > 0


def test_learnable_scale_is_identity_at_init():
    torch.manual_seed(1)
    a = make(compression_mode="block_dct")
    torch.manual_seed(1)
    b = make(compression_mode="block_dct", pool_learnable_scale=True)
    assert torch.allclose(a(X), b(X), atol=1e-6)


def test_fourier_vlm_matches_global_lowpass_definition():
    from llava.model.visual_compressor.dct import dct_matrix
    m = make(compression_mode="fourier_vlm", target_visual_tokens=64)
    y = m(X)
    assert y.shape == (2, 64, 1024)
    assert sum(p.numel() for p in m.parameters()) == 0
    g = X.reshape(2, 24, 24, 1024)
    d24, d8 = dct_matrix(24), dct_matrix(8)
    f = torch.einsum("uh,bhwd,vw->buvd", d24, g, d24)[:, :8, :8]
    ref = torch.einsum("uh,buvd,vw->bhwd", d8, f, d8) * (8 / 24)
    assert torch.allclose(y, ref.reshape(2, 64, 1024), atol=1e-4)
    c = make(compression_mode="fourier_vlm", target_visual_tokens=64)
    assert torch.allclose(c(torch.full((1, 576, 1024), 2.0)),
                          torch.full((1, 64, 1024), 2.0), atol=1e-4)
    noscale = make(compression_mode="fourier_vlm", fourier_scale=False)
    assert torch.allclose(noscale(torch.full((1, 576, 1024), 2.0)),
                          torch.full((1, 64, 1024), 6.0), atol=1e-3)


def test_avgpool_equals_residual_base():
    a = make(compression_mode="avgpool", target_visual_tokens=64)(X)
    r = make(compression_mode="block_dct", residual_base="avgpool")
    base = r.residual_base(X.reshape(2, 24, 24, 1024)).reshape(2, 64, 1024)
    # same 3x3-cell averages; block order differs from raster order
    assert torch.allclose(a.sort(dim=1).values, base.sort(dim=1).values,
                          atol=1e-5)


def test_parameter_free_budget_errors():
    with pytest.raises(ValueError):
        VisualCompressorConfig(compression_mode="fourier_vlm",
                               target_visual_tokens=60).resolve(24, 1024)
