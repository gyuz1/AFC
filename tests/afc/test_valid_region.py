import pytest
import torch

from llava.model.visual_compressor import VisualCompressorConfig
from llava.model.visual_compressor.diagnostics import summarize
from llava.model.visual_compressor.valid_region import interp_rows, valid_box

torch.manual_seed(0)
X = torch.randn(2, 576, 1024)
MODES = [dict(compression_mode="global_dct"),
         dict(compression_mode="block_dct"),
         dict(compression_mode="spatial_block")]


def make(**kw):
    from llava.model.visual_compressor.builder import create_compressor
    kw.setdefault("residual_base", "avgpool")
    kw.setdefault("pool_learnable_scale", True)
    m = create_compressor(VisualCompressorConfig(**kw).resolve(24, 1024))
    # non-zero adapter output so the learned path is actually compared
    torch.nn.init.normal_(m.adapter.net[-1].weight if hasattr(m.adapter, "net")
                          else [p for p in m.adapter.parameters()][-2], std=0.02)
    return m


@pytest.mark.parametrize("size,box", [
    ((640, 427), (4, 20, 0, 24)),      # 3:2, border on a patch boundary
    ((640, 480), (3, 21, 0, 24)),      # 4:3
    ((500, 500), (0, 24, 0, 24)),      # square: nothing cut
    ((1920, 1080), (5, 19, 0, 24)),    # 16:9: border inside a patch (74 %)
    ((426, 640), (0, 24, 4, 20)),      # portrait: columns cut
])
def test_valid_box(size, box):
    assert valid_box(*size, 24) == box


def test_valid_box_min_side_for_extreme_aspect():
    r0, r1, c0, c1 = valid_box(2000, 50, 24, min_side=4)
    assert r1 - r0 == 4 and (c0, c1) == (0, 24) and 0 <= r0 and r1 <= 24


def test_interp_rows_identity_and_endpoints():
    t = torch.randn(6, 8)
    assert torch.equal(interp_rows(t, 6), t)
    s = interp_rows(t, 3)                      # positions 0, 2, 4
    assert torch.allclose(s, t[[0, 2, 4]])
    u = interp_rows(t, 4)                      # positions 0, 1.5, 3, 4.5
    assert torch.allclose(u[1], (t[1] + t[2]) / 2)


@pytest.mark.parametrize("kw", MODES)
def test_square_images_match_the_square_path(kw):
    m = make(**kw).eval()
    ref = m(X)
    m.cfg.valid_region = True
    out = m(X, image_sizes=[(500, 500), (336, 336)])
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


@pytest.mark.parametrize("kw", MODES)
def test_padding_rows_do_not_affect_the_output(kw):
    m = make(valid_region=True, **kw).eval()
    x = X[:1].clone()
    y = m(x, image_sizes=[(640, 427)])         # rows 0-3 and 20-23 are padding
    grid = x.view(1, 24, 24, 1024).clone()
    grid[:, :4] = torch.randn_like(grid[:, :4]) * 50
    grid[:, 20:] = torch.randn_like(grid[:, 20:]) * 50
    y2 = m(grid.view(1, 576, 1024), image_sizes=[(640, 427)])
    assert torch.equal(y, y2)
    grid[:, 10, 5] += 1.0                      # an image patch does matter
    assert not torch.equal(m(grid.view(1, 576, 1024), image_sizes=[(640, 427)]), y)


@pytest.mark.parametrize("kw", MODES)
def test_shapes_mixed_batch_and_gradients(kw):
    m = make(valid_region=True, **kw)
    m.keep_attention = True
    sizes = [(640, 427), (1920, 1080)]
    y = m(X, image_sizes=sizes)
    assert y.shape == (2, 64, 1024)
    attn = m.last_attention
    assert attn.shape[:4] == (2, m.cfg.num_blocks, m.cfg.pool_heads,
                              m.cfg.queries_per_block)
    assert torch.allclose(attn.sum(-1), torch.ones(()))       # padded zeros
    summarize(attn)
    y.pow(2).mean().backward()
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and p.grad is None]
    if kw["compression_mode"] == "global_dct":     # no block positions
        missing = [n for n in missing if "block_pos" not in n]
    assert trainable and not missing, missing


def test_extreme_aspect_block_mode():
    m = make(valid_region=True, compression_mode="block_dct")
    assert m(X[:1], image_sizes=[(3000, 60)]).shape == (1, 64, 1024)


def test_requires_image_sizes_and_supported_modes():
    m = make(valid_region=True, compression_mode="global_dct")
    with pytest.raises(ValueError):
        m(X)
    with pytest.raises(ValueError):
        VisualCompressorConfig(compression_mode="avgpool", valid_region=True
                               ).resolve(24, 1024)
    with pytest.raises(ValueError):
        VisualCompressorConfig(compression_mode="block_dct", valid_region=True,
                               query_sharing="block_specific").resolve(24, 1024)
