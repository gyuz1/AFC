import pytest
import torch

from llava.model.visual_compressor.blocks import partition_blocks
from llava.model.visual_compressor.dct import SpatialDCT2d, dct_matrix

torch.manual_seed(0)


@pytest.mark.parametrize("n", [4, 6, 8, 24])
def test_orthonormal(n):
    d = dct_matrix(n)
    assert torch.allclose(d @ d.T, torch.eye(n), atol=1e-6)


@pytest.mark.parametrize("n", [4, 6, 8, 24])
def test_inverse_reconstructs(n):
    t = SpatialDCT2d(n)
    x = torch.randn(2, 3, n, n, 16)
    assert torch.allclose(t.inverse(t(x)), x, atol=1e-4)


def test_constant_is_dc_only():
    t = SpatialDCT2d(6)
    f = t(torch.full((1, 6, 6, 8), 2.0))
    assert torch.allclose(f[..., 0, 0, :], torch.full((1, 8), 12.0), atol=1e-5)
    rest = f.clone()
    rest[..., 0, 0, :] = 0
    assert rest.abs().max() < 1e-5


def test_channels_untouched():
    t = SpatialDCT2d(6)
    x = torch.randn(1, 6, 6, 32)
    perm = torch.randperm(32)
    assert torch.allclose(t(x[..., perm]), t(x)[..., perm], atol=1e-5)


def test_block24_equals_global():
    x = torch.randn(1, 24, 24, 8)
    t = SpatialDCT2d(24)
    assert torch.allclose(t(partition_blocks(x, 24))[:, 0], t(x), atol=1e-5)


def test_gradient_flows():
    t = SpatialDCT2d(6)
    x = torch.randn(1, 6, 6, 4, requires_grad=True)
    t(x).pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_dct_matrix_not_in_state_dict():
    assert "mat" not in SpatialDCT2d(6).state_dict()
