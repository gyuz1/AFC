import pytest
import torch
import torch.nn.functional as F

from llava.model.visual_compressor import VisualCompressorConfig
from llava.model.visual_compressor.builder import create_compressor

torch.manual_seed(0)


def make(k):
    return create_compressor(VisualCompressorConfig(
        compression_mode="prumerge", target_visual_tokens=k).resolve(24, 1024))


def inputs(b=1, c=1024):
    return torch.randn(b, 576, c), torch.randn(b, 577, c), torch.randn(b, 577, c)


@pytest.mark.parametrize("k", [64, 144, 256])
def test_exact_budget(k):
    x, q, kk = inputs(2)
    assert make(k)(x, q=q, k=kk).shape == (2, k, 1024)


def test_selection_order_and_summary_token():
    x, q, kk = inputs()
    y = make(8)(x, q=q, k=kk)
    attn = F.softmax(q @ kk.transpose(-2, -1) * 1024 ** -0.5, dim=-1)[:, 0, 1:]
    top = attn[0].topk(7).indices                    # attention order (official)
    # each kept token = itself + attention-weighted neighbours, so it stays
    # closest to its own feature among the kept candidates
    d = torch.cdist(y[0, :7], x[0, top])
    assert torch.equal(d.argmin(1), torch.arange(7))
    pruned = torch.ones(576, dtype=torch.bool)
    pruned[top] = False
    summary = (x[0, pruned] * attn[0, pruned, None]).sum(0)
    assert torch.allclose(y[0, 7], summary, atol=1e-4)


def test_requires_qk():
    with pytest.raises(ValueError):
        make(64)(torch.randn(1, 576, 1024))


def test_output_follows_feature_dtype():
    x, q, kk = inputs()
    y = make(16)(x.half(), q=q, k=kk)          # tower internals fp32, features fp16
    assert y.dtype == torch.float16
