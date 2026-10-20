import torch

from llava.model.visual_compressor.positional import Factorized2dEmbedding
from llava.model.visual_compressor.query_pool import QueryPooling

torch.manual_seed(0)


def test_factorized_embedding():
    emb = Factorized2dEmbedding(4, 6, 8)
    grid = emb()
    assert grid.shape == (24, 8)
    assert torch.allclose(grid[1 * 6 + 2], emb.emb_a[1] + emb.emb_b[2])
    assert emb.emb_a.std() > 0                       # not all-zero init
    assert not torch.allclose(grid[0], grid[1])


def test_query_pool_shapes_and_normalisation():
    pool = QueryPooling(dim=128, num_queries=4, heads=4)
    h = torch.randn(2, 16, 36, 128)
    out, attn = pool(h)
    assert out.shape == (2, 16, 4, 128)
    assert attn.shape == (2, 16, 4, 4, 36)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 16, 4, 4), atol=1e-5)
    assert pool.queries.shape == (1, 4, 128)         # ONE shared set


def test_shared_queries_still_differ_per_block():
    pool = QueryPooling(dim=32, num_queries=2, heads=2)
    out, _ = pool(torch.randn(1, 3, 9, 32))
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_block_specific_queries():
    pool = QueryPooling(dim=32, num_queries=2, heads=2, num_blocks=16,
                        block_specific=True)
    assert pool.queries.shape == (16, 2, 32)
    out, attn = pool(torch.randn(2, 16, 36, 32))
    assert out.shape == (2, 16, 2, 32) and attn.shape == (2, 16, 2, 2, 36)


def test_pool_gradients():
    pool = QueryPooling(dim=32, num_queries=2, heads=2)
    h = torch.randn(1, 4, 9, 32, requires_grad=True)
    out, _ = pool(h)
    out.sum().backward()
    assert pool.queries.grad.abs().sum() > 0 and h.grad.abs().sum() > 0
