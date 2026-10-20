import pytest
import torch

from llava.model.visual_compressor.blocks import (block_coords, merge_blocks,
                                                  partition_blocks,
                                                  seq_to_grid)


@pytest.mark.parametrize("b", [4, 6, 8, 12, 24])
def test_roundtrip(b):
    x = torch.randn(2, 24, 24, 5)
    blocks = partition_blocks(x, b)
    assert blocks.shape == (2, (24 // b) ** 2, b, b, 5)
    assert torch.equal(merge_blocks(blocks), x)


def test_raster_order_and_coords():
    ids = torch.arange(576).reshape(1, 24, 24, 1)
    blocks = partition_blocks(ids, 6)[0, :, :, :, 0]      # (16, 6, 6)
    assert blocks[0, 0, 0] == 0 and blocks[1, 0, 0] == 6  # right neighbour
    assert blocks[4, 0, 0] == 6 * 24                      # next block row
    coords = block_coords(24, 6)
    assert coords.shape == (16, 2)
    assert coords[1].tolist() == [0, 1] and coords[4].tolist() == [1, 0]


def test_seq_to_grid_and_errors():
    x = torch.randn(1, 576, 3)
    assert torch.equal(seq_to_grid(x)[0, 1, 2], x[0, 26])
    with pytest.raises(ValueError):
        partition_blocks(torch.randn(1, 24, 24, 1), 5)
    with pytest.raises(ValueError):
        seq_to_grid(torch.randn(1, 575, 3))
