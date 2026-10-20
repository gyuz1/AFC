import pytest
import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX
from llava.train.compressor_kd import answer_logits, kd_loss

V = 11


def _batch(prefix_len, answer):
    labels = torch.full((1, prefix_len + len(answer)), IGNORE_INDEX)
    labels[0, prefix_len:] = torch.tensor(answer)
    return labels


def test_alignment_by_answer_ordinal_not_position():
    answer = [3, 7, 2]
    s_lab, t_lab = _batch(5, answer), _batch(40, answer)  # different lengths
    s_log = torch.randn(1, s_lab.shape[1], V)
    t_log = torch.randn(1, t_lab.shape[1], V)
    (zs, ys), = answer_logits(s_log, s_lab)
    (zt, yt), = answer_logits(t_log, t_lab)
    assert ys.tolist() == yt.tolist() == answer
    assert torch.equal(zs, s_log[0, 4:7]) and torch.equal(zt, t_log[0, 39:42])


def test_kd_zero_when_identical_and_matches_manual():
    answer = [1, 4]
    s_lab, t_lab = _batch(3, answer), _batch(9, answer)
    t_log = torch.randn(1, 11, V)
    s_log = torch.zeros(1, 5, V)
    s_log[0, 2:4] = t_log[0, 8:10]
    assert kd_loss(s_log, s_lab, t_log, t_lab, 2.0).abs() < 1e-6
    s_log = torch.randn(1, 5, V)
    T = 2.0
    manual = (F.softmax(t_log[0, 8:10] / T, -1) *
              (F.log_softmax(t_log[0, 8:10] / T, -1)
               - F.log_softmax(s_log[0, 2:4] / T, -1))).sum() / 2 * T * T
    assert torch.allclose(kd_loss(s_log, s_lab, t_log, t_lab, T), manual,
                          atol=1e-5)


def test_mismatched_targets_raise():
    s_lab, t_lab = _batch(3, [1, 2]), _batch(9, [1, 3])
    with pytest.raises(RuntimeError):
        kd_loss(torch.randn(1, 5, V), s_lab, torch.randn(1, 11, V), t_lab)
