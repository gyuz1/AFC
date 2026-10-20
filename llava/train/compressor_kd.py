"""Teacher-student logit distillation aligned by ANSWER TOKEN ORDINAL.

Teacher (576 visual tokens) and student (e.g. 64) sequences have different
lengths, so absolute positions are never compared. With causal shift, the
logit at position t predicts label t+1; for every sample we keep the shifted
positions whose label is not IGNORE_INDEX (the answer tokens), check that
teacher and student targets are identical and in the same order, and compare
the m-th answer prediction of the teacher with the m-th of the student.

    L_KD = T^2 * KL(softmax(z_T / T) || softmax(z_S / T))   (exact, full vocab)
averaged over all answer tokens of the batch.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX


def answer_logits(logits: torch.Tensor, labels: torch.Tensor
                  ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Per sample: (logits predicting answer tokens (n_i, V), targets (n_i,)).

    ``logits`` ``(B, L, V)`` and ``labels`` ``(B, L)`` are the multimodal
    (post image insertion) tensors of one forward pass.
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)} vs labels "
                         f"{tuple(labels.shape)}")
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    out = []
    for b in range(labels.shape[0]):
        keep = shift_labels[b] != IGNORE_INDEX
        out.append((shift_logits[b][keep], shift_labels[b][keep]))
    return out


def kd_loss(student_logits: torch.Tensor, student_labels: torch.Tensor,
            teacher_logits: torch.Tensor, teacher_labels: torch.Tensor,
            temperature: float = 2.0) -> torch.Tensor:
    """Exact (full-vocabulary) temperature-scaled KL, answer-aligned."""
    s_list = answer_logits(student_logits, student_labels)
    t_list = answer_logits(teacher_logits, teacher_labels)
    total = student_logits.new_zeros((), dtype=torch.float32)
    count = 0
    for b, ((zs, ys), (zt, yt)) in enumerate(zip(s_list, t_list)):
        if not torch.equal(ys, yt.to(ys.device)):
            raise RuntimeError(
                f"sample {b}: teacher/student answer targets differ "
                f"({ys.numel()} vs {yt.numel()} tokens); cannot align KD")
        if ys.numel() == 0:
            continue
        log_ps = F.log_softmax(zs.float() / temperature, dim=-1)
        log_pt = F.log_softmax(zt.float().to(zs.device) / temperature, dim=-1)
        total = total + F.kl_div(log_ps, log_pt, log_target=True,
                                 reduction="sum")
        count += ys.numel()
    if count == 0:
        return total
    return temperature ** 2 * total / count
