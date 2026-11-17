"""Answer-token KD: L_KD = T^2 * KL(p_teacher || p_student), averaged over answer
tokens (teacher and student sequences are aligned by answer-token order)."""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX


def answer_logits(logits: torch.Tensor, labels: torch.Tensor
                  ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Per sample: logits predicting answer tokens (n_i, V) and their targets."""
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
