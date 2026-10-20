"""Training objective of the visual compressor as ONE module forward.

    embeddings (CLIP -> compressor -> projector -> <image> insertion)
    -> LLaMA decoder -> hidden states (B, L, 4096)
    -> keep only positions whose shifted label is an answer token
    -> lm_head on those positions only (n, |V|) -> CE  (+ KD vs. the
       576-token teacher, computed the same way under no_grad)

Equivalent to HF's causal-LM loss (mean CE over non-ignored shifted labels)
but never materialises (B, L, |V|) logits: with ~700 positions and a few
dozen answer tokens per sample this removes most of the LM-head time and
memory, for the student and the teacher. Running everything inside this
module's forward lets DDP see every trainable parameter in one graph.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from llava.constants import IGNORE_INDEX
from llava.model.visual_compressor.builder import compressor_disabled


def answer_positions(hidden: torch.Tensor, labels: torch.Tensor):
    """Hidden states that predict answer tokens and their targets.

    ``hidden`` (B, L, H), ``labels`` (B, L) post-insertion. Position t
    predicts label t+1. Returns ``(h (n, H), targets (n,), counts (B,))``
    flattened in batch order (sample 0's answer tokens first).
    """
    h = hidden[:, :-1]
    y = labels[:, 1:]
    keep = y != IGNORE_INDEX
    return h[keep], y[keep], keep.sum(dim=1)


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
          temperature: float) -> torch.Tensor:
    """``T^2 * KL(p_T || p_S)`` averaged over answer tokens (exact, full
    vocabulary)."""
    log_ps = F.log_softmax(student_logits.float() / temperature, dim=-1)
    log_pt = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    kl = F.kl_div(log_ps, log_pt, log_target=True, reduction="sum")
    return temperature ** 2 * kl / max(student_logits.shape[0], 1)


class CompressorObjective(nn.Module):
    """Wraps a LLaVA model; ``forward`` returns the loss and its parts."""

    def __init__(self, llava: nn.Module, kd_lambda: float = 0.0,
                 kd_temperature: float = 2.0):
        super().__init__()
        self.llava = llava
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature

    # --- attributes the HF Trainer expects ------------------------------
    @property
    def config(self):
        return self.llava.config

    @property
    def dtype(self):
        return self.llava.dtype

    def gradient_checkpointing_enable(self, **kwargs):
        self.llava.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.llava.gradient_checkpointing_disable()

    # ------------------------------------------------------------------
    def _answer_logits(self, input_ids, labels, attention_mask, images,
                       image_sizes):
        llava = self.llava
        (_, position_ids, attn, _, embeds, mm_labels) = \
            llava.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, labels, images,
                image_sizes)
        hidden = llava.model(inputs_embeds=embeds, attention_mask=attn,
                             position_ids=position_ids, use_cache=False,
                             return_dict=True).last_hidden_state
        h, targets, counts = answer_positions(hidden, mm_labels)
        return llava.lm_head(h), targets, counts, mm_labels.shape

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                images=None, image_sizes=None) -> Dict[str, torch.Tensor]:
        dtype = self.llava.dtype
        images = ([x.to(dtype) for x in images] if isinstance(images, list)
                  else images.to(dtype))
        amp = dtype in (torch.bfloat16, torch.float16)
        # Explicit autocast: fp32 trainable compressor / projector meet bf16
        # CLIP features and LLM weights.
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp):
            zs, ys, counts, seq_shape = self._answer_logits(
                input_ids, labels, attention_mask, images, image_sizes)
            ce = F.cross_entropy(zs.float(), ys)
            out = {"ce": ce.detach(), "answer_tokens": counts.sum().detach(),
                   "seq_len": torch.tensor(seq_shape[1], device=ce.device)}
            loss = ce
            if self.kd_lambda > 0:
                with torch.no_grad(), compressor_disabled(self.llava):
                    zt, yt, t_counts, _ = self._answer_logits(
                        input_ids, labels, attention_mask, images,
                        image_sizes)
                if not (torch.equal(ys, yt) and torch.equal(counts,
                                                            t_counts)):
                    raise RuntimeError("teacher/student answer targets "
                                       "differ; cannot align KD")
                kd = kd_kl(zs, zt, self.kd_temperature)
                loss = loss + self.kd_lambda * kd
                out["kd"] = kd.detach()
        out["loss"] = loss
        return out
