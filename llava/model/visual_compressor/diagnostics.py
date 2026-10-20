"""Query-collapse diagnostics for the pooling attention.

Attention weights describe query-frequency affinity / utilisation patterns;
they are not causal importance.
"""

from typing import Dict

import torch
import torch.nn.functional as F


def head_mean(attn: torch.Tensor) -> torch.Tensor:
    """``(B, J, H, k, N)`` -> ``(B, J, k, N)``."""
    return attn.mean(dim=2)


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Entropy (nats) of each query distribution: ``(..., k, N)`` -> ``(..., k)``."""
    return -(attn * attn.clamp_min(1e-12).log()).sum(-1)


def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine similarity of the k rows of ``(..., k, F)``."""
    x = F.normalize(x.float(), dim=-1)
    sim = x @ x.transpose(-1, -2)                          # (..., k, k)
    k = sim.shape[-1]
    if k < 2:
        return torch.zeros(sim.shape[:-2], device=sim.device)
    off = sim.sum((-1, -2)) - sim.diagonal(dim1=-2, dim2=-1).sum(-1)
    return off / (k * (k - 1))


def diversity_penalty(attn: torch.Tensor) -> torch.Tensor:
    """``|| A A^T - I ||_F^2`` on head-averaged, L2-normalised rows (optional
    regulariser; disabled by default)."""
    a = F.normalize(head_mean(attn).float(), dim=-1)       # (B, J, k, N)
    gram = a @ a.transpose(-1, -2)
    eye = torch.eye(gram.shape[-1], device=gram.device)
    return (gram - eye).pow(2).sum((-1, -2)).mean()


@torch.no_grad()
def summarize(attn: torch.Tensor, pooled: torch.Tensor = None) -> Dict:
    """Scalar summary for logging. ``attn`` ``(B, J, H, k, N)``."""
    a = head_mean(attn.float())                            # (B, J, k, N)
    n = a.shape[-1]
    out = {
        "attn_entropy": attention_entropy(a).mean().item(),
        "attn_entropy_max": float(torch.log(torch.tensor(float(n)))),
        "query_attn_cosine": pairwise_cosine(a).mean().item(),
        # spread of the same query's map across blocks / images
        "block_variation": a.std(dim=1).mean().item(),
        "image_variation": a.std(dim=0).mean().item() if a.shape[0] > 1
        else float("nan"),
    }
    if pooled is not None:
        out["pooled_output_cosine"] = pairwise_cosine(pooled).mean().item()
    return out
