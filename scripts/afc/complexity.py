"""Paper 2.4: theoretical KV-cache / prefill cost of the frozen LLM and the
measured FLOPs of AFC itself (torch FlopCounterMode, matmul-type ops).

    python scripts/afc/complexity.py   # prints markdown tables

LLM: Vicuna-7B (L=32, d=4096, MHA, FP16). Prefill FLOPs = 2*P*N + 2*L*d*N^2
(linear layers + causal attention scores/values); P = non-embedding params.
"""

import json

import torch
from torch.utils.flop_counter import FlopCounterMode

from llava.model.visual_compressor import VisualCompressorConfig
from llava.model.visual_compressor.builder import create_compressor

L, D, BYTES = 32, 4096, 2
VOCAB, FFN = 32000, 11008
P_TOTAL = 6.738e9                     # Vicuna-7B incl. embeddings + lm_head
P_NONEMB = P_TOTAL - 2 * VOCAB * D    # linear-layer params used per token
N_VIS, N_TEXT = 576, 64               # N_TEXT: illustrative prompt length
BUDGETS = [576, 256, 144, 64]
CKPT = "checkpoints/llava-v1.5-7b-afc-{}_all/config.json"
MIB, G, T = 2 ** 20, 1e9, 1e12


def kv_per_token():
    return 2 * L * D * BYTES


def prefill_flops(n):
    return 2 * P_NONEMB * n + 2 * L * D * n * n


def afc(mode, k):
    cfg = json.load(open(CKPT.format(mode)))["visual_compressor"]
    cfg["target_visual_tokens"] = k
    m = create_compressor(VisualCompressorConfig(**cfg).resolve(24, 1024)).eval()
    x = torch.randn(1, 576, 1024)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops(), sum(p.numel() for p in m.parameters())


def main():
    kv = kv_per_token()
    print(f"KV/token = {kv / MIB:.3f} MiB, P_nonemb = {P_NONEMB / G:.2f} B\n")
    print("| #I | visual KV (MiB) | KV red. | KV red. (+text) | prefill TFLOPs | prefill red. |")
    print("|---|---|---|---|---|---|")
    base_all = prefill_flops(N_VIS + N_TEXT)
    for k in BUDGETS:
        kvv = k * kv / MIB
        red = 1 - k / N_VIS
        red_t = 1 - (k + N_TEXT) / (N_VIS + N_TEXT)
        pf = prefill_flops(k + N_TEXT)
        print(f"| {k} | {kvv:.0f} | {red:.1%} | {red_t:.1%} | {pf / T:.2f} | {1 - pf / base_all:.1%} |")
    print("\n| mode | #I | AFC GFLOPs | params (M) | saved LLM prefill TFLOPs | AFC / saved |")
    print("|---|---|---|---|---|---|")
    for mode in ("global", "block"):
        for k in BUDGETS[1:]:
            f, p = afc(mode, k)
            saved = prefill_flops(N_VIS + N_TEXT) - prefill_flops(k + N_TEXT)
            print(f"| {mode} | {k} | {f / G:.3f} | {p / 1e6:.2f} | {saved / T:.2f} | {f / saved:.3%} |")
    # DCT alone (matrix form, all 1024 channels)
    g = 2 * 2 * 24 ** 3 * 1024
    b = 16 * 2 * 2 * 6 ** 3 * 1024
    print(f"\nDCT only: global {g / 1e6:.1f} MFLOPs, block {b / 1e6:.1f} MFLOPs")
    clip = 2 * 303e6 * 577 + 2 * 24 * 1024 * 577 ** 2
    print(f"CLIP ViT-L/14-336 forward ~ {clip / G:.0f} GFLOPs")
    # decode: batch size bound by KV capacity on a 24 GiB GPU
    w = P_TOTAL * BYTES
    for k in (576, 64):
        ctx = k + N_TEXT + 512
        free = 24 * 2 ** 30 - w - 2 * 2 ** 30   # 2 GiB other
        print(f"#I={k}: ctx {ctx}, KV/req {ctx * kv / MIB:.0f} MiB, "
              f"max batch on 24GiB ~ {int(free // (ctx * kv))}")


if __name__ == "__main__":
    main()
