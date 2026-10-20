"""Answer-token CE / KD on the held-out validation samples of a run.

Measures reference points for the training curves:
    vanilla        original LLaVA, 576 tokens, original projector
    vanilla+proj   576 tokens with the projector of a trained checkpoint
    avgpool        parameter-free 8x8 average pooling, original projector
    <ckpt>         a trained compressor checkpoint (compressor + projector)

    CUDA_VISIBLE_DEVICES=1 python scripts/afc/val_ce.py --run global75k \
        --ckpt checkpoints/llava-v1.5-7b-afc-global75k --n 200
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from llava import conversation as conversation_lib
from llava.eval.afc import MODEL_PATH
from llava.model import LlavaLlamaForCausalLM
from llava.model.visual_compressor import VisualCompressorConfig
from llava.model.visual_compressor.builder import (compressor_disabled,
                                                   load_compressor_weights)
from llava.train.compressor_objective import CompressorObjective
from llava.train.train import (DataArguments, DataCollatorForSupervisedDataset,
                               LazySupervisedDataset)
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]


@torch.no_grad()
def mean_ce(model, data, collator, kd=False):
    obj = CompressorObjective(model, kd_lambda=1.0 if kd else 0.0)
    ce, kds, n = 0.0, 0.0, 0
    for i in range(len(data)):
        batch = collator([data[i]])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = obj(**batch)
        tok = int(out["answer_tokens"])
        ce += float(out["ce"]) * tok
        if kd:
            kds += float(out["kd"]) * tok
        n += tok
    return ce / n, (kds / n if kd else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run whose eval_subset.json "
                    "defines the held-out samples")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data_path", default=str(
        Path(MODEL_PATH).parents[1] / "llava_train/llava_v1_5_available.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), model_max_length=768,
                                        padding_side="right", use_fast=False)
    tok.pad_token = tok.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
    model = LlavaLlamaForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa").cuda().eval()
    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt.to(device="cuda", dtype=torch.bfloat16)

    da = DataArguments(data_path=args.data_path, lazy_preprocess=True,
                       is_multimodal=True,
                       image_folder=str(Path(args.data_path).parent),
                       image_aspect_ratio="pad")
    da.image_processor = vt.image_processor
    da.mm_use_im_start_end = False
    data = LazySupervisedDataset(args.data_path, tok, da)
    pool = data.list_data_dict
    ev = json.load(open(REPO / f"checkpoints/llava-v1.5-7b-afc-{args.run}/"
                        "eval_subset.json"))["indices"][:args.n]
    data.list_data_dict = [pool[i] for i in ev]
    coll = DataCollatorForSupervisedDataset(tokenizer=tok)
    orig_proj = copy.deepcopy(model.get_model().mm_projector.state_dict())

    res = {"vanilla (576, original projector)": mean_ce(model, data, coll)[0]}
    model.get_model().initialize_visual_compressor(
        VisualCompressorConfig(compression_mode="avgpool",
                               target_visual_tokens=64))
    res["avgpool 64 (original projector)"] = mean_ce(model, data, coll)[0]
    if args.ckpt:
        cfg = json.load(open(Path(args.ckpt) / "config.json"))["visual_compressor"]
        model.get_model().initialize_visual_compressor(
            VisualCompressorConfig.from_dict(
                {k: v for k, v in cfg.items()
                 if k not in ("grid_size", "input_dim")}))
        model.get_model().visual_compressor.to(torch.bfloat16)
        load_compressor_weights(model, args.ckpt)
        ce, kd = mean_ce(model, data, coll, kd=True)
        res[f"trained {Path(args.ckpt).name} (CE, KD vs current teacher)"] = (ce, kd)
        with compressor_disabled(model):
            res["vanilla 576 + TRAINED projector"] = mean_ce(model, data,
                                                              coll)[0]
        model.get_model().mm_projector.load_state_dict(orig_proj)
        ce, kd = mean_ce(model, data, coll, kd=True)
        res[f"trained compressor + ORIGINAL projector (CE, KD)"] = (ce, kd)
    for k, v in res.items():
        print(f"{k:58s} {v}")


if __name__ == "__main__":
    main()
