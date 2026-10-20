"""Train the visual token compressor on a frozen LLaVA-1.5 (AFC / V2).

Everything except the compressor is frozen: CLIP, mm_projector, Vicuna and
the LM head. The student LLM forward is NOT under no_grad, so the answer CE
gradient flows through the frozen LLM and projector into the compressor.
Data, conversation preprocessing and label masking are LLaVA's own
(``LazySupervisedDataset``, ``preprocess``, ``DataCollatorForSupervisedDataset``).

    python -m llava.train.train_compressor --config configs/afc/block_dct_64.yaml \
        [--config configs/afc/ablations/<one axis>.yaml] \
        --model_name_or_path liuhaotian/llava-v1.5-7b \
        --data_path <llava json> --image_folder <images> \
        --output_dir checkpoints/llava-v1.5-7b-afc-block_dct_64

Several --config files are applied in order (later ones override earlier
ones); CLI flags override all YAML. One GPU: ``python -m ...``; several GPUs (DDP):
``torchrun --nproc_per_node N -m llava.train.train_compressor ...``.
With ``--train_mm_projector`` the projector is trained too (its own
``--mm_projector_lr``) and saved in the compressor checkpoint.
"""

import copy
import dataclasses
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
import yaml
from transformers import AutoTokenizer, HfArgumentParser

from llava import conversation as conversation_lib
from llava.model import LlavaLlamaForCausalLM
from llava.model.visual_compressor import VisualCompressorConfig
from llava.model.visual_compressor.builder import (WEIGHTS_NAME,
                                                   load_compressor_weights)
from llava.model.visual_compressor.diagnostics import (diversity_penalty,
                                                       summarize)
from llava.train.compressor_objective import CompressorObjective
from llava.train.llava_trainer import LLaVATrainer
from llava.train.train import (DataArguments, DataCollatorForSupervisedDataset,
                               LazySupervisedDataset, TrainingArguments)

class SizedSupervisedDataset(LazySupervisedDataset):
    """LLaVA's dataset plus the original ``(width, height)`` of the image
    (read from the file header), for ``valid_region``."""

    def __getitem__(self, i):
        item = super().__getitem__(i)
        src = self.list_data_dict[i]
        if "image" in src:
            from PIL import Image
            with Image.open(os.path.join(self.data_args.image_folder,
                                         src["image"])) as im:
                item["image_size"] = im.size
        return item


@dataclass
class SizedCollator(DataCollatorForSupervisedDataset):
    def __call__(self, instances):
        batch = super().__call__(instances)
        if all("image_size" in x for x in instances):
            batch["image_sizes"] = [x["image_size"] for x in instances]
        return batch


# Files of the base checkpoint that are not linked into the compressor dir.
_NOT_LINKED = {"config.json", "README.md", WEIGHTS_NAME}


@dataclass
class CompressorArguments:
    model_name_or_path: str = field(default="liuhaotian/llava-v1.5-7b")
    version: str = field(default="v1", metadata={
        "help": "conversation template used by LLaVA-1.5 fine-tuning"})
    compression_mode: str = "block_dct"
    block_size: int = 6
    target_visual_tokens: int = 64
    compressor_hidden_dim: int = 128
    compressor_layers: int = 2
    compressor_heads: int = 4
    compressor_ffn_dim: int = 512
    compressor_dropout: float = 0.0
    pool_heads: int = 4
    pool_prenorm: bool = True
    pool_learnable_scale: bool = False
    residual_base: str = "none"
    fourier_scale: bool = True
    pos_mode: str = "freq_and_block"
    use_frequency_transformer: bool = True
    adapter_type: str = "mlp"
    adapter_hidden_dim: int = 512
    query_sharing: str = "shared"
    valid_region: bool = field(default=False, metadata={
        "help": "compress only the image area of the square-padded input "
                "(needs the original image sizes; added to every batch)"})
    train_mm_projector: bool = field(default=False, metadata={
        "help": "also train the (pretrained) mm_projector; lr from "
                "--mm_projector_lr"})
    kd_lambda: float = 0.0
    kd_temperature: float = 2.0
    diversity_lambda: float = 0.0
    return_compressor_attention: bool = field(default=False, metadata={
        "help": "log query-attention diagnostics at logging steps"})
    max_train_samples: Optional[int] = None
    sample_seed: int = 0
    eval_holdout: int = field(default=0, metadata={
        "help": "hold out N samples of the data pool (disjoint from the "
                "training subset) for validation loss"})
    subset_strategy: str = field(default="random", metadata={
        "help": "random | longest (longest samples first; for worst-case "
                "memory profiling only)"})

    def compressor_config(self) -> VisualCompressorConfig:
        names = {f.name for f in dataclasses.fields(VisualCompressorConfig)}
        return VisualCompressorConfig(**{k: v for k, v in
                                         dataclasses.asdict(self).items()
                                         if k in names})


@dataclass
class CompressorDataArguments(DataArguments):
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    image_aspect_ratio: str = "pad"


@dataclass
class CompressorTrainingArguments(TrainingArguments):
    model_max_length: int = 2048
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    report_to: str = "none"
    save_safetensors: bool = False
    ddp_find_unused_parameters: bool = False


# ---------------------------------------------------------------------------
def parse_args():
    parser = HfArgumentParser((CompressorArguments, CompressorDataArguments,
                               CompressorTrainingArguments))
    argv, defaults = sys.argv[1:], {}
    while "--config" in argv:
        i = argv.index("--config")
        with open(argv[i + 1]) as f:
            defaults.update({k.replace("-", "_"): v
                             for k, v in (yaml.safe_load(f) or {}).items()})
        argv = argv[:i] + argv[i + 2:]
    parser.set_defaults(**defaults)
    return parser.parse_args_into_dataclasses(args=argv)


def param_summary(model) -> str:
    groups: Dict[str, list] = {}
    for name, p in model.named_parameters():
        if "visual_compressor" in name:
            key = "compressor." + name.split("visual_compressor.")[1].split(
                ".")[0]
        elif "vision_tower" in name:
            key = "vision_tower (CLIP)"
        elif "mm_projector" in name:
            key = "mm_projector"
        else:
            key = "language model"
        g = groups.setdefault(key, [0, 0])
        g[0] += p.numel()
        g[1] += p.numel() if p.requires_grad else 0
    total = sum(g[0] for g in groups.values())
    train = sum(g[1] for g in groups.values())
    lines = [f"{'module':<28}{'params':>15}{'trainable':>15}"]
    lines += [f"{k:<28}{v[0]:>15,}{v[1]:>15,}" for k, v in sorted(
        groups.items())]
    lines.append(f"{'TOTAL':<28}{total:>15,}{train:>15,}  "
                 f"(frozen {total - train:,})")
    return "\n".join(lines)


def local_model_dir(path: str) -> str:
    if os.path.isdir(path):
        return os.path.abspath(path)
    from huggingface_hub import snapshot_download
    return snapshot_download(path)


def save_compressor_checkpoint(model, output_dir: str, base_dir: str):
    """Save compressor weights + config; link the frozen base files.

    The result is a complete LLaVA model directory (loadable by
    ``load_pretrained_model``, the official eval scripts and lmms-eval) that
    stores only the compressor.
    """
    os.makedirs(output_dir, exist_ok=True)
    inner = model.get_model()
    state = {f"compressor.{k}": v.detach().cpu()
             for k, v in inner.visual_compressor.state_dict().items()}
    if getattr(model.config, "afc_trained_mm_projector", False):
        state.update({f"mm_projector.{k}": v.detach().cpu()
                      for k, v in inner.mm_projector.state_dict().items()})
    torch.save(state, os.path.join(output_dir, WEIGHTS_NAME))
    config = model.config
    use_cache = config.use_cache
    config.use_cache = True
    config.afc_base_model = base_dir
    config.save_pretrained(output_dir)
    config.use_cache = use_cache
    for name in os.listdir(base_dir):
        if name in _NOT_LINKED or name.startswith("."):
            continue
        dst = os.path.join(output_dir, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.relpath(os.path.join(base_dir, name),
                                       output_dir), dst)


def unwrap(model):
    """DDP -> CompressorObjective."""
    return model.module if hasattr(model, "module") else model


def llava_of(model):
    """The LLaVA model inside a (DDP-wrapped) CompressorObjective."""
    return unwrap(model).llava


class CompressorTrainer(LLaVATrainer):
    def __init__(self, *args, comp_args: CompressorArguments,
                 base_dir: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.comp_args = comp_args
        self.base_dir = base_dir
        self._extra: Dict[str, list] = {}
        self._eval_extra: Dict[str, list] = {}
        self._first_batch = True

    def _record(self, **values):
        for k, v in values.items():
            self._extra.setdefault(k, []).append(float(v))

    def compute_loss(self, model, inputs, return_outputs=False):
        ca = self.comp_args
        compressor = llava_of(model).get_model().visual_compressor
        want_attn = ca.diversity_lambda > 0 or (
            ca.return_compressor_attention and self._should_log_next())
        compressor.keep_attention = want_attn

        out = model(**inputs)               # CompressorObjective.forward
        loss = out["loss"]
        if not model.training:              # validation: separate bucket
            for k in ("ce", "kd"):
                if k in out:
                    self._eval_extra.setdefault(k, []).append(out[k].item())
            return (loss, out) if return_outputs else loss
        self._record(ce=out["ce"].item())
        if "kd" in out:
            self._record(kd=out["kd"].item())
        if want_attn and compressor.last_attention is not None:
            attn = compressor.last_attention
            if ca.diversity_lambda > 0:
                div = diversity_penalty(attn)
                loss = loss + ca.diversity_lambda * div
                self._record(diversity=div.item())
            if ca.return_compressor_attention:
                for k, v in summarize(attn).items():
                    self._record(**{k: v})
        if self._first_batch and self.is_world_process_zero():
            self._first_batch = False
            print("[afc] first batch:")
            for k, v in compressor.last_shapes.items():
                print(f"  {k:<14}{v}")
            print(f"  sequence      {int(out['seq_len'])} (multimodal, max "
                  f"in batch)")
            print(f"  answer tokens {int(out['answer_tokens'])}")
            print(f"  loss          {loss.item():.4f}")
        return (loss, out) if return_outputs else loss

    def _should_log_next(self) -> bool:
        step = self.state.global_step + 1
        return self.args.logging_steps > 0 and \
            step % self.args.logging_steps == 0

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self.state.global_step == 0 and self.is_world_process_zero() \
                and not getattr(self, "_grad_shown", False):
            self._grad_shown = True
            print("[afc] grad norms after the first backward:")
            for name, p in llava_of(model).get_model().visual_compressor \
                    .named_parameters():
                if p.grad is not None and ("weight" in name or "queries" in name
                                           or "emb_" in name):
                    print(f"  {name:<52}{p.grad.norm().item():.3e}")
            frozen = [n for n, p in llava_of(model).named_parameters()
                      if not p.requires_grad and p.grad is not None]
            print(f"  frozen params with .grad: {len(frozen)}")
        return loss

    def log(self, logs: Dict[str, float]):
        if "eval_loss" in logs:
            for k, v in self._eval_extra.items():
                logs[f"eval_{k}"] = sum(v) / len(v)
            self._eval_extra = {}
        else:
            for k, v in self._extra.items():
                logs[k] = sum(v) / len(v)
            self._extra = {}
        super().log(logs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        save_compressor_checkpoint(llava_of(self.model), output_dir,
                                   self.base_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Resume: only the compressor (+ projector) differs from the base."""
        load_compressor_weights(llava_of(model or self.model),
                                resume_from_checkpoint)


def select_subset(dataset, n: Optional[int], seed: int, out_dir: str,
                  strategy: str = "random", write: bool = True):
    total = len(dataset.list_data_dict)
    if n is None or n >= total:
        idx = list(range(total))
    elif strategy == "longest":
        lengths = dataset.lengths                  # LLaVA's word-count proxy
        idx = sorted(range(total), key=lambda i: -lengths[i])[:n]
        dataset.list_data_dict = [dataset.list_data_dict[i] for i in idx]
    elif strategy == "random":
        idx = sorted(random.Random(seed).sample(range(total), n))
        dataset.list_data_dict = [dataset.list_data_dict[i] for i in idx]
    else:
        raise ValueError(f"unknown subset_strategy {strategy!r}")
    if not write:
        return idx
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train_subset.json"), "w") as f:
        json.dump({"seed": seed, "source_size": total, "indices": idx,
                   "ids": [d.get("id") for d in dataset.list_data_dict]}, f)
    return idx


def main():
    comp_args, data_args, training_args = parse_args()
    if training_args.world_size == 1 and torch.cuda.device_count() > 1:
        raise SystemExit("several GPUs visible: use torchrun for DDP or set "
                         "CUDA_VISIBLE_DEVICES to one device")
    is_main = training_args.process_index == 0
    if training_args.gradient_checkpointing_kwargs is None:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    transformers.set_seed(training_args.seed)
    ccfg = comp_args.compressor_config()
    if not ccfg.enabled:
        raise SystemExit("compression_mode=none: nothing to train")

    dtype = (torch.bfloat16 if training_args.bf16 else
             torch.float16 if training_args.fp16 else torch.float32)
    base_dir = local_model_dir(comp_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_dir, model_max_length=training_args.model_max_length,
        padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = \
        conversation_lib.conv_templates[comp_args.version]

    # Ranks load one after another and move to their GPU at once, so host
    # RAM holds a single CPU copy of the 7B weights at a time.
    with training_args.main_process_first(local=True, desc="load LLaVA"):
        model = LlavaLlamaForCausalLM.from_pretrained(
            base_dir, torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation="sdpa")
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=dtype)
        model.to(training_args.device)
    model.config.use_cache = False

    model.requires_grad_(False)
    compressor = model.get_model().initialize_visual_compressor(ccfg)
    compressor.requires_grad_(True)                 # fp32 master weights
    model.config.afc_trained_mm_projector = comp_args.train_mm_projector
    if comp_args.train_mm_projector:   # fp32 master weights for AdamW
        model.get_model().mm_projector.float().requires_grad_(True)
        compressor.output_dtype = torch.float32
    if is_main:
        print("[afc]", compressor.describe().replace("\n", "\n[afc] "))
        print(f"[afc] KD lambda {comp_args.kd_lambda} "
              f"(T={comp_args.kd_temperature}) | diversity lambda "
              f"{comp_args.diversity_lambda} | train mm_projector "
              f"{comp_args.train_mm_projector} (lr "
              f"{training_args.mm_projector_lr}) | world size "
              f"{training_args.world_size}")
        print(param_summary(model))

    data_args.image_processor = vision_tower.image_processor
    data_args.mm_use_im_start_end = False
    data_args.image_aspect_ratio = getattr(model.config, "image_aspect_ratio",
                                           data_args.image_aspect_ratio)
    dataset_cls = (SizedSupervisedDataset if comp_args.valid_region
                   else LazySupervisedDataset)
    dataset = dataset_cls(data_args.data_path, tokenizer, data_args)
    pool = list(dataset.list_data_dict)
    idx = select_subset(dataset, comp_args.max_train_samples,
                        comp_args.sample_seed, training_args.output_dir,
                        comp_args.subset_strategy, write=is_main)
    eval_dataset = None
    if comp_args.eval_holdout > 0:
        rest = sorted(set(range(len(pool))) - set(idx))
        if len(rest) < comp_args.eval_holdout:
            # Training uses (nearly) the whole pool: carve the held-out
            # samples out of the training subset so they stay disjoint.
            ev = sorted(random.Random(comp_args.sample_seed + 1).sample(
                idx, comp_args.eval_holdout))
            held = set(ev)
            idx = [i for i in idx if i not in held]
            dataset.list_data_dict = [pool[i] for i in idx]
            if is_main:
                with open(os.path.join(training_args.output_dir,
                                       "train_subset.json"), "w") as f:
                    json.dump({"seed": comp_args.sample_seed,
                               "source_size": len(pool), "indices": idx,
                               "ids": [d.get("id")
                                       for d in dataset.list_data_dict]}, f)
            rest = []
        ev = ev if not rest else sorted(random.Random(
            comp_args.sample_seed + 1).sample(
                rest, min(comp_args.eval_holdout, len(rest))))
        eval_dataset = copy.copy(dataset)
        eval_dataset.list_data_dict = [pool[i] for i in ev]
        if is_main:
            with open(os.path.join(training_args.output_dir,
                                   "eval_subset.json"), "w") as f:
                json.dump({"indices": ev}, f)
    del pool
    if is_main:
        print(f"[afc] training samples: {len(idx)} | validation samples: "
              f"{0 if eval_dataset is None else len(eval_dataset)}")

    objective = CompressorObjective(model, comp_args.kd_lambda,
                                    comp_args.kd_temperature)
    trainer = CompressorTrainer(
        model=objective, args=training_args, train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=(SizedCollator if comp_args.valid_region else
                       DataCollatorForSupervisedDataset)(tokenizer=tokenizer),
        tokenizer=tokenizer, comp_args=comp_args, base_dir=base_dir)
    torch.cuda.reset_peak_memory_stats()
    resume = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume)
    trained = [p for p in model.parameters() if p.requires_grad]
    checksum = sum(float(p.detach().double().sum()) for p in trained)
    print(f"[afc] rank {training_args.process_index} peak GPU memory: "
          f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB | "
          f"trainable checksum {checksum:.6f}")
    if trainer.is_world_process_zero():
        trainer.save_state()
        trainer._save(training_args.output_dir)
        print(f"[afc] saved compressor checkpoint to "
              f"{training_args.output_dir}")


if __name__ == "__main__":
    main()
