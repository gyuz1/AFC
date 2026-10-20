# AFC: frequency-domain visual token compression for LLaVA-1.5

This fork adds a trainable visual token compressor between the CLIP vision
tower and the frozen `mm_projector` of LLaVA-1.5-7B. Vanilla behaviour is
unchanged when no compressor is configured (verified bitwise against the
unmodified LLaVA code).

## Method (main configuration)

```
CLIP ViT-L/14-336 (frozen)        (B, 576, 1024)
-> 24x24 grid -> 6x6 blocks       (B, 16, 6, 6, 1024)
-> block 2-D DCT (spatial axes)   (B, 16, 6, 6, 1024)
-> 36 frequency tokens / block    (B, 16, 36, 1024)
-> LayerNorm + Linear 1024->128   (B, 16, 36, 128)    channel bottleneck
   + e_freq_u[u] + e_freq_v[v] + e_block_row[r] + e_block_col[c]
-> frequency Transformer          (B, 16, 36, 128)    2 Pre-LN layers, 4 heads
-> learnable query pooling        (B, 16, 4, 128)     4 shared queries,
                                                      cross-attention over all 36
-> adapter 128->512->1024 (MLP)   (B, 16, 4, 1024)
-> raster flatten                 (B, 64, 1024)
-> mm_projector (frozen)          (B, 64, 4096)
-> LLaVA <image> insertion, Vicuna-7B (frozen), answer-token CE (+ optional KD)
```

* Fully differentiable: no top-k, STE, Gumbel or masks. Each pooled token is
  a learned soft combination of all frequency tokens of its block.
* Query pooling uses Pre-LN cross-attention (LayerNorm on queries and on
  keys/values). Without it the small query init gives uniform attention and
  four identical outputs per block (measured: query-attention cosine 1.0000
  at init and after 30 steps); `pool_no_prenorm.yaml` keeps that variant.
* Trainable: 1.19 M parameters (encoder 532 K, pooling 67 K, adapter 592 K).
  Frozen: CLIP, projector, Vicuna, LM head (7.06 B).
* The student LLM forward is not under `no_grad`; gradients pass through the
  frozen LLM and projector into the compressor (tested).

## Code

| path | role |
|---|---|
| `llava/model/visual_compressor/` | `dct.py`, `blocks.py`, `positional.py`, `frequency_encoder.py`, `query_pool.py`, `adapter.py`, `compressor.py` (all modes), `config.py`, `builder.py`, `diagnostics.py` |
| `llava/model/llava_arch.py` | `encode_images`: `compressor(features)` between tower and projector; compressor construction (**modified**) |
| `llava/model/builder.py` | loads `visual_compressor.bin` from a checkpoint dir (**modified**) |
| `llava/train/train_compressor.py` | training entry point (LLaVA dataset / preprocess / collator, HF Trainer) |
| `llava/train/compressor_kd.py` | KD aligned by answer-token ordinal |
| `llava/eval/afc/` | paths, model I/O, MMBench CircularEval + tuning-split scorers, efficiency |
| `scripts/afc/` | setup, data download / checks, debug, train, evaluation, ablations, plots |
| `configs/afc/` | `block_dct_64.yaml` + one-axis overrides in `ablations/` |
| `splits/afc/` | fixed tuning subsets (TextVQA train, GQA train; never evaluation splits) |
| `tests/afc/` | 50 CPU tests + 6 GPU tests on the real checkpoint |

Only two original LLaVA files are modified (`git diff upstream -- llava/model/builder.py llava/model/llava_arch.py`).

### Checkpoint format

`train_compressor` writes `config.json` (with a `visual_compressor` entry)
and `visual_compressor.bin`, and symlinks every other file of the base
model. The directory is a normal LLaVA model directory, so the official eval
scripts and lmms-eval load it unchanged. A directory without the
`visual_compressor` entry (e.g. `liuhaotian/llava-v1.5-7b`) is vanilla.

## Setup

```bash
bash scripts/afc/setup_env.sh                 # conda env `afc`
conda activate afc
export AFC_DATA=/path/to/afc_common/data      # default: <repo>/../afc_common/data
python scripts/afc/download_data.py           # benchmarks, tuning sources, weights
python scripts/afc/check_data.py
```

Versions: torch 2.1.2+cu121, transformers 4.37.2, lmms-eval v0.2.0,
accelerate 0.29.3, peft 0.10.0.

Training data: any LLaVA-format JSON (`--data_path`, `--image_folder`);
`download_llava_train.sh` fetches `llava_v1_5_mix665k.json` with COCO, GQA
and VG images (OCR-VQA is not downloaded; those samples are dropped).

## Commands

```bash
source scripts/afc/env.sh                     # every new shell: env + paths
bash scripts/afc/download_llava_train.sh      # LLaVA-1.5 665k data (~57 GB zips)

# shapes of the vanilla / compressed pipeline for one image
python scripts/afc/debug_llava_multimodal_pipeline.py --model-path $MODEL
python scripts/afc/debug_llava_multimodal_pipeline.py --model-path $MODEL --compression-mode block_dct

# unattended run: [profile ->] train -> quick eval -> full eval -> table (+W&B)
RUN_NAME=block75k EXTRA_CONFIG=configs/afc/full.yaml MAX_TRAIN_SAMPLES=75000 \
TRAIN_GPU=0 TRAIN_GPUS=0 EVAL_GPUS=0 TARGET_EFFECTIVE_BATCH=24 START_STAGE=train \
setsid nohup bash scripts/afc/pipeline_main.sh > /dev/null 2>&1 &
#   TRAIN_GPUS=0,1 -> DDP on two GPUs; RESUME=<checkpoint dir> resumes the train stage
#   ablation: EXTRA_CONFIG=<yaml with the full recipe + one changed key>

# watch
bash scripts/afc/live.sh block75k global75k   # live iter + latest losses
python scripts/afc/history.py block75k --eval # full train / validation history
tail -f outputs/pipeline/block75k/status      # stage events

# combined table once runs finish (Fourier-VLM paper rows, retention %)
setsid nohup bash scripts/afc/final_report.sh block75k global75k &
python -m llava.eval.afc.table outputs/eval/* --paper   # any time

# single pieces
GPU=0 DATA_PATH=... IMAGE_FOLDER=... bash scripts/afc/train.sh <name> configs/afc/full.yaml
GPU=0 LIMIT=500 bash scripts/afc/run_eval.sh <model dir>
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.afc.efficiency --model-path <dir> --n 50
python scripts/afc/visualize_attention.py --model-path <dir> --n 32

# tests
pytest tests/afc -m "not gpu"
LLAVA_MODEL_PATH=$MODEL PRISTINE_LLAVA_DIR=<unmodified LLaVA checkout> \
CUDA_VISIBLE_DEVICES=0 AFC_SECOND_GPU=1 pytest tests/afc
```

## Controlled comparison (same frozen-LLM recipe for every row)

Every method is trained with `configs/afc/full.yaml` (frozen CLIP and Vicuna,
trained projector, KD, same data / steps); only `compression_mode` differs.

| config | compressor | trainable |
|---|---|---|
| `fourier_full.yaml` | Fourier-VLM FFC: global DCT, keep 8x8, C-point iDCT x C/N | projector |
| `avgpool_full.yaml` | 8x8 average pooling (= the residual base) | projector |
| `global_full.yaml` | global DCT + 64 learned queries + residual | compressor + projector |
| `full.yaml` | block DCT (16 x 4 queries) + residual | compressor + projector |

Fourier-VLM's paper numbers come from retraining the LLM and are listed only
as reference rows (`table.py --paper`).

## Ablations (`configs/afc/ablations/`, each overrides one axis of the main config)

| id | files | question |
|---|---|---|
| A1 | `spatial_64` | DCT vs. spatial tokens (same blocks, encoder, pooling, adapter, parameter count) |
| A2 | `global_dct_64` | 24x24 global DCT + 64 queries vs. block DCT (16 x 4) |
| A3 | `no_freq_transformer` | frequency self-attention |
| A4 | `fixed_lowpass_64`, `spatial_average_64` | learned pooling vs. fixed 2x2 low-pass / 2x2 average (same projection + adapter) |
| A5 | `pos_none`, `pos_freq_only` (main = freq + block) | positional embeddings |
| A6 | `budget_32`, main (64), `budget_128`, `budget_144` | token budget |
| A7 | `ce_kd` | CE vs. CE + KD (T = 2, lambda = 1) |
| A8 | `block4_144`, `budget_144`, `block8_144` | block size at exactly 144 tokens (36x4, 16x9, 9x16) |
| A9 | `query_block_specific` | shared vs. block-specific queries |
| A10 | `adapter_linear` | linear vs. MLP adapter |
| — | `pool_no_prenorm` | pooling without Pre-LN |

Keep data subset (`max_train_samples`, `sample_seed`), seed, optimiser,
schedule, steps and precision identical across compared runs.

## Measured so far (pipeline checks, not results)

| check | value |
|---|---|
| vanilla `inputs_embeds` (official example) | (1, 634, 4096) = 35 + 576 + 23 |
| main compressor `inputs_embeds` | (1, 122, 4096) = 35 + 64 + 23 |
| vanilla logits vs. unmodified LLaVA | bitwise identical |
| train step, batch 4, bf16, grad. checkpointing (CE / CE+KD) | ~1.7 / 0.76 steps/s; peak 18.8 GB with KD |
| compressor latency (3090, fp16) | ~1.3 ms / image |

No benchmark numbers are reported until full training runs are done.

## Known limitations

* One process per GPU (no DDP); run ablations in parallel instead.
* Evaluation uses batch size 1 (as the official scripts and lmms-eval
  LLaVA wrapper do).
* Attention maps describe query-frequency affinity, not causal importance.
* The LLaVA-1.5 training mixture must be downloaded separately.
