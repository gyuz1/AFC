"""Real LLaVA-1.5-7B tests (GPU). Run with LLAVA_MODEL_PATH set.

Optional: PRISTINE_LLAVA_DIR = an unmodified LLaVA checkout, used to compare
the vanilla path bitwise against the original code in a subprocess.
"""

import os
import subprocess
import sys
from io import BytesIO

import pytest
import torch

from conftest import MODEL_PATH
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token

pytestmark = pytest.mark.gpu
HERE = os.path.dirname(os.path.abspath(__file__))
QUESTION = "What are the things I should be cautious about when I visit here?"
ANSWER = "Be careful on the wooden pier: it has no railings."


def _image():
    from PIL import Image
    path = os.environ.get("AFC_TEST_IMAGE")
    if path:
        return Image.open(path).convert("RGB")
    import requests
    url = "https://llava-vl.github.io/static/images/view.jpg"
    return Image.open(BytesIO(requests.get(url, timeout=30).content)
                      ).convert("RGB")


@pytest.fixture(scope="module")
def llava():
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    tok, model, proc, _ = load_pretrained_model(
        MODEL_PATH, None, get_model_name_from_path(MODEL_PATH),
        device_map="cuda:0")
    model.eval()
    return tok, model, proc


@pytest.fixture(scope="module")
def sample(llava):
    tok, model, proc = llava
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + QUESTION)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX,
                                return_tensors="pt").unsqueeze(0).cuda()
    image = _image()
    pixels = process_images([image], proc, model.config).cuda().half()
    return ids, pixels, image.size


def _logits(model, sample):
    ids, pixels, size = sample
    with torch.inference_mode():
        return model(input_ids=ids, images=pixels,
                     image_sizes=[size]).logits.float().cpu()


def _attach(model, **kw):
    from llava.model.visual_compressor import VisualCompressorConfig
    cfg = VisualCompressorConfig(compression_mode="block_dct", **kw)
    return model.get_model().initialize_visual_compressor(cfg)


def _detach(model):
    from llava.model.visual_compressor import VisualCompressorConfig
    model.get_model().initialize_visual_compressor(VisualCompressorConfig())


# --------------------------------------------------------------------------
def test_vanilla_model_has_no_compressor(llava):
    _, model, _ = llava
    assert model.get_model().get_visual_compressor() is None
    assert not any("visual_compressor" in k for k in model.state_dict())


def test_vanilla_matches_original_code_bitwise(llava, sample, tmp_path):
    """Original (unmodified) LLaVA in a subprocess vs. this fork."""
    pristine = os.environ.get("PRISTINE_LLAVA_DIR")
    if not pristine:
        pytest.skip("PRISTINE_LLAVA_DIR not set")
    ids, pixels, size = sample
    torch.save({"ids": ids.cpu(), "pixels": pixels.cpu(), "size": size},
               tmp_path / "in.pt")
    script = (
        "import sys, torch\n"
        "import llava; assert llava.__file__.startswith(sys.argv[3]), "
        "llava.__file__\n"
        "from llava.model.builder import load_pretrained_model\n"
        "from llava.mm_utils import get_model_name_from_path\n"
        "_, m, _, _ = load_pretrained_model(sys.argv[1], None, "
        "get_model_name_from_path(sys.argv[1]), device_map='cuda:0')\n"
        "d = torch.load(sys.argv[2] + '/in.pt')\n"
        "with torch.inference_mode():\n"
        "    out = m(input_ids=d['ids'].cuda(), images=d['pixels'].cuda(), "
        "image_sizes=[d['size']]).logits.float().cpu()\n"
        "torch.save(out, sys.argv[2] + '/out.pt')\n")
    env = dict(os.environ, PYTHONPATH=pristine,
               CUDA_VISIBLE_DEVICES=os.environ.get("AFC_SECOND_GPU", "1"))
    subprocess.run([sys.executable, "-c", script, MODEL_PATH, str(tmp_path),
                    os.path.abspath(pristine)], check=True, env=env,
                   cwd=tmp_path)  # cwd is sys.path[0] for -c
    original = torch.load(tmp_path / "out.pt")
    assert torch.equal(_logits(llava[1], sample), original)


def test_disabled_compressor_is_literal_bypass(llava, sample):
    from llava.model.visual_compressor.builder import compressor_disabled
    _, model, _ = llava
    ref = _logits(model, sample)
    _attach(model)
    try:
        with compressor_disabled(model):
            assert torch.equal(_logits(model, sample), ref)
    finally:
        _detach(model)
    assert torch.equal(_logits(model, sample), ref)


def test_real_forward_block_dct(llava, sample):
    _, model, _ = llava
    ids = sample[0]
    _attach(model)
    try:
        model.get_model().visual_compressor.eval()
        with torch.inference_mode():
            emb = model.encode_images(sample[1])
        logits = _logits(model, sample)
        again = _logits(model, sample)
    finally:
        _detach(model)
    assert emb.shape == (1, 64, 4096)
    assert logits.shape[1] == ids.shape[1] - 1 + 64
    assert torch.isfinite(logits).all()
    assert torch.equal(logits, again)                    # deterministic eval


def test_gradients_reach_compressor_through_frozen_llm(llava, sample):
    import llava.conversation as conversation_lib
    from llava.train.train import preprocess
    tok, model, _ = llava
    _, pixels, size = sample
    conversation_lib.default_conversation = conv_templates["v1"]
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    batch = preprocess([[{"from": "human",
                          "value": DEFAULT_IMAGE_TOKEN + "\n" + QUESTION},
                         {"from": "gpt", "value": ANSWER}]], tok,
                       has_image=True)
    model.requires_grad_(False)
    comp = _attach(model)
    comp.train()
    seen = {}
    comp.register_forward_hook(lambda m, i, o: o.retain_grad() or
                               seen.setdefault("out", o))
    try:
        out = model(input_ids=batch["input_ids"].cuda(),
                    labels=batch["labels"].cuda(), images=pixels,
                    image_sizes=[size])
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert seen["out"].grad is not None and seen["out"].grad.abs().sum() > 0
        params = dict(comp.named_parameters())
        for name in ("encoder.input_proj.weight", "encoder.inner_pos.emb_a",
                     "encoder.block_pos.emb_b",
                     "encoder.transformer.layers.1.linear2.weight",
                     "pool.queries", "pool.k_proj.weight", "pool.out_proj.weight",
                     "adapter.3.weight"):
            g = params[name].grad
            assert g is not None and g.abs().sum() > 0, name
        frozen_with_grad = [n for n, p in model.named_parameters()
                            if "visual_compressor" not in n and p.grad is not None]
        assert frozen_with_grad == []
    finally:
        _detach(model)


def test_checkpoint_roundtrip(llava, tmp_path):
    """Save a compressor dir, reload it with the official loader (GPU 1)."""
    from llava.train.train_compressor import (local_model_dir,
                                              save_compressor_checkpoint)
    _, model, _ = llava
    comp = _attach(model, target_visual_tokens=32)
    ckpt = tmp_path / "llava-v1.5-7b-afc-test"
    try:
        save_compressor_checkpoint(model, str(ckpt), local_model_dir(MODEL_PATH))
        ref = {k: v.detach().cpu() for k, v in comp.state_dict().items()}
        torch.save(ref, tmp_path / "ref.pt")
    finally:
        _detach(model)
    script = (
        "import sys, torch\n"
        "from llava.model.builder import load_pretrained_model\n"
        "from llava.mm_utils import get_model_name_from_path\n"
        "_, m, _, _ = load_pretrained_model(sys.argv[1], None, "
        "get_model_name_from_path(sys.argv[1]), device_map='cuda:0')\n"
        "c = m.get_model().get_visual_compressor()\n"
        "assert c is not None and c.cfg.target_visual_tokens == 32\n"
        "ref = torch.load(sys.argv[2])\n"
        "for k, v in c.state_dict().items():\n"
        "    assert torch.equal(v.cpu(), ref[k].to(v.dtype)), k\n"
        "x = torch.randn(1, 3, 336, 336, device='cuda', dtype=torch.float16)\n"
        "with torch.inference_mode():\n"
        "    assert m.encode_images(x).shape == (1, 32, 4096)\n"
        "print('roundtrip ok')\n")
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=os.environ.get("AFC_SECOND_GPU", "1"))
    res = subprocess.run([sys.executable, "-c", script, str(ckpt),
                          str(tmp_path / "ref.pt")], env=env,
                         capture_output=True, text=True)
    assert "roundtrip ok" in res.stdout, res.stderr[-3000:]


def test_objective_matches_hf_causal_lm_loss(llava, sample):
    """Answer-position-only CE == HF CE; KD path runs; grads reach compressor."""
    import llava.conversation as conversation_lib
    from llava.train.compressor_objective import CompressorObjective
    from llava.train.train import preprocess
    tok, model, _ = llava
    _, pixels, size = sample
    conversation_lib.default_conversation = conv_templates["v1"]
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    batch = preprocess([[{"from": "human",
                          "value": DEFAULT_IMAGE_TOKEN + "\n" + QUESTION},
                         {"from": "gpt", "value": ANSWER}]], tok,
                       has_image=True)
    ids, labels = batch["input_ids"].cuda(), batch["labels"].cuda()
    model.requires_grad_(False)
    # Same precision regime as the objective (autocast fp16) so the only
    # difference is answer-position-only vs. full-sequence logits.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        ref = model(input_ids=ids, labels=labels, images=pixels).loss.float()
    comp = _attach(model)
    try:
        with torch.no_grad(), \
                __import__("llava.model.visual_compressor.builder",
                           fromlist=["x"]).compressor_disabled(model):
            out = CompressorObjective(model)(ids, labels, images=pixels)
        # fp16 GEMMs of different shapes (all positions vs. answer rows)
        # round differently (max |logit diff| ~4e-3); relative loss error
        # ~1.5e-4 measured.
        assert torch.allclose(out["ce"].float(), ref, rtol=5e-4, atol=0), \
            (out["ce"], ref)
        obj = CompressorObjective(model, kd_lambda=1.0)
        res = obj(ids, labels, images=pixels)
        assert torch.isfinite(res["loss"]) and res["kd"] >= 0
        res["loss"].backward()
        assert comp.pool.queries.grad is not None
        assert comp.pool.queries.grad.abs().sum() > 0
    finally:
        _detach(model)
