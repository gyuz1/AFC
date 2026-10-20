"""Print the tensor shapes of the LLaVA-1.5 multimodal pipeline for one sample.

Vanilla:
    python scripts/afc/debug_llava_multimodal_pipeline.py \
        --model-path liuhaotian/llava-v1.5-7b
A trained compressor checkpoint directory (or a fresh, untrained compressor):
    python scripts/afc/debug_llava_multimodal_pipeline.py \
        --model-path <ckpt dir>
    python scripts/afc/debug_llava_multimodal_pipeline.py \
        --model-path liuhaotian/llava-v1.5-7b --compression-mode block_dct

Sequence lengths are reported, never assumed: the counts depend on the
tokenizer and conversation template.
"""

import argparse
from io import BytesIO

import requests
import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model

DEFAULT_PROMPT = ("What are the things I should be cautious about when I "
                  "visit here?")
DEFAULT_IMAGE = "https://llava-vl.github.io/static/images/view.jpg"


def load_image(src: str) -> Image.Image:
    if src.startswith("http"):
        return Image.open(BytesIO(requests.get(src, timeout=30).content)
                          ).convert("RGB")
    return Image.open(src).convert("RGB")


def shape(t) -> tuple:
    return tuple(t.shape)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compression-mode", default=None,
                        help="attach a fresh (untrained) compressor of this "
                             "mode to inspect shapes only")
    parser.add_argument("--target-visual-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=6)
    args = parser.parse_args()

    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path),
        device_map=args.device)
    model.eval()
    if args.compression_mode:
        from llava.model.visual_compressor import VisualCompressorConfig
        cfg = VisualCompressorConfig(
            compression_mode=args.compression_mode,
            block_size=args.block_size,
            target_visual_tokens=args.target_visual_tokens)
        model.get_model().initialize_visual_compressor(cfg)
        print("[untrained compressor attached: shapes only]")

    cfg = model.config
    vt = model.get_vision_tower()
    compressor = model.get_model().get_visual_compressor()
    print("== model")
    print(f"vision tower   : {vt.vision_tower_name} | select_layer "
          f"{vt.select_layer} | select_feature {vt.select_feature}")
    print(f"CLIP           : hidden {vt.config.hidden_size} | grid "
          f"{vt.num_patches_per_side}x{vt.num_patches_per_side} = "
          f"{vt.num_patches} patches")
    print(f"projector      : {cfg.mm_projector_type} "
          f"{cfg.mm_hidden_size} -> {cfg.hidden_size}")
    print(f"LLM            : {cfg.num_hidden_layers} layers | hidden "
          f"{cfg.hidden_size} | vocab {cfg.vocab_size}")
    print("compressor     :", "none (vanilla path)" if compressor is None
          else compressor.describe())

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                return_tensors="pt").unsqueeze(0)
    ids = ids.to(model.device)
    img_pos = (ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten().tolist()
    print("\n== text")
    print(prompt)
    print("input_ids      :", shape(ids))
    print("IMAGE_TOKEN_INDEX", IMAGE_TOKEN_INDEX, "at", img_pos)
    for i, t in enumerate(ids[0].tolist()):
        tok = "<image>" if t == IMAGE_TOKEN_INDEX else \
            tokenizer.convert_ids_to_tokens(t)
        print(f"  {i:3d} {t:6d} {tok!r}")

    image = load_image(args.image)
    pixels = process_images([image], processor, model.config)
    pixels = pixels.to(model.device, dtype=model.dtype)
    print("\n== vision")
    print("image (W, H)   :", image.size, "| aspect:",
          cfg.image_aspect_ratio)
    print("pixel_values   :", shape(pixels), pixels.dtype)
    clip_out = vt(pixels)
    print("CLIP features  :", shape(clip_out), "(CLS removed)")
    feats = clip_out
    if compressor is not None:
        feats = compressor(clip_out, return_attention=True)
        print("compressed     :", shape(feats))
        for name, s in compressor.last_shapes.items():
            print(f"  {name:<22}{s}")
    proj = model.get_model().mm_projector(feats)
    print("projector in   :", shape(feats), "-> out", shape(proj))
    assert torch.equal(proj, model.encode_images(pixels))

    out = model.prepare_inputs_labels_for_multimodal(
        ids, None, None, None, None, pixels, image_sizes=[image.size])
    embeds = out[4]
    n_txt = ids.shape[1] - len(img_pos)
    print("\n== multimodal sequence")
    print(f"inputs_embeds  : {shape(embeds)} = {n_txt} text + "
          f"{proj.shape[1]} visual")
    if len(img_pos) == 1:
        p = img_pos[0]
        print(f"  layout       : {p} text | {proj.shape[1]} visual | "
              f"{ids.shape[1] - p - 1} text")
    logits = model(input_ids=ids, images=pixels,
                   image_sizes=[image.size]).logits
    print("logits         :", shape(logits))


if __name__ == "__main__":
    main()
