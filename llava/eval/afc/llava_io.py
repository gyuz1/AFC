"""LLaVA-1.5 loading and single-sample input construction (batch size 1)."""

import torch
from PIL import Image

from llava.eval.afc import MODEL_PATH

CONV_MODE = "vicuna_v1"


def load_model(model_path=MODEL_PATH, device: str = "cuda:0"):
    """Returns ``(tokenizer, model, image_processor)``; model in eval mode.

    ``device`` must be a string: LLaVA's builder also passes it to the
    vision tower's ``.to()``.
    """
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    tok, model, proc, _ = load_pretrained_model(
        str(model_path), None, get_model_name_from_path(str(model_path)),
        device_map=device)
    return tok, model.eval(), proc


def build_prompt(question: str, answer: str = None) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], answer)
    return conv.get_prompt()


def build_inputs(tok, model, proc, image, question: str):
    """``(input_ids (1, T), pixel_values (1, 3, 336, 336) fp16)`` on device.

    ``image`` is a PIL image or a path.
    """
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    ids = tokenizer_image_token(build_prompt(question), tok,
                                IMAGE_TOKEN_INDEX, return_tensors="pt")
    pixels = process_images([image], proc, model.config)
    return (ids.unsqueeze(0).to(model.device),
            pixels.to(model.device, dtype=torch.float16))
