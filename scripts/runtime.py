"""Offline wrapper around the pinned official legacy loader, without weight conversion."""

import os
import sys
from pathlib import Path
from common import ROOT


def official_imports():
    sys.path.insert(0, str(ROOT / "third_party/TinyLLaVA_Factory_legacy"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load(device="cuda", model_path=None, vision_path=None):
    official_imports()
    from tinyllava.model.language_model.llava_llama import TinyLlavaConfig
    from tinyllava.model.builder import load_pretrained_model

    path = Path(model_path or ROOT / "model/TinyLLaVA-1.5B").resolve()
    cfg = TinyLlavaConfig.from_pretrained(path, local_files_only=True)
    # In-memory path mapping only. Original config and original weights stay intact.
    # Full fine-tuning exports an updated vision encoder beside the main weights.
    # Prefer it even if the checkpoint was moved to a different directory.
    # The legacy builder dispatches by the substring "siglip" in the path.
    saved_vision = path / "vision_encoder-SigLIP"
    cfg.mm_vision_tower = str(Path(vision_path).resolve() if vision_path else (
        saved_vision if saved_vision.is_dir()
        else ROOT / "model/TinyLLaVA-1.5B-SigLIP"
    ))
    tokenizer, model, processor, context = load_pretrained_model(
        str(path),
        None,
        "TinyLLaVA-1.5B",
        config=cfg,
        device=device,
        device_map={"": device},
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model, processor, context


def inputs(tokenizer, model, processor, image_path, question, device="cuda"):
    from PIL import Image
    from tinyllava.conversation import conv_templates
    from tinyllava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from tinyllava.mm_utils import tokenizer_image_token, process_images
    import torch

    conv = conv_templates["v1"].copy()
    conv.append_message(
        conv.roles[0], (DEFAULT_IMAGE_TOKEN + "\n" if image_path else "") + question
    )
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )
    images = None
    if image_path:
        with Image.open(image_path) as im:
            images = process_images([im.convert("RGB")], processor, model.config).to(
                device=device, dtype=torch.float16
            )
    return ids, images, prompt


def batch_inputs(tokenizer, model, processor, image_paths, questions, device="cuda"):
    """Left-pad original prompts; keep the multimodal attention mask explicit."""
    import torch

    if not questions or len(image_paths) != len(questions):
        raise ValueError("A batch needs matching nonempty image/question lists")
    prepared = [inputs(tokenizer, model, processor, path, question, device="cpu")
                for path, question in zip(image_paths, questions)]
    lengths = [ids.shape[-1] for ids, _, _ in prepared]
    ids = torch.full((len(prepared), max(lengths)), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, (sample_ids, _, _) in enumerate(prepared):
        ids[i, -lengths[i]:] = sample_ids[0]
        mask[i, -lengths[i]:] = True
    images = [image for _, image, _ in prepared]
    if any(image is None for image in images) and not all(image is None for image in images):
        raise ValueError("A batch must contain either all image prompts or all text prompts")
    pixels = torch.cat(images).to(device) if images[0] is not None else None
    expanded_lengths = [length + (728 if path else 0) for length, path in zip(lengths, image_paths)]
    return ids.to(device), pixels, mask.to(device), expanded_lengths
