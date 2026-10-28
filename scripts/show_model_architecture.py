"""Print the local legacy architecture using metadata tensors, without loading weights."""

import argparse
from pathlib import Path

import torch

from common import ROOT
from runtime import official_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    official_imports()
    from tinyllava.model.language_model.llava_llama import TinyLlavaConfig, TinyLlavaLlamaForCausalLM
    from tinyllava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionModel

    config = TinyLlavaConfig.from_pretrained(ROOT / "model/TinyLLaVA-1.5B", local_files_only=True)
    config.mm_vision_tower = str(ROOT / "model/TinyLLaVA-1.5B-SigLIP")
    # Use the same automatic attention selection as the FP16 checkpoint loader.
    config = TinyLlavaLlamaForCausalLM._autoset_attn_implementation(config, torch_dtype=torch.float16)
    vision_config = SigLipVisionConfig.from_pretrained(config.mm_vision_tower, local_files_only=True)
    with torch.device("meta"):
        model = TinyLlavaLlamaForCausalLM(config)
        tower = model.get_vision_tower()
        tower.vision_tower = SigLipVisionModel(vision_config)
        tower.is_loaded = True
    if any(parameter.device.type != "meta" for parameter in model.parameters()):
        raise RuntimeError("Architecture inspection must not allocate real weight tensors")
    vision_layers = len(tower.vision_tower.vision_model.encoder.layers)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = (f"Local TinyLLaVA-1.5B architecture (meta tensors; no weights/GPU allocation)\n"
               f"Vision: config={vision_config.num_hidden_layers}, actual={vision_layers} encoder blocks\n"
               f"LLM: {len(model.model.layers)} decoder blocks; attention={type(model.model.layers[0].self_attn).__name__}\n"
               f"Parameters: {parameters:,}\n")
    output = summary + "\n" + str(model) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(summary + f"\nSaved: {args.output}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
