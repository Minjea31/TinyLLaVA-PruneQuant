"""GPU integration check: saved FP32 masters == all reloaded FP16 parameters."""

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import write_json
from runtime import load
from train_full import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    torch.set_num_threads(4)
    tokenizer, model, processor, _ = load("cuda", model_path=checkpoint)
    state = torch.load(str(checkpoint / "training_state.pt"), map_location="cpu",
                       weights_only=False, mmap=True)
    masters = state["offload"]["masters"]
    parameters = dict(model.named_parameters())
    if set(masters) != set(parameters):
        raise RuntimeError("Reloaded parameter names differ from saved training state")
    for name, parameter in parameters.items():
        if not torch.equal(parameter.detach().cpu(), masters[name].to(parameter.dtype)):
            raise RuntimeError(f"Checkpoint reload changed parameter {name}")
    result = {"checkpoint": str(checkpoint), "matched_parameter_tensors": len(parameters),
              "matched_parameters": sum(p.numel() for p in parameters.values()),
              "cursor": state["cursor"], "vision_path": model.config.mm_vision_tower}
    del state, masters
    settings = json.loads((checkpoint.parent / "arguments.json").read_text())
    data_args = SimpleNamespace(
        data_path=Path(settings["data_path"]), image_folder=Path(settings["image_folder"]),
        max_length=settings["max_length"], max_samples=settings["max_samples"],
    )
    dataset, collator, _ = prepare_data(data_args, tokenizer, model, processor)
    batch = {n: t.to("cuda", dtype=torch.float16 if n == "images" else t.dtype)
             for n, t in collator([dataset[0]]).items()}
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        result["reloaded_image_loss"] = float(model(**batch, use_cache=False).loss)
    if not math.isfinite(result["reloaded_image_loss"]):
        raise RuntimeError("Reloaded image forward produced a non-finite loss")
    result["status"] = "passed"
    write_json(checkpoint.parent / "reload_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
