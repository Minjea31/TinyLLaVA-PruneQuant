"""Inspect safetensors headers/parameter counts without loading multi-GB tensors."""

import argparse
from collections import Counter
import json
import math
import pickletools
import struct
import zipfile
from common import ROOT, make_dirs, now, write_json


def compare_vision():
    from contextlib import ExitStack
    from safetensors import safe_open
    import torch

    model = ROOT / "model/TinyLLaVA-1.5B"
    vision = ROOT / "model/TinyLLaVA-1.5B-SigLIP"
    main_index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    vi = vision / "model.safetensors.index.json"
    if vi.exists():
        vision_index = json.loads(vi.read_text())["weight_map"]
    else:
        filename = next(vision.glob("*.safetensors")).name
        with safe_open(vision / filename, framework="pt", device="cpu") as f:
            vision_index = {k: filename for k in f.keys()}
    prefix = "model.vision_tower.vision_tower."
    corresponding = {
        k[len(prefix) :]: v for k, v in main_index.items() if k.startswith(prefix)
    }
    if set(corresponding) != set(vision_index):
        raise RuntimeError("Vision tensor keys differ between snapshots")
    with ExitStack() as stack:
        full = {
            n: stack.enter_context(safe_open(model / n, framework="pt", device="cpu"))
            for n in sorted(set(corresponding.values()))
        }
        external = {
            n: stack.enter_context(safe_open(vision / n, framework="pt", device="cpu"))
            for n in sorted(set(vision_index.values()))
        }
        mismatched = []
        for key in sorted(corresponding):
            a = full[corresponding[key]].get_tensor(prefix + key)
            b = external[vision_index[key]].get_tensor(key)
            if not torch.equal(a, b):
                mismatched.append(key)
    report = dict(
        time=now(),
        compared_tensors=len(corresponding),
        identical=not mismatched,
        mismatched=mismatched,
    )
    write_json(ROOT / "docs/vision_weight_comparison.json", report)
    print("Main vs separate vision:", report)


def inspect(path):
    config = json.loads((path / "config.json").read_text())
    index = path / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else [p.name for p in path.glob("*.safetensors")]
    )
    if not files:
        raise RuntimeError(f"No completed safetensors files in {path}")
    groups = Counter()
    dtypes = Counter()
    tensor_shapes = {}
    keys = {}
    tensor_bytes = 0
    for name in files:
        p = path / name
        with p.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("Truncated header")
            size = struct.unpack("<Q", raw)[0]
            if size > 100_000_000:
                raise ValueError("Oversized header")
            header = json.loads(f.read(size))
        offsets = []
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in keys:
                raise ValueError(f"Duplicate tensor key: {key}")
            a, b = value["data_offsets"]
            offsets.append((a, b))
            if b + size + 8 > p.stat().st_size:
                raise ValueError("Tensor extends past EOF")
            count = math.prod(value["shape"])
            tensor_bytes += b - a
            group = (
                "vision"
                if "vision_tower" in key or key.startswith("vision_model")
                else ("projector" if "mm_projector" in key else "llm")
            )
            groups[group] += count
            dtypes[value["dtype"]] += count
            keys[key] = name
            tensor_shapes[key] = value
        offsets.sort()
        if offsets and (
            offsets[0][0] != 0
            or any(x[1] != y[0] for x, y in zip(offsets, offsets[1:]))
            or offsets[-1][1] + size + 8 != p.stat().st_size
        ):
            raise ValueError("Non-contiguous tensor offsets or unexpected file size")
    if index.exists() and keys != json.loads(index.read_text())["weight_map"]:
        raise ValueError("Index/weight key mismatch")
    report = dict(
        time=now(),
        path=str(path.relative_to(ROOT)),
        config=config,
        parameters=dict(groups),
        total_parameters=sum(groups.values()),
        dtypes=dict(dtypes),
        tensor_bytes=tensor_bytes,
        tensors=len(keys),
        files=files,
        header_index_validation="passed",
        tensor_shapes=tensor_shapes,
    )
    args = path / "training_args.bin"
    if args.exists():
        # Inspect pickle opcodes only; never execute/unpickle remote TrainingArguments.
        with zipfile.ZipFile(args) as z:
            member = next(x for x in z.namelist() if x.endswith("/data.pkl"))
            strings = [
                arg
                for op, arg, pos in pickletools.genops(z.read(member))
                if isinstance(arg, str)
            ]
        report["training_run_identifiers"] = [
            s
            for s in strings
            if "tiny-llava-tinyllama-sharegpt4v" in s and "/runs/" not in s
        ]
    state = path / "trainer_state.json"
    if state.exists():
        state = json.loads(state.read_text())
        report["trainer_state"] = {k: v for k, v in state.items() if k != "log_history"}
    write_json(ROOT / "docs" / (path.name + "_analysis.json"), report)
    print(
        path.name,
        dict(groups),
        "total",
        report["total_parameters"],
        "dtype",
        dict(dtypes),
    )
    return report


if __name__ == "__main__":
    make_dirs()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    args = p.parse_args()
    for name in (
        ["TinyLLaVA-1.5B", "TinyLLaVA-1.5B-SigLIP"]
        if args.model == "all"
        else [args.model]
    ):
        inspect(ROOT / "model" / name)
    if args.model == "all":
        compare_vision()
