"""Full TinyLLaVA SFT: GPU compute, CPU FP32 masters/gradients/Adafactor.

All model parameters are unfrozen; no LoRA or quantization is used. The legacy
SigLIP post_layernorm is unused by its hidden_states[-1] feature path, so its
two parameters legitimately receive no SFT gradient. See docs/full_training.md.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

import psutil
import torch
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor

from common import ROOT, now, sha256, write_json
from runtime import load, official_imports


def component(name):
    if ".vision_tower." in name:
        return "vision"
    if ".mm_projector." in name:
        return "projector"
    return "llm"


def unused_legacy_parameter(name):
    return ".vision_tower." in name and name.endswith((
        ".vision_model.post_layernorm.weight", ".vision_model.post_layernorm.bias",
    ))


class CPUOffloadedAdafactor:
    """Accumulate gradients on CPU, then update FP32 masters once per window.

    Post-accumulate hooks release GPU gradients during backward. Unlike stepping
    in backward, this preserves gradient accumulation and global overflow checks.
    Parameter values never change during checkpoint recomputation/backward.
    """

    def __init__(self, model, learning_rate, vision_learning_rate, weight_decay=0.0):
        self.parameters = dict(model.named_parameters())
        if not all(p.requires_grad for p in self.parameters.values()):
            raise ValueError("Full training requires every parameter to be unfrozen")
        self.masters = {
            n: torch.nn.Parameter(p.detach().to(device="cpu", dtype=torch.float32).clone())
            for n, p in self.parameters.items()
        }
        groups = []
        for label in ("vision", "projector", "llm"):
            values = [p for n, p in self.masters.items() if component(n) == label]
            if values:
                groups.append({"params": values, "name": label,
                               "lr": vision_learning_rate if label == "vision" else learning_rate})
        self.optimizer = Adafactor(
            groups, scale_parameter=False, relative_step=False, warmup_init=False,
            beta1=None, weight_decay=weight_decay,
        )
        self.handles = [p.register_post_accumulate_grad_hook(self._hook(n))
                        for n, p in self.parameters.items()]

    def _hook(self, name):
        def offload(parameter):
            gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
            master = self.masters[name]
            if master.grad is None:
                master.grad = gradient
            else:
                master.grad.add_(gradient)
            parameter.grad = None
        return offload

    @torch.no_grad()
    def step(self, loss_scale):
        missing = [n for n, p in self.masters.items()
                   if p.grad is None and not unused_legacy_parameter(n)]
        if missing:
            raise RuntimeError(f"Disconnected trainable parameters: {missing[:12]}")
        squared_norms = dict.fromkeys(("vision", "projector", "llm"), 0.0)
        for name, master in self.masters.items():
            if master.grad is None:
                continue
            if not bool(torch.isfinite(master.grad).all()):
                self.zero_grad()
                return False, {"overflow": True}
            master.grad.div_(loss_scale)
            squared_norms[component(name)] += float(master.grad.norm()) ** 2
        norms = {n: math.sqrt(v) for n, v in squared_norms.items()}
        for label in {component(n) for n in self.parameters}:
            if not math.isfinite(norms[label]) or norms[label] == 0:
                raise RuntimeError(f"Invalid or zero gradient norm in {label}: {norms[label]}")
        # Adafactor applies its own RMS update clipping; no extra global clipping.
        self.optimizer.step()
        self.copy_to_model()
        self.zero_grad()
        return True, {"grad_norms": norms}

    @torch.no_grad()
    def copy_to_model(self):
        for name, parameter in self.parameters.items():
            parameter.copy_(self.masters[name])

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return {"masters": {n: p.detach() for n, p in self.masters.items()},
                "optimizer": self.optimizer.state_dict()}

    @torch.no_grad()
    def load_state_dict(self, state):
        if set(state["masters"]) != set(self.masters):
            raise ValueError("Resume parameter names do not match the model")
        for name, value in state["masters"].items():
            self.masters[name].copy_(value)
        self.optimizer.load_state_dict(state["optimizer"])
        self.copy_to_model()


def configure_model(model, max_length):
    model.requires_grad_(True)
    model.config.use_cache = False
    model.config.tokenizer_model_max_length = max_length
    model.config.tokenizer_padding_side = "right"
    options = {"use_reentrant": False}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=options)
    model.get_vision_tower().vision_tower.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=options,
    )
    model.train()


def prepare_data(args, tokenizer, model, processor):
    official_imports()
    from tinyllava import conversation as conversation_lib
    from tinyllava.arguments import DataArguments
    from tinyllava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
    from tinyllava.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
    from tinyllava.data.process import preprocess, preprocess_multimodal

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    # Tokenize before filtering against the expanded (image + text) limit.
    tokenizer.model_max_length = model.config.max_position_embeddings
    conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
    data_args = DataArguments(
        data_path=str(args.data_path), image_folder=str(args.image_folder),
        is_multimodal=True, image_aspect_ratio=getattr(model.config, "image_aspect_ratio", "pad"),
    )
    data_args.image_processor = processor
    data_args.mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    dataset = LazySupervisedDataset(str(args.data_path), tokenizer, data_args)
    selected, samples, skipped = [], [], Counter()
    image_hashes = {}
    patches = model.get_vision_tower().num_patches
    for index, row in enumerate(dataset.list_data_dict):
        if not isinstance(row.get("image"), str) or not row["image"]:
            skipped["not_single_image"] += 1
            continue
        image = args.image_folder / row["image"]
        if not image.is_file():
            skipped["missing_image"] += 1
            continue
        sources = preprocess_multimodal(copy.deepcopy([row["conversations"]]), data_args)
        item = preprocess(sources, tokenizer, has_image=True)
        ids, labels = item["input_ids"][0], item["labels"][0]
        expanded = ids.numel() + patches - 1
        if int(ids.eq(IMAGE_TOKEN_INDEX).sum()) != 1:
            skipped["image_token_count"] += 1
            continue
        if expanded > args.max_length:
            skipped["too_long"] += 1
            continue
        if int(labels[1:].ne(IGNORE_INDEX).sum()) == 0:
            skipped["no_assistant_labels"] += 1
            continue
        if str(image) not in image_hashes:
            image_hashes[str(image)] = sha256(image)
        selected.append(row)
        samples.append({"index": index, "id": row.get("id"), "image": row["image"],
                        "image_sha256": image_hashes[str(image)], "expanded_tokens": expanded})
        if args.max_samples and len(selected) >= args.max_samples:
            break
    if not selected:
        raise RuntimeError(f"No usable single-image SFT records; skipped={dict(skipped)}")
    original_count = len(dataset.list_data_dict)
    dataset.list_data_dict = selected
    manifest = {"source": str(args.data_path), "source_sha256": sha256(args.data_path),
                "image_folder": str(args.image_folder), "original_records": original_count,
                "selected_records": len(selected), "skipped": dict(skipped),
                "max_length": args.max_length, "samples": samples}
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    # The accepted records already fit; collating cannot silently cut the image/answer.
    tokenizer.model_max_length = args.max_length
    return dataset, DataCollatorForSupervisedDataset(tokenizer), manifest


def probes(model):
    suffixes = {"vision": ".embeddings.patch_embedding.weight",
                "projector": "mm_projector.0.weight",
                "llm": "layers.0.self_attn.q_proj.weight"}
    return {label: (name, p.detach().flatten()[:4096].cpu().clone())
            for label, suffix in suffixes.items()
            for name, p in model.named_parameters() if name.endswith(suffix)}


def save_checkpoint(path, model, tokenizer, offload, cursor, signature, report):
    """Export both updated model components plus exact FP32 resume state."""
    temporary = path.with_name(path.name + ".incomplete")
    temporary.mkdir(parents=True, exist_ok=False)
    if path.exists():
        raise FileExistsError(path)
    saved_config = copy.deepcopy(model.config)
    try:
        model.config.mm_vision_tower = str(path / "vision_encoder-SigLIP")
        model.config.use_cache = True
        cpu_state = {n: t.detach().cpu() for n, t in model.state_dict().items()}
        model.save_pretrained(temporary, state_dict=cpu_state,
                              safe_serialization=True, max_shard_size="1GB")
        prefix = "model.vision_tower.vision_tower."
        vision_state = {n[len(prefix):]: t for n, t in cpu_state.items() if n.startswith(prefix)}
        if not vision_state:
            raise RuntimeError("No trained vision weights found for checkpoint export")
        model.get_vision_tower().vision_tower.save_pretrained(
            temporary / "vision_encoder-SigLIP", state_dict=vision_state, safe_serialization=True,
        )
        tokenizer.save_pretrained(temporary)
        del cpu_state, vision_state
        state = {"offload": offload.state_dict(), "cursor": cursor, "signature": signature,
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                 "python_rng": random.getstate()}
        torch.save(state, temporary / "training_state.pt")
        write_json(temporary / "training_report.json", report)
        temporary.rename(path)
    finally:
        model.config = saved_config


def run(args, report):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run train_full.sh in the GPU-enabled host terminal.")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    tokenizer, model, processor, _ = load("cuda", model_path=args.resume_from or args.model_path)
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("max-length exceeds model context length")
    configure_model(model, args.max_length)
    dataset, collator, manifest = prepare_data(args, tokenizer, model, processor)
    write_json(args.output / "training_manifest.json", manifest)
    counts = Counter()
    for name, parameter in model.named_parameters():
        counts[component(name)] += parameter.numel()
    # Masters + CPU accumulated gradients are 8 bytes/parameter. Adafactor's
    # factored state is small; reserve another 3 GiB for work buffers/runtime.
    required = sum(counts.values()) * 8 + 3 * 2**30
    if psutil.virtual_memory().available < required:
        raise RuntimeError(f"Need about {required / 2**30:.1f} GiB free CPU RAM for full training")
    report.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  mode="full_sft_cpu_fp32_adafactor_gpu_fp16", trainable_parameters=dict(counts),
                  inactive_legacy_parameters=[n for n, _ in model.named_parameters()
                                              if unused_legacy_parameter(n)],
                  selected_records=len(dataset), data_fingerprint=manifest["fingerprint"])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    offload = CPUOffloadedAdafactor(model, args.learning_rate, args.vision_learning_rate,
                                   args.weight_decay)
    signature = {"data": manifest["fingerprint"], "seed": args.seed,
                 "gradient_accumulation": args.gradient_accumulation,
                 "learning_rate": args.learning_rate, "vision_learning_rate": args.vision_learning_rate,
                 "weight_decay": args.weight_decay}
    cursor = {"epoch": 0, "batch_offset": 0, "global_step": 0,
              "attempted_updates": 0, "loss_scale": args.loss_scale}
    if args.resume_from:
        state = torch.load(args.resume_from / "training_state.pt", map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Resume data/seed/accumulation/optimizer settings differ from checkpoint")
        offload.load_state_dict(state["offload"])
        cursor = state["cursor"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        del state
    if args.max_steps and cursor["global_step"] >= args.max_steps:
        raise ValueError("max-steps must exceed the resumed global step")
    if cursor["epoch"] >= args.epochs:
        raise ValueError("epochs must exceed the resumed completed epoch count")
    initial_probes = probes(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(cursor["epoch"], args.epochs):
        order = list(range(len(dataset)))
        random.Random(args.seed + epoch).shuffle(order)
        offset = cursor["batch_offset"] if epoch == cursor["epoch"] else 0
        # A dedicated generator keeps DataLoader setup from changing model RNG on resume.
        loader = DataLoader(dataset, batch_size=1, sampler=order[offset:], collate_fn=collator,
                            num_workers=0, generator=torch.Generator().manual_seed(args.seed + epoch))
        iterator = iter(loader)
        while offset < len(order):
            window = min(args.gradient_accumulation, len(order) - offset)
            losses = []
            step_start = time.perf_counter()
            for _ in range(window):
                batch = {n: t.to("cuda", dtype=torch.float16 if n == "images" else t.dtype)
                         for n, t in next(iterator).items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(**batch, use_cache=False).loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite training loss")
                losses.append(float(loss.detach()))
                (loss * (cursor["loss_scale"] / window)).backward()
                del loss, batch
            updated, gradients = offload.step(cursor["loss_scale"])
            cursor["attempted_updates"] += 1
            if updated:
                cursor["global_step"] += 1
            else:
                cursor["loss_scale"] /= 2
                if cursor["loss_scale"] < 1 / 65536:
                    raise RuntimeError("Repeated FP16 gradient overflows")
            offset += window
            cursor.update(epoch=epoch + int(offset == len(order)),
                          batch_offset=0 if offset == len(order) else offset)
            torch.cuda.synchronize()
            record = {**cursor, "loss": sum(losses) / len(losses), "microbatches": window,
                      "updated": updated, **gradients, "seconds": time.perf_counter() - step_start,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
            print(json.dumps(record), flush=True)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            report.update(cursor=dict(cursor), last_update=record,
                          elapsed_seconds=time.perf_counter() - started,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            if updated:
                current = dict(model.named_parameters())
                report["updated_probe_elements"] = {
                    label: int(current[name].detach().flatten()[:4096].cpu().ne(before).sum())
                    for label, (name, before) in initial_probes.items()
                }
            write_json(args.output / "status.json", report)
            if updated and args.save_every and cursor["global_step"] % args.save_every == 0:
                save_checkpoint(args.output / f"checkpoint-{cursor['global_step']:06d}",
                                model, tokenizer, offload, dict(cursor), signature, report)
            if args.max_steps and cursor["global_step"] >= args.max_steps:
                break
        if args.max_steps and cursor["global_step"] >= args.max_steps:
            break
    if cursor["global_step"] == 0 or not all(report.get("updated_probe_elements", {}).values()):
        raise RuntimeError("No verified updates in all three model components")
    checkpoint = args.output / "final"
    save_checkpoint(checkpoint, model, tokenizer, offload, dict(cursor), signature, report)
    report.update(status="complete", checkpoint=str(checkpoint))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "model/TinyLLaVA-1.5B")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/train/annotations/llava_sft_coco1024.json")
    parser.add_argument("--image-folder", type=Path, default=ROOT / "data/train/llava")
    parser.add_argument("--output", type=Path, default=ROOT / "results/full_training" / time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = all epochs; counts successful optimizer updates")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = every eligible record")
    parser.add_argument("--max-length", type=int, default=1024, help="Includes all 729 image patches")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--vision-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-scale", type=float, default=128.0)
    parser.add_argument("--save-every", type=int, default=0, help="0 = final checkpoint only")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for key in ("epochs", "gradient_accumulation", "cpu_threads"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    for key in ("max_steps", "max_samples", "save_every"):
        if getattr(args, key) < 0:
            parser.error(f"{key} cannot be negative")
    for key in ("learning_rate", "vision_learning_rate", "loss_scale"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if args.max_length < 761:
        parser.error("max-length needs room for 729 image patches and text (>=761)")
    for key in ("model_path", "data_path", "image_folder", "output", "resume_from"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    if not args.data_path.is_file() or not args.image_folder.is_dir():
        parser.error("Training annotation or image folder is missing")
    if args.resume_from and not (args.resume_from / "training_state.pt").is_file():
        parser.error("resume-from must contain training_state.pt")
    # Do not create output inside an input model directory, or overwrite any run.
    for source in (args.model_path, args.resume_from):
        if source and (args.output == source or source in args.output.parents):
            parser.error("output must be separate from input model/checkpoint")
    return args


def main():
    args = parse_args()
    print(f"Output: {args.output}", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "arguments.json", {n: str(v) if isinstance(v, Path) else v
                                                for n, v in vars(args).items()})
    report = {"time": now(), "status": "running"}
    start = time.perf_counter()
    try:
        run(args, report)
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_at=now(), elapsed_seconds=time.perf_counter() - start)
        write_json(args.output / "status.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
