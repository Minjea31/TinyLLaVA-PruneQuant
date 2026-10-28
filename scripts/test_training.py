"""Offline CUDA smoke test: real image SFT, LLM LoRA + projector, frozen vision.

This checks optimizer updates, not full-model fine-tuning or MegaTrain support.
Original model weights and training annotations are never overwritten.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

from common import ROOT, now, sha256, write_json
from runtime import load
import torch


def prepare_batches(tokenizer, base, processor, output, count):
    from tinyllava import conversation as conversation_lib
    from tinyllava.arguments import DataArguments
    from tinyllava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
    from tinyllava.data.dataset import (
        DataCollatorForSupervisedDataset,
        LazySupervisedDataset,
    )

    source = ROOT / "data/train/annotations/llava_sft_coco1024.json"
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 1024
    base.config.tokenizer_model_max_length = 1024
    base.config.tokenizer_padding_side = "right"
    conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
    args = DataArguments(
        data_path=str(source),
        image_folder=str(ROOT / "data/train/llava"),
        is_multimodal=True,
        image_aspect_ratio=getattr(base.config, "image_aspect_ratio", "pad"),
    )
    args.image_processor = processor
    args.mm_use_im_start_end = getattr(base.config, "mm_use_im_start_end", False)
    dataset = LazySupervisedDataset(str(source), tokenizer, args)
    collator = DataCollatorForSupervisedDataset(tokenizer)
    batches, samples = [], []
    for index, row in enumerate(dataset.list_data_dict):
        image = Path(args.image_folder) / row.get("image", "")
        if not image.is_file():
            continue
        # Keep complete conversations and all 729 image patches; skip long rows.
        if sum(len(x["value"]) for x in row["conversations"]) > 600:
            continue
        item = dataset[index]
        tokens = item["input_ids"].numel()
        supervised = int(item["labels"].ne(IGNORE_INDEX).sum())
        if tokens + 728 > 1024 or supervised < 8:
            continue
        if int(item["input_ids"].eq(IMAGE_TOKEN_INDEX).sum()) != 1:
            continue
        batch = collator([item])
        batches.append({
            key: value.to("cuda", dtype=torch.float16 if key == "images" else value.dtype)
            for key, value in batch.items()
        })
        samples.append({
            "id": row["id"],
            "image": str(image.relative_to(ROOT)),
            "image_sha256": sha256(image),
            "text_tokens": tokens,
            "expanded_tokens": tokens + 728,
            "supervised_tokens": supervised,
            "conversations": row["conversations"],
        })
        if len(batches) == count:
            break
    if len(batches) != count:
        raise RuntimeError(f"Only {len(batches)} usable image/text training samples")
    write_json(output / "samples.json", samples)
    return batches, {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "count": count,
        "samples": samples,
    }


def run(args, report):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run outside the GPU-restricted sandbox")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    report.update(gpu=smi.stdout.strip(), torch=torch.__version__, cuda=torch.version.cuda)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    tokenizer, base, processor, _ = load("cuda")
    base.config.use_cache = False
    batches, report["data"] = prepare_batches(tokenizer, base, processor, args.output, 2)

    from peft import LoraConfig, TaskType, get_peft_model

    # Fully qualified names prevent accidentally adding LoRA to vision layers.
    targets = [
        name for name, _ in base.named_modules()
        if name.startswith("model.layers.") and name.endswith(("q_proj", "v_proj"))
    ]
    if len(targets) != 44:
        raise RuntimeError(f"Expected 44 LLM q/v projections, found {len(targets)}")
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        lora_dropout=0.0, bias="none", target_modules=targets,
    ))
    base.get_model().mm_projector.float().requires_grad_(True)
    # GradScaler requires FP32 trainable parameters; base weights stay FP16.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    base.get_vision_tower().eval()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    groups = {
        "llm_lora": {n: p for n, p in trainable.items() if "lora_" in n},
        "projector": {n: p for n, p in trainable.items() if ".mm_projector." in n},
    }
    if not all(groups.values()) or sum(map(len, groups.values())) != len(trainable):
        raise RuntimeError("Unexpected trainable parameter groups")
    if any(p.requires_grad for p in base.get_vision_tower().parameters()):
        raise RuntimeError("Vision tower must stay frozen in this test")
    before = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(init_scale=128.0)
    report.update(
        mode="fp16_base_fp32_lora_and_projector_frozen_vision",
        batch_size=1, sequence_limit=1024, image_patches=729,
        seed=2026, gradient_checkpointing=True, learning_rate=args.learning_rate,
        trainable_parameters={k: sum(p.numel() for p in v.values()) for k, v in groups.items()},
        full_model_finetuning_tested=False, megatrain_tested=False,
    )

    def evaluate():
        model.eval()
        values = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for batch in batches:
                values.append(float(model(**batch, use_cache=False).loss))
        if not all(math.isfinite(x) for x in values):
            raise RuntimeError("Non-finite evaluation loss")
        model.train()
        base.get_vision_tower().eval()
        return values

    torch.cuda.reset_peak_memory_stats()
    report["initial_losses"] = evaluate()
    print("Initial losses:", report["initial_losses"], flush=True)
    report["steps"] = []
    for step in range(args.steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**batches[step % len(batches)], use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {}
        for group, parameters in groups.items():
            if any(p.grad is None for p in parameters.values()):
                raise RuntimeError(f"Missing gradient in {group}")
            if any(not bool(p.grad.isfinite().all()) for p in parameters.values()):
                raise RuntimeError(f"Non-finite gradient in {group}")
            norm = math.sqrt(sum(float(p.grad.float().norm()) ** 2 for p in parameters.values()))
            if norm == 0:
                raise RuntimeError(f"Zero gradient in {group}")
            gradients[group] = norm
        torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            raise RuntimeError("GradScaler skipped the optimizer step")
        torch.cuda.synchronize()
        record = dict(step=step + 1, sample=step % len(batches), loss=float(loss.detach()),
                      grad_norms=gradients, seconds=time.perf_counter() - start)
        report["steps"].append(record)
        print(json.dumps(record), flush=True)
        write_json(args.output / "status.json", report)
        del loss
    report["final_losses"] = evaluate()
    report["loss_decreased_on_each_sample"] = all(
        end < start for start, end in zip(report["initial_losses"], report["final_losses"])
    )
    report["updated_tensors"] = {}
    for group, parameters in groups.items():
        changed = sum(not torch.equal(p.detach().cpu(), before[n]) for n, p in parameters.items())
        report["updated_tensors"][group] = dict(changed=changed, total=len(parameters))
        if not changed:
            raise RuntimeError(f"No parameter updates in {group}")
    if any(p.grad is not None for p in base.get_vision_tower().parameters()):
        raise RuntimeError("Frozen vision unexpectedly accumulated gradients")
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    # Save only the new adapter/projector, separate from the source checkpoint.
    checkpoint = args.output / "checkpoint"
    model.save_pretrained(checkpoint)
    torch.save({n: p.detach().cpu() for n, p in groups["projector"].items()},
               checkpoint / "non_lora_trainables.bin")
    tokenizer.save_pretrained(checkpoint)
    report["checkpoint"] = str(checkpoint)
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=ROOT / "results/training_smoke" / time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if args.steps < 2 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Need >=2 steps and a positive finite learning rate")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(time=now(), status="running", requested_steps=args.steps)
    start = time.perf_counter()
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        report["finished_at"] = now()
        write_json(args.output / "status.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
