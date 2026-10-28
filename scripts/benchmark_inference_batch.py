"""Measure batch memory/throughput separately from the fixed-batch accuracy runs."""

import argparse
import json
import statistics
import subprocess
import time

import torch

from common import ROOT, now, write_json
from runtime import inputs, load
from test_inference import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--precision", choices=["fp16", "int4-all"], default="int4-all")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.samples < 1 or any(size < 1 for size in args.batch_sizes):
        parser.error("Samples and batch sizes must be positive")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    tokenizer, model, processor, context = load("cuda")
    revisions = {name: json.loads((ROOT / "docs" / f"{name}_snapshot.json").read_text())["revision"]
                 for name in ("TinyLLaVA-1.5B", "TinyLLaVA-1.5B-SigLIP")}
    audit = None
    if args.precision == "int4-all":
        from int4 import prepare_checkpoint
        checkpoint = ROOT / "model/quantized/TinyLLaVA-1.5B-W4A16-all-RTN-g128"
        audit = prepare_checkpoint(model, checkpoint, 128, "triton", revisions)
    # The decoder reads the last prompt position; left padding is required.
    # An explicit mask retains padding after multimodal expansion. Generation
    # derives positions from that mask again at each decoding step.
    model.config.tokenizer_padding_side = "left"
    source = ROOT / "data/eval/pope/questions.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()][:args.samples]
    prepared = [inputs(tokenizer, model, processor, ROOT / "data/eval/pope/val2014" / row["image"],
                       row["text"], device="cpu")[:2] for row in rows]
    if any(ids.shape[-1] + 728 + 32 > context for ids, _ in prepared):
        raise RuntimeError("Sample exceeds fixed context")
    reference_root = "results/baseline" if args.precision == "fp16" else "results/int4_all_w4a16"
    reference_path = ROOT / reference_root / "pope/predictions.jsonl"
    old = ({row["question_id"]: row for row in map(json.loads, reference_path.read_text().splitlines())}
           if reference_path.exists() else {})
    reference_conditions = ROOT / reference_root / "pope/run.json"
    reference_batch_size = (json.loads(reference_conditions.read_text()).get("batch_size", 1)
                            if reference_conditions.exists() else None)
    report = {"time": now(), "benchmark": "pope_first_samples", "samples": len(rows),
              "precision": args.precision, "model_revisions": revisions,
              "max_new_tokens": 32, "do_sample": False, "padding_side": "left", "runs": {}}
    if audit:
        report["checkpoint_sha256"] = audit["checkpoint_sha256"]
    output_path = ROOT / (args.output or f"results/batch_probe/{args.precision}_pope.json")
    for size in args.batch_sizes:
        torch.cuda.empty_cache()
        durations, allocated, reserved, smi_memory, predictions = [], [], [], [], []
        for start in range(0, len(rows), size):
            chunk = prepared[start:start + size]
            length = max(ids.shape[-1] for ids, _ in chunk)
            ids = torch.full((len(chunk), length), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for i, (sample_ids, _) in enumerate(chunk):
                ids[i, -sample_ids.shape[-1]:] = sample_ids[0]
                mask[i, -sample_ids.shape[-1]:] = True
            ids, mask = (tensor.to("cuda") for tensor in (ids, mask))
            images = torch.cat([image for _, image in chunk]).to("cuda")
            kwargs = dict(images=images, attention_mask=mask,
                          do_sample=False, num_beams=1, max_new_tokens=32, use_cache=True,
                          pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            with torch.inference_mode():
                if start == 0:
                    model.generate(ids, **kwargs)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
                output = model.generate(ids, **kwargs)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - begin)
            allocated.append(torch.cuda.max_memory_allocated())
            reserved.append(torch.cuda.max_memory_reserved())
            observed = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                                "--format=csv,noheader,nounits"], text=True)
            smi_memory.append(int(observed.strip()))
            for row, answer in zip(rows[start:start + size], tokenizer.batch_decode(output, skip_special_tokens=True)):
                predictions.append(dict(row, text=answer.strip()))
            del ids, mask, images, kwargs, output
        reference = [old[row["question_id"]] for row in rows] if all(row["question_id"] in old for row in rows) else None
        item = dict(batch_size=size, sample_count=len(rows), seconds=sum(durations),
                    samples_per_second=len(rows) / sum(durations),
                    mean_batch_seconds=statistics.mean(durations),
                    peak_allocated_gib=max(allocated) / 2**30, peak_reserved_gib=max(reserved) / 2**30,
                    observed_device_memory_mib=max(smi_memory),
                    reference_batch_size=reference_batch_size,
                    exact_saved_reference_answers=(sum(a["text"] == b["text"] for a, b in zip(predictions, reference))
                                                   if reference is not None else None),
                    metrics=score(predictions, "pope"),
                    reference_metrics=score(reference, "pope") if reference is not None else None,
                    predictions=predictions)
        report["runs"][str(size)] = item
        write_json(output_path, report)
        print(json.dumps({k: v for k, v in item.items() if k not in ("predictions", "metrics", "reference_metrics")} ), flush=True)


if __name__ == "__main__":
    main()
