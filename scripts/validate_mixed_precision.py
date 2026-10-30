"""Independent calibration check of W-only, A-only and joint FP16 exceptions.

Uses 64 previously unused training/calibration examples. It performs forward
evaluation only. Vision, Projector, embedding and all quantized LLM layers are
recomputed for every configuration; no FP16 prefix is reused for mixed models.
"""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import ROOT, now, sha256, write_json
from runtime import load
from int4 import prepare_checkpoint
from mixed_precision import (DEFAULT_POLICY, read_policy, capture_fp16,
    enable_mixed_precision, mixed_execution_counts)
from analyze_llm_sensitivity import prepare_samples, cache_batches, score_batch, aggregate


CHECKPOINT = ROOT / "model/quantized/TinyLLaVA-1.5B-W4A16-all-RTN-g128"


def configure(model, max_length):
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    model.config.tokenizer_padding_side = "right"
    model.config.tokenizer_model_max_length = max_length


@torch.inference_mode()
def full_log_probs(model, batch):
    _, positions, mask, _, embeds, labels = model.prepare_inputs_labels_for_multimodal(
        batch["input_ids"].cuda(), None, batch["input_mask"].cuda(), None,
        batch["input_labels"].cuda(), batch["pixels"].cuda())
    if not torch.equal(labels.cpu(), batch["expanded_labels"]):
        raise RuntimeError("Mixed precision changed assistant target alignment")
    hidden = model.model(inputs_embeds=embeds, position_ids=positions, attention_mask=None,
                         use_cache=False, return_dict=True).last_hidden_state
    selected = hidden.reshape(-1, hidden.shape[-1]).index_select(0, batch["selected_rows"].cuda())
    logits = model.lm_head(selected).float()
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Non-finite mixed-precision validation logits")
    return F.log_softmax(logits, dim=-1)


@torch.inference_mode()
def evaluate(model, batches):
    rows, minimum = [], 0.0
    for index, batch in enumerate(batches):
        scored, local_minimum = score_batch(full_log_probs(model, batch), batch)
        rows.extend(scored)
        minimum = min(minimum, local_minimum)
        print(f"  validation batch {index+1}/{len(batches)}", flush=True)
    return {"metrics": aggregate(rows), "examples": rows, "minimum_unclamped_token_kl": minimum}


def add_pixels(batches, processor, config):
    from PIL import Image
    from tinyllava.mm_utils import process_images

    calibration = {row["id"]: row for row in json.loads((ROOT / "data/calibration/calibration.json").read_text())}
    for batch in batches:
        pixels = []
        for sample in batch["samples"]:
            with Image.open(ROOT / calibration[sample["id"]]["image"]) as image:
                pixels.append(process_images([image.convert("RGB")], processor, config).half())
        batch["pixels"] = torch.cat(pixels)
        # These FP16 caches are only needed for initial reference checks, not mixed evaluation.
        for key in ("features", "prefix_hidden", "head_input", "embeds"):
            batch.pop(key, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=32)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/mixed_precision/calibration")
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.target_tokens) < 1:
        parser.error("Samples, batch and target tokens must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    policy = read_policy(args.policy)
    prior_path = ROOT / "results/sensitivity_llm/run.json"
    prior = json.loads(prior_path.read_text())
    prior_ids = {s["id"] for s in prior["samples"]}
    args.max_length, args.seed = prior["max_length"], prior["seed"]
    args.calibration = ROOT / "data/calibration/calibration.json"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"time": now(), "status": "running", "finetuning_steps": 0})
    tokenizer, model, processor, _ = load()
    configure(model, args.max_length)
    requested = args.samples
    args.samples = len(prior_ids) + requested
    pool, skipped = prepare_samples(args, tokenizer, model)
    args.samples = requested
    if not prior_ids <= {s["id"] for s in pool}:
        raise RuntimeError("Expected the original sensitivity selection as a prefix of the larger pool")
    samples = [s for s in pool if s["id"] not in prior_ids]
    if len(samples) != requested:
        raise RuntimeError("Could not select the requested unused calibration examples")
    for sample in samples:
        sample["fold"] = "holdout"
    sample_manifest = [{k: s[k] for k in ("id", "image", "image_sha256", "source", "length_bin", "fold")} for s in samples]
    paths = [Path(__file__), ROOT / "scripts/mixed_precision.py", ROOT / "scripts/analyze_llm_sensitivity.py",
             ROOT / "scripts/int4.py", ROOT / "scripts/w4a4.py", ROOT / "scripts/int4_cuda.py",
             ROOT / "kernels/int4_gemm.cu"]
    spec = {"samples": sample_manifest, "excluded_sensitivity_ids": sorted(prior_ids),
            "prior_sensitivity_spec_sha256": sha256(prior_path), "policy_sha256": sha256(args.policy),
            "policy_rules": policy["rules"], "calibration_sha256": sha256(args.calibration),
            "seed": args.seed, "batch_size": args.batch_size, "max_length": args.max_length,
            "target_tokens": args.target_tokens, "finetuning_steps": 0,
            "source_hashes": {str(p): sha256(p) for p in paths}, "skipped": skipped,
            "arithmetic": "native S4 GEMM for remaining W4A4; explicit FP16 GEMM exceptions",
            "attention_mask_policy": "same causal right-padding diagnostic as sensitivity; scored valid assistant positions only"}
    if (output / "run.json").exists() and json.loads((output / "run.json").read_text()) != spec:
        raise RuntimeError("Validation conditions changed; choose a new output directory")
    write_json(output / "run.json", spec)
    batches, public_control = cache_batches(args, model, tokenizer, processor, samples)
    add_pixels(batches, processor, model.config)
    print("Independent FP16 reference control", flush=True)
    fp16 = evaluate(model, batches)
    if fp16["metrics"]["all"]["kl"] != 0 or fp16["metrics"]["all"]["nll_delta"] != 0:
        raise RuntimeError("Uncached public FP16 replay differs from the diagnostic reference")
    fp16.update(public_forward_control=public_control)
    write_json(output / "fp16.json", fp16)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    results = {"fp16": fp16}
    revisions = json.loads((CHECKPOINT / "manifest.json").read_text())["model_revisions"]
    for variant in ("none", "weights", "activations", "both"):
        destination = output / f"{variant}.json"
        if destination.exists():
            results[variant] = json.loads(destination.read_text())
            continue
        print(f"Mixed-precision holdout: {variant}", flush=True)
        _, model, _, _ = load()
        configure(model, args.max_length)
        originals = capture_fp16(model, policy, variant)
        base_audit = prepare_checkpoint(model, CHECKPOINT, 128, "triton", revisions)
        audit = enable_mixed_precision(model, policy, variant, originals)
        del originals
        start = time.perf_counter()
        result = evaluate(model, batches)
        result.update(variant=variant, quantization=audit, checkpoint_sha256=base_audit["checkpoint_sha256"],
                      execution=mixed_execution_counts(model), diagnostic_seconds=time.perf_counter()-start)
        native = result["execution"]["native"]
        if native["current_process_called_layers"] != audit["native_layer_count"]:
            raise RuntimeError("Some native layers were not executed on holdout")
        if not all(row["calls"] > 0 for row in result["execution"]["exceptions"].values()):
            raise RuntimeError("A mixed exception layer was not executed")
        write_json(destination, result)
        results[variant] = result
        print(f"{variant}: KL={result['metrics']['all']['kl']:.6g}, "
              f"NLL={result['metrics']['all']['nll']:.6g}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    ids = {s["id"] for s in samples}
    assert not ids & prior_ids
    for mode, result in results.items():
        if {row["id"] for row in result["examples"]} != ids:
            raise RuntimeError(f"Holdout IDs mismatch: {mode}")
    summary = {"time": now(), "status": "complete", "examples": len(samples),
               "overlap_with_sensitivity": 0, "finetuning_steps": 0,
               "source_counts": dict(Counter(s["source"] for s in samples)),
               "metrics": {mode: result["metrics"]["all"] for mode, result in results.items()}}
    write_json(output / "summary.json", summary)
    lines = ["# FP16 예외 기준 — 독립 calibration 검증", "",
             f"민감도 선정에 쓰지 않은 {len(samples)}개 예제에서 같은 정답 문맥·응답 위치를 비교했다.",
             "모든 quantized 모드의 Vision·Projector·embedding·LLM을 다시 계산했다. 학습은 없다.",
             "W4A4는 native INT4×INT4→INT32, 예외는 명시한 FP16 GEMM 경로다.",
             "이 값은 teacher-forced calibration 지표이며 전체 벤치마크 정확도가 아니다.", "",
             "| 설정 | FP16 대비 KL | 정답 토큰 NLL | FP16 대비 ΔNLL |",
             "| --- | ---: | ---: | ---: |"]
    for mode, m in summary["metrics"].items():
        lines.append(f"| {mode} | {m['kl']:.6f} | {m['nll']:.6f} | {m['nll_delta']:+.6f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "status.json", {"time": now(), "status": "complete", "finetuning_steps": 0})
    print(f"Verified independent holdout: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
