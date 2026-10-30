"""Official greedy inference: FP16, W4A16, or native INT4 Linear/Conv W4A4."""

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import time
from common import ROOT, make_dirs, now, sha256, write_json
import torch
from runtime import load, inputs, batch_inputs


def score(rows, benchmark):
    if benchmark == "scienceqa":
        correct = 0
        invalid = 0
        for r in rows:
            match = re.match(r"^\s*\(?([A-E])\)?(?:[.\s:]|$)", r["text"])
            invalid += match is None
            correct += bool(match and match.group(1) == r["answer"])
        return dict(
            accuracy=correct / len(rows),
            correct=correct,
            count=len(rows),
            invalid=invalid,
        )
    if benchmark == "pope":
        result = {}
        for category in sorted({x["category"] for x in rows}):
            subset = [x for x in rows if x["category"] == category]
            tp = tn = fp = fn = 0
            for r in subset:
                # Match the official TinyLLaVA POPE evaluator's first-sentence rule.
                words = r["text"].split(".")[0].replace(",", "").split(" ")
                pred = not any(x in words for x in ["No", "not", "no"])
                label = r["label"] == "yes"
                tp += pred and label
                fp += pred and not label
                tn += not pred and not label
                fn += not pred and label
            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / (tp + fn) if tp + fn else 0
            result[category] = dict(
                count=len(subset),
                accuracy=(tp + tn) / len(subset),
                precision=precision,
                recall=recall,
                f1=2 * precision * recall / (precision + recall)
                if precision + recall
                else 0,
                yes_ratio=(tp + fp) / len(subset),
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
            )
        return result
    return None


def main():
    make_dirs()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--validate-load",
        action="store_true",
        help="CPU weight/tokenizer/processor validation; no FP16 CPU forward",
    )
    p.add_argument(
        "--benchmark", choices=["smoke", "pope", "scienceqa"], default="smoke"
    )
    p.add_argument("--image", type=Path)
    p.add_argument("--question", default="Describe the image briefly.")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = full fixed benchmark; nonzero is a smoke subset",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", type=Path)
    p.add_argument("--precision", choices=["fp16", "int4-all", "int4-w4a4", "int4-mixed"], default="fp16")
    p.add_argument("--mixed-variant", choices=["weights", "activations", "both"], default="both")
    p.add_argument("--mixed-policy", type=Path,
                   default=ROOT / "results/sensitivity_llm/fp16_threshold_candidates.json")
    p.add_argument("--quant-group-size", type=int, default=128)
    p.add_argument("--quant-backend", choices=["triton", "torch"], default="triton")
    p.add_argument("--quant-checkpoint", type=Path,
                   default=ROOT / "model/quantized/TinyLLaVA-1.5B-W4A16-all-RTN-g128")
    p.add_argument("--reuse-identical-inputs", action="store_true",
                   help="Greedy evaluation: reuse predictions for identical image/question inputs")
    args = p.parse_args()
    if args.quant_group_size < 1 or args.limit < 0 or args.max_new_tokens < 1 or args.warmup < 0 or args.batch_size < 1:
        p.error("Invalid group size, limit, generation length or warmup")
    if args.validate_load and args.precision != "fp16":
        p.error("--validate-load is the original FP16 CPU load check")
    result_root = {"fp16": "results/baseline", "int4-all": "results/int4_all_w4a16",
                   "int4-w4a4": "results/int4_all_w4a4",
                   "int4-mixed": f"results/mixed_precision/{args.mixed_variant}"}[args.precision]
    out = args.output or ROOT / result_root / args.benchmark
    out.mkdir(parents=True, exist_ok=True)
    if args.validate_load:
        if args.image is None:
            calibration = ROOT / "data/calibration/calibration.json"
            if calibration.exists():
                args.image = ROOT / json.loads(calibration.read_text())[0]["image"]
        tokenizer, model, processor, context = load("cpu")
        report = dict(
            time=now(),
            status="cpu_load_passed",
            parameters=sum(x.numel() for x in model.parameters()),
            dtype=str(next(model.parameters()).dtype),
            context=context,
            vision_local=model.get_vision_tower().vision_tower_name,
            tokenizer_vocab=len(tokenizer),
            fp16_inference="not_tested_on_cpu",
        )
        if args.image:
            ids, images, _ = inputs(
                tokenizer, model, processor, args.image, args.question, "cpu"
            )
            report.update(
                image_path=str(args.image),
                image_sha256=sha256(args.image),
                question=args.question,
                image_shape=list(images.shape),
                finite=bool(images.isfinite().all()),
                input_shape=list(ids.shape),
            )
        write_json(ROOT / "docs/local_load_validation.json", report)
        print(json.dumps(report, indent=2))
        return
    try:
        smi = subprocess.run(
            ["nvidia-smi"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        nvml_ok = smi.returncode == 0
        nvml_detail = smi.stdout.strip()
    except Exception as e:
        nvml_ok = False
        nvml_detail = repr(e)
    if not nvml_ok or not torch.cuda.is_available():
        report = dict(
            time=now(),
            status="blocked_gpu",
            reason="GPU evaluation requires healthy nvidia-smi and CUDA; run python scripts/check_environment.py and see docs/environment.json",
            nvidia_smi=nvml_detail,
            cuda_available=torch.cuda.is_available(),
            fp16_baseline_completed=False,
        )
        write_json(out / "status.json", report)
        print(json.dumps(report, indent=2))
        raise SystemExit(2)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    if args.benchmark == "smoke":
        image = args.image
        if image is None:
            cal = ROOT / "data/calibration/calibration.json"
            if not cal.exists():
                raise SystemExit("Provide --image or prepare calibration first")
            image = ROOT / json.loads(cal.read_text())[0]["image"]
        if not image.is_file():
            raise FileNotFoundError(image)
        rows = [
            dict(question_id="smoke", image=str(image.resolve()), text=args.question)
        ]
        source_hash = sha256(image)
        base = ROOT
    else:
        name = (
            "pope/questions.jsonl"
            if args.benchmark == "pope"
            else "scienceqa/questions_test_image.jsonl"
        )
        question_file = ROOT / "data/eval" / name
        rows = [json.loads(x) for x in question_file.read_text().splitlines()]
        source_hash = sha256(question_file)
        base = (
            ROOT
            / "data/eval"
            / ("pope/val2014" if args.benchmark == "pope" else "scienceqa/images/test")
        )
        if args.limit:
            rows = rows[: args.limit]
    snapshots = {
        n: json.loads((ROOT / "docs" / (n + "_snapshot.json")).read_text())["revision"]
        for n in ["TinyLLaVA-1.5B", "TinyLLaVA-1.5B-SigLIP"]
    }
    spec = dict(
        benchmark=args.benchmark,
        source_sha256=source_hash,
        count=len(rows),
        limit=args.limit,
        seed=2026,
        model_revisions=snapshots,
        dtype="float16",
        do_sample=False,
        num_beams=1,
        conv_mode="v1",
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        batch_size=args.batch_size,
        padding_side="left",
        reuse_identical_inputs=args.reuse_identical_inputs,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    if args.precision != "fp16":
        from int4 import FORMAT
        spec.update(dtype="int4", activation_dtype="float16", quantization_format="W4A16",
                    quantization_method="RTN", quantization_version=FORMAT,
                    quant_group_size=args.quant_group_size, quant_backend=args.quant_backend,
                    quantization_scope="all_model_parameters", finetuning_steps=0)
    if args.precision in ("int4-w4a4", "int4-mixed"):
        from w4a4 import FORMAT as NATIVE_FORMAT
        spec.update(quantization_format="W4A4_linear_conv", native_quantization_version=NATIVE_FORMAT,
                    activation_dtype="int4_at_weighted_gemm_inputs", activation_scaling="dynamic_per_row_and_original_scale_aligned_K_segment_absmax",
                    gemm_backend="cuda_wmma_s4", multiply_dtype="int4", accumulator_dtype="int32",
                    native_compute_scope="all_Linear_and_Conv2d", attention_matmul_dtype="original_fp16",
                    non_gemm_dtype="original_fp16_fp32",
                    native_implementation_sha256={str(path): sha256(ROOT / path) for path in (
                        Path("kernels/int4_gemm.cu"), Path("scripts/int4_cuda.py"), Path("scripts/w4a4.py"))})
    mixed_policy = None
    if args.precision == "int4-mixed":
        from mixed_precision import FORMAT as MIXED_FORMAT, read_policy
        mixed_policy = read_policy(args.mixed_policy)
        spec.update(dtype="mixed_int4_fp16", quantization_format="mixed_WA_from_W4A4",
                    mixed_precision_version=MIXED_FORMAT, mixed_variant=args.mixed_variant,
                    mixed_policy_sha256=sha256(args.mixed_policy), mixed_rules=mixed_policy["rules"],
                    quantization_scope="all_model_parameters_except_policy_FP16_weights",
                    activation_dtype="int4_except_policy_FP16_Linear_inputs",
                    native_compute_scope="remaining_W4A4_Linear_Conv2d",
                    multiply_dtype="int4_native_W4A4_and_FP16_exceptions",
                    accumulator_dtype="int32_native_W4A4_and_original_FP16_operator_exceptions",
                    exception_compute="W16A4_A_QDQ_then_FP16_GEMM;W4A16_W_DQ_then_FP16_GEMM;W16A16_FP16_GEMM",
                    mixed_implementation_sha256={str(path): sha256(ROOT / path) for path in (
                        Path("scripts/mixed_precision.py"), Path("scripts/analyze_llm_sensitivity.py"),
                        Path("scripts/int4.py"))})
    if args.reuse_identical_inputs:
        spec["reuse_identical_inputs"] = True
    old = out / "run.json"
    answers = out / "predictions.jsonl"
    if old.exists() and json.loads(old.read_text()) != spec:
        raise SystemExit(
            "Existing output uses different conditions. Choose a new --output directory."
        )
    write_json(old, spec)
    done = (
        {
            x["question_id"]: x
            for x in (json.loads(s) for s in answers.read_text().splitlines())
        }
        if answers.exists()
        else {}
    )
    summary_path = out / "summary.json"
    if summary_path.exists() and set(done) == {row["question_id"] for row in rows}:
        previous = json.loads(summary_path.read_text())
        if previous.get("status") == "complete" and previous.get("conditions") == spec:
            print(f"Already complete: {out}; preserved original measurements and execution counts", flush=True)
            return
    tokenizer, model, processor, context = load("cuda")
    model.config.tokenizer_padding_side = "left"
    if mixed_policy is not None:
        from mixed_precision import capture_fp16
        fp16_originals = capture_fp16(model, mixed_policy, args.mixed_variant)
    if args.precision != "fp16":
        from int4 import prepare_checkpoint
        print("Preparing packed INT4 weights for Vision / Projector / LLM...", flush=True)
        audit = prepare_checkpoint(model, args.quant_checkpoint, args.quant_group_size,
                                   args.quant_backend, snapshots)
        if args.precision == "int4-w4a4":
            from w4a4 import enable_w4a4
            audit["integer_compute"] = enable_w4a4(model)
            audit["compute"] = "native_s4_s4_s32_Linear_Conv2d_original_FP16_FP32_other_operations"
            audit["activation_dtype"] = "int4_at_weighted_gemm_inputs"
            print("Native INT4 Tensor Core weighted GEMM enabled: "
                  f"{audit['integer_compute']['linear_layers']} Linear, "
                  f"{audit['integer_compute']['conv2d_layers']} Conv2d; no FP16 operand GEMM fallback", flush=True)
        elif mixed_policy is not None:
            from mixed_precision import enable_mixed_precision
            audit["mixed_precision"] = enable_mixed_precision(model, mixed_policy, args.mixed_variant, fp16_originals)
            del fp16_originals
            audit["compute"] = "native_W4A4_except_explicit_policy_FP16_GEMM_exceptions"
            audit["mixed_policy_sha256"] = sha256(args.mixed_policy)
            print("Mixed precision enabled: " + json.dumps(
                audit["mixed_precision"]["weighted_layer_precision_counts"]), flush=True)
        write_json(out / "quantization.json", audit)
        print(json.dumps({k: audit[k] for k in ("total_parameters", "total_packed_bytes", "components")}),
              flush=True)
        torch.cuda.empty_cache()
    prediction_cache = {(row["image"], row["question"]): row for row in done.values()}
    warmed = False
    pending = [row for row in rows if row["question_id"] not in done]
    for offset in range(0, len(pending), args.batch_size):
        chunk = pending[offset:offset + args.batch_size]
        fresh, seen = [], set()
        for row in chunk:
            key = (row.get("image"), row["text"])
            if not args.reuse_identical_inputs or (key not in prediction_cache and key not in seen):
                fresh.append(row)
                seen.add(key)
        generated = {}
        if fresh:
            start = time.perf_counter()
            paths = [base / row["image"] if row.get("image") else None for row in fresh]
            ids, images, attention_mask, lengths = batch_inputs(
                tokenizer, model, processor, paths, [row["text"] for row in fresh])
            preprocess = time.perf_counter() - start
            if max(lengths) + args.max_new_tokens > context:
                raise RuntimeError("Sample exceeds fixed context; do not silently truncate the benchmark")
            kwargs = dict(images=images, attention_mask=attention_mask, do_sample=False, num_beams=1,
                          max_new_tokens=args.max_new_tokens, use_cache=True,
                          pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            with torch.inference_mode():
                if not warmed:
                    for _ in range(args.warmup):
                        model.generate(ids, **kwargs)
                    warmed = True
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                output = model.generate(ids, **kwargs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            texts = tokenizer.batch_decode(output, skip_special_tokens=True)
            if len(texts) != len(fresh) or output.shape[0] != len(fresh):
                raise RuntimeError("Generation output count differs from the input batch")
            tokens = output.tolist()
            for row, text, sequence in zip(fresh, texts, tokens):
                # The legacy loader returns BOS plus new tokens. Trim padding after
                # the first EOS, since other items can keep the batch decoding.
                ending = next((i + 1 for i, token in enumerate(sequence[1:], 1)
                               if token == tokenizer.eos_token_id), len(sequence))
                result = dict(row)
                result["question"] = result.pop("text")
                result.update(text=text.strip(), seconds=elapsed / len(fresh),
                              preprocess_seconds=preprocess / len(fresh),
                              batch_seconds=elapsed, batch_size=len(fresh),
                              batch_id=fresh[0]["question_id"],
                              peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                              generated_tokens=ending)
                if args.reuse_identical_inputs:
                    result["reused_from_question_id"] = None
                generated[row["question_id"]] = result
                prediction_cache[(row.get("image"), row["text"])] = result
            del ids, images, attention_mask, output, kwargs
        for row in chunk:
            if row["question_id"] in generated:
                result = generated[row["question_id"]]
            else:
                previous = prediction_cache[(row.get("image"), row["text"])]
                result = dict(row)
                result["question"] = result.pop("text")
                for key in ("text", "seconds", "preprocess_seconds", "peak_allocated_bytes",
                            "peak_reserved_bytes", "generated_tokens", "batch_id", "batch_size", "batch_seconds"):
                    result[key] = previous[key]
                result["reused_from_question_id"] = previous.get("reused_from_question_id") or previous["question_id"]
            with answers.open("a") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            done[row["question_id"]] = result
        print(f"{len(done)}/{len(rows)}: batch {len(fresh)}, "
              + (f"{elapsed:.3f}s" if fresh else "reused identical inputs"), flush=True)
    completed = [done[x["question_id"]] for x in rows]
    batches = {row["batch_id"]: row["batch_seconds"] for row in completed}
    fresh_count = sum(not row.get("reused_from_question_id") for row in completed)
    native_execution = {}
    if args.precision == "int4-w4a4":
        from w4a4 import execution_counts
        native_execution["native_execution"] = execution_counts(model)
    elif args.precision == "int4-mixed":
        from mixed_precision import mixed_execution_counts
        native_execution["mixed_execution"] = mixed_execution_counts(model)
    write_json(
        out / "summary.json",
        dict(
            time=now(),
            status="complete",
            conditions=spec,
            metrics=score(completed, args.benchmark),
            mean_seconds=statistics.mean(x["seconds"] for x in completed),
            median_seconds=statistics.median(x["seconds"] for x in completed),
            peak_allocated_bytes=max(x["peak_allocated_bytes"] for x in completed),
            peak_reserved_bytes=max(x["peak_reserved_bytes"] for x in completed),
            gpu=torch.cuda.get_device_name(),
            generation_seconds=sum(batches.values()),
            fresh_samples_per_second=fresh_count / sum(batches.values()),
            measurement=("per-item seconds = batch generation seconds / actual batch size; includes "
                         "vision/projector/prefill/decode; preprocessing, model load and warmup excluded"
                         + ("; repeated-input timing observations reused and weighted by question multiplicity"
                            if args.reuse_identical_inputs else "")),
            execution={
                "unique_input_count": len(prediction_cache),
                "fresh_generation_count": fresh_count,
                "reused_prediction_count": sum(bool(row.get("reused_from_question_id")) for row in completed),
            },
            **native_execution,
        ),
    )


if __name__ == "__main__":
    main()
