"""Verify fixed mixed precision against FP16, W4A16 and native W4A4 full runs."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
from safetensors import safe_open

from common import ROOT, now, sha256, write_json
from mixed_precision import DEFAULT_POLICY, read_policy, selected_names
from report_int4 import load_verified


def original_fp16_hashes(names):
    root = ROOT / "model/TinyLLaVA-1.5B"
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    hashes = {}
    for name in sorted(names):
        selected = [key for key in index if key.startswith(name + ".") and key.rsplit(".", 1)[-1] in ("weight", "bias")]
        hashes[name] = {}
        for key in selected:
            with safe_open(root / index[key], framework="pt", device="cpu") as source:
                value = source.get_tensor(key).to(torch.float16).contiguous()
            hashes[name][key.rsplit(".", 1)[-1]] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["weights", "activations", "both"], default="both")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    policy = read_policy(args.policy)
    weights, activations = selected_names(policy, args.variant)
    fp16_hashes = original_fp16_hashes(weights)
    destination = ROOT / "results/mixed_precision" / args.variant
    roots = {"FP16": ROOT / "results/baseline", "W4A16": ROOT / "results/int4_all_w4a16",
             "W4A4": ROOT / "results/int4_all_w4a4", "Mixed": destination}
    report = {"time": now(), "status": "complete", "variant": args.variant, "policy_sha256": sha256(args.policy),
              "finetuning_steps": 0, "benchmarks": {}, "original_public_fp16_weight_hashes": fp16_hashes}
    lines = ["# KL 기준 weight·activation FP16 유지 — 전체 평가", "",
             f"설정: {args.variant}. Weight KL ≥ {policy['rules']['weight']['threshold']}, "
             f"activation KL ≥ {policy['rules']['activation']['threshold']}를 두 선정 집합에서 모두 만족한 대상을 사용했다.",
             f"FP16 weight 레이어 {len(weights)}개, A16 Linear 입력 {len(activations)}개. 학습·LoRA·optimizer update는 없다.",
             "나머지 W4A4 Linear·Conv는 실제 INT4×INT4→INT32를 사용한다. W16A4·W4A16·W16A16 예외는 FP16 GEMM 경로다.",
             "Attention QK/PV·Softmax·RoPE 등 비대상 연산은 기존 정밀도를 유지한다.", "",
             "| 평가 | FP16 | W4A16 | W4A4 | Mixed |", "| --- | ---: | ---: | ---: | ---: |"]
    performance = ["", "| 평가·설정 | peak allocated GiB | 문항당 처리 시간 ms | 처리량 문항/초 |",
                   "| --- | ---: | ---: | ---: |"]
    common = ("benchmark", "source_sha256", "count", "limit", "seed", "model_revisions", "do_sample", "num_beams",
              "conv_mode", "max_new_tokens", "warmup", "torch", "cuda", "batch_size", "padding_side", "reuse_identical_inputs")
    audit_reference = None
    for benchmark in ("pope", "scienceqa"):
        entries = {}
        for label, root in roots.items():
            summary, predictions, checks = load_verified(root, benchmark)
            score = (statistics.mean(r["f1"] for r in summary["metrics"].values()) if benchmark == "pope"
                     else summary["metrics"]["accuracy"])
            entries[label] = {"summary": summary, "verified": checks, "score": score}
        baseline = entries["FP16"]["summary"]
        mixed = entries["Mixed"]["summary"]
        spec = mixed["conditions"]
        for entry in entries.values():
            assert all(baseline["conditions"][key] == entry["summary"]["conditions"][key] for key in common)
        assert spec["mixed_variant"] == args.variant and spec["mixed_policy_sha256"] == sha256(args.policy)
        assert spec["finetuning_steps"] == 0
        for key in ("mixed_implementation_sha256", "native_implementation_sha256"):
            assert all(sha256(ROOT / path) == digest for path, digest in spec[key].items())
        quantization = json.loads((destination / benchmark / "quantization.json").read_text())
        source_quantization = json.loads((roots["W4A4"] / benchmark / "quantization.json").read_text())
        assert quantization["checkpoint_sha256"] == source_quantization["checkpoint_sha256"]
        audit = quantization["mixed_precision"]
        assert set(audit["fp16_weight_layers"]) == weights and set(audit["fp16_activation_layers"]) == activations
        assert audit["fp16_parameter_sha256"] == fp16_hashes
        assert audit["weighted_layer_count"] == 314
        expected_native = sum(r["precision"] == "W4A4" for r in audit["weighted_layers"])
        assert audit["native_layer_count"] == expected_native
        assert any("S4.S4" in line for line in audit["kernel"]["imma_by_kernel"]["scaled_int4_gemm"])
        execution = mixed["mixed_execution"]
        assert execution["native"]["native_layer_count"] == expected_native
        assert execution["native"]["current_process_called_layers"] == expected_native
        assert all(r["calls"] > 0 for r in execution["exceptions"].values())
        assert len(execution["exceptions"]) == 314 - expected_native
        entries["Mixed"]["quantization"] = quantization
        old_score, new_score = entries["W4A4"]["score"], entries["Mixed"]["score"]
        entries["Mixed"]["delta_vs_w4a4_pp"] = 100 * (new_score-old_score)
        report["benchmarks"][benchmark] = entries
        label = "POPE macro F1" if benchmark == "pope" else "ScienceQA accuracy"
        lines.append(f"| {label} | " + " | ".join(f"{entry['score']*100:.2f}%" for entry in entries.values()) + " |")
        for label, entry in entries.items():
            summary = entry["summary"]
            performance.append(f"| {benchmark} / {label} | {summary['peak_allocated_bytes']/2**30:.4f} | "
                               f"{summary['mean_seconds']*1000:.2f} | {summary['fresh_samples_per_second']:.2f} |")
        audit_reference = audit
    lines += performance + ["", "공통 조건: batch 8, left padding, v1 prompt, greedy, seed 2026, max_new_tokens 32, warmup 1.",
        "문항당 시간은 batch 생성 시간을 실제 batch 크기로 나눈 값이며 개별 요청 latency가 아니다.",
        "Peak allocated는 CUDA tensor 할당량이며 프로세스 전체 메모리와 구분한다. 각 설정은 단일 실행 측정이다.", "",
        "## 실제 정밀도 구성", "", "| 계산 정밀도 | Linear·Conv 개수 |", "| --- | ---: |"]
    for precision, count in audit_reference["weighted_layer_precision_counts"].items():
        lines.append(f"| {precision} | {count} |")
    lines += ["", "FP16 weight는 공개 원본 safetensors를 FP16으로 변환한 값의 hash와 직접 일치했다.",
              "같은 INT4 checkpoint·입력·생성 조건, 예측 재채점, 레이어 호출과 native 명령을 검증했다.",
              "독립 calibration 비교: [보고서](../calibration/report.md). Weight-only·activation-only는 해당 calibration에서 비교했다.",
              "결합 설정의 전체 평가 결과를 기록했으며, threshold를 전체 평가 점수에 맞추어 변경하지 않았다."]
    write_json(destination / "comparison.json", report)
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    write_json(destination / "status.json", {"time": now(), "status": "complete", "finetuning_steps": 0,
        "variant": args.variant, "full_verified_predictions": 11017})
    print(f"Verified mixed-precision comparison: {destination / 'report.md'}")


if __name__ == "__main__":
    main()
