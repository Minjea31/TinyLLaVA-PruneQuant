"""Verify full native W4A4 results and compare FP16 / W4A16 / W4A4 scopes."""

import json
import statistics

from common import ROOT, now, sha256, write_json
from report_int4 import load_verified


def main():
    roots = {"FP16": ROOT / "results/baseline", "W4A16": ROOT / "results/int4_all_w4a16",
             "W4A4 Linear+Conv": ROOT / "results/int4_all_w4a4"}
    report = {"time": now(), "status": "complete", "finetuning_steps": 0,
              "native_scope": "all Linear and Conv2d; original attention QK/PV and non-GEMM precision",
              "benchmarks": {}}
    lines = ["# 실제 INT4 곱셈 W4A4 평가", "",
             "모든 가중치는 기존 INT4 RTN 체크포인트의 정수 값과 scale을 그대로 사용한다.",
             "313개 Linear와 SigLIP patch Conv2d 1개는 INT4×INT4 Tensor Core 곱셈·INT32 누산을 수행한다.",
             "activation은 각 입력 행 안에서 원래 weight scale 경계에 맞춘 K 그룹마다 동적으로 INT4 양자화한다.",
             "Attention의 QK/PV 행렬곱·Softmax·Norm·RoPE·embedding은 기존 정밀도를 유지한다.",
             "모델 전체의 모든 연산을 INT4로 수행한 결과가 아니다. 파인튜닝은 없다.", "",
             "| 평가 | FP16 | 기존 W4A16 | 실제 W4A4 Linear+Conv |",
             "| --- | ---: | ---: | ---: |"]
    performance = ["", "| 평가·모드 | peak allocated GiB | 문항당 처리 시간 ms | 처리량 문항/초 |",
                   "| --- | ---: | ---: | ---: |"]
    same_conditions = ("benchmark", "source_sha256", "count", "limit", "seed", "model_revisions",
                       "do_sample", "num_beams", "conv_mode", "max_new_tokens", "warmup", "torch", "cuda",
                       "batch_size", "padding_side", "reuse_identical_inputs")
    for benchmark in ("pope", "scienceqa"):
        entries = {}
        for label, root in roots.items():
            summary, rows, checks = load_verified(root, benchmark)
            entries[label] = {"summary": summary, "verified": checks}
        baseline = entries["FP16"]["summary"]
        native = entries["W4A4 Linear+Conv"]["summary"]
        for label, entry in entries.items():
            if any(baseline["conditions"][key] != entry["summary"]["conditions"][key] for key in same_conditions):
                raise RuntimeError(f"{benchmark}/{label} uses different evaluation conditions")
        native_conditions = native["conditions"]
        if native_conditions.get("gemm_backend") != "cuda_wmma_s4" or native_conditions.get("finetuning_steps") != 0:
            raise RuntimeError("Expected native INT4 weighted GEMM without fine-tuning")
        for path, expected_hash in native_conditions["native_implementation_sha256"].items():
            if sha256(ROOT / path) != expected_hash:
                raise RuntimeError(f"Native implementation changed after this run: {path}")
        audit = json.loads((roots["W4A4 Linear+Conv"] / benchmark / "quantization.json").read_text())
        previous = json.loads((roots["W4A16"] / benchmark / "quantization.json").read_text())
        if audit["checkpoint_sha256"] != previous["checkpoint_sha256"]:
            raise RuntimeError("W4A16 and native W4A4 used different packed source weights")
        compute = audit["integer_compute"]
        if compute["linear_layers"] != 313 or compute["conv2d_layers"] != 1:
            raise RuntimeError("Incomplete native INT4 layer coverage")
        instructions = compute["kernel"].get("imma_by_kernel", {}).get("scaled_int4_gemm", [])
        if not any("S4.S4" in line for line in instructions):
            raise RuntimeError("No signed INT4 instruction evidence for weighted GEMM")
        if native["native_execution"]["native_layer_count"] != 314:
            raise RuntimeError("Native execution layer topology mismatch")
        entries["W4A4 Linear+Conv"]["quantization"] = audit
        scores = []
        for label, entry in entries.items():
            summary = entry["summary"]
            metric = (statistics.mean(row["f1"] for row in summary["metrics"].values())
                      if benchmark == "pope" else summary["metrics"]["accuracy"])
            entry["score"] = metric
            scores.append(f"{metric*100:.2f}%")
            performance.append(f"| {benchmark} / {label} | {summary['peak_allocated_bytes']/2**30:.4f} | "
                               f"{summary['mean_seconds']*1000:.2f} | {summary['fresh_samples_per_second']:.2f} |")
        label = "POPE macro F1" if benchmark == "pope" else "ScienceQA accuracy"
        lines.append(f"| {label} | " + " | ".join(scores) + " |")
        report["benchmarks"][benchmark] = entries
    lines += performance + ["", "공통 조건은 batch 8, 같은 질문·이미지·greedy·max_new_tokens 32·warmup 1이다.",
                            "문항당 시간은 batch generation 시간을 실제 batch 크기로 나눈 값이며 개별 요청 latency가 아니다.",
                            "이 커널은 정수 정확성을 먼저 검증한 구현이며 속도 최적화는 완료하지 않았다.",
                            "타일 padding·activation quantize/pack·group scale·bias 비용도 포함한다.",
                            "원자료: `results/int4_all_w4a4/comparison.json`."]
    output = roots["W4A4 Linear+Conv"] / "comparison.json"
    write_json(output, report)
    write_json(roots["W4A4 Linear+Conv"] / "status.json", {
        "time": now(), "status": "complete", "finetuning_steps": 0, "comparison": str(output),
        "scope": report["native_scope"]})
    path = ROOT / "docs/int4_all_w4a4_results.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Verified native INT4 comparison: {output}\nReport: {path}")


if __name__ == "__main__":
    main()
