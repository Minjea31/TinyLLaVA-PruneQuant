"""Verify completed FP16/INT4 runs and write the pre-fine-tuning comparison."""

import argparse
import json
from pathlib import Path
import statistics

from common import ROOT, now, sha256, write_json
from test_inference import score


def load_verified(root, benchmark):
    directory = root / benchmark
    summary = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    source = ROOT / "data/eval" / ("pope/questions.jsonl" if benchmark == "pope"
                                  else "scienceqa/questions_test_image.jsonl")
    questions = [json.loads(line) for line in source.read_text().splitlines()]
    expected = {row["question_id"]: row for row in questions}
    ids = [row["question_id"] for row in rows]
    conditions = summary["conditions"]
    checks = {
        "complete": summary["status"] == "complete",
        "full_split": conditions["limit"] == 0 and len(rows) == len(questions) == conditions["count"],
        "unique_ids": len(set(ids)) == len(ids),
        "exact_ids": set(ids) == set(expected),
        "source_hash": sha256(source) == conditions["source_sha256"],
        "run_conditions": json.loads((directory / "run.json").read_text()) == conditions,
        "recomputed_metrics": score(rows, benchmark) == summary["metrics"],
        "questions_labels_images": all(
            row["question"] == expected[row["question_id"]]["text"]
            and all(row.get(key) == expected[row["question_id"]].get(key)
                    for key in ("answer", "label", "category", "image")) for row in rows),
        "mean_latency": statistics.mean(row["seconds"] for row in rows) == summary["mean_seconds"],
        "peak_allocated": max(row["peak_allocated_bytes"] for row in rows) == summary["peak_allocated_bytes"],
        "peak_reserved": max(row["peak_reserved_bytes"] for row in rows) == summary["peak_reserved_bytes"],
        "batch_size_8": conditions["batch_size"] == 8,
        "batch_timing": all(row["seconds"] == row["batch_seconds"] / row["batch_size"] for row in rows),
        "generation_seconds": sum({row["batch_id"]: row["batch_seconds"] for row in rows}.values()) == summary["generation_seconds"],
    }
    if conditions.get("reuse_identical_inputs"):
        fresh = {row["question_id"]: row for row in rows if not row.get("reused_from_question_id")}
        reused = [row for row in rows if row.get("reused_from_question_id")]
        same_fields = ("image", "question", "text", "seconds", "preprocess_seconds",
                       "peak_allocated_bytes", "peak_reserved_bytes", "generated_tokens")
        checks["reused_predictions_have_exact_real_source"] = all(
            row["reused_from_question_id"] in fresh and all(
                row[key] == fresh[row["reused_from_question_id"]][key] for key in same_fields)
            for row in reused)
        checks["execution_counts"] = summary["execution"] == {
            "unique_input_count": len({(row["image"], row["question"]) for row in rows}),
            "fresh_generation_count": len(fresh), "reused_prediction_count": len(reused),
        }
    if not all(checks.values()):
        raise RuntimeError(f"Invalid {root.name}/{benchmark}: {checks}")
    return summary, rows, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Publish verified completed benchmarks and mark missing ones pending")
    args = parser.parse_args()
    result = {"time": now(), "status": "complete", "finetuning_steps": 0, "benchmarks": {}}
    lines = ["# 전체 가중치 INT4 / activation FP16 — 파인튜닝 전 결과", "",
             "공개 TinyLLaVA-1.5B의 모든 parameter를 RTN INT4로 압축한 W4A16 실험이다.",
             "학습·gradient update·activation 양자화·AWQ/GPTQ 보정은 수행하지 않았다.", "",
             "| 평가 | FP16 Accuracy | INT4 Accuracy | FP16 F1 | INT4 F1 | F1 변화(pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    perf_lines = ["", "**GPU 메모리·시간**", "",
                  "| 평가 | FP16 / INT4 peak allocated GiB | allocated 절감률 | FP16 / INT4 평균 시간(ms) | 시간 변화율 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
    for benchmark in ("pope", "scienceqa"):
        if not (ROOT / "results/int4_all_w4a16" / benchmark / "summary.json").is_file():
            if not args.allow_partial:
                raise RuntimeError(f"INT4 {benchmark} is incomplete; use --allow-partial to label a partial report")
            result["status"] = "partial"
            predictions = ROOT / "results/int4_all_w4a16" / benchmark / "predictions.jsonl"
            done = len(predictions.read_text().splitlines()) if predictions.exists() else 0
            result.setdefault("pending_benchmarks", {})[benchmark] = {"completed_predictions": done,
                                                                     "expected_count": 9000 if benchmark == "pope" else 2017}
            continue
        fp16, old_rows, old_checks = load_verified(ROOT / "results/baseline", benchmark)
        quant, new_rows, new_checks = load_verified(ROOT / "results/int4_all_w4a16", benchmark)
        comparable = ("benchmark", "source_sha256", "count", "limit", "seed", "model_revisions",
                      "do_sample", "num_beams", "conv_mode", "max_new_tokens", "warmup", "torch", "cuda")
        comparable += ("batch_size", "padding_side", "reuse_identical_inputs")
        if any(fp16["conditions"][key] != quant["conditions"][key] for key in comparable):
            raise RuntimeError(f"FP16/INT4 conditions differ for {benchmark}")
        if quant["conditions"].get("finetuning_steps") != 0:
            raise RuntimeError("This report requires a pre-fine-tuning run")
        row = {"fp16": fp16, "int4": quant, "verified": {"fp16": old_checks, "int4": new_checks}}
        if benchmark == "pope":
            row["macro_f1_fp16"] = statistics.mean(m["f1"] for m in fp16["metrics"].values())
            row["macro_f1_int4"] = statistics.mean(m["f1"] for m in quant["metrics"].values())
            row["macro_f1_delta_pp"] = (row["macro_f1_int4"] - row["macro_f1_fp16"]) * 100
            for category in ("random", "popular", "adversarial"):
                a, b = fp16["metrics"][category], quant["metrics"][category]
                lines.append(f"| POPE {category} | {a['accuracy']*100:.2f} | {b['accuracy']*100:.2f} | "
                             f"{a['f1']*100:.2f} | {b['f1']*100:.2f} | {(b['f1']-a['f1'])*100:+.2f} |")
            lines.append(f"| POPE macro F1 | — | — | {row['macro_f1_fp16']*100:.2f} | "
                         f"{row['macro_f1_int4']*100:.2f} | {row['macro_f1_delta_pp']:+.2f} |")
            duplicates = {}
            for previous in old_rows:
                key = (previous["image"], previous["question"])
                duplicates.setdefault(key, set()).add(previous["text"])
            row["fp16_identical_input_check"] = {
                "unique_inputs": len(duplicates),
                "repeated_inputs_with_different_predictions": sum(len(values) != 1 for values in duplicates.values()),
            }
        else:
            a, b = fp16["metrics"], quant["metrics"]
            row["accuracy_delta_pp"] = (b["accuracy"] - a["accuracy"]) * 100
            lines.append(f"| ScienceQA image-test | {a['accuracy']*100:.2f} | {b['accuracy']*100:.2f} | — | — | — |")
            old = {r["question_id"]: r for r in old_rows}
            gained = lost = 0
            for new in new_rows:
                previous = old[new["question_id"]]
                previous_correct = score([previous], "scienceqa")["correct"] == 1
                current_correct = score([new], "scienceqa")["correct"] == 1
                gained += current_correct and not previous_correct
                lost += previous_correct and not current_correct
            row["paired_correct_gained"] = gained
            row["paired_correct_lost"] = lost
        row["allocated_reduction_percent"] = (1 - quant["peak_allocated_bytes"] / fp16["peak_allocated_bytes"]) * 100
        row["reserved_reduction_percent"] = (1 - quant["peak_reserved_bytes"] / fp16["peak_reserved_bytes"]) * 100
        row["mean_time_change_percent"] = (quant["mean_seconds"] / fp16["mean_seconds"] - 1) * 100
        # Returned generation sequences include the initial BOS token in this loader.
        row["fp16_mean_returned_tokens"] = statistics.mean(r["generated_tokens"] for r in old_rows)
        row["int4_mean_returned_tokens"] = statistics.mean(r["generated_tokens"] for r in new_rows)
        row["fp16_token_cap_hits"] = sum(r["generated_tokens"] == 33 for r in old_rows)
        row["int4_token_cap_hits"] = sum(r["generated_tokens"] == 33 for r in new_rows)
        result["benchmarks"][benchmark] = row
        perf_lines.append(f"| {benchmark} | {fp16['peak_allocated_bytes']/2**30:.4f} / "
                          f"{quant['peak_allocated_bytes']/2**30:.4f} | {row['allocated_reduction_percent']:.2f}% | "
                          f"{fp16['mean_seconds']*1000:.2f} / {quant['mean_seconds']*1000:.2f} | "
                          f"{row['mean_time_change_percent']:+.2f}% |")
    checkpoint = ROOT / "model/quantized/TinyLLaVA-1.5B-W4A16-all-RTN-g128"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if manifest["total_parameters"] != 1504353824 or manifest["unquantized_parameter_tensors"]:
        raise RuntimeError("Incomplete all-model INT4 coverage")
    if sha256(checkpoint / "model.safetensors") != manifest["checkpoint_sha256"]:
        raise RuntimeError("INT4 checkpoint file hash mismatch")
    result["quantization"] = manifest
    output = ROOT / "results/int4_all_w4a16/comparison.json"
    write_json(output, result)
    if result["status"] == "complete":
        write_json(ROOT / "results/int4_all_w4a16/status.json", {
            "time": now(), "status": "complete", "scienceqa": "complete", "pope": "complete",
            "finetuning_steps": 0, "comparison": str(output),
        })
    lines += perf_lines
    qa = result["benchmarks"].get("scienceqa")
    if qa:
        lines += ["", f"ScienceQA accuracy 변화: {qa['accuracy_delta_pp']:+.2f}pp. "
                  f"FP16에서 오답→INT4에서 정답 {qa['paired_correct_gained']}개, "
                  f"정답→오답 {qa['paired_correct_lost']}개."]
    execution = result["benchmarks"].get("pope", {}).get("int4", {}).get("execution")
    if execution and result["benchmarks"]["pope"]["int4"]["conditions"]["reuse_identical_inputs"]:
        lines += ["", f"POPE는 동일 이미지·질문 입력이 반복되어 {execution['fresh_generation_count']:,}번 실제 추론하고 "
                  f"{execution['reused_prediction_count']:,}개 답변을 재사용해 전체 9,000문항을 채점했다.",
                  "각 재사용 예측이 실제 생성한 원본과 같은 이미지·질문·답변인지 검증했다. "
                  "배치 문맥에 따른 수치 차이가 있을 수 있으므로 동일 입력 답변의 일치 여부는 comparison.json에 별도 기록했다.",
                  "POPE INT4 평균 시간은 고유 입력의 실측 시간을 문항 반복 횟수로 가중한 값이다. "
                  "전체 문항을 각각 다시 실행한 9,000개 독립 시간 관측값은 아니다. "
                  "속도 결론은 같은 고유 입력을 별도로 반복 측정해 확인한다."]
    lines += ["", "**적용 범위와 해석**", "",
              "Vision·Projector·LLM의 626개 parameter tensors, 총 1,504,353,824개 값을 모두 INT4로 변환했다.",
              "Embedding 2개, Linear 313개, Conv2d 1개, LayerNorm 53개, RMSNorm 45개와 해당 bias를 포함한다.",
              "연속된 128개 값마다 FP16 absmax/7 scale을 저장하고, FP32에서 round-to-nearest 후 [-7,7]로 제한한다.",
              "두 integer code를 uint8 한 개에 packing한다. [-7,7]의 15개 수준을 쓰는 대칭 narrow-range INT4다.",
              "activation·KV cache는 원래 정밀도를 유지하고, softmax·normalization 내부 FP32 누산 등도 원래 동작을 유지한다.",
              "RoPE 등 비학습 arithmetic buffer와 FP16 scale metadata는 4-bit parameter 범위에 포함하지 않는다.",
              f"packed payload는 {manifest['total_packed_bytes']:,} bytes "
              f"({manifest['total_packed_bytes']/2**30:.4f} GiB), safetensors 파일은 {manifest['checkpoint_file_bytes']:,} bytes다.",
              "각 연산 직전에 필요한 weight를 Triton으로 FP16 복원하고 원래 FP16 연산을 실행한다.",
              "FP16 weights 전체를 계속 보관하는 fake-quant 방식은 아니지만, 정수 GEMM 전용 최적화 커널도 아니다.",
              "따라서 시간은 이 reference backend의 실측값이며 최적화된 INT4의 일반적인 속도 효과로 해석하지 않는다.",
              "각 문항 1회 측정이며 답변 길이가 달라질 수 있으므로 출력 길이·반복 측정을 추가해 속도를 평가한다.",
              "FP16과 INT4 모두 batch 8·left padding을 사용한다. 문항당 시간은 batch 생성 시간을 실제 batch 크기로 나눈 값이다.",
              "ScienceQA는 image-test 2,017문항과 로컬 CQM letter-only prompt다.", "",
              "**원자료와 재현**", "", "```bash", "bash scripts/eval_int4_all.sh",
              "python scripts/report_int4.py", "```", "",
              "- `results/int4_all_w4a16/comparison.json`: 재채점·조건 검증·정밀 수치.",
              "- `results/int4_all_w4a16/{pope,scienceqa}/`: 예측·summary·run·quantization audit.",
              "- `model/quantized/TinyLLaVA-1.5B-W4A16-all-RTN-g128/`: packed checkpoint·manifest.",
              "- `docs/int4_implementation.md`: 구현 방식과 적용 범위.",
              "- `docs/fp16_baseline.md`: 비교에 사용한 FP16 기준 성능.", "",
              "이 결과는 전체 가중치를 단순 RTN으로 압축했을 때의 출발점이다. "
              "후속 AWQ/GPTQ·혼합 정밀도·복구 학습 결과와 구분한다. W4A4는 아직 실행하지 않았다."]
    path = ROOT / "docs/int4_all_w4a16_results.md"
    if result["status"] == "partial":
        pending = "; ".join(f"{name}: {value['completed_predictions']:,}/{value['expected_count']:,}문항"
                            for name, value in result["pending_benchmarks"].items())
        lines[2:2] = [f"**일부 완료: {pending}. 미완료 benchmark의 최종 점수는 아직 없다.**", ""]
    path.write_text("\n".join(lines) + "\n")
    print(f"Verified comparison: {output}\nReport: {path}")


if __name__ == "__main__":
    main()
