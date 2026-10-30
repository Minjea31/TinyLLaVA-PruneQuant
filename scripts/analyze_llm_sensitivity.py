"""Single-target LLM sensitivity, with fixed FP16 context and no weight updates.

QDQ interventions use the same FP16 arithmetic to separate W/A quantization.
Teacher-forced assistant positions are scored, never image/prompt/pad positions.
Existing RTN weights are read-only; public FP16 parameters are restored in finally.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, ExitStack
import copy
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

from common import ROOT, now, sha256, write_json
from int4 import PackedLayer
from runtime import load, official_imports


MODES = ("w4a16", "w16a4", "w4a4")
FORMAT = "llm_single_target_teacher_forced_sensitivity_v1"
CHECKPOINT = ROOT / "model/quantized/TinyLLaVA-1.5B-W4A16-all-RTN-g128"


@torch.no_grad()
def activation_qdq(inputs, group_size=128, rows_per_chunk=256):
    """Native-matching A4 values, returned in floating point for controlled GEMM."""
    group = math.gcd(inputs.shape[-1], group_size)
    flat = inputs.reshape(-1, inputs.shape[-1])
    result = torch.empty_like(flat)
    for start in range(0, len(flat), rows_per_chunk):
        values = flat[start:start+rows_per_chunk].float().reshape(-1, group)
        maximum = values.abs().amax(-1, keepdim=True)
        scale = (maximum / 7).clamp_min(2**-24).half()
        scale[maximum == 0] = 1
        quantized = (values / scale.float()).round().clamp(-7, 7)
        result[start:start+rows_per_chunk] = (quantized * scale.float()).reshape(
            -1, inputs.shape[-1]).to(inputs.dtype)
    return result.reshape(inputs.shape)


def targets_for_model(model, include_groups=True):
    """Discover LLM only: seven projections + two RMSNorms per decoder block."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    targets = []
    for name, module in model.named_modules():
        if not (name.startswith("model.layers.") or name in ("model.embed_tokens", "model.norm", "lm_head")):
            continue
        if not isinstance(module, (nn.Linear, nn.Embedding, LlamaRMSNorm)):
            continue
        match = re.match(r"model\.layers\.(\d+)\.(.+)", name)
        block = int(match.group(1)) if match else None
        projection = match.group(2).split(".")[-1] if match else name.split(".")[-1]
        family = ("attention" if ".self_attn." in name else "ffn" if ".mlp." in name
                  else "norm" if isinstance(module, LlamaRMSNorm) else projection)
        kind = "linear" if isinstance(module, nn.Linear) else "embedding" if isinstance(module, nn.Embedding) else "rmsnorm"
        targets.append({"name": name, "modules": [name], "kind": kind, "level": "leaf",
                        "block": block, "family": family, "projection": projection,
                        "parameters": sum(p.numel() for p in module.parameters(recurse=False))})
    if include_groups:
        for index, _ in enumerate(model.model.layers):
            for family, part in (("attention", "self_attn"), ("ffn", "mlp")):
                prefix = f"model.layers.{index}.{part}."
                selected = [target for target in targets if target["level"] == "leaf"
                            and target["name"].startswith(prefix) and target["kind"] == "linear"]
                if selected:
                    targets.append({"name": prefix[:-1], "modules": [t["name"] for t in selected],
                                    "kind": "linear", "level": "group", "block": index,
                                    "family": family, "projection": part,
                                    "parameters": sum(t["parameters"] for t in selected)})
    return targets


def load_packed(module, name, checkpoint, kind="linear"):
    packed = PackedLayer(module, kind, 128, "triton", initialize=False)
    for parameter in ("weight", "bias"):
        tensor = getattr(packed, parameter + "_q")
        if tensor is None:
            continue
        for field in ("codes", "scales"):
            getattr(tensor, field).copy_(checkpoint.get_tensor(f"{name}.{parameter}_q.{field}"))
    return packed


@contextmanager
def intervention(model, target, mode, checkpoint):
    """Quantize just the requested modules; restore even if scoring raises."""
    if mode not in MODES or (target["kind"] != "linear" and mode != "w4a16"):
        raise ValueError(f"Unsupported intervention {target['kind']}/{mode}")
    originals, handles = [], []
    try:
        for name in target["modules"]:
            module = model.get_submodule(name)
            if mode != "w16a4":
                packed = load_packed(module, name, checkpoint, target["kind"])
                for parameter in ("weight", "bias"):
                    value = getattr(module, parameter, None)
                    if value is not None:
                        originals.append((value, value.detach().cpu().clone()))
                        with torch.no_grad():
                            value.copy_(getattr(packed, parameter + "_q").dequantize())
                del packed
            if mode != "w4a16":
                def quantize_input(_module, args, kwargs):
                    if args:
                        return (activation_qdq(args[0]), *args[1:]), kwargs
                    kwargs = dict(kwargs)
                    kwargs["input"] = activation_qdq(kwargs["input"])
                    return args, kwargs
                handles.append(module.register_forward_pre_hook(quantize_input, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for parameter, original in originals:
                parameter.copy_(original)


def response_positions(labels, max_tokens=32):
    """Hidden position t predicts label t+1; evenly sample assistant labels only."""
    rows, values, counts = [], [], []
    for batch_index, label in enumerate(labels):
        positions = label[1:].ne(-100).nonzero().flatten()
        if not len(positions):
            raise ValueError("Every example must contain assistant target tokens")
        if len(positions) > max_tokens:
            positions = positions[torch.linspace(0, len(positions)-1, max_tokens).round().long()]
        rows.append(positions + batch_index * labels.shape[1])
        values.append(label[positions + 1])
        counts.append(len(positions))
    return torch.cat(rows), torch.cat(values), counts


def prepare_samples(args, tokenizer, model):
    official_imports()
    from tinyllava import conversation as conversation_lib
    from tinyllava.arguments import DataArguments
    from tinyllava.data.process import preprocess, preprocess_multimodal

    rows = json.loads(args.calibration.read_text())
    manifest = json.loads((ROOT / "data/manifests/calibration.json").read_text())
    if sha256(args.calibration) != manifest["sha256"] or manifest["eval_overlap"] != 0:
        raise ValueError("Expected verified calibration data disjoint from evaluation images")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = model.config.max_position_embeddings
    conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
    data_args = DataArguments(is_multimodal=True)
    data_args.mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    pools, skipped = defaultdict(list), Counter()
    for row in rows:
        sources = preprocess_multimodal(copy.deepcopy([row["conversations"]]), data_args)
        item = preprocess(sources, tokenizer, has_image=True)
        ids, labels = item["input_ids"][0], item["labels"][0]
        if int(ids.eq(-200).sum()) != 1:
            skipped["image_token_count"] += 1
            continue
        if len(ids) + 728 > args.max_length:
            skipped["too_long"] += 1
            continue
        if not bool(labels[1:].ne(-100).any()):
            skipped["no_assistant_labels"] += 1
            continue
        pools[(row["source"], row["length_bin"])].append((row, ids, labels))
    rng = random.Random(args.seed)
    for pool in pools.values():
        rng.shuffle(pool)
    chosen = []
    # Alternate each stratum between discovery and confirmation before reshuffling.
    while len(chosen) < args.samples and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(chosen) < args.samples:
                chosen.append(pools[key].pop())
    if len(chosen) != args.samples:
        raise ValueError(f"Only {len(chosen)} eligible calibration samples; skipped={dict(skipped)}")
    fold_counts, fold_totals, first_folds = Counter(), Counter(), {}
    samples = []
    for row, ids, labels in chosen:
        key = (row["source"], row["length_bin"])
        if fold_counts[key] % 2 == 0:
            first_folds[key] = min(("discovery", "confirmation"), key=lambda f: fold_totals[f])
            fold = first_folds[key]
        else:
            fold = "confirmation" if first_folds[key] == "discovery" else "discovery"
        fold_counts[key] += 1
        fold_totals[fold] += 1
        image = ROOT / row["image"]
        if sha256(image) != row["image_sha256"]:
            raise ValueError(f"Calibration image hash changed: {image}")
        samples.append({"id": row["id"], "image": row["image"], "image_sha256": row["image_sha256"],
                        "source": row["source"], "length_bin": row["length_bin"], "fold": fold,
                        "input_ids": ids, "labels": labels})
    rng.shuffle(samples)
    return samples, dict(skipped)


def fuse_cached_features(model, ids, mask, labels, features):
    """Reuse unchanged FP16 Vision/Projector, while still executing token embedding."""
    dummy = torch.empty((len(ids), 0, 0, 0), device=ids.device)
    with patch.object(model, "encode_images", return_value=features):
        _, positions, expanded_mask, _, embeds, expanded_labels = model.prepare_inputs_labels_for_multimodal(
            ids, None, mask, None, labels, dummy)
    return positions, expanded_mask, embeds, expanded_labels


@torch.inference_mode()
def selected_log_probs(model, batch, target=None, capture=False):
    if target is not None and target["name"] == "lm_head":
        logits = model.lm_head(batch["head_input"].cuda()).float()
        return F.log_softmax(logits, dim=-1)
    start = (len(model.model.layers) if target is not None and target["name"] == "model.norm"
             else target["block"] // 4 * 4 if target is not None and target["block"] is not None else 0)
    if target is not None and target["kind"] == "embedding":
        positions, mask, embeds, labels = fuse_cached_features(model,
            batch["input_ids"].cuda(), batch["input_mask"].cuda(), batch["input_labels"].cuda(),
            batch["features"].cuda())
        if not torch.equal(labels.cpu(), batch["expanded_labels"]):
            raise RuntimeError("Embedding intervention changed target alignment")
    else:
        positions = batch["positions"].cuda() if batch["positions"] is not None else None
        mask = batch["mask"].cuda()
        embeds = (batch["prefix_hidden"][start] if start else batch["embeds"]).cuda()
    layers = model.model.layers
    try:
        if start:
            model.model.layers = layers[start:]
        # All padding is on the right. A causal decoder cannot attend to future
        # padding at any scored position. Omitting its explicit mask enables SDPA
        # FlashAttention; the public masked forward is checked before scanning.
        hidden = model.model(inputs_embeds=embeds, position_ids=positions,
            attention_mask=None, use_cache=False, return_dict=True).last_hidden_state
    finally:
        model.model.layers = layers
    selected = hidden.reshape(-1, hidden.shape[-1]).index_select(0, batch["selected_rows"].cuda())
    if capture:
        batch["head_input"] = selected.cpu()
    logits = model.lm_head(selected).float()
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Non-finite diagnostic logits")
    return F.log_softmax(logits, dim=-1)


@torch.inference_mode()
def cache_batches(args, model, tokenizer, processor, samples):
    from PIL import Image
    from tinyllava.mm_utils import process_images

    batches, control = [], {}
    for start in range(0, len(samples), args.batch_size):
        selected = samples[start:start+args.batch_size]
        ids = nn.utils.rnn.pad_sequence([s["input_ids"] for s in selected], batch_first=True,
                                       padding_value=tokenizer.pad_token_id).cuda()
        labels = nn.utils.rnn.pad_sequence([s["labels"] for s in selected], batch_first=True,
                                          padding_value=-100).cuda()
        lengths = torch.tensor([len(s["input_ids"]) for s in selected], device="cuda")
        mask = torch.arange(ids.shape[1], device="cuda")[None, :] < lengths[:, None]
        pixels = []
        for sample in selected:
            with Image.open(ROOT / sample["image"]) as image:
                pixels.append(process_images([image.convert("RGB")], processor, model.config))
        pixels = torch.cat(pixels).cuda().half()
        features = model.encode_images(pixels)
        positions, expanded_mask, embeds, expanded_labels = fuse_cached_features(model, ids, mask, labels, features)
        rows, values, counts = response_positions(expanded_labels.cpu(), args.target_tokens)
        batch = {"input_ids": ids.cpu(), "input_mask": mask.cpu(), "input_labels": labels.cpu(),
                 "features": features.cpu(), "positions": positions.cpu() if positions is not None else None,
                 "mask": expanded_mask.cpu(), "embeds": embeds.cpu(), "expanded_labels": expanded_labels.cpu(),
                 "selected_rows": rows, "targets": values, "counts": counts,
                 "samples": [{k: s[k] for k in ("id", "fold")} for s in selected]}
        batch["prefix_hidden"] = {}
        with ExitStack() as stack:
            for index in [*range(4, len(model.model.layers), 4), len(model.model.layers)]:
                module = model.model.norm if index == len(model.model.layers) else model.model.layers[index]
                def capture_input(_module, args, key=index):
                    batch["prefix_hidden"][key] = args[0].detach().cpu().clone()
                handle = module.register_forward_pre_hook(capture_input)
                stack.callback(handle.remove)
            reference = selected_log_probs(model, batch, capture=True)
        batch["reference"] = reference.cpu()
        if start == 0:
            # Check the cached path against the public model forward, including Vision.
            full = model(input_ids=ids, attention_mask=mask, images=pixels, use_cache=False,
                         return_dict=True).logits
            public = F.log_softmax(full.reshape(-1, full.shape[-1])[rows.cuda()].float(), dim=-1)
            discrepancy = (reference.exp() * (reference - public)).sum(-1)
            control = {"public_forward_mean_kl": float(discrepancy.mean()),
                       "public_forward_max_abs_log_probability_difference": float((reference-public).abs().max()),
                       "checked_examples": len(selected)}
            if abs(control["public_forward_mean_kl"]) > 1e-4:
                raise RuntimeError(f"Cached teacher path does not match public forward: {control}")
            del full, public
            prefix_checks = {}
            for index in batch["prefix_hidden"]:
                target = {"name": "model.norm" if index == len(model.model.layers) else f"model.layers.{index}",
                          "kind": "rmsnorm" if index == len(model.model.layers) else "linear", "block": index}
                replay = selected_log_probs(model, batch, target)
                if not torch.equal(reference, replay):
                    raise RuntimeError(f"Cached prefix {index} does not reproduce FP16 outputs exactly")
                prefix_checks[str(index)] = "bit_exact_log_probabilities"
            if not torch.equal(reference, selected_log_probs(model, batch,
                {"name": "lm_head", "kind": "linear", "block": None})):
                raise RuntimeError("Cached lm_head input does not reproduce FP16 outputs")
            control["prefix_replay_checks"] = prefix_checks
            control["cached_lm_head_input"] = "bit_exact_log_probabilities"
        batches.append(batch)
        print(f"Cached FP16 calibration {min(start+args.batch_size, len(samples))}/{len(samples)}", flush=True)
        del ids, labels, mask, pixels, features, embeds, reference
    return batches, control


def score_batch(candidate, batch):
    reference = batch["reference"].to(candidate.device)
    kl = (reference.exp() * (reference - candidate)).sum(-1)
    minimum = float(kl.min())
    if minimum < -2e-5:
        raise RuntimeError(f"KL numerical error exceeds tolerance: {minimum}")
    kl = kl.clamp_min(0)
    targets = batch["targets"].to(candidate.device)
    nll = -candidate.gather(-1, targets[:, None]).squeeze(-1)
    reference_nll = -reference.gather(-1, targets[:, None]).squeeze(-1)
    agreement = candidate.argmax(-1).eq(reference.argmax(-1)).float()
    result, offset = [], 0
    for sample, count in zip(batch["samples"], batch["counts"]):
        section = slice(offset, offset + count)
        result.append({**sample, "tokens": count, "kl": float(kl[section].mean()),
                       "nll": float(nll[section].mean()), "reference_nll": float(reference_nll[section].mean()),
                       "nll_delta": float((nll-reference_nll)[section].mean()),
                       "argmax_agreement": float(agreement[section].mean())})
        offset += count
    return result, minimum


def aggregate(rows):
    output = {}
    for fold in ("all", "discovery", "confirmation"):
        subset = [r for r in rows if fold == "all" or r["fold"] == fold]
        if subset:
            output[fold] = {"examples": len(subset), "tokens": sum(r["tokens"] for r in subset),
                            **{key: statistics.mean(r[key] for r in subset)
                               for key in ("kl", "nll", "reference_nll", "nll_delta", "argmax_agreement")}}
    return output


@torch.inference_mode()
def score_model(model, batches, target=None):
    rows, minimum = [], 0.0
    for batch in batches:
        candidate = selected_log_probs(model, batch, target)
        scored, batch_minimum = score_batch(candidate, batch)
        rows.extend(scored)
        minimum = min(minimum, batch_minimum)
    return aggregate(rows), rows, minimum


def llm_fingerprint(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if name.startswith("model.layers.") or name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def validate_native(args, model, batches, results):
    if not args.native_topk:
        return {"status": "not_requested"}
    from w4a4 import NativeInt4Layer
    import int4_cuda

    candidates = sorted((r for r in results if r["mode"] == "w4a4" and r["target"]["level"] == "leaf"),
                        key=lambda r: r["metrics"].get("discovery", r["metrics"]["all"])["kl"], reverse=True)
    records = []
    with safe_open(CHECKPOINT / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        for row in candidates[:args.native_topk]:
            name = row["target"]["name"]
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            original = getattr(parent, child_name)
            native = NativeInt4Layer(load_packed(original, name, checkpoint))
            native.eval().requires_grad_(False)
            try:
                setattr(parent, child_name, native)
                metrics, examples, minimum = score_model(model, batches, row["target"])
                if native.calls == 0:
                    raise RuntimeError(f"Native validation layer was not called: {name}")
                records.append({"name": name, "qdq_metrics": row["metrics"], "native_metrics": metrics,
                                "native_calls": native.calls, "examples": examples,
                                "minimum_unclamped_token_kl": minimum})
                print(f"Native INT4 verification {name}: KL={metrics['all']['kl']:.6g}", flush=True)
            finally:
                setattr(parent, child_name, original)
                del native
    audit = {"status": "complete", "selection": "top W4A4 leaf targets by discovery KL",
             "kernel": int4_cuda.kernel_info(include_instructions=True), "results": records}
    write_json(args.output_dir / "native_checks.json", audit)
    # The replacement path must also restore the original FP16 model.
    control, _, _ = score_model(model, batches)
    if control["all"]["kl"] != 0 or control["all"]["nll_delta"] != 0:
        raise RuntimeError("Native validation did not restore FP16 outputs")
    return {"status": "complete", "layers": len(records), "checks": "native_checks.json"}


def ranked(results, mode, level="leaf", fold="discovery"):
    return sorted((r for r in results if r["mode"] == mode and r["target"]["level"] == level),
                  key=lambda r: r["metrics"].get(fold, r["metrics"]["all"])["kl"], reverse=True)


def write_report(directory, results, summary):
    import csv

    with (directory / "ranking.csv").open("w") as output:
        fields = ["name", "level", "block", "family", "projection", "mode", "parameters",
                  "discovery_kl", "confirmation_kl", "all_kl", "all_nll_delta", "all_argmax_agreement"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in results:
            target, metrics = row["target"], row["metrics"]
            writer.writerow({**{key: target[key] for key in fields[:5]}, "mode": row["mode"],
                "parameters": target["parameters"],
                **{fold+"_kl": metrics.get(fold, {}).get("kl", "") for fold in ("discovery", "confirmation", "all")},
                "all_nll_delta": metrics["all"]["nll_delta"], "all_argmax_agreement": metrics["all"]["argmax_agreement"]})
    lines = ["# LLM 레이어별 양자화 민감도", "",
             f"{summary['examples']}개 calibration 예제, {summary['cases']}개 단일 대상 개입을 완료했다.",
             f"집합 구성: {summary['fold_counts']}. 양자화 대상 이외의 Vision·Projector·LLM은 FP16으로 고정했다.",
             "각 개입은 하나의 leaf 또는 하나의 블록 안 Attention/FFN 묶음만 변경한다. 전체 INT4 모델에서 복구한 실험은 아니다.",
             "같은 정답 문맥에서 assistant 토큰만 평가한다. 이미지·사용자 prompt·padding 위치는 점수에서 제외한다.",
             "KL은 FP16 다음 토큰 분포와 비교한 값이며, NLL 변화는 정답 토큰에 대한 손실 차이다. 예제별 평균을 다시 평균한다.",
             "이 점수는 POPE F1·ScienceQA accuracy가 아니며, 최종 벤치마크 개선을 보장하지 않는다.",
             "W/A를 분리하기 위해 같은 FP16 연산에서 INT4 양자화·복원을 시뮬레이션했다. 이 실행의 시간·메모리는 속도 비교에 사용하지 않는다.",
             "A4는 Linear 입력에만 적용한다. RMSNorm·embedding의 A4 점수는 정의하지 않았다.",
             "가중치는 기존 group 128 RTN 체크포인트를 읽었으며 학습·LoRA·optimizer update는 없다.",
             "FP16 파라미터 복구와 개입 전후 FP16 출력 일치를 검증했다.", "",
             "## 개별 레이어 순위", "",
             "순위는 discovery 집합의 KL로 정하고, 별도 confirmation 집합의 값을 나란히 표시한다."]
    stability = {}
    for mode in MODES:
        ordering = ranked(results, mode)
        other = ranked(results, mode, fold="confirmation")
        top = min(10, len(ordering))
        overlap = len({r['target']['name'] for r in ordering[:top]} & {r['target']['name'] for r in other[:top]})
        stability[mode] = {"topk": top, "discovery_confirmation_overlap": overlap}
        lines += ["", f"### {mode.upper()} — 상위 {top}개", "",
                  f"두 집합의 top-{top} 공통 레이어는 {overlap}개다.", "",
                  "| 레이어 | Discovery KL | Confirmation KL | 전체 ΔNLL | 파라미터 M |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for row in ordering[:top]:
            metric = row["metrics"]
            lines.append(f"| `{row['target']['name']}` | {metric.get('discovery',metric['all'])['kl']:.6g} | "
                         f"{metric.get('confirmation',metric['all'])['kl']:.6g} | "
                         f"{metric['all']['nll_delta']:+.6g} | {row['target']['parameters']/1e6:.3f} |")
    lines += ["", "## 블록 안 Attention과 FFN 묶음", "",
              "각 블록의 Attention projection 전체 또는 FFN projection 전체를 함께 바꾼 경우다.",
              "FFN과 Attention은 파라미터 수가 다르므로 아래 원시 KL은 같은 저장 비용의 비교가 아니다.", "",
              "| 모드 | 묶음 | 블록 수 | KL 중앙값 | ΔNLL 중앙값 | 블록당 파라미터 M |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    group_statistics = []
    for mode in MODES:
        for family in ("attention", "ffn"):
            rows = [r for r in results if r['mode'] == mode and r['target']['level'] == 'group'
                    and r['target']['family'] == family]
            if not rows:
                continue
            record = {"mode": mode, "family": family, "blocks": len(rows),
                      "median_kl": statistics.median(r['metrics']['all']['kl'] for r in rows),
                      "median_nll_delta": statistics.median(r['metrics']['all']['nll_delta'] for r in rows),
                      "mean_parameters": statistics.mean(r['target']['parameters'] for r in rows)}
            group_statistics.append(record)
            lines.append(f"| {mode.upper()} | {family} | {len(rows)} | {record['median_kl']:.6g} | "
                         f"{record['median_nll_delta']:+.6g} | {record['mean_parameters']/1e6:.3f} |")
    lines += ["", "## 블록 위치별 비교", "",
              "개별 projection의 KL 중앙값이다. 앞(0–6), 중간(7–14), 뒤(15–21)를 비교하며 Norm·embedding·head는 제외한다.", "",
              "| 모드 | 위치 | 개별 projection 수 | KL 중앙값 |", "| --- | --- | ---: | ---: |"]
    bands = []
    for mode in MODES:
        for label, low, high in (("앞",0,6),("중간",7,14),("뒤",15,21)):
            rows = [r for r in results if r['mode'] == mode and r['target']['level'] == 'leaf'
                    and r['target']['kind'] == 'linear' and r['target']['block'] is not None
                    and low <= r['target']['block'] <= high]
            if rows:
                median = statistics.median(r['metrics']['all']['kl'] for r in rows)
                bands.append({"mode": mode, "band": label, "median_kl": median, "count": len(rows)})
                lines.append(f"| {mode.upper()} | {label} | {len(rows)} | {median:.6g} |")
    write_json(directory / "analysis.json", {"top10_stability": stability, "groups": group_statistics, "bands": bands})
    if summary['native']['status'] == 'complete':
        checks = json.loads((directory/'native_checks.json').read_text())
        lines += ["", "## 실제 INT4 커널 확인", "",
                  "Discovery W4A4 상위 leaf를 기존 native 커널로 교체해 같은 조건에서 재측정했다. 나머지 모델은 FP16이다.", "",
                  "| 레이어 | QDQ 전체 KL | Native 전체 KL |", "| --- | ---: | ---: |"]
        for row in checks['results']:
            lines.append(f"| `{row['name']}` | {row['qdq_metrics']['all']['kl']:.6g} | {row['native_metrics']['all']['kl']:.6g} |")
    plot_heatmaps(directory, results)
    lines += ["", "## Heatmap", "", "색은 최종 출력 KL이다. 큰 값일수록 이번 조건에서 출력 변화가 크다. 회색은 해당 조건을 적용하지 않은 칸이다.",
              "", "![개별 레이어 민감도](layers.png)", "", "![Attention·FFN 묶음 민감도](groups.png)",
              "", "원자료: `run.json`, `baseline.json`, `measurements.jsonl`, `ranking.csv`, `native_checks.json`, `summary.json`.",
              "집합 간 순위가 안정적인 대상을 후속 calibration 개선 후보로 삼고, 최종 설정은 별도 전체 평가로 확인한다."]
    (directory / "report.md").write_text("\n".join(lines)+"\n")


def plot_heatmaps(directory, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np

    for level, labels, filename in (
        ("leaf", ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","input_layernorm","post_attention_layernorm"], "layers"),
        ("group", ["self_attn","mlp"], "groups")):
        arrays = []
        for mode in MODES:
            values = np.full((22, len(labels)), np.nan)
            for row in results:
                target = row['target']
                if row['mode'] == mode and target['level'] == level and target['block'] is not None and target['projection'] in labels:
                    values[target['block'], labels.index(target['projection'])] = row['metrics']['all']['kl']
            arrays.append(values)
        positives = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in arrays])
        if not len(positives):
            continue
        floor, ceiling = max(float(positives.min()), 1e-8), float(positives.max())
        ceiling = max(ceiling, floor * 10)
        fig, axes = plt.subplots(1, 3, figsize=(18 if level == 'leaf' else 10, 10), layout="constrained")
        cmap = plt.get_cmap('magma').copy()
        cmap.set_bad('#d0d0d0')
        for ax, mode, values in zip(axes, MODES, arrays):
            display = np.where(np.isfinite(values), np.maximum(values, floor), np.nan)
            im = ax.imshow(display, aspect='auto', cmap=cmap, norm=LogNorm(floor, ceiling))
            ax.set_title(mode.upper())
            names = [x.replace('input_layernorm','input_norm').replace('post_attention_layernorm','post_norm') for x in labels]
            ax.set_xticks(range(len(labels)), names, rotation=55, ha='right')
            ax.set_yticks(range(22))
            ax.set_ylabel('LLM block index')
        fig.suptitle('Single-target sensitivity: mean KL to FP16 (FP16 QDQ simulation)')
        fig.colorbar(im, ax=axes, label='KL (log scale)', shrink=0.8)
        fig.savefig(directory / f'{filename}.png', dpi=160)
        fig.savefig(directory / f'{filename}.svg')
        plt.close(fig)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model, processor, _ = load()
    model.eval().requires_grad_(False)
    if model.config.pretraining_tp != 1 or model.config.num_hidden_layers != 22:
        raise ValueError("This experiment expects the public 22-block TinyLlama checkpoint")
    model.config.use_cache = False
    model.config.tokenizer_padding_side = "right"
    model.config.tokenizer_model_max_length = args.max_length
    targets = targets_for_model(model, not args.no_groups)
    if len([t for t in targets if t["level"] == "leaf"]) != 201:
        raise RuntimeError("Incomplete LLM layer coverage")
    if args.target_pattern:
        targets = [t for t in targets if re.search(args.target_pattern, t["name"])]
    if args.limit_targets:
        targets = targets[:args.limit_targets]
    if not targets:
        raise ValueError("No sensitivity targets selected")
    manifest = json.loads((CHECKPOINT / "manifest.json").read_text())
    if sha256(CHECKPOINT / "model.safetensors") != manifest["checkpoint_sha256"]:
        raise ValueError("Original RTN checkpoint hash mismatch")
    samples, skipped = prepare_samples(args, tokenizer, model)
    sample_manifest = [{k: s[k] for k in ("id", "image", "image_sha256", "source", "length_bin", "fold")} for s in samples]
    sources = [Path(__file__), ROOT / "scripts/int4.py", ROOT / "scripts/runtime.py",
               ROOT / "third_party/TinyLLaVA_Factory_legacy/tinyllava/model/llava_arch.py",
               ROOT / "third_party/TinyLLaVA_Factory_legacy/tinyllava/model/language_model/llava_llama.py"]
    import transformers.models.llama.modeling_llama as llama
    sources.append(Path(llama.__file__))
    spec = {"format": FORMAT, "samples": sample_manifest, "skipped": skipped,
            "calibration_sha256": sha256(args.calibration), "seed": args.seed,
            "batch_size": args.batch_size, "max_length": args.max_length,
            "max_target_tokens_per_example": args.target_tokens, "targets": targets,
            "group_size": 128, "activation_group": "gcd(in_features,128)",
            "checkpoint_sha256": manifest["checkpoint_sha256"], "model_revisions": manifest["model_revisions"],
            "finetuning_steps": 0, "arithmetic": "FP16_QDQ_simulation_with_same_GEMM_path",
            "context": "all non-target modules FP16; cached unchanged FP16 Vision/Projector and prefixes at stride 4",
            "attention_mask_policy": "causal; right padding omitted from attention mask; only valid response positions scored",
            "temperature": 1, "score": "per-example mean over evenly selected assistant target positions",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "source_hashes": {str(p): sha256(p) for p in sources},
            "native_topk": args.native_topk}
    spec_path = args.output_dir / "run.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Sensitivity run conditions changed; use a separate output directory")
    write_json(spec_path, spec)
    write_json(args.output_dir / "status.json", {"time": now(), "status": "running", "targets": len(targets)})
    original_fingerprint = llm_fingerprint(model)
    batches, public_control = cache_batches(args, model, tokenizer, processor, samples)
    baseline, _, _ = score_model(model, batches)
    if baseline["all"]["kl"] != 0 or baseline["all"]["nll_delta"] != 0:
        raise RuntimeError(f"FP16 repeat is not deterministic: {baseline}")
    write_json(args.output_dir / "baseline.json", {"time": now(), "metrics": baseline,
               "public_forward_control": public_control, "llm_parameter_sha256": original_fingerprint})
    path = args.output_dir / "measurements.jsonl"
    results = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r["target"]["name"], r["mode"]) for r in results}
    cases = [(target, mode) for target in targets for mode in MODES
             if target["kind"] == "linear" or mode == "w4a16"]
    start = time.perf_counter()
    with safe_open(CHECKPOINT / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        for index, (target, mode) in enumerate(cases):
            if (target["name"], mode) in done:
                continue
            step = time.perf_counter()
            with intervention(model, target, mode, checkpoint):
                metrics, rows, minimum = score_model(model, batches, target)
            result = {"target": target, "mode": mode, "metrics": metrics,
                      "examples": rows, "minimum_unclamped_token_kl": minimum,
                      "diagnostic_seconds": time.perf_counter() - step}
            with path.open("a") as output:
                output.write(json.dumps(result) + "\n")
            results.append(result)
            print(f"[{index+1}/{len(cases)}] {target['name']} {mode}: "
                  f"KL={metrics['all']['kl']:.6g}, NLL_delta={metrics['all']['nll_delta']:+.6g}, "
                  f"{result['diagnostic_seconds']:.2f}s", flush=True)
    if len(results) != len(cases) or len({(r['target']['name'],r['mode']) for r in results}) != len(cases):
        raise RuntimeError("Missing or duplicate sensitivity interventions")
    final_fingerprint = llm_fingerprint(model)
    if final_fingerprint != original_fingerprint:
        raise RuntimeError("FP16 model parameters were not restored")
    final_control, _, _ = score_model(model, batches)
    if final_control != baseline:
        raise RuntimeError("FP16 outputs changed after restoring all interventions")
    native = validate_native(args, model, batches, results)
    summary = {"time": now(), "status": "complete", "targets": len(targets), "cases": len(cases),
               "examples": len(samples), "fold_counts": dict(Counter(s["fold"] for s in samples)),
               "baseline": baseline, "model_parameters_restored": True,
               "fp16_control_repeated_exactly": True, "llm_parameter_sha256": final_fingerprint,
               "finetuning_steps": 0, "native": native,
               "diagnostic_elapsed_seconds_this_process": time.perf_counter()-start}
    write_json(args.output_dir / "summary.json", summary)
    write_report(args.output_dir, results, summary)
    write_json(args.output_dir / "status.json", {"time": now(), "status": "complete",
               "cases": len(cases), "finetuning_steps": 0})
    print(f"Verified sensitivity report: {args.output_dir / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--target-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration", type=Path, default=ROOT / "data/calibration/calibration.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/sensitivity_llm")
    parser.add_argument("--target-pattern")
    parser.add_argument("--limit-targets", type=int, default=0)
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--native-topk", type=int, default=3)
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.target_tokens) < 1 or args.native_topk < 0:
        parser.error("Sample, batch and target-token counts must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
