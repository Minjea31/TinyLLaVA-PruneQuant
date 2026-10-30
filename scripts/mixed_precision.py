"""Independent FP16 weight/input exceptions over the unchanged native W4A4 base.

Only remaining W4A4 Linear/Conv modules use S4xS4 IMMA. W16A4 uses dynamic
activation QDQ then FP16 GEMM; W4A16 uses the existing packed-weight DQ backend.
Selected FP16 weights come from the public model BEFORE any RTN conversion.
"""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, sha256
from int4 import PackedLayer
from w4a4 import NativeInt4Layer, execution_counts
from analyze_llm_sensitivity import activation_qdq
import int4_cuda


FORMAT = "kl_separate_weight_activation_fp16_exceptions_v1"
DEFAULT_POLICY = ROOT / "results/sensitivity_llm/fp16_threshold_candidates.json"
VARIANTS = ("none", "weights", "activations", "both")


def read_policy(path=DEFAULT_POLICY):
    policy = json.loads(Path(path).read_text())
    measurements = ROOT / "results/sensitivity_llm/measurements.jsonl"
    if sha256(measurements) != policy["measurements_sha256"]:
        raise ValueError("Sensitivity measurements changed after candidate selection")
    rows = [json.loads(line) for line in measurements.read_text().splitlines()]
    for category, mode in (("weight", "w4a16"), ("activation", "w16a4")):
        rule = policy["rules"][category]
        if rule["metric_mode"] != mode or not 0 < rule["threshold"] < float("inf"):
            raise ValueError(f"Invalid {category} selection rule")
        expected = {r["target"]["name"] for r in rows if r["mode"] == mode and r["target"]["level"] == "leaf"
                    and all(r["metrics"][fold]["kl"] >= rule["threshold"]
                            for fold in ("discovery", "confirmation"))}
        candidates = policy["candidate_layers"][category]
        names = [r["name"] for r in candidates]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError(f"Policy {category} targets do not match the measured thresholds")
    return policy


def selected_names(policy, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown precision variant {variant}")
    weights = {r["name"] for r in policy["candidate_layers"]["weight"]} if variant in ("weights", "both") else set()
    activations = {r["name"] for r in policy["candidate_layers"]["activation"]} if variant in ("activations", "both") else set()
    return weights, activations


def capture_fp16(model, policy, variant):
    """CPU copies of public FP16 modules; call BEFORE prepare_checkpoint."""
    weights, activations = selected_names(policy, variant)
    for name in activations:
        if not isinstance(model.get_submodule(name), nn.Linear):
            raise ValueError(f"A16 exceptions apply only to measured Linear inputs: {name}")
    originals = {}
    for name in sorted(weights):
        module = model.get_submodule(name)
        if not hasattr(module, "weight") or module.weight.dtype != torch.float16:
            raise ValueError(f"Expected original FP16 module: {name}")
        if isinstance(module, (PackedLayer, NativeInt4Layer)):
            raise ValueError("Cannot reconstruct original FP16 weights from a quantized module")
        originals[name] = copy.deepcopy(module).cpu().eval().requires_grad_(False)
    return originals


class FloatWeightLinear(nn.Module):
    def __init__(self, original, activation_bits, group_size=128):
        super().__init__()
        if not isinstance(original, nn.Linear) or activation_bits not in (4, 16):
            raise ValueError("Expected an original Linear and A4/A16")
        self.weight, self.bias = original.weight, original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        self.activation_bits, self.group_size = activation_bits, group_size
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        if self.activation_bits == 4:
            inputs = activation_qdq(inputs, self.group_size)
        return F.linear(inputs, self.weight, self.bias)


class PackedWeightA16Linear(nn.Module):
    def __init__(self, packed):
        super().__init__()
        if not isinstance(packed, PackedLayer) or packed.kind != "linear":
            raise ValueError("Expected a packed Linear")
        self.packed = packed
        self.in_features, self.out_features = packed.in_features, packed.out_features
        self.calls = 0

    @property
    def weight(self):
        return self.packed.weight

    @property
    def bias(self):
        return self.packed.bias

    def forward(self, inputs):
        self.calls += 1
        return self.packed(inputs)


def enable_mixed_precision(model, policy, variant, originals):
    weights, activations = selected_names(policy, variant)
    if getattr(model.config, "pretraining_tp", 1) != 1:
        raise ValueError("Mixed precision requires pretraining_tp=1; sliced F.linear bypasses wrappers")
    if set(originals) != weights:
        raise ValueError("Missing original FP16 module snapshots")
    packed = {name: layer for name, layer in model.named_modules() if isinstance(layer, PackedLayer)}
    if not (weights | activations) <= set(packed):
        raise ValueError("A policy target is missing from the packed model")
    restored, weighted = {}, []
    expected_fp16_parameters = 0
    for name, layer in packed.items():
        parent_name, key = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        if name in weights:
            original = originals[name].to(device=layer.weight_q.codes.device)
            if tuple(original.weight.shape) != tuple(layer.weight_q.shape):
                raise ValueError(f"Original FP16 shape mismatch: {name}")
            expected_fp16_parameters += sum(p.numel() for p in original.parameters())
            restored[name] = {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                              for n, p in original.named_parameters()}
            if layer.kind == "linear":
                replacement = FloatWeightLinear(original, 16 if name in activations else 4,
                                                layer.weight_q.group_size)
                precision = "W16A16" if name in activations else "W16A4"
            elif layer.kind in ("rmsnorm", "layernorm", "embedding"):
                replacement = original
                precision = "W16_original_non_GEMM"
            else:
                raise ValueError(f"FP16 exception kind not implemented: {layer.kind}")
        elif name in activations:
            replacement = PackedWeightA16Linear(layer)
            precision = "W4A16"
        elif layer.kind in ("linear", "conv2d"):
            replacement = NativeInt4Layer(layer)
            precision = "W4A4"
        else:
            continue
        replacement.eval().requires_grad_(False)
        setattr(parent, key, replacement)
        if layer.kind in ("linear", "conv2d"):
            weighted.append({"name": name, "kind": layer.kind, "precision": precision,
                             "backend": "cuda_wmma_s4" if precision == "W4A4" else "original_FP16_GEMM"})
    expected_weighted = sum(layer.kind in ("linear", "conv2d") for layer in packed.values())
    if len(weighted) != expected_weighted:
        raise RuntimeError("Incomplete mixed-precision weighted-layer coverage")
    actual_fp16_parameters = sum(p.numel() for p in model.parameters())
    if actual_fp16_parameters != expected_fp16_parameters:
        raise RuntimeError("Unexpected unquantized or missing FP16 parameters")
    counts = Counter(row["precision"] for row in weighted)
    model.eval().requires_grad_(False)
    return {"format": FORMAT, "variant": variant, "finetuning_steps": 0,
            "fp16_weight_layers": sorted(weights), "fp16_activation_layers": sorted(activations),
            "weighted_layer_count": len(weighted), "weighted_layer_precision_counts": dict(counts),
            "native_layer_count": counts["W4A4"], "weighted_layers": weighted,
            "fp16_parameter_values": actual_fp16_parameters,
            "fp16_weight_source": "public FP16 model captured before RTN conversion",
            "fp16_parameter_sha256": restored,
            "remaining_weight_source": "unchanged original packed RTN INT4 checkpoint",
            "W16A4_compute": "dynamic A4 QDQ then FP16 GEMM with original FP16 weight",
            "W4A16_compute": "packed W4 dequantization then FP16 GEMM with original activation",
            "W4A4_compute": "native signed INT4 multiply / INT32 accumulation",
            "other_operations": "original attention QK/PV, Softmax, RoPE and non-GEMM arithmetic",
            "kernel": int4_cuda.kernel_info(next(model.parameters()).device, include_instructions=True)}


def mixed_execution_counts(model):
    exceptions = {}
    for name, module in model.named_modules():
        if isinstance(module, (FloatWeightLinear, PackedWeightA16Linear)):
            exceptions[name] = {"calls": module.calls,
                "precision": f"W16A{module.activation_bits}" if isinstance(module, FloatWeightLinear) else "W4A16"}
    return {"native": execution_counts(model), "exceptions": exceptions}
