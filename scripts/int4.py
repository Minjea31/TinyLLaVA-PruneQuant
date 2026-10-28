"""Uniform packed INT4 RTN for every TinyLLaVA parameter, with FP16 compute.

All Linear, Embedding, Conv2d, LayerNorm and LlamaRMSNorm weights AND biases are
quantized. Activations, KV cache and arithmetic buffers retain the original
floating-point behavior. This is a dequantize-then-compute reference backend,
not an optimized integer GEMM implementation and not NF4/FP4 quantization.
"""

from collections import Counter
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common import now, sha256, write_json


FORMAT = "uniform_symmetric_int4_rtn_all_parameters_v1"


def component(name):
    if ".vision_tower." in name:
        return "vision"
    if ".mm_projector." in name:
        return "projector"
    return "llm"


class PackedTensor(nn.Module):
    def __init__(self, tensor, group_size=128, backend="triton", initialize=True):
        super().__init__()
        if group_size < 1 or tensor.numel() < 1:
            raise ValueError("Need a nonempty tensor and positive group size")
        self.shape = tuple(tensor.shape)
        self.numel = tensor.numel()
        self.group_size = group_size
        self.output_dtype = tensor.dtype
        self.backend = backend
        self.register_buffer("codes", torch.empty((self.numel + 1) // 2,
                                                  device=tensor.device, dtype=torch.uint8))
        self.register_buffer("scales", torch.empty(math.ceil(self.numel / group_size),
                                                   device=tensor.device, dtype=torch.float16))
        if initialize:
            self.pack(tensor)

    @torch.no_grad()
    def pack(self, tensor):
        # Compute RTN on CPU in FP32. Quantize using the actual stored FP16
        # scale, so both reference and Triton reconstruct identical values.
        values = tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Cannot quantize non-finite model weights")
        padded = F.pad(values, (0, (-self.numel) % self.group_size))
        groups = padded.view(-1, self.group_size)
        maximum = groups.abs().amax(dim=1)
        scales = (maximum / 7).clamp(min=2**-24).half()
        scales[maximum == 0] = 1
        if not bool(torch.isfinite(scales).all()):
            raise ValueError("INT4 scale cannot be represented in FP16")
        integers = (groups / scales.float().unsqueeze(1)).round().clamp_(-7, 7)
        codes = (integers.reshape(-1)[:self.numel] + 8).to(torch.uint8)
        if self.numel % 2:
            codes = F.pad(codes, (0, 1), value=8)
        packed = codes[0::2] | (codes[1::2] << 4)
        self.codes.copy_(packed)
        self.scales.copy_(scales)

    def dequantize(self):
        if self.codes.is_cuda and self.backend == "triton":
            from int4_kernel import unpack_int4
            return unpack_int4(self.codes, self.scales, self.numel,
                               self.group_size, self.output_dtype).view(self.shape)
        codes = torch.stack((self.codes & 15, self.codes >> 4), dim=1).reshape(-1)[:self.numel]
        scales = self.scales.repeat_interleave(self.group_size)[:self.numel]
        return ((codes.to(torch.float32) - 8) * scales.float()).to(self.output_dtype).view(self.shape)

    def storage_bytes(self):
        return self.codes.numel() + self.scales.numel() * 2


class PackedLayer(nn.Module):
    """Preserve leaf-layer semantics while storing only packed parameters."""

    def __init__(self, original, kind, group_size, backend, initialize):
        super().__init__()
        self.kind = kind
        # Upstream model.device/dtype and SigLipVisionTower.dtype inspect
        # parameters. This zero-element anchor carries metadata, no weights.
        self.register_parameter("_dtype_anchor", nn.Parameter(
            torch.empty(0, device=original.weight.device, dtype=original.weight.dtype),
            requires_grad=False,
        ))
        self.weight_q = PackedTensor(original.weight, group_size, backend, initialize)
        self.bias_q = (PackedTensor(original.bias, group_size, backend, initialize)
                       if getattr(original, "bias", None) is not None else None)
        if kind == "linear":
            self.in_features, self.out_features = original.in_features, original.out_features
        elif kind == "embedding":
            self.num_embeddings, self.embedding_dim = original.num_embeddings, original.embedding_dim
            self.padding_idx = original.padding_idx
            self.max_norm, self.norm_type = original.max_norm, original.norm_type
            self.scale_grad_by_freq, self.sparse = original.scale_grad_by_freq, original.sparse
            if self.max_norm is not None:
                raise ValueError("Embedding max_norm would mutate dequantized weights")
        elif kind == "conv2d":
            for attr in ("stride", "padding", "dilation", "groups", "padding_mode"):
                setattr(self, attr, getattr(original, attr))
            if self.padding_mode != "zeros":
                raise ValueError("Only zero-padding Conv2d is supported")
        elif kind == "layernorm":
            self.normalized_shape, self.eps = original.normalized_shape, original.eps
        elif kind == "rmsnorm":
            self.variance_epsilon = original.variance_epsilon

    @property
    def weight(self):
        return self.weight_q.dequantize()

    @property
    def bias(self):
        return self.bias_q.dequantize() if self.bias_q is not None else None

    def forward(self, inputs):
        if self.kind == "linear":
            return F.linear(inputs, self.weight, self.bias)
        if self.kind == "embedding":
            return F.embedding(inputs, self.weight, self.padding_idx, self.max_norm,
                               self.norm_type, self.scale_grad_by_freq, self.sparse)
        if self.kind == "conv2d":
            return F.conv2d(inputs, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        if self.kind == "layernorm":
            return F.layer_norm(inputs, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.kind == "rmsnorm":
            dtype = inputs.dtype
            values = inputs.float()
            variance = values.pow(2).mean(-1, keepdim=True)
            values = values * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * values.to(dtype)
        raise RuntimeError(f"Unsupported packed operation {self.kind}")


def convert_model(model, group_size=128, backend="triton", initialize=True):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    original = {name: {"shape": list(p.shape), "numel": p.numel(),
                       "component": component(name)} for name, p in model.named_parameters()}
    converted = set()
    counts = Counter()
    tensors = []
    supported = ((nn.Linear, "linear"), (nn.Embedding, "embedding"),
                 (nn.Conv2d, "conv2d"), (nn.LayerNorm, "layernorm"),
                 (LlamaRMSNorm, "rmsnorm"))

    def replace(parent, prefix=""):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            kind = next((label for cls, label in supported if isinstance(child, cls)), None)
            if kind is None:
                replace(child, full_name)
                continue
            replacement = PackedLayer(child, kind, group_size, backend, initialize)
            replacement.train(child.training)
            setattr(parent, name, replacement)
            counts[kind] += 1
            for parameter in ("weight", "bias"):
                if getattr(child, parameter, None) is not None:
                    key = f"{full_name}.{parameter}"
                    packed = getattr(replacement, parameter + "_q")
                    tensors.append({"name": key, **original[key],
                                    "packed_bytes": packed.storage_bytes()})
                    converted.add(key)

    replace(model)
    remaining = [n for n, p in model.named_parameters() if p.numel()]
    if remaining or converted != set(original):
        raise RuntimeError(f"Incomplete all-parameter INT4 conversion: remaining={remaining}, "
                           f"unconverted={sorted(set(original) - converted)}")
    model.eval().requires_grad_(False)
    components = {}
    for label in ("vision", "projector", "llm"):
        selected = [row for row in tensors if row["component"] == label]
        components[label] = {"parameters": sum(row["numel"] for row in selected),
                             "parameter_tensors": len(selected),
                             "packed_bytes": sum(row["packed_bytes"] for row in selected)}
    return {"format": FORMAT, "quantization": "RTN", "group_size": group_size,
            "integer_range": [-7, 7], "nibble_encoding": "signed_value_plus_8",
            "scale_dtype": "float16", "weight_bits": 4, "activation_dtype": "float16",
            "coverage": "all_model_parameters_including_embeddings_conv_norm_and_bias",
            "unquantized_parameter_tensors": [], "finetuning_steps": 0,
            "calibration": "none; weight-only absolute-maximum RTN",
            "backend": backend, "compute": "dequantize_per_layer_then_original_FP16_operators",
            "optimized_integer_matmul": False, "layers": dict(counts),
            "components": components, "tensors": sorted(tensors, key=lambda row: row["name"]),
            "total_parameters": sum(row["numel"] for row in tensors),
            "total_packed_bytes": sum(row["packed_bytes"] for row in tensors)}


def prepare_checkpoint(model, checkpoint, group_size, backend, revisions):
    from safetensors.torch import load_file, save_file

    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / "manifest.json"
    original_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        if checkpoint.exists():
            manifest = json.loads(manifest_path.read_text())
            if (manifest["format"] != FORMAT or manifest["group_size"] != group_size
                    or manifest["model_revisions"] != revisions):
                raise ValueError("INT4 checkpoint settings differ; use a new checkpoint directory")
            if sha256(checkpoint / "model.safetensors") != manifest["checkpoint_sha256"]:
                raise ValueError("INT4 checkpoint hash mismatch")
            audit = convert_model(model, group_size, backend, initialize=False)
            if audit["tensors"] != manifest["tensors"]:
                raise ValueError("INT4 checkpoint topology does not match the model")
            state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
            model.load_state_dict(state, strict=True)
            del state
            manifest["backend"] = backend
            return manifest
        audit = convert_model(model, group_size, backend)
        checkpoint.mkdir(parents=True, exist_ok=False)
        state = {n: t.detach().cpu().contiguous() for n, t in model.state_dict().items()}
        save_file(state, str(checkpoint / "model.safetensors"))
        del state
        audit.update(time=now(), model_revisions=revisions,
                     checkpoint_sha256=sha256(checkpoint / "model.safetensors"),
                     checkpoint_file_bytes=(checkpoint / "model.safetensors").stat().st_size)
        write_json(manifest_path, audit)
        return audit
    finally:
        torch.set_num_threads(original_threads)
