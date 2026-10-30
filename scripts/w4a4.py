"""Real INT4 multiplication for packed RTN Linear and patch Conv2d layers.

Original weight integer values and scales are retained exactly. Native tile
padding handles flat groups that cross a row boundary, without requantization.
Activations use a dynamic absmax scale per row and original-scale-aligned K group.
Attention QK/PV, softmax, normalization, RoPE and embedding lookup stay original.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from int4 import PackedLayer
import int4_cuda


FORMAT = "rtn_original_weights_dynamic_group_a4_native_wmma_v2"


class NativePackedWeight(nn.Module):
    def __init__(self, original):
        super().__init__()
        if not original.codes.is_cuda or original.output_dtype != torch.float16:
            raise ValueError("Native INT4 weights require an existing CUDA FP16 RTN checkpoint")
        self.shape = original.shape
        self.n = self.shape[0]
        self.k = math.prod(self.shape[1:])
        self.weight_group = original.group_size
        # This subdivision retains every original flat-group weight scale.
        self.group = math.gcd(self.k, self.weight_group)
        self.physical_group = (self.group + 31) // 32 * 32
        codes = int4_cuda.repack_weight(original.codes, self.n, self.k, self.group, self.physical_group)
        self.register_buffer("codes", codes)
        self.register_buffer("scales", original.scales)
        self.original_packed_bytes = original.codes.numel() + original.scales.numel() * 2

    def integer_values(self):
        """Inspection/reference only; the native forward never calls this."""
        values = torch.stack((self.codes & 15, self.codes >> 4), dim=-1).reshape(self.codes.shape[0], -1)
        values = values.to(torch.int8)
        values = torch.where(values >= 8, values - 16, values)
        values = values.reshape(values.shape[0], -1, self.physical_group)[:, :, :self.group]
        return values.reshape(values.shape[0], self.k)[:self.n]

    def dequantize(self):
        """Explicit inspection API, not used as a GEMM fallback."""
        values = self.integer_values().float().reshape(-1)
        indices = torch.arange(values.numel(), device=values.device) // self.weight_group
        return (values * self.scales[indices].float()).half().reshape(self.shape)

    def forward(self, inputs, bias=None):
        if not inputs.is_cuda or inputs.dtype != torch.float16 or inputs.device != self.codes.device:
            raise ValueError("Native W4A4 requires FP16 activation input on the weight's CUDA device")
        if self.scales.dtype != torch.float16:
            raise ValueError("Native W4A4 weight scales must remain FP16")
        if bias is not None and (bias.dtype != torch.float16 or bias.device != inputs.device
                                 or bias.shape != (self.n,) or not bias.is_contiguous()):
            raise ValueError("Native W4A4 bias must be a contiguous FP16 output vector on the same device")
        if inputs.shape[-1] != self.k:
            raise ValueError(f"Expected input last dimension {self.k}; got {inputs.shape}")
        if torch.is_grad_enabled() and inputs.requires_grad:
            raise RuntimeError("The native INT4 backend is inference-only; no autograd implementation")
        shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.k).contiguous()
        if not flat.shape[0]:
            return torch.empty((*shape, self.n), device=inputs.device, dtype=inputs.dtype)
        a, scales = int4_cuda.quantize_activation(flat, self.group, self.physical_group)
        output = int4_cuda.scaled_gemm(a, self.codes, scales, self.scales, bias,
                                       flat.shape[0], self.n, self.k,
                                       self.weight_group, self.group, self.physical_group)
        return output.reshape(*shape, self.n)


class NativeInt4Layer(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.kind = original.kind
        self.register_parameter("_dtype_anchor", original._dtype_anchor)
        self.weight_q = NativePackedWeight(original.weight_q)
        self.bias_q = original.bias_q
        self.calls = 0
        if self.kind == "linear":
            self.in_features, self.out_features = original.in_features, original.out_features
        elif self.kind == "conv2d":
            for attribute in ("stride", "padding", "dilation", "groups", "padding_mode"):
                setattr(self, attribute, getattr(original, attribute))
            if self.groups != 1 or self.padding_mode != "zeros" or self.padding == "same":
                raise ValueError("Native patch Conv2d supports groups=1 and zero/valid padding")
            self.kernel_size = original.weight_q.shape[-2:]
        else:
            raise ValueError(f"Native INT4 weighted GEMM does not implement {self.kind}")

    @property
    def weight(self):
        return self.weight_q.dequantize()

    @property
    def bias(self):
        return self.bias_q.dequantize() if self.bias_q is not None else None

    def forward(self, inputs):
        self.calls += 1
        if self.kind == "linear":
            return self.weight_q(inputs, self.bias)
        padding = (0, 0) if self.padding == "valid" else self.padding
        patches = F.unfold(inputs, kernel_size=self.kernel_size, dilation=self.dilation,
                           padding=padding, stride=self.stride).transpose(1, 2)
        output = self.weight_q(patches, self.bias).transpose(1, 2)
        height, width = inputs.shape[-2:]
        out_h = (height + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_w = (width + 2 * padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return output.reshape(inputs.shape[0], self.weight_q.n, out_h, out_w).contiguous()


def enable_w4a4(model):
    """Replace every packed Linear/Conv2d, refusing silent FP16 GEMM fallback."""
    if getattr(getattr(model, "config", None), "pretraining_tp", 1) != 1:
        raise ValueError("Native W4A4 requires pretraining_tp=1; sliced F.linear would bypass this backend")
    converted = []

    def replace(parent, prefix=""):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, PackedLayer) and child.kind in ("linear", "conv2d"):
                native = NativeInt4Layer(child)
                native.eval().requires_grad_(False)
                setattr(parent, name, native)
                converted.append((full_name, native))
            else:
                replace(child, full_name)

    replace(model)
    if not converted:
        raise ValueError("Load and convert the packed RTN checkpoint before enabling native W4A4")
    if any(isinstance(layer, PackedLayer) and layer.kind in ("linear", "conv2d") for layer in model.modules()):
        raise RuntimeError("A weighted GEMM layer was left on the FP16 fallback")
    model.eval().requires_grad_(False)
    info = int4_cuda.kernel_info(next(model.parameters()).device, include_instructions=True)
    return {"format": FORMAT, "kernel": info, "finetuning_steps": 0,
            "weight_values_preserved_from_w4a16": True,
            "activation_quantization": "dynamic_per_row_and_K_segment_absmax_symmetric_-7_to_7_fp16_scale",
            "activation_segment_size": "gcd(in_features, original_weight_group_size); <=128 at g128",
            "activation_quantization_scope": "all_linear_inputs_and_unfolded_conv_patches",
            "weighted_gemm_compute": "signed_int4_times_signed_int4_accumulate_int32",
            "remaining_operations": "original_FP16_FP32_attention_QK_PV_softmax_norm_RoPE_embedding",
            "linear_layers": sum(layer.kind == "linear" for _, layer in converted),
            "conv2d_layers": sum(layer.kind == "conv2d" for _, layer in converted),
            "native_packed_weight_bytes": sum(layer.weight_q.codes.numel() + layer.weight_q.scales.numel() * 2 for _, layer in converted),
            "tile_padding_extra_bytes": sum(layer.weight_q.codes.numel() + layer.weight_q.scales.numel() * 2 - layer.weight_q.original_packed_bytes for _, layer in converted),
            "layer_names": [name for name, _ in converted]}


def execution_counts(model):
    layers = {name: layer.calls for name, layer in model.named_modules() if isinstance(layer, NativeInt4Layer)}
    return {"current_process_total_calls": sum(layers.values()),
            "current_process_called_layers": sum(value > 0 for value in layers.values()),
            "native_layer_count": len(layers)}
