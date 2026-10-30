"""Native S4 IMMA parity, original flat-group preservation and multimodal use."""

import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from int4 import PackedLayer, convert_model
from int4_cuda import raw_gemm, kernel_info
from w4a4 import NativeInt4Layer, NativePackedWeight, enable_w4a4, execution_counts


def pack_signed(values):
    codes = (values & 15).to(torch.uint8).contiguous()
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def reference_linear(inputs, original_weight, original_scales, group_size, bias=None):
    """Independent CPU INT32 dot products; original flat scales, no float GEMM."""
    n, k = original_weight.shape
    shape = inputs.shape[:-1]
    x = inputs.detach().reshape(-1, k).cpu().float()
    qw = original_weight.cpu().int()
    ws = original_scales.cpu().float()
    group = math.gcd(k, group_size)
    result = torch.zeros(x.shape[0], n)
    for start in range(0, k, group):
        values = x[:, start:start+group]
        maximum = values.abs().amax(1)
        scale_x = (maximum / 7).clamp_min(2**-24).half()
        scale_x[maximum == 0] = 1
        quantized_x = (values / scale_x.float()[:, None]).round().clamp(-7, 7).int()
        dot = quantized_x @ qw[:, start:start+group].T
        scale = ws[(torch.arange(n) * k + start) // group_size]
        result += dot.float() * scale * scale_x.float()[:, None]
    if bias is not None:
        result += bias.detach().cpu().float()
    return result.half().to(inputs.device).reshape(*shape, n)


def original_integer_weight(packed):
    codes = torch.stack((packed.codes & 15, packed.codes >> 4), dim=-1).reshape(-1)[:packed.numel]
    return (codes.to(torch.int8) - 8).reshape(packed.shape[0], -1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for actual INT4 Tensor Cores")
class W4A4Tests(unittest.TestCase):
    def test_raw_int32_gemm_exact_with_negative_extremes_and_tail_tiles(self):
        for m, n, k in ((1, 1, 32), (13, 19, 96), (16, 24, 256)):
            torch.manual_seed(k)
            a = torch.randint(-8, 8, ((m+7)//8*8, k), dtype=torch.int8, device="cuda")
            b = torch.randint(-8, 8, ((n+7)//8*8, k), dtype=torch.int8, device="cuda")
            actual = raw_gemm(pack_signed(a), pack_signed(b), m, n)
            expected = a.cpu().int() @ b.cpu().int().T
            torch.testing.assert_close(actual.cpu(), expected[:m, :n], rtol=0, atol=0)
        self.assertTrue(kernel_info(include_instructions=True)["imma_instructions"])

    def test_linear_flat_groups_bias_zero_and_no_dequantized_weight_fallback(self):
        for m, n, k in ((3, 11, 128), (9, 17, 80), (1, 7, 33), (8, 16, 4304)):
            torch.manual_seed(k)
            packed = PackedLayer(nn.Linear(k, n, bias=True).cuda().half(), "linear", 128, "triton", True)
            qw = original_integer_weight(packed.weight_q).clone()
            scales, bias = packed.weight_q.scales.clone(), packed.bias.clone()
            native = NativeInt4Layer(packed)
            torch.testing.assert_close(native.weight_q.integer_values(), qw, rtol=0, atol=0)
            x = torch.randn(m, 2, k, dtype=torch.float16, device="cuda").transpose(0, 1)
            x[0, 0].zero_()
            expected = reference_linear(x, qw, scales, 128, bias)
            with torch.inference_mode(), patch.object(NativePackedWeight, "dequantize", side_effect=AssertionError("FP16 fallback")):
                actual = native(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_patch_conv_uses_native_gemm_with_original_flat_weight_scales(self):
        original = nn.Conv2d(3, 11, 2, stride=2, bias=True).cuda().half()
        packed = PackedLayer(original, "conv2d", 128, "triton", True)
        qw = original_integer_weight(packed.weight_q).clone()
        scales, bias = packed.weight_q.scales.clone(), packed.bias.clone()
        native = NativeInt4Layer(packed)
        image = torch.randn(2, 3, 8, 10, device="cuda", dtype=torch.float16)
        patches = F.unfold(image, 2, stride=2).transpose(1, 2)
        expected = reference_linear(patches, qw, scales, 128, bias).transpose(1, 2).reshape(2, 11, 4, 5)
        with torch.inference_mode():
            actual = native(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_current_cuda_stream_and_inference_only_contract(self):
        packed = PackedLayer(nn.Linear(128, 16, bias=False).cuda().half(), "linear", 128, "triton", True)
        native = NativeInt4Layer(packed)
        x = torch.randn(7, 128, device="cuda", dtype=torch.float16)
        expected = reference_linear(x, original_integer_weight(packed.weight_q), packed.weight_q.scales, 128)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            actual = native(x)
        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            native(x.requires_grad_())

    def test_complete_legacy_multimodal_native_layers_and_finite_logits(self):
        from runtime import official_imports
        official_imports()
        from tinyllava.model.language_model.llava_llama import TinyLlavaConfig, TinyLlavaLlamaForCausalLM
        from tinyllava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionModel
        from tinyllava.constants import IMAGE_TOKEN_INDEX

        torch.manual_seed(17)
        config = TinyLlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024,
                                 mm_vision_tower="bczhou/SigLIP-test", mm_hidden_size=8,
                                 mm_projector_type="mlp2x_gelu", mm_patch_merge_type="flat")
        model = TinyLlavaLlamaForCausalLM(config).cuda().half().eval()
        tower = model.get_vision_tower()
        tower.config = SigLipVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=3,
                                          num_attention_heads=2, image_size=54, patch_size=2)
        tower.vision_tower = SigLipVisionModel(tower.config).cuda().half().eval()
        nn.init.normal_(tower.vision_tower.vision_model.embeddings.position_embedding.weight, std=0.02)
        tower.is_loaded = True
        convert_model(model)
        audit = enable_w4a4(model)
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([[1, IMAGE_TOKEN_INDEX, 3, 4]], device="cuda"),
                           images=torch.randn(1, 3, 54, 54, device="cuda", dtype=torch.float16), use_cache=False).logits
        self.assertTrue(bool(torch.isfinite(logits).all()))
        counts = execution_counts(model)
        self.assertEqual(counts["native_layer_count"], audit["linear_layers"] + audit["conv2d_layers"])
        self.assertEqual(counts["current_process_called_layers"], counts["native_layer_count"])
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
