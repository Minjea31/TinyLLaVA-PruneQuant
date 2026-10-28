"""Packed integer correctness, full parameter coverage and CUDA kernel parity."""

import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from int4 import PackedTensor, convert_model


class Int4Tests(unittest.TestCase):
    def test_roundtrip_zero_odd_and_partial_groups(self):
        for size in (1, 17, 128, 257):
            torch.manual_seed(size)
            original = torch.randn(size)
            packed = PackedTensor(original, group_size=128, backend="torch")
            result = packed.dequantize()
            scale = packed.scales.repeat_interleave(128)[:size].float()
            expected = (original / scale).round().clamp(-7, 7) * scale
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            self.assertEqual(packed.codes.numel(), (size + 1) // 2)
            self.assertTrue(bool((result - original).abs().le(scale / 2 + 1e-6).all()))
        zero = PackedTensor(torch.zeros(129), backend="torch")
        torch.testing.assert_close(zero.dequantize(), torch.zeros(129), rtol=0, atol=0)

    def test_linear_conv_embedding_norm_and_bias_have_no_float_parameters(self):
        model = nn.ModuleDict({"linear": nn.Linear(16, 8), "embedding": nn.Embedding(17, 16),
                               "conv": nn.Conv2d(3, 8, 2), "norm": nn.LayerNorm(8)})
        reference = copy.deepcopy(model)
        count = sum(p.numel() for p in model.parameters())
        for parameter in reference.parameters():
            parameter.data.copy_(PackedTensor(parameter, backend="torch").dequantize())
        audit = convert_model(model, backend="torch")
        self.assertEqual(audit["total_parameters"], count)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 0)
        for name, data in [("linear", torch.randn(2, 16)), ("embedding", torch.tensor([1, 4, 8])),
                           ("conv", torch.randn(1, 3, 8, 8)), ("norm", torch.randn(2, 8))]:
            torch.testing.assert_close(model[name](data), reference[name](data), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_triton_matches_reference_exactly(self):
        for size in (17, 128, 729, 1152 * 588):
            original = torch.randn(size, device="cuda", dtype=torch.float16)
            packed = PackedTensor(original, backend="torch")
            expected = packed.dequantize()
            packed.backend = "triton"
            torch.testing.assert_close(packed.dequantize(), expected, rtol=0, atol=0)

    def test_complete_legacy_multimodal_logits_match_dense_rtn(self):
        from runtime import official_imports
        official_imports()
        from tinyllava.model.language_model.llava_llama import TinyLlavaConfig, TinyLlavaLlamaForCausalLM
        from tinyllava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionModel
        from tinyllava.constants import IMAGE_TOKEN_INDEX

        torch.manual_seed(17)
        config = TinyLlavaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024,
            mm_vision_tower="bczhou/SigLIP-test", mm_hidden_size=8,
            mm_projector_type="mlp2x_gelu", mm_patch_merge_type="flat",
        )
        packed = TinyLlavaLlamaForCausalLM(config).eval()
        tower = packed.get_vision_tower()
        tower.config = SigLipVisionConfig(hidden_size=8, intermediate_size=16,
                                          num_hidden_layers=3, num_attention_heads=2,
                                          image_size=54, patch_size=2)
        tower.vision_tower = SigLipVisionModel(tower.config).eval()
        torch.nn.init.normal_(tower.vision_tower.vision_model.embeddings.position_embedding.weight,
                              std=0.02)
        tower.is_loaded = True
        dense = copy.deepcopy(packed)
        for parameter in dense.parameters():
            parameter.data.copy_(PackedTensor(parameter, backend="torch").dequantize())
        count = sum(p.numel() for p in packed.parameters())
        audit = convert_model(packed, backend="torch")
        self.assertEqual(audit["total_parameters"], count)
        self.assertTrue(all(audit["components"][name]["parameters"] > 0
                            for name in ("vision", "projector", "llm")))
        kwargs = {"input_ids": torch.tensor([[1, IMAGE_TOKEN_INDEX, 3, 4]]),
                  "images": torch.randn(1, 3, 54, 54), "use_cache": False}
        with torch.inference_mode():
            expected = dense(**kwargs).logits
            actual = packed(**kwargs).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
