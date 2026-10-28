"""Numerical checks for offloading and the real legacy multimodal gradient path."""

import copy
from pathlib import Path
import sys
import unittest

import torch
from transformers.optimization import Adafactor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_full import CPUOffloadedAdafactor, configure_model, unused_legacy_parameter
from runtime import official_imports


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.vision_tower = torch.nn.Linear(3, 4)
        self.model.mm_projector = torch.nn.Linear(4, 4)
        self.output = torch.nn.Linear(4, 2)

    def forward(self, inputs):
        return self.output(self.model.mm_projector(self.model.vision_tower(inputs)).tanh())


class OffloadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_accumulation_matches_fp32_adafactor_and_resume(self):
        actual = SmallModel()
        reference = copy.deepcopy(actual)
        offload = CPUOffloadedAdafactor(actual, 1e-3, 1e-3)
        optimizer = Adafactor(reference.parameters(), lr=1e-3, scale_parameter=False,
                              relative_step=False, warmup_init=False, beta1=None)
        for count in (3, 1):  # Full accumulation window and final short window.
            for _ in range(count):
                inputs, target = torch.randn(2, 3), torch.randn(2, 2)
                ((actual(inputs) - target).square().mean() * 128 / count).backward()
                ((reference(inputs) - target).square().mean() / count).backward()
            self.assertTrue(all(p.grad is None for p in actual.parameters()))
            updated, _ = offload.step(128)
            self.assertTrue(updated)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for a, b in zip(actual.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        restored = SmallModel()
        resumed = CPUOffloadedAdafactor(restored, 1e-3, 1e-3)
        resumed.load_state_dict(copy.deepcopy(offload.state_dict()))
        inputs = torch.randn(2, 3)
        actual(inputs).square().mean().backward()
        restored(inputs).square().mean().backward()
        offload.step(1)
        resumed.step(1)
        for a, b in zip(actual.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_overflow_skips_every_component_and_clears_gradients(self):
        model = SmallModel()
        offload = CPUOffloadedAdafactor(model, 1e-3, 1e-3)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        model(torch.randn(2, 3)).sum().backward()
        next(iter(offload.masters.values())).grad.fill_(float("inf"))
        updated, result = offload.step(128)
        self.assertFalse(updated)
        self.assertTrue(result["overflow"])
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in offload.masters.values()))

    def test_legacy_siglip_and_llama_receive_gradients(self):
        official_imports()
        from tinyllava.model.language_model.llava_llama import TinyLlavaConfig, TinyLlavaLlamaForCausalLM
        from tinyllava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionModel
        from tinyllava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

        config = TinyLlavaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024,
            mm_vision_tower="bczhou/SigLIP-test", mm_hidden_size=8, mm_projector_type="mlp2x_gelu",
            mm_patch_merge_type="flat", attention_dropout=0.0,
        )
        model = TinyLlavaLlamaForCausalLM(config)
        tower = model.get_vision_tower()
        # 27 x 27 patches exercise the actual 729-token image expansion.
        tower.config = SigLipVisionConfig(hidden_size=8, intermediate_size=16,
                                          num_hidden_layers=3, num_attention_heads=2,
                                          image_size=54, patch_size=2)
        tower.vision_tower = SigLipVisionModel(tower.config)
        # The upstream vision class has an empty _init_weights: initialize its
        # standalone position embedding explicitly for this synthetic test.
        torch.nn.init.normal_(tower.vision_tower.vision_model.embeddings.position_embedding.weight,
                              std=0.02)
        tower.is_loaded = True
        configure_model(model, 1024)
        offload = CPUOffloadedAdafactor(model, 1e-3, 1e-3)
        ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 3, 4, 5, 6]])
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5, 6]])
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                     images=torch.randn(1, 3, 54, 54), use_cache=False).loss
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        missing = {n for n, p in offload.masters.items() if p.grad is None}
        self.assertTrue(missing)
        self.assertTrue(all(unused_legacy_parameter(n) for n in missing), missing)
        updated, result = offload.step(1)
        self.assertTrue(updated)
        self.assertTrue(all(value > 0 for value in result["grad_norms"].values()))


if __name__ == "__main__":
    unittest.main()
