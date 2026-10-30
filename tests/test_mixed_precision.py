"""Original FP16 restoration, independent W/A selection and native coverage."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from int4 import convert_model, PackedLayer
from mixed_precision import (capture_fp16, enable_mixed_precision, selected_names,
    FloatWeightLinear, PackedWeightA16Linear, mixed_execution_counts)
from w4a4 import NativeInt4Layer, NativePackedWeight


POLICY = {"candidate_layers": {
    "weight": [{"name": name} for name in ("a", "c", "norm")],
    "activation": [{"name": name} for name in ("b", "c")]}}


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pretraining_tp=1)
        for name in ("a", "b", "c", "d"):
            setattr(self, name, nn.Linear(128, 16, bias=True))
        self.norm = LlamaRMSNorm(128)
        with torch.no_grad():
            self.norm.weight.copy_(torch.linspace(.65, 1.5, 128))


class MixedTests(unittest.TestCase):
    def test_weight_activation_selection_is_independent(self):
        self.assertEqual(selected_names(POLICY, "weights"), ({"a", "c", "norm"}, set()))
        self.assertEqual(selected_names(POLICY, "activations"), (set(), {"b", "c"}))
        self.assertEqual(selected_names(POLICY, "both"), ({"a", "c", "norm"}, {"b", "c"}))
        self.assertEqual(selected_names(POLICY, "none"), (set(), set()))
        with self.assertRaises(ValueError):
            selected_names(POLICY, "training")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_all_four_weight_activation_modes_and_original_fp16_values(self):
        for variant in ("none", "weights", "activations", "both"):
            torch.manual_seed(19)
            model = Toy().cuda().half().eval().requires_grad_(False)
            original = {name: p.detach().clone() for name, p in model.named_parameters()}
            originals = capture_fp16(model, POLICY, variant)
            self.assertTrue(all(p.device.type == 'cpu' for m in originals.values() for p in m.parameters()))
            convert_model(model)
            packed_weight = {name: model.get_submodule(name).weight.clone() for name in ("a", "b", "c", "d", "norm")}
            packed_bias = {name: model.get_submodule(name).bias.clone() for name in ("a", "b", "c", "d")}
            audit = enable_mixed_precision(model, POLICY, variant, originals)
            weights, activations = selected_names(POLICY, variant)
            expected = {"none": {"W4A4": 4}, "weights": {"W4A4": 2, "W16A4": 2},
                        "activations": {"W4A4": 2, "W4A16": 2},
                        "both": {"W4A4": 1, "W16A4": 1, "W4A16": 1, "W16A16": 1}}
            self.assertEqual(audit['weighted_layer_precision_counts'], expected[variant])
            x = torch.randn(3, 128, device='cuda', dtype=torch.float16)
            # Independent CPU QDQ, using the native scale's actual FP16 value.
            values = x.cpu().float().reshape(-1, 128)
            scales = (values.abs().amax(1, keepdim=True)/7).clamp_min(2**-24).half().float()
            x_qdq = ((values/scales).round().clamp(-7, 7)*scales).half().cuda()
            with torch.inference_mode(), patch.object(NativePackedWeight, 'dequantize', side_effect=AssertionError('native fallback')):
                for name in ("a", "b", "c", "d"):
                    module = model.get_submodule(name)
                    output = module(x)
                    self.assertTrue(torch.isfinite(output).all())
                    if name in weights:
                        self.assertIsInstance(module, FloatWeightLinear)
                        self.assertTrue(torch.equal(module.weight, original[name+'.weight']))
                        self.assertFalse(torch.equal(module.weight, packed_weight[name]))
                        reference = F.linear(x if name in activations else x_qdq,
                                             original[name+'.weight'], original[name+'.bias'])
                        torch.testing.assert_close(output, reference, rtol=0, atol=0)
                    elif name in activations:
                        self.assertIsInstance(module, PackedWeightA16Linear)
                        reference = F.linear(x, packed_weight[name], packed_bias[name])
                        torch.testing.assert_close(output, reference, rtol=0, atol=0)
                    else:
                        self.assertIsInstance(module, NativeInt4Layer)
                if 'norm' in weights:
                    self.assertIsInstance(model.norm, LlamaRMSNorm)
                    self.assertTrue(torch.equal(model.norm.weight, original['norm.weight']))
                else:
                    self.assertIsInstance(model.norm, PackedLayer)
            calls = mixed_execution_counts(model)
            self.assertEqual(calls['native']['current_process_called_layers'], audit['native_layer_count'])
            self.assertTrue(all(row['calls'] > 0 for row in calls['exceptions'].values()))
            self.assertFalse(any(p.requires_grad for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_missing_snapshot_and_pretraining_tp_are_rejected(self):
        model = Toy().cuda().half()
        snapshots = capture_fp16(model, POLICY, 'both')
        convert_model(model)
        with self.assertRaisesRegex(ValueError, 'snapshots'):
            enable_mixed_precision(model, POLICY, 'both', {})
        model.config.pretraining_tp = 2
        with self.assertRaisesRegex(ValueError, 'pretraining_tp'):
            enable_mixed_precision(model, POLICY, 'both', snapshots)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
