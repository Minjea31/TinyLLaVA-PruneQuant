"""Exercise the complete greedy evaluator's reuse and resume without a GPU."""

import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import test_inference as evaluator


class ReuseTests(unittest.TestCase):
    def test_exact_input_reuse_respects_image_and_survives_resume(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data/eval/pope").mkdir(parents=True)
            (root / "docs").mkdir()
            rows = [
                {"question_id": "first", "image": "a.jpg", "text": "Is there a cat?",
                 "category": "random", "label": "yes"},
                {"question_id": "duplicate", "image": "a.jpg", "text": "Is there a cat?",
                 "category": "popular", "label": "yes"},
                {"question_id": "different_image", "image": "b.jpg", "text": "Is there a cat?",
                 "category": "adversarial", "label": "yes"},
            ]
            (root / "data/eval/pope/questions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            for name in ("TinyLLaVA-1.5B", "TinyLLaVA-1.5B-SigLIP"):
                (root / "docs" / f"{name}_snapshot.json").write_text('{"revision":"fixture"}')
            tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
                                        batch_decode=lambda *a, **kw: ["Yes."])
            calls = []

            def generate(*a, **kw):
                calls.append(True)
                return torch.tensor([[1, 2]])

            with (patch.object(evaluator, "ROOT", root),
                  patch.object(evaluator, "make_dirs"),
                  patch.object(evaluator.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="ok")),
                  patch.object(evaluator, "load", return_value=(tokenizer, SimpleNamespace(generate=generate, config=SimpleNamespace()), None, 2048)),
                  patch.object(evaluator, "inputs", return_value=(torch.tensor([[1, 2]]), None, "prompt")),
                  patch.object(evaluator, "batch_inputs", return_value=(torch.tensor([[1, 2]]), None, torch.ones((1, 2), dtype=torch.bool), [2])),
                  patch.object(torch.cuda, "is_available", return_value=True),
                  patch.object(torch.cuda, "manual_seed_all"),
                  patch.object(torch.cuda, "synchronize"),
                  patch.object(torch.cuda, "reset_peak_memory_stats"),
                  patch.object(torch.cuda, "max_memory_allocated", return_value=1000),
                  patch.object(torch.cuda, "max_memory_reserved", return_value=2000),
                  patch.object(torch.cuda, "get_device_name", return_value="test"),
                  patch.object(sys, "argv", ["test_inference.py", "--benchmark", "pope", "--batch-size", "1", "--reuse-identical-inputs"]),
                  contextlib.redirect_stdout(io.StringIO())):
                evaluator.main()
                self.assertEqual(len(calls), 3)  # One warmup plus two distinct inputs.
                evaluator.main()
                self.assertEqual(len(calls), 3)  # Completed run is resumed, not recomputed.
            directory = root / "results/baseline/pope"
            predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(predictions), 3)
            self.assertEqual(predictions[1]["reused_from_question_id"], "first")
            self.assertIsNone(predictions[2]["reused_from_question_id"])
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["execution"], {"unique_input_count": 2,
                                                   "fresh_generation_count": 2,
                                                   "reused_prediction_count": 1})
            self.assertTrue(all(value["accuracy"] == 1 for value in summary["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
