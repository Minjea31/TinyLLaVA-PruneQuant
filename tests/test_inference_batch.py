"""Check multimodal padding, final partial batches, timing and safe resume."""

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
import runtime
import test_inference as evaluator


class BatchTests(unittest.TestCase):
    def test_left_padding_preserves_image_sentinel_and_mask(self):
        prepared = [(torch.tensor([[1, -200, 7]]), torch.ones(1, 3, 2, 2), "a"),
                    (torch.tensor([[1, -200, 8, 9, 10]]), torch.zeros(1, 3, 2, 2), "b")]
        with patch.object(runtime, "inputs", side_effect=prepared):
            ids, images, mask, lengths = runtime.batch_inputs(
                SimpleNamespace(pad_token_id=0), None, None, ["a.jpg", "b.jpg"], ["a", "b"], "cpu")
        self.assertEqual(ids.tolist(), [[0, 0, 1, -200, 7], [1, -200, 8, 9, 10]])
        self.assertEqual(mask.tolist(), [[False, False, True, True, True], [True] * 5])
        self.assertEqual(lengths, [731, 733])
        self.assertEqual(images.shape, (2, 3, 2, 2))
        self.assertEqual(images[0].sum().item(), 12)

    def test_complete_evaluator_keeps_tail_eos_lengths_and_resume_conditions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data/eval/scienceqa").mkdir(parents=True)
            (root / "docs").mkdir()
            rows = [dict(question_id=f"q{i}", image=f"{i}.png", text="question " * (i + 1),
                         answer="A") for i in range(9)]
            (root / "data/eval/scienceqa/questions_test_image.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            for name in ("TinyLLaVA-1.5B", "TinyLLaVA-1.5B-SigLIP"):
                (root / "docs" / f"{name}_snapshot.json").write_text('{"revision":"fixture"}')
            tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
                                        batch_decode=lambda output, **kw: ["A"] * output.shape[0])
            calls = []

            def prepare(tok, model, processor, paths, questions):
                size = len(questions)
                return torch.ones(size, 5, dtype=torch.long), torch.zeros(size, 3, 2, 2), torch.ones(size, 5, dtype=torch.bool), [733] * size

            def generate(ids, **kw):
                self.assertEqual(kw["attention_mask"].shape, ids.shape)
                self.assertFalse(kw["do_sample"])
                self.assertEqual(kw["max_new_tokens"], 32)
                calls.append(ids.shape[0])
                return torch.tensor([[1, 12, 2, 0, 0]] * ids.shape[0])

            argv = ["test_inference.py", "--benchmark", "scienceqa", "--batch-size", "8"]
            with (patch.object(evaluator, "ROOT", root), patch.object(evaluator, "make_dirs"),
                  patch.object(evaluator.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="ok")),
                  patch.object(evaluator, "load", return_value=(tokenizer, SimpleNamespace(generate=generate, config=SimpleNamespace()), None, 2048)),
                  patch.object(evaluator, "batch_inputs", side_effect=prepare),
                  patch.object(torch.cuda, "is_available", return_value=True),
                  patch.object(torch.cuda, "manual_seed_all"), patch.object(torch.cuda, "synchronize"),
                  patch.object(torch.cuda, "reset_peak_memory_stats"),
                  patch.object(torch.cuda, "max_memory_allocated", return_value=1000),
                  patch.object(torch.cuda, "max_memory_reserved", return_value=2000),
                  patch.object(torch.cuda, "get_device_name", return_value="test"),
                  patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO())):
                evaluator.main()
                self.assertEqual(calls, [8, 8, 1])
                evaluator.main()
                self.assertEqual(calls, [8, 8, 1])
                argv[-1] = "4"
                with self.assertRaisesRegex(SystemExit, "different conditions"):
                    evaluator.main()
            directory = root / "results/baseline/scienceqa"
            answers = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual([row["question_id"] for row in answers], [row["question_id"] for row in rows])
            self.assertTrue(all(row["generated_tokens"] == 3 for row in answers))
            self.assertEqual([row["batch_size"] for row in answers], [8] * 8 + [1])
            self.assertAlmostEqual(sum(row["seconds"] for row in answers), summary["generation_seconds"])
            self.assertEqual(summary["metrics"]["correct"], 9)
            self.assertEqual(summary["execution"]["fresh_generation_count"], 9)


if __name__ == "__main__":
    unittest.main()
