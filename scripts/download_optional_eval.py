"""Download secondary benchmark assets from primary sources; record manual prerequisites."""

import argparse
import json
import logging
import shutil
from common import ROOT, make_dirs, now, write_json
from download_data import fetch, extract, hf_file
from huggingface_hub import HfApi, snapshot_download


def main():
    make_dirs()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        choices=[
            "textvqa",
            "gqa",
            "vqav2",
            "mme",
            "mmvet",
            "mmmu",
            "ocr-vqa",
            "sharegpt4v",
        ],
    )
    p.add_argument(
        "--revision", help="Reuse a pinned HF revision for subsequent downloads"
    )
    args = p.parse_args()
    name = args.dataset
    dest = ROOT / "data/eval" / name
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / f"logs/download_optional_{name}.log"),
        ],
    )
    report = dict(
        time=now(), dataset=name, status="downloaded_assets_need_official_adapter"
    )
    try:
        if name == "textvqa":
            fetch(
                "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json",
                dest / "TextVQA_0.5.1_val.json",
            )
            report["next"] = (
                "Use official eval.zip OCR-augmented questions and shared train_images; do not silently replace OCR inputs."
            )
        elif name == "vqav2":
            archive = fetch(
                "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
                dest / "v2_Questions_Test_mscoco.zip",
            )
            extract(archive, dest)
            report["next"] = (
                "Use official VQAv2 question conversion and TinyLLaVA script; submit answers to EvalAI."
            )
        elif name == "gqa":
            archive = fetch(
                "https://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip",
                dest / "eval.zip",
            )
            extract(archive, dest)
            report["next"] = (
                "Use testdev_balanced_questions.json, shared GQA images and official eval.py; set paths in upstream gqa.sh."
            )
        elif name == "mme":
            repo = "darkyarding/MME"
            info = HfApi().dataset_info(repo, revision=args.revision)
            # This download is linked by the MME authors in their official README.
            archive = hf_file(repo, "MME_Benchmark_release_version.zip", dest, info.sha)
            extract(archive, dest)
            tool = fetch(
                "https://raw.githubusercontent.com/BradyFU/Awesome-Multimodal-Large-Language-Models/Evaluation/tools/eval_tool.zip",
                dest / "eval_tool.zip",
            )
            extract(tool, dest)
            report["revision"] = info.sha
            report["next"] = (
                "Prepare official llava_mme.jsonl and conversion script from TinyLLaVA eval.zip, then run eval_tool/calculation.py."
            )
        elif name == "mmvet":
            # Image archive is downloaded by download_data --group mmvet.
            annotations = list(dest.rglob("mm-vet.json"))
            if not annotations:
                raise RuntimeError(
                    "Download/extract the official mm-vet.zip first: download_data.py --group mmvet"
                )
            report["annotation"] = str(annotations[0].relative_to(ROOT))
            report["next"] = (
                "Use upstream model_vqa and convert_mmvet_for_eval.py; official GPT-4 grader/API or evaluation service required."
            )
        elif name == "mmmu":
            repo = "MMMU/MMMU"
            api = HfApi()
            info = api.dataset_info(repo, revision=args.revision, files_metadata=True)
            total = sum(s.size or 0 for s in info.siblings)
            if shutil.disk_usage(ROOT).free < total + 10 * 1024**3:
                raise RuntimeError("Insufficient disk plus reserve")
            snapshot_download(
                repo,
                repo_type="dataset",
                revision=info.sha,
                local_dir=dest / "raw",
                max_workers=3,
            )
            report.update(
                revision=info.sha,
                next="Use official MMMU evaluator; parquet image export needs datasets/pyarrow, not installed in baseline runtime. Legacy TinyLLaVA requires customized MMMU.zip annotations.",
            )
        else:
            report["status"] = "manual_source_workflow_required"
            if name == "ocr-vqa":
                report.update(
                    source="https://ocr-vqa.github.io/",
                    next="Use the official download script linked on the project page; retrieve original book-cover URLs, save .jpg under data/train/llava/ocr_vqa/images. Dead external image URLs must be logged as missing.",
                )
            else:
                report.update(
                    source="https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V",
                    next="Full SAM 000000~000050 and WebData require external acquisition/access and substantial extra disk. Use official Google Drive 9K SAM subset for SFT only; it does not complete 1.246M pretraining images.",
                )
        write_json(ROOT / f"data/manifests/optional_{name}.json", report)
        print(json.dumps(report, indent=2))
    except Exception as e:
        logging.exception("Optional source failed")
        write_json(
            ROOT / f"data/manifests/optional_{name}.json",
            dict(time=now(), dataset=name, status="failed", error=repr(e)),
        )
        raise


if __name__ == "__main__":
    main()
