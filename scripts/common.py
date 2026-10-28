"""Shared paths, environment checks and atomic research manifests."""

import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path("/home/a/anaconda3/envs/LLAVA")
if Path(sys.prefix).resolve() != PREFIX.resolve():
    raise SystemExit(f"Use {PREFIX}/bin/python; current prefix: {sys.prefix}")
os.environ.setdefault("HF_HOME", str(ROOT / "model/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "model/.cache/huggingface/hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/.cache/huggingface"))
os.environ.setdefault("HF_XET_CACHE", str(ROOT / "model/.cache/xet"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_dirs():
    for directory in [
        "model/TinyLLaVA-1.5B",
        "model/TinyLLaVA-1.5B-SigLIP",
        "model/quantized",
        "model/checkpoints",
        "data/train/llava",
        "data/train/sharegpt4v",
        "data/train/annotations",
        *[
            "data/eval/" + s
            for s in [
                "pope",
                "scienceqa",
                "textvqa",
                "gqa",
                "vqav2",
                "mme",
                "mmvet",
                "mmmu",
            ]
        ],
        "data/calibration",
        "data/manifests",
        "configs",
        "logs",
        "results/baseline",
        "docs",
    ]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)


def iter_array(path):
    """Stream a JSON array of objects with bounded memory, using the stdlib decoder."""
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    eof = False
    with open(path, encoding="utf-8") as f:

        def fill():
            nonlocal buffer, pos, eof
            chunk = f.read(2 * 1024 * 1024)
            buffer = buffer[pos:] + chunk
            pos = 0
            eof = not chunk

        def skip():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                fill()

        fill()
        skip()
        if pos >= len(buffer) or buffer[pos] != "[":
            raise ValueError(f"Expected JSON array: {path}")
        pos += 1
        first = True
        while True:
            skip()
            if pos < len(buffer) and buffer[pos] == "]":
                if buffer[pos + 1 :].strip() or f.read().strip():
                    raise ValueError("Trailing JSON data")
                return
            if not first:
                if pos >= len(buffer) or buffer[pos] != ",":
                    raise ValueError("Missing JSON array separator")
                pos += 1
                skip()
            try:
                item, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError as error:
                while True:
                    if eof:
                        raise ValueError(
                            f"Truncated/invalid JSON array: {path}"
                        ) from error
                    fill()
                    try:
                        item, end = decoder.raw_decode(buffer, pos)
                        break
                    except json.JSONDecodeError:
                        pass
            yield item
            pos = end
            first = False
