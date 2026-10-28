"""Download immutable complete HF snapshots directly into model/, then verify hashes."""

import argparse
import logging
import os
import shutil
import sys

if "--fast" in sys.argv:
    import importlib.util

    if importlib.util.find_spec("hf_transfer") is None:
        raise SystemExit(
            "Install the optional pinned hf-transfer package before using --fast"
        )
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from common import ROOT, make_dirs, now, sha256, write_json
from huggingface_hub import HfApi, snapshot_download
from filelock import FileLock


def download(repo, revision=None):
    locks = ROOT / "model/.cache/download-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(str(locks / (repo.replace("/", "--") + ".lock"))):
        return download_locked(repo, revision)


def download_locked(repo, revision=None):
    api = HfApi()
    pins = __import__("json").loads((ROOT / "configs/sources.json").read_text())[
        "models"
    ]
    revision = revision or pins.get(repo)
    info = api.model_info(repo, revision=revision, files_metadata=True)
    dest = ROOT / "model" / repo.split("/")[-1]
    manifest = ROOT / "docs" / (repo.split("/")[-1] + "_snapshot.json")
    files = [s for s in info.siblings if s.size is not None]
    missing = sum(s.size for s in files if not (dest / s.rfilename).exists())
    if shutil.disk_usage(ROOT).free < missing + 10 * 1024**3:
        raise RuntimeError(f"Not enough disk: need {missing} bytes plus 10 GiB reserve")
    logging.info(
        "%s @ %s: %s bytes, missing %s",
        repo,
        info.sha,
        sum(s.size for s in files),
        missing,
    )
    if "--repair-incomplete" in sys.argv:
        from huggingface_hub import hf_hub_url
        from segmented_download import repair_sparse

        for entry in files:
            if not entry.lfs or (dest / entry.rfilename).exists():
                continue
            partials = list(
                (dest / ".cache/huggingface/download").glob(
                    "*." + entry.lfs.sha256 + ".incomplete"
                )
            )
            if partials:
                repair_sparse(
                    hf_hub_url(repo, entry.rfilename, revision=info.sha),
                    partials[0],
                    entry.size,
                    entry.lfs.sha256,
                )
    if "--parallel" in sys.argv:
        from huggingface_hub import hf_hub_url
        from segmented_download import transfer

        for entry in files:
            target = dest / entry.rfilename
            if not entry.lfs or entry.size < 100 * 1024**2 or target.exists():
                continue
            partials = list(
                (dest / ".cache/huggingface/download").glob(
                    "*." + entry.lfs.sha256 + ".incomplete"
                )
            )
            cache = dest / ".cache/parallel" / entry.rfilename
            completed = transfer(
                hf_hub_url(repo, entry.rfilename, revision=info.sha),
                cache,
                entry.size,
                entry.lfs.sha256,
                workers=96,
                chunk=8 * 1024 * 1024,
                existing=partials[0] if partials else None,
            )
            completed.replace(target)
    snapshot_download(repo, revision=info.sha, local_dir=dest, max_workers=3)
    old = __import__("json").loads(manifest.read_text()) if manifest.exists() else {}
    previous = (
        {x["path"]: x for x in old.get("files", [])}
        if old.get("revision") == info.sha
        else {}
    )
    records = []
    for s in files:
        p = dest / s.rfilename
        if not p.is_file() or p.stat().st_size != s.size:
            raise RuntimeError(f"Missing or invalid size: {p}")
        cached = previous.get(s.rfilename, {})
        digest = (
            cached.get("sha256")
            if cached.get("mtime_ns") == p.stat().st_mtime_ns
            else None
        )
        digest = digest or sha256(p)
        expected = s.lfs.sha256 if s.lfs else None
        if expected and digest != expected:
            raise RuntimeError(f"Invalid LFS SHA256: {p}")
        records.append(
            dict(
                path=s.rfilename,
                bytes=s.size,
                sha256=digest,
                lfs_sha256=expected,
                mtime_ns=p.stat().st_mtime_ns,
            )
        )
    write_json(
        manifest,
        dict(
            repo=repo,
            revision=info.sha,
            verified_at=now(),
            bytes=sum(x["bytes"] for x in records),
            files=records,
        ),
    )
    return info.sha


if __name__ == "__main__":
    make_dirs()
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "logs/download_model.log"),
        ],
    )
    p = argparse.ArgumentParser()
    p.add_argument("--repo", action="append")
    p.add_argument("--revision")
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Bounded range transport using official HF URLs/LFS hashes; finalize with snapshot_download",
    )
    p.add_argument(
        "--repair-incomplete",
        action="store_true",
        help="Repair missing extents in interrupted official transport files, then validate full LFS hash",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Official hf-transfer backend; incomplete HTTP files may restart, complete files are reused",
    )
    args = p.parse_args()
    for repo in args.repo or ["bczhou/TinyLLaVA-1.5B", "bczhou/TinyLLaVA-1.5B-SigLIP"]:
        try:
            download(repo, args.revision)
        except Exception:
            logging.exception("Snapshot download failed for %s; rerun to resume", repo)
            raise
