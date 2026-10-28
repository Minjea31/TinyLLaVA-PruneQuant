"""Official-source, resumable downloads with receipts, disk guards and path mapping."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import shutil
import time
import threading
import zipfile
import requests
from common import ROOT, iter_array, make_dirs, now, sha256, write_json
from huggingface_hub import HfApi, hf_hub_download

GIB = 1024**3
_sessions = threading.local()


def session():
    if not hasattr(_sessions, "value"):
        _sessions.value = requests.Session()
    return _sessions.value


ANNOTATIONS = [
    (
        "liuhaotian/LLaVA-Pretrain",
        "blip_laion_cc_sbu_558k.json",
        "70f9d1e5e1a697fe35830875cfc7de1dd590d727",
    ),
    (
        "liuhaotian/LLaVA-Instruct-150K",
        "llava_v1_5_mix665k.json",
        "9d451dc7629cfe0469f6ae4432b765cd603d5fcb",
    ),
    (
        "Lin-Chen/ShareGPT4V",
        "share-captioner_coco_lcs_sam_1246k_1107.json",
        "55d02b0bc53a2754095a14110dda6daedd95671d",
    ),
    (
        "Lin-Chen/ShareGPT4V",
        "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json",
        "55d02b0bc53a2754095a14110dda6daedd95671d",
    ),
]


def receipt(path):
    return path.with_name(path.name + ".receipt.json")


def fetch(url, path):
    # Same official COCO S3 bucket, with a hostname covered by its TLS certificate.
    if url.startswith("https://images.cocodataset.org/"):
        url = url.replace(
            "https://images.cocodataset.org/",
            "https://s3.amazonaws.com/images.cocodataset.org/",
            1,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = receipt(path)
    if path.is_file() and record.exists():
        old = json.loads(record.read_text())
        if (
            old["url"] == url
            and old["bytes"] == path.stat().st_size
            and old["sha256"] == sha256(path)
        ):
            return path
    part = path.with_name(path.name + ".partial")
    for attempt in range(5):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {"Accept-Encoding": "identity"}
            if start:
                headers["Range"] = f"bytes={start}-"
            with session().get(
                url, headers=headers, stream=True, timeout=(10, 120)
            ) as r:
                if r.status_code == 416 and start:
                    part.unlink()
                    continue
                r.raise_for_status()
                resumed = start and r.status_code == 206
                if resumed and not r.headers.get("Content-Range", "").startswith(
                    f"bytes {start}-"
                ):
                    raise RuntimeError("Server returned an incorrect byte range")
                expected = (
                    int(r.headers.get("Content-Length", 0))
                    if r.headers.get("Content-Encoding", "identity") == "identity"
                    else 0
                )
                if shutil.disk_usage(ROOT).free < expected + 10 * GIB:
                    raise RuntimeError(f"Insufficient disk for {path}; reserve 10 GiB")
                with part.open("ab" if resumed else "wb") as f:
                    for chunk in r.iter_content(4 * 1024 * 1024):
                        f.write(chunk)
                total = expected + (start if resumed else 0)
                if expected and part.stat().st_size != total:
                    raise RuntimeError("Incomplete HTTP transfer")
                part.replace(path)
                write_json(
                    record,
                    dict(
                        url=url,
                        bytes=path.stat().st_size,
                        sha256=sha256(path),
                        etag=r.headers.get("ETag"),
                        downloaded_at=now(),
                    ),
                )
                return path
        except Exception:
            logging.exception("Download attempt %d failed: %s", attempt + 1, url)
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def hf_file(repo, name, dest, revision):
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    entry = next(s for s in info.siblings if s.rfilename == name)
    if (
        shutil.disk_usage(ROOT).free < entry.size + 10 * GIB
        and not (dest / name).exists()
    ):
        raise RuntimeError("Insufficient disk for HF data download")
    path = Path(
        hf_hub_download(
            repo, name, repo_type="dataset", revision=info.sha, local_dir=dest
        )
    )
    digest = sha256(path)
    if path.stat().st_size != entry.size or (entry.lfs and entry.lfs.sha256 != digest):
        raise RuntimeError(f"HF integrity failure: {path}")
    write_json(
        receipt(path),
        dict(
            repo=repo,
            revision=info.sha,
            path=name,
            bytes=entry.size,
            sha256=digest,
            verified_at=now(),
        ),
    )
    return path


def extract(archive, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / (".extracted-" + archive.name + ".json")
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["archive_sha256"] == sha256(archive) and all(
            (dest / x["name"]).is_file()
            and (dest / x["name"]).stat().st_size == x["bytes"]
            for x in old["files"]
        ):
            return
    with zipfile.ZipFile(archive) as z:
        files = [x for x in z.infolist() if not x.is_dir()]
        if shutil.disk_usage(ROOT).free < sum(x.file_size for x in files) + 10 * GIB:
            raise RuntimeError(
                "Insufficient disk for extracted data plus 10 GiB reserve"
            )
        for entry in files:
            target = (dest / entry.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError("Unsafe archive path")
            # Read each member through ZipFile: CRC checked during extraction.
            target.parent.mkdir(parents=True, exist_ok=True)
            with (
                z.open(entry) as src,
                target.with_name(target.name + ".partial").open("wb") as out,
            ):
                shutil.copyfileobj(src, out, 4 * 1024 * 1024)
            target.with_name(target.name + ".partial").replace(target)
    write_json(
        marker,
        dict(
            archive_sha256=sha256(archive),
            verified_at=now(),
            files=[
                dict(name=x.filename, bytes=x.file_size, crc32=x.CRC) for x in files
            ],
        ),
    )


def github_revision(repo):
    r = session().get(
        f"https://api.github.com/repos/{repo}/commits/HEAD", timeout=(10, 60)
    )
    r.raise_for_status()
    return r.json()["sha"]


def annotations():
    with ThreadPoolExecutor(max_workers=4) as ex:
        tasks = [
            ex.submit(hf_file, repo, name, ROOT / "data/train/annotations", rev)
            for repo, name, rev in ANNOTATIONS
        ]
        for task in tasks:
            logging.info("Annotation complete: %s", task.result())


def pretrain():
    from huggingface_hub import hf_hub_url
    from segmented_download import transfer

    repo = "liuhaotian/LLaVA-Pretrain"
    revision = ANNOTATIONS[0][2]
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    entry = next(s for s in info.siblings if s.rfilename == "images.zip")
    dest = ROOT / "data/train/llava/images.zip"
    previous = list(
        (dest.parent / ".cache/huggingface/download").glob(
            "*." + entry.lfs.sha256 + ".incomplete"
        )
    )
    archive = transfer(
        hf_hub_url(repo, "images.zip", repo_type="dataset", revision=revision),
        dest,
        entry.size,
        entry.lfs.sha256,
        workers=96,
        existing=previous[0] if previous else None,
    )
    extract(archive, ROOT / "data/train/llava/llava_pretrain/images")


def pope():
    dest = ROOT / "data/eval/pope"
    rev = github_revision("RUCAIBox/POPE")
    all_rows = []
    metadata = {}
    for category in ["random", "popular", "adversarial"]:
        name = f"coco_pope_{category}.json"
        path = fetch(
            f"https://raw.githubusercontent.com/RUCAIBox/POPE/{rev}/output/coco/{name}",
            dest / "coco" / name,
        )
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        metadata[category] = len(rows)
        for row in rows:
            all_rows.append(
                dict(
                    question_id=f"{category}_{row['question_id']}",
                    image=row["image"],
                    text=row["text"],
                    label=row["label"],
                    category=category,
                )
            )
    images = sorted({r["image"] for r in all_rows})

    def image(name):
        p = fetch(
            "https://images.cocodataset.org/val2014/" + name, dest / "val2014" / name
        )
        from PIL import Image

        with Image.open(p) as im:
            im.verify()

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(image, images))
    (dest / "questions.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in all_rows)
    )
    write_json(
        ROOT / "data/manifests/pope.json",
        dict(
            status="ready",
            source="RUCAIBox/POPE",
            revision=rev,
            categories=metadata,
            questions=len(all_rows),
            images=len(images),
            split="COCO val2014 POPE",
            questions_sha256=sha256(dest / "questions.jsonl"),
        ),
    )


def scienceqa():
    dest = ROOT / "data/eval/scienceqa"
    rev = github_revision("lupantech/ScienceQA")
    for name in ["problems.json", "pid_splits.json"]:
        fetch(
            f"https://raw.githubusercontent.com/lupantech/ScienceQA/{rev}/data/scienceqa/{name}",
            dest / name,
        )
    url = "https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip"
    with session().get(
        url,
        headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(10, 60),
    ) as probe:
        probe.raise_for_status()
        size = (
            int(probe.headers["Content-Range"].split("/")[-1])
            if probe.status_code == 206
            else None
        )
    if size:
        from segmented_download import transfer

        old = dest / "test.zip.partial"
        archive = transfer(
            url,
            dest / "test.zip",
            size,
            None,
            workers=8,
            existing=old if old.exists() else None,
            chunk=8 * 1024 * 1024,
        )
    else:
        archive = fetch(url, dest / "test.zip")
    extract(archive, dest / "images")
    problems = json.loads((dest / "problems.json").read_text())
    splits = json.loads((dest / "pid_splits.json").read_text())
    rows = []
    image_rows = []
    for pid in splits["test"]:
        p = problems[str(pid)]
        name = f"{pid}/{p['image']}" if p["image"] else None
        if name:
            path = dest / "images/test" / name
            if not path.is_file():
                raise RuntimeError(f"Missing ScienceQA image {path}")
            from PIL import Image

            with Image.open(path) as im:
                im.verify()
        choices = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(p["choices"]))
        prompt = (
            (("Context: " + p["hint"] + "\n") if p["hint"] else "")
            + p["question"]
            + "\n"
            + choices
            + "\nAnswer with the option letter only."
        )
        row = dict(
            question_id=str(pid), image=name, text=prompt, answer=chr(65 + p["answer"])
        )
        rows.append(row)
        if name:
            image_rows.append(row)
    for name, items in [("questions_test", rows), ("questions_test_image", image_rows)]:
        (dest / (name + ".jsonl")).write_text(
            "".join(json.dumps(x) + "\n" for x in items)
        )
    write_json(
        ROOT / "data/manifests/scienceqa.json",
        dict(
            status="ready",
            source="lupantech/ScienceQA",
            revision=rev,
            split="test",
            questions=len(rows),
            image_questions=len(image_rows),
            protocol="local CQM letter-only (report separately from official LLaVA eval.zip CQM-A)",
            questions_sha256=sha256(dest / "questions_test_image.jsonl"),
        ),
    )


def training_subset():
    import random

    rng = random.Random(2026)
    pool = []
    seen = set()
    n = 0
    annotation = ROOT / "data/train/annotations/llava_v1_5_mix665k.json"
    pope_file = ROOT / "data/eval/pope/coco/coco_pope_random.json"
    excluded = (
        {
            json.loads(x)["image"].split("_")[-1].split(".")[0]
            for x in pope_file.read_text().splitlines()
        }
        if pope_file.exists()
        else set()
    )
    for row in iter_array(annotation):
        image = row.get("image", "")
        if not image.startswith("coco/train2017/") or image in seen:
            continue
        if Path(image).stem in excluded:
            continue
        seen.add(image)
        n += 1
        if len(pool) < 1024:
            pool.append(row)
        else:
            j = rng.randrange(n)
            if j < 1024:
                pool[j] = row

    def get_image(row):
        path = fetch(
            "https://images.cocodataset.org/train2017/" + Path(row["image"]).name,
            ROOT / "data/train/llava" / row["image"],
        )
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        return dict(id=row["id"], image=row["image"], sha256=sha256(path))

    with ThreadPoolExecutor(max_workers=8) as ex:
        records = list(ex.map(get_image, pool))
    write_json(ROOT / "data/train/annotations/llava_sft_coco1024.json", pool)
    write_json(
        ROOT / "data/manifests/training_subset.json",
        dict(
            status="ready",
            seed=2026,
            samples=len(pool),
            source_sha256=sha256(annotation),
            source="LLaVA-v1.5 SFT / COCO train2017",
            excluded_pope_ids=len(excluded),
            files=records,
        ),
    )


# Full archives are opt-in and disk-checked. Images shared by train/eval via symlinks.
ARCHIVES = {
    "coco-train": (
        "https://images.cocodataset.org/zips/train2017.zip",
        "data/train/llava/coco/train2017.zip",
        "data/train/llava/coco",
    ),
    "gqa-images": (
        "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
        "data/train/llava/gqa/images.zip",
        "data/train/llava/gqa",
    ),
    "textvqa": (
        "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
        "data/train/llava/textvqa/train_val_images.zip",
        "data/train/llava/textvqa",
    ),
    "vg1": (
        "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
        "data/train/llava/vg/images.zip",
        "data/train/llava/vg",
    ),
    "vg2": (
        "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
        "data/train/llava/vg/images2.zip",
        "data/train/llava/vg",
    ),
    "vqav2": (
        "https://images.cocodataset.org/zips/test2015.zip",
        "data/eval/vqav2/test2015.zip",
        "data/eval/vqav2",
    ),
    "gqa-questions": (
        "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
        "data/eval/gqa/questions1.2.zip",
        "data/eval/gqa",
    ),
    "mmvet": (
        "https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip",
        "data/eval/mmvet/mm-vet.zip",
        "data/eval/mmvet",
    ),
}


def main():
    make_dirs()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        choices=[
            "annotations",
            "pretrain",
            "pope",
            "scienceqa",
            "initial-eval",
            "training-subset",
            *ARCHIVES,
        ],
        required=True,
    )
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / f"logs/download_data_{args.group}.log"),
        ],
    )
    try:
        if args.group in ARCHIVES:
            url, archive, dest = ARCHIVES[args.group]
            extract(fetch(url, ROOT / archive), ROOT / dest)
        elif args.group == "initial-eval":
            # Independent sources: do not let a failed benchmark suppress the other.
            errors = []
            for fn in [pope, scienceqa]:
                try:
                    fn()
                except Exception as e:
                    logging.exception("%s failed", fn.__name__)
                    errors.append(str(e))
            if errors:
                raise RuntimeError(errors)
        else:
            globals()[args.group.replace("-", "_")]()
    except Exception as e:
        write_json(
            ROOT / f"data/manifests/failure_{args.group}.json",
            dict(time=now(), error=repr(e)),
        )
        logging.exception("Group %s failed; rerun to resume", args.group)
        raise


if __name__ == "__main__":
    main()
