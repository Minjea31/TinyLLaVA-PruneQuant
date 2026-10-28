"""Fixed-seed stratified PTQ image/text selection from downloaded training data."""

import argparse
from collections import Counter, defaultdict
import random
from PIL import Image
from common import ROOT, iter_array, make_dirs, now, sha256, write_json


def image_path(name, source):
    name = name.removeprefix("./")
    candidates = [
        ROOT / "data/train/llava" / name,
        ROOT / "data/train/sharegpt4v" / name,
    ]
    if source == "llava":
        candidates.insert(0, ROOT / "data/train/llava/llava_pretrain/images" / name)
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def main():
    make_dirs()
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=512)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if not 256 <= args.count <= 512:
        raise SystemExit("Initial calibration count must be between 256 and 512")
    rng = random.Random(args.seed)
    pools = defaultdict(list)
    seen = Counter()
    eligible = Counter()
    scanned = Counter()
    sources = [
        ("llava", "blip_laion_cc_sbu_558k.json"),
        ("sharegpt4v", "share-captioner_coco_lcs_sam_1246k_1107.json"),
    ]
    subset = ROOT / "data/train/annotations/llava_sft_coco1024.json"
    if subset.exists():
        sources.append(("llava_sft", "llava_sft_coco1024.json"))
    provenance = {}
    for source, name in sources:
        annotation = ROOT / "data/train/annotations" / name
        if not annotation.is_file():
            continue
        provenance[source] = dict(
            annotation=str(annotation.relative_to(ROOT)), sha256=sha256(annotation)
        )
        for row in iter_array(annotation):
            scanned[source] += 1
            if not row.get("image") or not row.get("conversations"):
                continue
            image = image_path(row["image"], source)
            if image is None:
                continue
            text = "\n".join(t["value"] for t in row["conversations"])
            length = len(text)
            bucket = (
                0
                if length <= 256
                else (1 if length <= 512 else (2 if length <= 1024 else 3))
            )
            key = (source, bucket)
            seen[key] += 1
            eligible[source] += 1
            item = dict(
                id=f"{source}:{row['id']}",
                image=str(image.relative_to(ROOT)),
                conversations=row["conversations"],
                source_id=row["id"],
                source=source,
                characters=length,
                length_bin=bucket,
            )
            # Oversample each stratum to resolve duplicate images and exclusions later.
            if len(pools[key]) < args.count:
                pools[key].append(item)
            else:
                j = rng.randrange(seen[key])
                if j < args.count:
                    pools[key][j] = item
    if not pools:
        raise SystemExit(
            "No usable training image/annotation pairs; download images first"
        )
    # Hash evaluation images to exclude byte-identical content, not only equal paths.
    excluded = set()
    for directory in [
        ROOT / "data/eval/pope/val2014",
        ROOT / "data/eval/scienceqa/images/test",
    ]:
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                excluded.add(sha256(path))
    for pool in pools.values():
        rng.shuffle(pool)
    keys = sorted(pools)
    chosen = []
    used = set()
    rejected = Counter()
    while len(chosen) < args.count and any(pools.values()):
        for key in keys:
            if not pools[key] or len(chosen) == args.count:
                continue
            item = pools[key].pop()
            path = ROOT / item["image"]
            digest = sha256(path)
            if digest in excluded or digest in used:
                rejected["duplicate_or_eval_overlap"] += 1
                continue
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    width, height = im.size
            except Exception:
                rejected["invalid_image"] += 1
                continue
            used.add(digest)
            item.update(image_sha256=digest, width=width, height=height)
            chosen.append(item)
    if len(chosen) != args.count:
        raise SystemExit(
            f"Only {len(chosen)} valid unique pairs; requested {args.count}"
        )
    rng.shuffle(chosen)
    # Token lengths use the exact downloaded LLM tokenizer without any model allocation.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "model/TinyLLaVA-1.5B", local_files_only=True, use_fast=False
    )
    for item in chosen:
        item["text_tokens"] = len(
            tokenizer("\n".join(t["value"] for t in item["conversations"]))["input_ids"]
        )
    dest = ROOT / "data/calibration/calibration.json"
    write_json(dest, chosen)
    manifest = dict(
        time=now(),
        seed=args.seed,
        count=len(chosen),
        sources=provenance,
        scanned=dict(scanned),
        eligible=dict(eligible),
        strata={f"{s}/length_bin_{b}": n for (s, b), n in seen.items()},
        selected_sources=dict(Counter(x["source"] for x in chosen)),
        selected_bins=dict(Counter(x["length_bin"] for x in chosen)),
        rejected=dict(rejected),
        policy="Streaming reservoir per source and character bin (<=256, <=512, <=1024, >1024), round-robin strata, unique image SHA256; exclude evaluation SHA256.",
        eval_overlap=0,
        sha256=sha256(dest),
        samples=[{k: v for k, v in x.items() if k != "conversations"} for x in chosen],
    )
    write_json(ROOT / "data/manifests/calibration.json", manifest)
    write_json(
        ROOT / "docs/calibration_summary.json",
        {k: v for k, v in manifest.items() if k != "samples"},
    )
    print(
        "Prepared",
        len(chosen),
        "calibration samples:",
        dict(Counter(x["source"] for x in chosen)),
    )


if __name__ == "__main__":
    main()
