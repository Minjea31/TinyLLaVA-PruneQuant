"""Audit original annotation counts and actual local image coverage (bounded RAM)."""

from collections import Counter
import json
from common import ROOT, iter_array, now, write_json
from download_data import ANNOTATIONS


def main():
    inventory = dict(time=now(), training={}, evaluation={}, calibration={})
    for repo, name, rev in ANNOTATIONS:
        path = ROOT / "data/train/annotations" / name
        if not path.is_file():
            inventory["training"][name] = dict(status="missing")
            continue
        counts = Counter()
        missing = Counter()
        for row in iter_array(path):
            counts["records"] += 1
            image = row.get("image")
            if not image:
                counts["text_only"] += 1
                continue
            counts["image_records"] += 1
            if name.startswith("blip_"):
                local = ROOT / "data/train/llava/llava_pretrain/images" / image
            elif repo.startswith("Lin-Chen"):
                local = ROOT / "data/train/sharegpt4v" / image
            else:
                local = ROOT / "data/train/llava" / image
            if local.is_file():
                counts["linked_image_records"] += 1
            else:
                missing[image.split("/")[0]] += 1
        receipt = path.with_name(path.name + ".receipt.json")
        r = json.loads(receipt.read_text())
        inventory["training"][name] = dict(
            status="ready"
            if counts["image_records"] == counts["linked_image_records"]
            else "partial_images",
            counts=dict(counts),
            missing_by_prefix=dict(missing),
            revision=r["revision"],
            sha256=r["sha256"],
            bytes=path.stat().st_size,
        )
        print(name, dict(counts), flush=True)
    for name in ["pope", "scienceqa", "training_subset", "calibration"]:
        path = ROOT / f"data/manifests/{name}.json"
        if path.exists():
            r = json.loads(path.read_text())
            inventory["calibration" if name == "calibration" else "evaluation"][
                name
            ] = {k: v for k, v in r.items() if k not in ["samples", "files"]}
        else:
            inventory["evaluation"][name] = dict(status="missing")
    cal = ROOT / "data/calibration/calibration.json"
    if cal.exists():
        from PIL import Image
        from common import sha256

        rows = json.loads(cal.read_text())
        hashes = set()
        for row in rows:
            p = ROOT / row["image"]
            if sha256(p) != row["image_sha256"]:
                raise RuntimeError(f"Calibration image changed: {p}")
            with Image.open(p) as im:
                im.verify()
            if row["image_sha256"] in hashes:
                raise RuntimeError("Duplicate calibration image")
            hashes.add(row["image_sha256"])
        inventory["calibration"]["verification"] = dict(
            valid_images=len(rows), unique_hashes=len(hashes), status="passed"
        )
    write_json(ROOT / "docs/data_inventory.json", inventory)


if __name__ == "__main__":
    main()
