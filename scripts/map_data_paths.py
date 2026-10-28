"""Create nonduplicating path aliases expected by official training/evaluation code."""

import os
from common import ROOT, make_dirs


def link(source, target):
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Existing different mapping: {target}")
        return
    if target.exists():
        if target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        else:
            raise RuntimeError(f"Will not overwrite existing path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)


def main():
    make_dirs()
    llava = ROOT / "data/train/llava"
    share = ROOT / "data/train/sharegpt4v"
    # ShareGPT4V annotations contain llava/llava_pretrain/images/... and coco/... .
    link(llava, share / "llava")
    for name in ["coco", "gqa", "ocr_vqa", "textvqa", "vg"]:
        (llava / name).mkdir(exist_ok=True)
        link(llava / name, share / name)
    # Different upstream releases capitalize these names differently.
    link(ROOT / "data/eval/mme", ROOT / "data/eval/MME")
    link(ROOT / "data/eval/mmmu", ROOT / "data/eval/MMMU")
    link(ROOT / "data/eval/mmvet", ROOT / "data/eval/mm-vet")
    link(llava / "gqa/images", ROOT / "data/eval/gqa/images")
    link(llava / "textvqa/train_images", ROOT / "data/eval/textvqa/train_images")
    print("Data path aliases ready; absent image corpora remain explicitly absent.")


if __name__ == "__main__":
    main()
