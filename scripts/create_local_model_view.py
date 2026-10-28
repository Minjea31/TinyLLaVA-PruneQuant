"""A derived config plus weight symlinks for unmodified official evaluation CLIs."""

import json
import os
from common import ROOT, write_json


def main():
    source = ROOT / "model/TinyLLaVA-1.5B"
    dest = ROOT / "model/checkpoints/TinyLLaVA-1.5B-local"
    dest.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    config["mm_vision_tower"] = str(ROOT / "model/TinyLLaVA-1.5B-SigLIP")
    write_json(dest / "config.json", config)
    for p in source.iterdir():
        if not p.is_file() or p.name == "config.json":
            continue
        target = dest / p.name
        if target.is_symlink():
            if target.resolve() != p.resolve():
                raise RuntimeError(f"Different symlink already exists: {target}")
        elif target.exists():
            raise RuntimeError(f"Will not overwrite {target}")
        else:
            target.symlink_to(os.path.relpath(p, dest))
    print(dest)


if __name__ == "__main__":
    main()
