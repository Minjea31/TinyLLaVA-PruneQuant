"""Restore recorded upstream commits into absent directories; preserve existing clones."""

import argparse
import json
import subprocess
from common import ROOT


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    pins = json.loads((ROOT / "configs/sources.json").read_text()).get(
        "repositories", {}
    )
    if not pins:
        raise SystemExit("Repository pins have not been finalized yet")
    for name, spec in pins.items():
        path = ROOT / "third_party" / name
        if path.exists():
            actual = git("-C", str(path), "rev-parse", "HEAD")
            if actual != spec["commit"]:
                raise RuntimeError(
                    f"Preserving existing {path}; commit differs: {actual}"
                )
            print(name, actual, "verified")
            continue
        if args.verify_only:
            raise RuntimeError(f"Missing repository {path}")
        branch = "tinyllava_bench" if name.endswith("_legacy") else "main"
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "--depth",
                "1",
                "--branch",
                branch,
                spec["remote"],
                str(path),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "fetch", "--depth", "1", "origin", spec["commit"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", spec["commit"]], check=True
        )
        print(name, spec["commit"], "restored")


if __name__ == "__main__":
    main()
