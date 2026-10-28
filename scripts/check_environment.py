"""Read-only LLAVA environment audit; GPU failure is recorded, not fatal to downloads."""

import importlib
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
import psutil
from common import ROOT, make_dirs, now, write_json


def command(args):
    try:
        p = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=40,
        )
        return dict(returncode=p.returncode, output=p.stdout.strip())
    except Exception as e:
        return dict(error=repr(e))


def main():
    make_dirs()
    r = dict(
        time=now(),
        python=sys.version,
        executable=sys.executable,
        prefix=sys.prefix,
        platform=platform.platform(),
        os_release=platform.freedesktop_os_release(),
        kernel=platform.release(),
        ram=psutil.virtual_memory()._asdict(),
        swap=psutil.swap_memory()._asdict(),
        disk=shutil.disk_usage(ROOT)._asdict(),
        conda_prefix=os.environ.get("CONDA_PREFIX"),
        inherited_pythonpath=os.environ.get("PYTHONPATH"),
        packages={},
        imports={},
    )
    for package in [
        "torch",
        "torchvision",
        "transformers",
        "tokenizers",
        "accelerate",
        "peft",
        "huggingface_hub",
        "hf-transfer",
        "hf-xet",
        "safetensors",
        "numpy",
        "Pillow",
        "bitsandbytes",
        "timm",
        "sentencepiece",
        "datasets",
        "deepspeed",
        "flash-attn",
        "psutil",
    ]:
        try:
            r["packages"][package] = md.version(package)
        except md.PackageNotFoundError:
            r["packages"][package] = None
    for module in [
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "peft",
        "timm",
        "huggingface_hub",
    ]:
        try:
            m = importlib.import_module(module)
            r["imports"][module] = dict(status="ok", file=m.__file__)
            if module == "torch":
                available = m.cuda.is_available()
                r["cuda"] = dict(
                    torch_cuda=m.version.cuda,
                    available=available,
                    count=m.cuda.device_count(),
                    gpus=[],
                )
                if available:
                    for i in range(m.cuda.device_count()):
                        p = m.cuda.get_device_properties(i)
                        r["cuda"]["gpus"].append(
                            dict(
                                name=p.name,
                                bytes=p.total_memory,
                                capability=[p.major, p.minor],
                            )
                        )
        except Exception as e:
            r["imports"][module] = dict(status="error", error=repr(e))
    for name, args in dict(
        nvidia_smi=["nvidia-smi"],
        lspci=["lspci", "-nn"],
        nvcc=["nvcc", "--version"],
        modinfo=["modinfo", "nvidia"],
        pip_check=[sys.executable, "-m", "pip", "check"],
    ).items():
        r[name] = command(args)
    for path in [
        "/proc/driver/nvidia/version",
        "/proc/driver/nvidia/gpus/0000:01:00.0/information",
    ]:
        try:
            r[path] = open(path).read()
        except OSError as e:
            r[path] = repr(e)
    r["nvml_library"] = os.path.realpath("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
    r["repositories"] = {}
    for path in sorted((ROOT / "third_party").iterdir()):
        if (path / ".git").exists():
            r["repositories"][path.name] = dict(
                commit=command(["git", "-C", str(path), "rev-parse", "HEAD"]),
                remote=command(["git", "-C", str(path), "remote", "get-url", "origin"]),
                status=command(["git", "-C", str(path), "status", "--short"]),
            )
    write_json(ROOT / "docs/environment.json", r)
    frozen = command([sys.executable, "-m", "pip", "freeze"])
    if frozen.get("returncode") != 0:
        raise RuntimeError(frozen)
    # pip warnings go to stderr, use a separate stdout-only invocation for exact freeze.
    (ROOT / "docs/environment_requirements.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    (ROOT / "docs/conda_packages.txt").write_text(
        command(
            ["/home/a/anaconda3/bin/conda", "list", "-p", sys.prefix, "--explicit"]
        )["output"]
        + "\n"
    )
    print("Environment report:", ROOT / "docs/environment.json")
    print(
        "CUDA available:",
        r["cuda"]["available"],
        "; pip check:",
        r["pip_check"]["output"],
    )
    print(
        "Import failures:",
        {k: v for k, v in r["imports"].items() if v["status"] != "ok"},
    )


if __name__ == "__main__":
    main()
