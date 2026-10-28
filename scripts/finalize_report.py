"""Generate a factual, local completion summary from verified artifacts."""

import json
import os
from pathlib import Path
import shutil
import subprocess
from common import ROOT, now, write_json


def read(path):
    path = ROOT / path
    return json.loads(path.read_text()) if path.exists() else None


def size(directory):
    total = 0
    for base, dirs, files in os.walk(ROOT / directory, followlinks=False):
        for name in files:
            p = Path(base) / name
            if not p.is_symlink():
                total += p.stat().st_size
    return total


def main():
    inventory = read("docs/data_inventory.json") or {}
    models = {
        n: read(f"docs/{n}_snapshot.json")
        for n in ["TinyLLaVA-1.5B", "TinyLLaVA-1.5B-SigLIP"]
    }
    download_bytes = sum((x or {}).get("bytes", 0) for x in models.values())
    receipts = 0
    assets = []
    for p in (ROOT / "data").rglob("*.receipt.json"):
        r = json.loads(p.read_text())
        download_bytes += r["bytes"]
        receipts += 1
        assets.append(dict(local_receipt=str(p.relative_to(ROOT)), **r))
    write_json(ROOT / "docs/data_sources.json", dict(time=now(), assets=assets))
    report = dict(
        time=now(),
        models={
            k: dict(revision=v["revision"], bytes=v["bytes"], files=len(v["files"]))
            if v
            else dict(status="missing")
            for k, v in models.items()
        },
        verified_download_bytes=download_bytes,
        data_receipts=receipts,
        workspace_bytes={
            d: size(d)
            for d in [
                "model",
                "data",
                "third_party",
                "docs",
                "scripts",
                "logs",
                "results",
            ]
        },
        disk=shutil.disk_usage(ROOT)._asdict(),
        local_load=read("docs/local_load_validation.json"),
        calibration=read("docs/calibration_summary.json"),
        baseline=read("results/baseline/smoke/status.json"),
        gpu="evaluation deferred: host NVML 595.99 vs loaded kernel module 595.91.07; host CUDA metadata works, VRAM 8222998528 bytes",
        unfinished=[
            "GPU FP16 generation/benchmark scores and peak inference VRAM",
            "Full ShareGPT4V SAM/WebData images and full 665K SFT image coverage",
            "TextVQA/GQA/VQAv2/MME/MM-Vet/MMMU adapters and full benchmark execution",
            "MegaTrain TinyLLaVA adapter and forward/backward: not directly compatible",
        ],
        repositories={},
    )
    for path in sorted((ROOT / "third_party").iterdir()):
        if (path / ".git").exists():

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(path), *args], text=True
                ).strip()

            report["repositories"][path.name] = dict(
                commit=git("rev-parse", "HEAD"),
                remote=git("remote", "get-url", "origin"),
                status=git("status", "--short"),
            )
    pins = read("configs/sources.json")
    pins["repositories"] = report["repositories"]
    write_json(ROOT / "configs/sources.json", pins)
    write_json(ROOT / "docs/completion_report.json", report)
    gib = lambda x: f"{x / 1024**3:.2f} GiB"
    checks = [
        ("시스템 환경 확인", True),
        ("폴더 구조 구성", True),
        ("TinyLLaVA 및 별도 Vision snapshot 다운로드/해시 검증", all(models.values())),
        (
            "safetensors 구조/인덱스 검증",
            all(read(f"docs/{n}_analysis.json") for n in models),
        ),
        (
            "학습 Annotation 네 종류",
            len(inventory.get("training", {})) == 4
            and all(x.get("sha256") for x in inventory.get("training", {}).values()),
        ),
        (
            "사용 가능한 학습 이미지/텍스트 연결",
            bool(read("data/manifests/training_subset.json")),
        ),
        (
            "POPE·ScienceQA 초기 평가 데이터",
            all(read(f"data/manifests/{n}.json") for n in ["pope", "scienceqa"]),
        ),
        (
            "Calibration 512 samples",
            report["calibration"] is not None and report["calibration"]["count"] == 512,
        ),
        (
            "로컬 모델 CPU load/실제 이미지 전처리",
            report["local_load"] is not None
            and report["local_load"].get("finite") is True,
        ),
        ("MegaTrain 코드 호환성 검토", bool(read("docs/megatrain_compatibility.json"))),
        ("FP16 GPU 추론/평가", False),
        ("README 및 실행 가이드", True),
    ]
    text = (
        "# 실제 완료 보고서\n\n"
        + f"검사 시각: {report['time']}. 기존 LLAVA 환경만 사용했다.\n\n"
    )
    text += (
        "\n".join(f"- [{'x' if ready else ' '}] {label}" for label, ready in checks)
        + "\n\n"
    )
    text += f"검증된 다운로드 파일 합계: **{gib(download_bytes)}** ({download_bytes:,} bytes). HTTP/API 소량 metadata traffic은 제외한다.\n\n"
    text += f"실제 보관 model: {gib(report['workspace_bytes']['model'])}, data (압축본+해제본 포함): {gib(report['workspace_bytes']['data'])}. "
    text += f"디스크 여유: **{gib(report['disk']['free'])}**.\n\n"
    text += "데이터 파일별 출처/해시는 `data_sources.json`, 모델별 해시는 `*_snapshot.json`에 보존했다.\n\n"
    text += "| Snapshot | Revision | 파일 크기 |\n| --- | --- | --- |\n"
    for name, v in models.items():
        if v:
            text += f"| {name} | `{v['revision']}` | {gib(v['bytes'])} |\n"
    text += "\n| 학습 Annotation | 실제 records | 이미지가 연결된 records | 상태 |\n| --- | ---: | ---: | --- |\n"
    for name, r in inventory.get("training", {}).items():
        c = r.get("counts", {})
        text += f"| {name} | {c.get('records', 0):,} | {c.get('linked_image_records', 0):,} | {r['status']} |\n"
    text += "\nAnnotation만 있는 records는 사용 가능한 학습 데이터로 집계하지 않았다. text-only records는 이미지 미연결과 구분한다.\n\n"
    for name in ["pope", "scienceqa"]:
        r = read(f"data/manifests/{name}.json")
        if r:
            text += (
                f"- {name}: "
                + json.dumps(
                    {
                        k: v
                        for k, v in r.items()
                        if k in ["questions", "images", "image_questions", "split"]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    text += "\n저장소 commit:\n\n"
    for name, r in report["repositories"].items():
        text += f"- {name}: `{r['commit']}` (수정 상태: {r['status'] or 'clean'})\n"
    text += "\n## 제약과 미완료 항목\n\n"
    text += "호스트 NVML 595.99 / loaded module 595.91.07 불일치가 남아 있어 사용자 지침에 따라 GPU 평가를 보류했다. 호스트 CUDA metadata 조회는 성공했고 GPU VRAM은 8,222,998,528 bytes로 확인했다. FP16 generation과 benchmark 점수는 미실행이다. "
    text += "드라이버/시스템 CUDA는 변경하지 않았다. 원본 모델 FP16 메모리 크기 약 2.80 GiB는 Peak VRAM 실측값이 아니다.\n\n"
    text += "전체 ShareGPT4V PT는 추가 SAM/WebData와 상당한 디스크가 필요해 보류했다. 전체 LLaVA SFT 이미지도 초기 1,024-image subset과 구분하며 추가 archive script를 제공했다. "
    text += "다른 6개 benchmark는 다운로드 도구·공식 평가 절차를 제공했으나 이번 초기 실행 완료 대상으로 집계하지 않는다. "
    text += "MegaTrain은 TinyLLaVA의 image token/gradient 경로가 직접 호환되지 않아 학습 backend 설치/실행을 보류했다.\n\n"
    text += "GPU 정상화 후 `source scripts/env.sh` → `python scripts/check_environment.py` → `python scripts/test_inference.py` → POPE/ScienceQA full baseline을 실행한다. "
    text += "Quantization 및 재학습은 실행하지 않았다.\n"
    (ROOT / "docs/completion_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
