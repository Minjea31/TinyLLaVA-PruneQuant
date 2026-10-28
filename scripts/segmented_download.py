"""Resumable byte-range transfer for large public archives; final full SHA256 required."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
import requests
from common import ROOT, now, sha256, write_json
from filelock import FileLock


def repair_sparse(url, path, size, expected_sha256, workers=64):
    """Finish interrupted HF transport extents; accept only the official full LFS hash."""
    path = Path(path)
    fd = os.open(path, os.O_RDWR)
    current = os.fstat(fd).st_size
    if current > size:
        raise RuntimeError("Partial file exceeds official size")
    missing = []
    offset = 0
    while offset < current:
        try:
            data = os.lseek(fd, offset, os.SEEK_DATA)
        except OSError:
            missing.append((offset, current))
            break
        if data > offset:
            missing.append((offset, data))
        offset = os.lseek(fd, data, os.SEEK_HOLE)
    if current < size:
        missing.append((current, size))
    segments = []
    for start, end in missing:
        while start < end:
            stop = min(end, start + 1024 * 1024)
            segments.append((start, stop))
            start = stop
    os.ftruncate(fd, size)
    logging.info("Repairing %d missing MiB ranges in %s", len(segments), path.name)
    local = threading.local()
    lock = threading.Lock()
    completed = 0

    def part(bounds):
        nonlocal completed
        start, end = bounds
        if not hasattr(local, "session"):
            local.session = requests.Session()
        for attempt in range(6):
            try:
                with local.session.get(
                    url,
                    headers={
                        "Range": f"bytes={start}-{end - 1}",
                        "Accept-Encoding": "identity",
                    },
                    stream=True,
                    timeout=(8, 20),
                ) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or not r.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {start}-{end - 1}/"):
                        raise RuntimeError("Range not honored by official server")
                    # Buffer only 1 MiB, then commit the complete extent in one pwrite.
                    data = r.content
                    if len(data) != end - start:
                        raise RuntimeError("Incomplete range")
                    if os.pwrite(fd, data, start) != len(data):
                        raise OSError("Short pwrite")
                with lock:
                    completed += 1
                    if completed % 100 == 0:
                        logging.info(
                            "Repaired %d/%d MiB ranges", completed, len(segments)
                        )
                return
            except Exception:
                if attempt == 5:
                    raise
                local.session.close()
                local.session = requests.Session()
                time.sleep(1 + attempt)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for task in as_completed([pool.submit(part, x) for x in segments]):
                task.result()
        os.fsync(fd)
    finally:
        os.close(fd)
    if sha256(path) != expected_sha256:
        raise RuntimeError("Repair completed but official LFS SHA256 did not match")
    logging.info("Repaired file matches official SHA256: %s", path.name)


def transfer(
    url, path, size, expected_sha256, workers=12, existing=None, chunk=64 * 1024 * 1024
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_name(path.name + ".download.lock"))):
        return transfer_locked(
            url, path, size, expected_sha256, workers, existing, chunk
        )


def transfer_locked(
    url, path, size, expected_sha256, workers=12, existing=None, chunk=64 * 1024 * 1024
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size:
        recorded = path.with_name(path.name + ".receipt.json")
        expected = expected_sha256 or (
            json.loads(recorded.read_text())["sha256"] if recorded.exists() else None
        )
        if expected and sha256(path) == expected:
            return path
    partial = path.with_name(path.name + ".ranged-partial")
    state = path.with_name(path.name + ".ranges.json")
    chunks = (size + chunk - 1) // chunk
    if state.exists():
        info = json.loads(state.read_text())
        if (
            info["size"] != size
            or info["sha256"] != expected_sha256
            or info["chunk"] != chunk
        ):
            raise RuntimeError("Range manifest mismatch")
        done = set(info["done"])
        if not partial.exists():
            raise RuntimeError("Range data missing")
    else:
        prefix = 0
        if existing and Path(existing).exists():
            prefix = Path(existing).stat().st_size
            Path(existing).replace(partial)
        done = set(range(prefix // chunk))
        info = dict(
            url=url, size=size, sha256=expected_sha256, chunk=chunk, done=sorted(done)
        )
        write_json(state, info)
    remaining = sum(
        min(chunk, size - i * chunk) for i in range(chunks) if i not in done
    )
    if shutil.disk_usage(ROOT).free < remaining + 10 * 1024**3:
        raise RuntimeError("Insufficient disk for segmented archive")
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, size)
    lock = threading.Lock()

    def worker(index):
        start = index * chunk
        end = min(size, start + chunk) - 1
        for attempt in range(5):
            try:
                with requests.get(
                    url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                    },
                    stream=True,
                    timeout=(30, 120),
                ) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or not r.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError(
                            f"Byte range not honored: {index} / status {r.status_code}"
                        )
                    offset = start
                    for data in r.iter_content(4 * 1024 * 1024):
                        if offset + len(data) > end + 1:
                            raise RuntimeError("Excess range bytes")
                        written = os.pwrite(fd, data, offset)
                        if written != len(data):
                            raise OSError("Short pwrite")
                        offset += written
                    if offset != end + 1:
                        raise RuntimeError("Incomplete range")
                with lock:
                    os.fsync(fd)
                    done.add(index)
                    info["done"] = sorted(done)
                    write_json(state, info)
                    if len(done) % 10 == 0:
                        logging.info(
                            "Archive ranges: %d/%d (%.1f%%)",
                            len(done),
                            chunks,
                            100 * len(done) / chunks,
                        )
                return
            except Exception as e:
                if attempt == 4:
                    raise
                logging.warning(
                    "Range %d retry %d: %s", index, attempt + 1, type(e).__name__
                )
                time.sleep(2**attempt)

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            tasks = [ex.submit(worker, i) for i in range(chunks) if i not in done]
            for task in as_completed(tasks):
                task.result()
    finally:
        os.close(fd)
    actual = sha256(partial)
    if expected_sha256 and actual != expected_sha256:
        raise RuntimeError("Final archive SHA256 mismatch")
    partial.replace(path)
    write_json(
        path.with_name(path.name + ".receipt.json"),
        dict(url=url, bytes=size, sha256=actual, verified_at=now()),
    )
    return path
