"""Download one official CoMEM trajectory shard directly on the training node.

The downloader writes a ``.part`` file and resumes it with HTTP Range when the
Hugging Face endpoint supports ranges.  A completed file is atomically renamed
only after its expected size is verified, so a partially transferred archive is
never mistaken for a training corpus.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DATASET = "WenyiWU0111/CoMEM-agent-memory-trajectories"


def build_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/main/{path}?download=true"


def open_url(url: str, proxy: str | None, offset: int):
    handlers = [urllib.request.ProxyHandler({"https": proxy})] if proxy else []
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, headers={"User-Agent": "gui-memory-specialization/1.0"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    return opener.open(request, timeout=120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="repository-relative CoMEM shard path")
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--proxy", help="HTTP proxy URL used by the node")
    args = parser.parse_args()
    if args.expected_bytes <= 0:
        raise ValueError("--expected-bytes must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > args.expected_bytes:
        raise ValueError(f"partial file is larger than expected: {offset} > {args.expected_bytes}")
    retries = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > args.expected_bytes:
            raise ValueError(f"partial file is larger than expected: {offset} > {args.expected_bytes}")
        try:
            response = open_url(build_url(args.path), args.proxy, offset)
            status = getattr(response, "status", response.getcode())
            if offset and status != 206:
                # The endpoint ignored Range.  Start cleanly rather than
                # appending a duplicate stream.
                partial.unlink()
                response.close()
                continue
            next_report = ((offset // (1024**3)) + 1) * (1024**3)
            mode = "ab" if offset else "wb"
            with response, partial.open(mode) as destination:
                while block := response.read(8 * 1024 * 1024):
                    destination.write(block)
                    offset += len(block)
                    if offset >= next_report:
                        print(f"downloaded_gib={offset / 1024**3:.2f}", flush=True)
                        next_report += 1024**3
            if offset == args.expected_bytes:
                written = offset
                break
            raise RuntimeError(f"stream ended at {offset}, expected {args.expected_bytes}")
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            retries += 1
            if retries > 100:
                raise RuntimeError("download exceeded 100 reconnect attempts") from error
            print(f"download_retry={retries} offset={offset} error={error}", flush=True)
            time.sleep(min(60, 2 ** min(retries, 5)))
    if written != args.expected_bytes:
        raise RuntimeError(f"incomplete download: got {written}, expected {args.expected_bytes}")
    os.replace(partial, args.output)
    print(f"COMEM_SHARD_DOWNLOAD_OK path={args.path} bytes={written} output={args.output}")


if __name__ == "__main__":
    main()
