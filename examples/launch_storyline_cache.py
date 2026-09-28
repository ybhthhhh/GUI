"""Detach eight independent GPU candidate scorers on one verified node."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--backend", choices=("qwen25vl", "llava_next", "internvl2"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    if socket.gethostname() != args.expected_hostname:
        raise RuntimeError(f"host mismatch: {socket.gethostname()} != {args.expected_hostname}")
    if not args.manifest.is_file() or not Path(args.model_path).is_dir():
        raise FileNotFoundError("manifest or model directory does not exist")
    import torch

    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"expected eight GPUs, found {torch.cuda.device_count()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = args.output_dir / "launch.json"
    if record.exists():
        raise RuntimeError(f"launch record already exists: {record}")
    scorer = Path(__file__).with_name("score_storyline_candidates.py")
    launched = []
    for rank in range(8):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(rank)
        command = [sys.executable, str(scorer), str(args.manifest), str(args.output_dir),
                   "--backend", args.backend, "--model-path", args.model_path,
                   "--rank", str(rank), "--world-size", "8", "--max-rows", str(args.max_rows)]
        log = (args.output_dir / f"rank-{rank}.log").open("ab")
        try:
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       close_fds=True)
        finally:
            log.close()
        launched.append({"rank": rank, "pid": process.pid, "gpu": rank})
    payload = {"backend": args.backend, "hostname": args.expected_hostname,
               "manifest": str(args.manifest), "max_rows_per_rank": args.max_rows,
               "started_utc": datetime.now(timezone.utc).isoformat(), "processes": launched}
    temporary = record.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(record)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
