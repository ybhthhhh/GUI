"""Launch disjoint VLM evaluation shards, one model replica per requested GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--model-path", required=True)
parser.add_argument("--gpus", default="0,1,2,3")
parser.add_argument("--max-events", type=int, required=True)
args = parser.parse_args()

gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
if not gpus or len(set(gpus)) != len(gpus):
    parser.error("--gpus must name one or more distinct GPU ids")
args.output_dir.mkdir(parents=True, exist_ok=True)
launched = []
for shard_index, gpu in enumerate(gpus):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["LD_LIBRARY_PATH"] = "/usr/local/corex-3.2.3.zte20250822/lib64"
    environment["PYTHONPATH"] = "src:/usr/local/corex-3.2.3.zte20250822/lib64/python3/dist-packages"
    command = [
        sys.executable, "-m", "gui_memory_specialization.cli", "vl-pilot", str(args.dataset),
        "--model-path", args.model_path, "--device", "cuda:0", "--dtype", "bfloat16",
        "--max-events", str(args.max_events), "--shard-index", str(shard_index), "--num-shards", str(len(gpus)),
    ]
    log_path = args.output_dir / f"shard-{shard_index}-gpu-{gpu}.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
    launched.append({"shard": shard_index, "gpu": gpu, "pid": process.pid, "log": str(log_path)})
print(json.dumps({"launched": launched, "samples_file": str(args.dataset), "max_events": args.max_events}))
