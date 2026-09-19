"""Wait for sharded frozen-policy caches, then run one complete training pass."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def completed(path: Path, consumer: str) -> bool:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("kind") == "consumer_candidate_reward_cache" and payload.get("consumer") == consumer:
            return True
    return False


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path)
parser.add_argument("cache_dir", type=Path)
parser.add_argument("terminal_rewards", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--poll-seconds", type=int, default=60)
args = parser.parse_args()
expected = {"qwen": 2, "llava": 4, "idefics": 2}

while True:
    failures = []
    done = True
    for consumer, count in expected.items():
        for shard in range(count):
            path = args.cache_dir / f"{consumer}-{shard}.log"
            if not path.exists():
                done = False
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "Traceback" in text or "CUDA out of memory" in text:
                failures.append(str(path))
            if not completed(path, consumer):
                done = False
    if failures:
        raise SystemExit(f"cache shard failed: {', '.join(failures)}")
    if done:
        break
    print("waiting for complete candidate caches", flush=True)
    time.sleep(args.poll_seconds)

command = [
    sys.executable,
    "examples/train_and_eval_specializers.py",
    str(args.dataset),
    str(args.cache_dir),
    str(args.output_dir),
    "--max-events", "2",
    "--terminal-rewards", str(args.terminal_rewards),
]
print("all caches complete; running outcome-weighted training", flush=True)
subprocess.run(command, check=True)
