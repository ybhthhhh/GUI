"""Aggregate weighted means from completed vl-pilot shard logs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("log_dir", type=Path)
args = parser.parse_args()

totals: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
logs = sorted(args.log_dir.glob("shard-*.log"))
if not logs:
    raise SystemExit("no shard logs found")
for log_path in logs:
    payload = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("kind") == "real_trajectory_single_consumer_pilot":
            payload = candidate
    if payload is None:
        raise SystemExit(f"{log_path} has no completed result")
    for result in payload["results"]:
        key = (result["baseline"], result["consumer"])
        totals[key][0] += result["mean_score"] * result["n"]
        totals[key][1] += result["mean_events"] * result["n"]
        totals[key][2] += result["n"]

results = [
    {"baseline": baseline, "consumer": consumer, "mean_score": weighted_score / n, "mean_events": weighted_events / n, "n": int(n)}
    for (baseline, consumer), (weighted_score, weighted_events, n) in sorted(totals.items())
]
print(json.dumps({"log_dir": str(args.log_dir), "shards": len(logs), "results": results}, indent=2))
