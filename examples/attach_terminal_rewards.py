"""Attach OSWorld Verified episode-level result.txt values to derived rows.

The original trajectory archives are read in place.  This script never infers a
per-step reward: every derived row receives only its source episode's terminal
evaluation result.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


def parse_archive(value: str) -> tuple[str, Path]:
    try:
        source, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("archive must be SOURCE=PATH") from error
    return source, Path(raw_path)


def terminal_index(path: Path) -> dict[tuple[str, str], float]:
    index: dict[tuple[str, str], float] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            parts = PurePosixPath(member).parts
            # Archives differ in whether they retain a top-level run directory;
            # both `app/uuid/result.txt` and `run/app/uuid/result.txt` are valid.
            if len(parts) < 3 or parts[-1] != "result.txt":
                continue
            key = (parts[-3], parts[-2])
            value = float(archive.read(member).decode("utf-8").strip())
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid terminal reward in {path}:{member}: {value}")
            if key in index:
                raise ValueError(f"duplicate terminal reward for {key} in {path}")
            index[key] = value
    return index


parser = argparse.ArgumentParser()
parser.add_argument("metadata", type=Path, help="derived metadata.jsonl")
parser.add_argument("output", type=Path)
parser.add_argument("--archive", type=parse_archive, action="append", required=True, metavar="SOURCE=ZIP")
args = parser.parse_args()

by_source = {source: terminal_index(path) for source, path in args.archive}
if len(by_source) != len(args.archive):
    raise SystemExit("archive source names must be unique")
rows = []
for line_number, line in enumerate(args.metadata.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
        continue
    row = json.loads(line)
    source = str(row["source"])
    key = (str(row["application"]), str(row["episode_id"]))
    if source not in by_source or key not in by_source[source]:
        raise ValueError(f"{args.metadata}:{line_number} has no terminal reward")
    rows.append({"sample_id": str(row["sample_id"]), "terminal_reward": by_source[source][key]})
if len({row["sample_id"] for row in rows}) != len(rows):
    raise ValueError("duplicate sample IDs in metadata")
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\n")
print(json.dumps({"output": str(args.output), "n": len(rows), "terminal_reward_counts": Counter(row["terminal_reward"] for row in rows)}))
