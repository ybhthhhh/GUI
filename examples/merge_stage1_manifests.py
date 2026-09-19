"""Merge independently prepared CoMEM stage-one manifests atomically.

The fixed episode-level split produced by ``prepare_comem_stage1.py`` is
preserved verbatim.  This script only combines disjoint source domains; it
never re-splits records after the fact.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    rows: list[dict] = []
    seen: set[str] = set()
    for source in args.inputs:
        for row in read_manifest(source):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"record without sample_id in {source}")
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id across manifests: {sample_id}")
            if row.get("split") not in {"train", "validation"}:
                raise ValueError(f"invalid split for {sample_id}")
            seen.add(sample_id)
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    print(
        "COMEM_STAGE1_MERGE_OK",
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(rows),
                "splits": Counter(row["split"] for row in rows),
                "sources": [str(source) for source in args.inputs],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
