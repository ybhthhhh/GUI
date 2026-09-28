"""Build one shared 224px canonical trajectory manifest for all three VLMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui_memory_specialization.storyline_canonical import build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-cache", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.source, args.output, args.image_cache)
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
