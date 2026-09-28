"""Count executable action-JSON coverage without printing trajectory text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def extract(value: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("name"), str) and isinstance(candidate.get("arguments"), dict):
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    total = Counter()
    names = Counter()
    argument_fields = Counter()
    target_names = Counter()
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        target = row.get("target_action", row.get("action"))
        for role, text in [("target", target)] + [("history", event["action"]) for event in row["history"]]:
            total[role] += 1
            parsed = extract(text)
            if parsed is None:
                total[role + "_unparsed"] += 1
                continue
            total[role + "_parsed"] += 1
            names[parsed["name"]] += 1
            if role == "target":
                target_names[parsed["name"]] += 1
            argument_fields.update(parsed["arguments"].keys())
    print(json.dumps({"counts": total, "names": names, "target_names": target_names,
                      "argument_fields": argument_fields}, indent=2))


if __name__ == "__main__":
    main()
