"""Rebuild canonical events using only parsed executable GUI commands."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from gui_memory_specialization.storyline_actions import executable_action
from gui_memory_specialization.storyline_canonical import (
    MAX_HISTORY_ACTION_CHARS,
    MAX_TARGET_CHARS,
    canonicalize,
)


def build(source: Path, output: Path, image_cache: Path) -> dict:
    counts = Counter()
    episodes = {"train": set(), "validation": set()}
    seen = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with source.open(encoding="utf-8") as incoming, temporary.open("w", encoding="utf-8") as outgoing:
        for line in incoming:
            if not line.strip():
                continue
            original = json.loads(line)
            if not 2 <= len(original["history"]) <= 3:
                counts["excluded_short_history"] += 1
                continue
            target = executable_action(original["action"])
            if target is None:
                counts["excluded_unparsed_target"] += 1
                continue
            if len(target) > MAX_TARGET_CHARS:
                counts["excluded_long_target"] += 1
                continue
            previous = [executable_action(item["action"]) for item in original["history"]]
            if any(action is None for action in previous):
                counts["excluded_unparsed_history"] += 1
                continue
            if any(len(action) > MAX_HISTORY_ACTION_CHARS for action in previous):
                counts["excluded_long_history_action"] += 1
                continue
            if original["sample_id"] in seen:
                raise ValueError(f"duplicate sample ID: {original['sample_id']}")
            seen.add(original["sample_id"])
            normalized = dict(original, action=target, history=[
                dict(item, action=action) for item, action in zip(original["history"], previous)
            ])
            row = canonicalize(normalized, image_cache)
            split = row["split"]
            if split not in episodes:
                raise ValueError(f"unknown split: {split}")
            episodes[split].add(row["episode_id"])
            outgoing.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    overlap = episodes["train"] & episodes["validation"]
    if overlap:
        temporary.unlink()
        raise ValueError(f"episode leakage: {len(overlap)}")
    temporary.replace(output)
    return {**counts, "train_episodes": len(episodes["train"]),
            "validation_episodes": len(episodes["validation"]),
            "action_interface": "parsed JSON name + operational arguments; no reasoning/description prose",
            "memory_budget": "one observed pre/action/post event, three 224px images with current state"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-cache", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.source, args.output, args.image_cache)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
