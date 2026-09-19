"""Convert an official CoMEM success shard into stage-one multimodal samples.

The input archive is read directly; only screenshots used by selected training
positions are materialized.  Records are derived solely from directories named
``success`` and retain an episode-stable validation split.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Iterator


NEGATIVE = ("early stop", "cannot", "not found", "not available", "can't")


def image_data_uri(value: Any) -> str | None:
    if isinstance(value, str):
        return value if value.startswith("data:image/") and ";base64," in value else None
    if isinstance(value, dict):
        for child in value.values():
            found = image_data_uri(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = image_data_uri(child)
            if found:
                return found
    return None


def cache_image(uri: str, image_root: Path) -> Path:
    header, encoded = uri.split(",", 1)
    suffix = ".png" if "png" in header.lower() else ".jpg"
    digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    path = image_root / f"{digest}{suffix}"
    if not path.exists():
        payload = base64.b64decode(encoded, validate=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
    return path


def valid_rounds(data: dict[str, Any]) -> list[dict[str, Any]]:
    rounds = data.get("rounds")
    if not isinstance(rounds, list) or not 3 <= len(rounds) < 10:
        return []
    result = []
    for item in rounds:
        if not isinstance(item, dict):
            return []
        action = item.get("response")
        image = image_data_uri(item)
        if not isinstance(action, str) or not image:
            return []
        result.append({"action": action, "image": image})
    if any(phrase in result[-1]["action"].lower() for phrase in NEGATIVE):
        return []
    return result


def eligible_members(archive: zipfile.ZipFile) -> list[str]:
    return [
        item.filename
        for item in archive.infolist()
        if not item.is_dir() and item.filename.endswith(".jsonl") and "/success/" in f"/{item.filename}"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=15_000)
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if args.max_samples <= 0 or args.history_steps <= 0:
        raise ValueError("sample and history limits must be positive")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.image_root.mkdir(parents=True, exist_ok=True)
    partial = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    kept = skipped = files = 0
    rng = random.Random(args.seed)
    with zipfile.ZipFile(args.archive) as archive, partial.open("w", encoding="utf-8") as output:
        members = eligible_members(archive)
        rng.shuffle(members)
        for member in members:
            if kept >= args.max_samples:
                break
            try:
                data = json.loads(archive.read(member))
                rounds = valid_rounds(data)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
                skipped += 1
                continue
            if not rounds:
                skipped += 1
                continue
            files += 1
            task = str(data.get("task_description", "Complete the webpage task."))
            episode = str(data.get("conversation_id", hashlib.sha256(member.encode()).hexdigest()[:16]))
            split = "validation" if int(hashlib.sha256(episode.encode()).hexdigest(), 16) % 10 == 0 else "train"
            # The first action has no history. Each later action supplies the
            # variable-length visual-action context consumed by the compressor.
            for index in range(1, len(rounds)):
                if kept >= args.max_samples:
                    break
                history = rounds[max(0, index - args.history_steps) : index]
                try:
                    record = {
                        "sample_id": f"{episode}-round-{index}",
                        "episode_id": episode,
                        "split": split,
                        "task": task,
                        "history": [
                            {
                                "image": str(cache_image(item["image"], args.image_root).resolve()),
                                "action": item["action"],
                            }
                            for item in history
                        ],
                        "current_image": str(cache_image(rounds[index]["image"], args.image_root).resolve()),
                        "action": rounds[index]["action"],
                    }
                except (ValueError, OSError):
                    skipped += 1
                    continue
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
    partial.replace(args.manifest)
    summary = {
        "kind": "comem_success_shard_stage1_manifest",
        "archive": str(args.archive.resolve()),
        "samples": kept,
        "eligible_success_files": len(members),
        "accepted_episodes": files,
        "skipped_files_or_samples": skipped,
        "history_steps": args.history_steps,
        "seed": args.seed,
    }
    args.manifest.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("COMEM_STAGE1_PREPARE_OK", json.dumps(summary))


if __name__ == "__main__":
    main()
