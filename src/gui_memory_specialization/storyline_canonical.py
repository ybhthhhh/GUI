"""Model-independent observed GUI transitions for the first storyline gate.

No outcome or success label is invented from CoMEM's recorded actions.  The
observable change between adjacent screenshots is the only transition signal.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


IMAGE_SIDE = 224
MAX_TARGET_CHARS = 2200
MAX_HISTORY_ACTION_CHARS = 512


def _rgb(image) -> list[float]:
    from PIL import ImageStat

    return [round(value / 255.0, 6) for value in ImageStat.Stat(image).mean]


def _letterbox(source: Path, cache: Path) -> tuple[str, list[float]]:
    from PIL import Image, ImageOps

    if not source.is_file():
        raise FileNotFoundError(source)
    key = hashlib.sha256((str(source.resolve()) + f"|letterbox{IMAGE_SIDE}").encode()).hexdigest()
    target = cache / f"{key}.png"
    if not target.is_file():
        with Image.open(source) as original:
            fitted = ImageOps.contain(original.convert("RGB"), (IMAGE_SIDE, IMAGE_SIDE), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (IMAGE_SIDE, IMAGE_SIDE), (0, 0, 0))
            canvas.paste(fitted, ((IMAGE_SIDE - fitted.width) // 2, (IMAGE_SIDE - fitted.height) // 2))
        temporary = target.with_suffix(".tmp")
        canvas.save(temporary, format="PNG")
        temporary.replace(target)
    with Image.open(target) as image:
        return str(target.resolve()), _rgb(image)


def canonicalize(row: dict, cache: Path) -> dict:
    """Link each prior pre-state/action to its actually observed next state."""
    history = row["history"]
    if not 2 <= len(history) <= 3:
        raise ValueError("first gate uses two or three candidate transitions")
    if len(row["action"]) > MAX_TARGET_CHARS:
        raise ValueError("target action exceeds the common cap")
    cache.mkdir(parents=True, exist_ok=True)
    source_images = [item["image"] for item in history] + [row["current_image"]]
    states = [_letterbox(Path(path), cache) for path in source_images]
    events = []
    for index, item in enumerate(history):
        before, pre_rgb = states[index]
        after, post_rgb = states[index + 1]
        events.append({
            "timestamp": index,
            "pre_state_image": before,
            "action": item["action"][:MAX_HISTORY_ACTION_CHARS],
            "post_state_image": after,
            "outcome": {
                "observed_rgb_l1": round(sum(abs(a - b) for a, b in zip(pre_rgb, post_rgb)) / 3.0, 6),
                "success": None,
            },
            "pre_rgb": pre_rgb,
            "post_rgb": post_rgb,
            "failure_status": None,
            "subgoal": None,
            "temporal_link_to_next": index + 1 if index + 1 < len(history) else None,
        })
    return {
        "sample_id": row["sample_id"], "episode_id": row["episode_id"], "split": row["split"],
        "task": row["task"], "history": events, "current_image": states[-1][0],
        "current_rgb": states[-1][1], "target_action": row["action"],
    }


def candidate_features(row: dict, event: dict) -> list[float]:
    """Observable features only; never inspect target_action or success labels."""
    action = event["action"].lower()
    current = row["current_rgb"]
    before = event["pre_rgb"]
    after = event["post_rgb"]
    patterns = (r"\bclick\b", r"\btype\b|\bwrite\b", r"\bscroll\b", r"\bpress\b|\bhotkey\b")
    return [
        event["timestamp"] / max(len(row["history"]) - 1, 1),
        (len(row["history"]) - 1 - event["timestamp"]) / 2.0,
        min(len(action), MAX_HISTORY_ACTION_CHARS) / MAX_HISTORY_ACTION_CHARS,
        *[float(bool(re.search(pattern, action))) for pattern in patterns],
        *before, *after,
        event["outcome"]["observed_rgb_l1"],
        sum(abs(a - b) for a, b in zip(after, current)) / 3.0,
    ]


def candidate_messages(row: dict, event: dict) -> list[dict]:
    """Same ordered visual evidence and literal field wording for all backends."""
    def message(path: str, description: str) -> dict:
        return {"role": "user", "content": [{"type": "image", "path": path}, {"type": "text", "text": description}]}

    outcome = event["outcome"]["observed_rgb_l1"]
    return [
        message(event["pre_state_image"], f"Event timestamp={event['timestamp']}; field=pre_state."),
        message(event["post_state_image"],
                f"Event action={event['action']}; field=post_state; observed_rgb_l1={outcome:.6f}; "
                "success=unknown; failure_status=unknown; subgoal=unknown."),
        message(row["current_image"], f"Task: {row['task']}\nField=current_state. Choose the next webpage action."),
    ]


def build(source: Path, output: Path, image_cache: Path) -> dict:
    counts = {"train": 0, "validation": 0, "excluded_short_history": 0, "excluded_long_target": 0}
    episodes = {"train": set(), "validation": set()}
    seen = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with source.open(encoding="utf-8") as input_file, temporary.open("w", encoding="utf-8") as output_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if not 2 <= len(row["history"]) <= 3:
                counts["excluded_short_history"] += 1
                continue
            if len(row["action"]) > MAX_TARGET_CHARS:
                counts["excluded_long_target"] += 1
                continue
            if row["sample_id"] in seen:
                raise ValueError(f"duplicate sample ID: {row['sample_id']}")
            seen.add(row["sample_id"])
            canonical = canonicalize(row, image_cache)
            split = canonical["split"]
            if split not in episodes:
                raise ValueError(f"unknown split: {split}")
            episodes[split].add(canonical["episode_id"])
            output_file.write(json.dumps(canonical, ensure_ascii=False) + "\n")
            counts[split] += 1
    overlap = episodes["train"] & episodes["validation"]
    if overlap:
        temporary.unlink()
        raise ValueError(f"train/validation episode leakage: {len(overlap)} episodes")
    temporary.replace(output)
    return {**counts, "train_episodes": len(episodes["train"]),
            "validation_episodes": len(episodes["validation"]), "image_side": IMAGE_SIDE,
            "memory_budget": "one complete observed pre/action/post transition",
            "outcome_semantics": "observed visual change only; no claimed task success"}
