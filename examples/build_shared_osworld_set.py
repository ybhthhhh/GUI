"""Build a compact, source-balanced OSWorld evaluation set directly from ZIPs.

Only the action JSONL and screenshots needed for one next-action prediction are
materialized.  This keeps the experiment independent of a full archive extract.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


def parse_archive(value: str) -> tuple[str, Path]:
    try:
        source, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("archive must be SOURCE=PATH") from exc
    if not source.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("source may contain only letters, digits, _ and -")
    return source, Path(raw_path)


def episode_index(archive: zipfile.ZipFile) -> dict[tuple[str, str], str]:
    """Map (application, episode UUID) to the member holding traj.jsonl."""
    result: dict[tuple[str, str], str] = {}
    for member in archive.namelist():
        parts = PurePosixPath(member).parts
        if len(parts) >= 3 and parts[-1] == "traj.jsonl":
            key = (parts[-3], parts[-2])
            result.setdefault(key, member)
    return result


def result_index(archive: zipfile.ZipFile) -> set[tuple[str, str]]:
    """Episode keys with an actual terminal `result.txt` in this archive."""
    result: set[tuple[str, str]] = set()
    for member in archive.namelist():
        parts = PurePosixPath(member).parts
        if len(parts) >= 3 and parts[-1] == "result.txt":
            result.add((parts[-3], parts[-2]))
    return result


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive-relative path: {value}")
    return path


def write_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def make_sample(rows: list[dict], target_index: int, episode_dir: Path, source: str, app: str, episode_id: str) -> dict:
    target = rows[target_index]
    history = []
    for row in rows[:target_index]:
        screenshot = episode_dir / safe_relative(str(row["screenshot_file"]))
        history.append(
            {
                "timestamp": row["step_num"],
                "pre_state": "previous GUI state",
                "action": row["action"],
                "outcome": "observed",
                "post_state": "next GUI state",
                "visual_evidence": [str(screenshot)],
                "subgoal": "unknown",
                "failure_status": "unknown",
                "temporal_links": [],
            }
        )
    target_screenshot = episode_dir / safe_relative(str(target["screenshot_file"]))
    return {
        "sample_id": f"{source}-{app}-{episode_id}-step-{target['step_num']}",
        "task": f"OSWorld {app} next-action prediction",
        "current_observation": "current OSWorld GUI screenshot",
        "gold_action": target["action"],
        "history": history,
        "current_screenshot": str(target_screenshot),
    }


parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--archive", type=parse_archive, action="append", required=True, metavar="SOURCE=ZIP")
parser.add_argument("--limit", type=int, default=20, help="number of episode IDs shared by every source")
parser.add_argument("--target-index", type=int, default=3, help="zero-based next action; requires that many prior events")
args = parser.parse_args()
if len(args.archive) < 2:
    parser.error("at least two --archive inputs are required")
if args.limit < 1 or args.target_index < 1:
    parser.error("limit and target-index must be positive")
if len({source for source, _ in args.archive}) != len(args.archive):
    parser.error("source names must be unique")

archives = [(source, path, zipfile.ZipFile(path)) for source, path in args.archive]
try:
    indexes = {source: episode_index(archive) for source, _, archive in archives}
    terminal_results = {source: result_index(archive) for source, _, archive in archives}
    common = set.intersection(*(set(index) for index in indexes.values()))
    eligible: list[tuple[str, str]] = []
    for key in sorted(common):
        if any(key not in terminal_results[source] for source, _, _ in archives):
            continue
        enough_steps = True
        for source, _, archive in archives:
            rows = [json.loads(line) for line in archive.read(indexes[source][key]).decode("utf-8").splitlines() if line]
            enough_steps &= len(rows) > args.target_index and all("screenshot_file" in row for row in rows[: args.target_index + 1])
        if enough_steps:
            eligible.append(key)
    selected = eligible[: args.limit]
    if len(selected) < args.limit:
        raise RuntimeError(f"only {len(selected)} eligible common episodes; requested {args.limit}")

    args.output.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output / "samples.jsonl"
    metadata_path = args.output / "metadata.jsonl"
    count = 0
    with dataset_path.open("w", encoding="utf-8") as dataset, metadata_path.open("w", encoding="utf-8") as metadata:
        for app, episode_id in selected:
            for source, _, archive in archives:
                traj_member = indexes[source][(app, episode_id)]
                prefix = str(PurePosixPath(traj_member).parent)
                episode_dir = args.output / "episodes" / source / app / episode_id
                rows = [json.loads(line) for line in archive.read(traj_member).decode("utf-8").splitlines() if line]
                needed = {"traj.jsonl"}
                needed.update(str(safe_relative(str(row["screenshot_file"]))) for row in rows[: args.target_index + 1])
                for relative in needed:
                    member = f"{prefix}/{relative}"
                    write_member(archive, member, episode_dir / safe_relative(relative))
                sample = make_sample(rows, args.target_index, episode_dir, source, app, episode_id)
                dataset.write(json.dumps(sample, ensure_ascii=False) + "\n")
                target = rows[args.target_index]
                metadata.write(json.dumps({"sample_id": sample["sample_id"], "source": source, "application": app, "episode_id": episode_id, "target_step": target.get("step_num"), "target_reward": target.get("reward"), "target_done": target.get("done")}) + "\n")
                count += 1
    print(json.dumps({"output": str(dataset_path), "metadata": str(metadata_path), "shared_episodes": len(selected), "sources": [source for source, _, _ in archives], "samples": count}))
finally:
    for _, _, archive in archives:
        archive.close()
