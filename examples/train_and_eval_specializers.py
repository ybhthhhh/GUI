"""Fit per-consumer offline selectors and report an episode-held-out 3x3 matrix.

Inputs are candidate-reward cache logs produced by the frozen VLM adapters.  The
cache is deliberately separated from fitting: each selector is fit using only
its training episode groups; matrix scores are read only for held-out groups.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from gui_memory_specialization.cli import load_jsonl
from gui_memory_specialization.specialization import (
    LinearCandidateRanker,
    candidate_event_sets,
    candidate_features,
    candidate_key,
    episode_group,
    fit_ranker,
    is_test_group,
)


def load_cache(cache_dir: Path, consumer: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for path in sorted(cache_dir.glob(f"{consumer}-*.log")):
        payload = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if candidate.get("kind") == "consumer_candidate_reward_cache":
                payload = candidate
        if payload is None:
            raise ValueError(f"{path} has no completed candidate cache")
        if payload.get("consumer") != consumer:
            raise ValueError(f"{path} consumer does not match {consumer}")
        for row in payload["rows"]:
            if row["sample_id"] in rows:
                raise ValueError(f"duplicate cached sample {row['sample_id']}")
            scores = {str(item["key"]): float(item["score"]) for item in row["candidates"]}
            if len(scores) != len(row["candidates"]):
                raise ValueError(f"duplicate candidate in {row['sample_id']}")
            rows[str(row["sample_id"])] = scores
    if not rows:
        raise ValueError(f"no cache logs found for {consumer}")
    return rows


def load_terminal_rewards(path: Path) -> dict[str, float]:
    rewards: dict[str, float] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        reward = float(row["terminal_reward"])
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"{path}:{line_number} terminal reward is outside [0, 1]")
        if sample_id in rewards:
            raise ValueError(f"{path}:{line_number} duplicates {sample_id}")
        rewards[sample_id] = reward
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-events", type=int, default=2)
    parser.add_argument("--consumers", nargs="+", default=["qwen", "llava", "idefics"])
    parser.add_argument("--terminal-rewards", type=Path, help="episode terminal rewards; enables outcome-weighted offline training")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.max_events != 2:
        raise ValueError("this first formal run uses the preregistered two-event budget")
    groups = {episode_group(sample.sample_id) for sample in samples}
    train_groups = sorted(group for group in groups if not is_test_group(group))
    test_groups = sorted(group for group in groups if is_test_group(group))
    if not train_groups or not test_groups:
        raise ValueError("episode split is empty")
    caches = {consumer: load_cache(args.cache_dir, consumer) for consumer in args.consumers}
    sample_ids = {sample.sample_id for sample in samples}
    for consumer, cache in caches.items():
        if set(cache) != sample_ids:
            raise ValueError(f"{consumer} cache does not cover exactly the dataset rows")
    terminal_rewards = load_terminal_rewards(args.terminal_rewards) if args.terminal_rewards else None
    if terminal_rewards is not None and set(terminal_rewards) != sample_ids:
        raise ValueError("terminal rewards do not cover exactly the dataset rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankers: dict[str, LinearCandidateRanker] = {}
    train_rows = [sample for sample in samples if not is_test_group(episode_group(sample.sample_id))]
    test_rows = [sample for sample in samples if is_test_group(episode_group(sample.sample_id))]
    for consumer in args.consumers:
        rows = []
        for sample in train_rows:
            candidates = candidate_event_sets(sample, args.max_events)
            cache = caches[consumer][sample.sample_id]
            keys = [candidate_key(events) for events in candidates]
            if set(cache) != set(keys):
                raise ValueError(f"candidate mismatch for {consumer}/{sample.sample_id}")
            reward = terminal_rewards[sample.sample_id] if terminal_rewards is not None else 1.0
            # result.txt evaluates the whole executed trajectory.  Failed rows
            # are deliberately not cloned as positive action demonstrations.
            if reward <= 0.0:
                continue
            center = sum(cache[key] for key in keys) / len(keys)
            rows.extend((candidate_features(sample, events), reward * (cache[candidate_key(events)] - center)) for events in candidates)
        if not rows:
            raise ValueError(f"{consumer} has no positive-return training rows")
        ranker = fit_ranker(f"M_{consumer}", rows)
        rankers[consumer] = ranker
        model_payload = {
            "kind": "terminal_reward_weighted_offline_memory_ranker" if terminal_rewards is not None else "offline_consumer_specialized_memory_ranker",
            "producer": ranker.name,
            "trained_consumer": consumer,
            "max_events": args.max_events,
            "train_episode_groups": train_groups,
            "heldout_episode_groups": test_groups,
            "ranker": ranker.to_dict(),
            "training_signal": "positive terminal-result weighted consumer likelihood" if terminal_rewards is not None else "consumer action likelihood",
        }
        (args.output_dir / f"{ranker.name}.json").write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    matrix = []
    selected = {consumer: Counter() for consumer in args.consumers}
    for producer_consumer, ranker in rankers.items():
        for sample in test_rows:
            chosen = tuple(ranker.select(sample, args.max_events))
            selected[producer_consumer][candidate_key(chosen)] += 1
        for scored_consumer in args.consumers:
            values = []
            for sample in test_rows:
                chosen = tuple(ranker.select(sample, args.max_events))
                key = candidate_key(chosen)
                values.append(caches[scored_consumer][sample.sample_id][key])
            matrix.append({
                "producer": ranker.name,
                "consumer": scored_consumer,
                "mean_score": sum(values) / len(values),
                "n": len(values),
            })
    contrasts = {}
    for consumer in args.consumers:
        own = next(cell["mean_score"] for cell in matrix if cell["producer"] == f"M_{consumer}" and cell["consumer"] == consumer)
        other = [cell["mean_score"] for cell in matrix if cell["producer"] != f"M_{consumer}" and cell["consumer"] == consumer]
        contrasts[consumer] = own - sum(other) / len(other)
    report = {
        "kind": "episode_heldout_terminal_reward_weighted_consumer_specialization_3x3" if terminal_rewards is not None else "episode_heldout_offline_consumer_specialization_3x3",
        "objective": "terminal-result weighted positive-trajectory preference; no consumer weights updated" if terminal_rewards is not None else "frozen-consumer action-likelihood preference; no consumer weights updated",
        "split": {"train_episode_groups": train_groups, "heldout_episode_groups": test_groups, "train_n": len(train_rows), "test_n": len(test_rows)},
        "max_events": args.max_events,
        "matrix": matrix,
        "matched_minus_transferred": contrasts,
        "heldout_selected_candidates": {producer: dict(counts) for producer, counts in selected.items()},
        "terminal_reward_training": (
            {"train_positive_n": sum(terminal_rewards[row.sample_id] > 0.0 for row in train_rows), "test_positive_n": sum(terminal_rewards[row.sample_id] > 0.0 for row in test_rows), "train_mean": sum(terminal_rewards[row.sample_id] for row in train_rows) / len(train_rows), "test_mean": sum(terminal_rewards[row.sample_id] for row in test_rows) / len(test_rows)}
            if terminal_rewards is not None else None
        ),
        "limitation": "Terminal results belong to logged trajectories, not counterfactual selector rollouts. The evaluation matrix remains held-out action likelihood; online task-success validation requires an executable OSWorld environment.",
    }
    (args.output_dir / "formal_3x3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
