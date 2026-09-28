"""Train independent canonical event selectors and evaluate the true 3x3.

Candidate gold-action losses are precomputed with frozen backbones. Optimizing
expected downstream loss differentiates through each selector's probabilities,
not through the expensive VLM. This is an offline decision-loss gate, not
online executed task-success evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from gui_memory_specialization.storyline_canonical import candidate_features


BACKENDS = ("qwen25vl", "llava_next", "internvl2")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate canonical sample IDs")
    train = {row["episode_id"] for row in rows if row["split"] == "train"}
    valid = {row["episode_id"] for row in rows if row["split"] == "validation"}
    if not train or not valid or train & valid:
        raise ValueError("empty or leaking episode split")
    return by_id


def load_cache(directory: Path, backend: str, manifest_sha: str, manifest: dict[str, dict]) -> dict[str, list[float]]:
    records = {}
    for rank in range(8):
        meta = json.loads((directory / f"rank-{rank}.meta.json").read_text(encoding="utf-8"))
        complete = json.loads((directory / f"rank-{rank}.complete.json").read_text(encoding="utf-8"))
        if meta != {key: complete[key] for key in meta} or meta["backend"] != backend:
            raise ValueError(f"{backend} rank {rank} metadata/completion mismatch")
        if meta["manifest_sha256"] != manifest_sha or meta["world_size"] != 8:
            raise ValueError(f"{backend} rank {rank} scored a different manifest")
        if (directory / f"rank-{rank}.errors.jsonl").read_text(encoding="utf-8").strip():
            raise ValueError(f"{backend} rank {rank} has errors")
        lines = [json.loads(line) for line in (directory / f"rank-{rank}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != complete["completed_rows"]:
            raise ValueError(f"{backend} rank {rank} completed count mismatch")
        for row in lines:
            sample_id = row["sample_id"]
            if sample_id not in manifest or sample_id in records:
                raise ValueError(f"{backend} cache has absent or duplicate sample {sample_id}")
            events = manifest[sample_id]["history"]
            candidates = row["candidates"]
            if [item["event_index"] for item in candidates] != [event["timestamp"] for event in events]:
                raise ValueError(f"{backend} candidate order mismatch for {sample_id}")
            values = [float(item["action_ce"]) for item in candidates]
            if not all(0 <= value < 100 for value in values):
                raise ValueError(f"{backend} invalid candidate loss for {sample_id}")
            records[sample_id] = values
    if set(records) != set(manifest):
        raise ValueError(f"{backend} candidate cache coverage is incomplete: {len(records)}/{len(manifest)}")
    return records


def calibration_episode(episode_id: str) -> bool:
    return int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16) % 10 == 0


def matrix_arrays(rows: list[dict], rewards: dict[str, list[float]], torch):
    features = torch.zeros((len(rows), 3, 15), dtype=torch.float32)
    losses = torch.zeros((len(rows), 3), dtype=torch.float32)
    mask = torch.zeros((len(rows), 3), dtype=torch.bool)
    for index, row in enumerate(rows):
        values = rewards[row["sample_id"]]
        for candidate, (event, value) in enumerate(zip(row["history"], values)):
            features[index, candidate] = torch.tensor(candidate_features(row, event))
            losses[index, candidate] = value
            mask[index, candidate] = True
    return features, losses, mask


def selector_module(torch):
    return torch.nn.Sequential(torch.nn.Linear(15, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 32), torch.nn.Tanh(),
                               torch.nn.Linear(32, 1))


def fit_selector(backend: str, manifest: dict[str, dict], rewards: dict[str, list[float]], torch):
    fit_rows = [row for row in manifest.values() if row["split"] == "train" and not calibration_episode(row["episode_id"])]
    cal_rows = [row for row in manifest.values() if row["split"] == "train" and calibration_episode(row["episode_id"])]
    if not fit_rows or not cal_rows:
        raise ValueError("fit or calibration episode split is empty")
    x, y, mask = matrix_arrays(fit_rows, rewards, torch)
    cx, cy, cmask = matrix_arrays(cal_rows, rewards, torch)
    present = x[mask]
    mean, std = present.mean(0), present.std(0).clamp_min(1e-4)
    x, cx = (x - mean) / std, (cx - mean) / std
    torch.manual_seed(20260928)
    model = selector_module(torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
    best = None
    best_calibration = float("inf")
    stale = 0
    history = []
    for epoch in range(300):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(x).squeeze(-1).masked_fill(~mask, -1e9)
        probabilities = scores.softmax(-1)
        centered = y - (y * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        objective = (probabilities * centered).sum(-1).mean()
        objective.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            cscores = model(cx).squeeze(-1).masked_fill(~cmask, -1e9)
            chosen = cscores.argmax(-1)
            cal_loss = float(cy.gather(1, chosen[:, None]).mean())
        history.append({"epoch": epoch + 1, "fit_centered_expected_ce": float(objective.detach()),
                        "calibration_action_ce": cal_loss})
        if cal_loss < best_calibration - 1e-6:
            best_calibration = cal_loss
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= 30:
                break
    assert best is not None
    model.load_state_dict(best)
    return model, mean, std, {"backend": backend, "fit_rows": len(fit_rows), "calibration_rows": len(cal_rows),
                              "best_epoch": best_epoch, "calibration_action_ce": best_calibration,
                              "trace": history}


def choose(model, mean, std, row: dict, torch) -> int:
    x = torch.tensor([candidate_features(row, event) for event in row["history"]], dtype=torch.float32)
    with torch.no_grad():
        return int(model(((x - mean) / std)).flatten().argmax())


def episode_bootstrap_advantage(heldout: list[dict], choices: dict[str, list[int]],
                                caches: dict[str, dict[str, list[float]]], *, draws: int = 2000) -> dict:
    """Paired uncertainty for matched-vs-transferred CE, resampling episodes."""
    episodes: dict[str, list[int]] = {}
    for index, row in enumerate(heldout):
        episodes.setdefault(row["episode_id"], []).append(index)
    groups = list(episodes.values())
    rng = random.Random(20260928)
    result = {}
    for consumer in BACKENDS:
        differences = []
        for index, row in enumerate(heldout):
            values = caches[consumer][row["sample_id"]]
            matched = values[choices[consumer][index]]
            transferred = sum(values[choices[producer][index]] for producer in BACKENDS if producer != consumer) / 2
            differences.append(transferred - matched)
        samples = []
        for _ in range(draws):
            sampled_indices = [index for _ in groups for index in rng.choice(groups)]
            samples.append(sum(differences[index] for index in sampled_indices) / len(sampled_indices))
        samples.sort()
        result[consumer] = {"episode_count": len(groups), "mean_advantage": sum(differences) / len(differences),
                            "ci95": [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]],
                            "bootstrap_fraction_nonpositive": sum(value <= 0 for value in samples) / draws}
    return result


def producer_consumer_means(heldout: list[dict], choices: dict[str, list[int]],
                            caches: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    """Rows are memory producers; columns are frozen consumers."""
    if not heldout or any(len(choices[producer]) != len(heldout) for producer in BACKENDS):
        raise ValueError("nonempty held-out rows and aligned producer choices are required")
    return {
        producer: {
            consumer: sum(caches[consumer][row["sample_id"]][choice]
                          for row, choice in zip(heldout, choices[producer])) / len(heldout)
            for consumer in BACKENDS
        }
        for producer in BACKENDS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--only-backend", choices=BACKENDS,
                        help="fit one independent selector as soon as its complete cache is ready")
    args = parser.parse_args()
    import torch

    manifest = load_manifest(args.manifest)
    digest = file_sha256(args.manifest)
    selected_backends = (args.only_backend,) if args.only_backend else BACKENDS
    caches = {backend: load_cache(args.cache_root / backend, backend, digest, manifest) for backend in selected_backends}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selectors = {}
    training = {}
    for backend in selected_backends:
        model, mean, std, metadata = fit_selector(backend, manifest, caches[backend], torch)
        selectors[backend] = model, mean, std
        training[backend] = metadata
        payload = {"kind": "storyline_observed_transition_selector", "backend": backend,
                   "manifest_sha256": digest, "feature_count": 15,
                   "input_mean": mean.tolist(), "input_std": std.tolist(),
                   "state_dict": {key: value.tolist() for key, value in model.state_dict().items()},
                   "training": {key: value for key, value in metadata.items() if key != "trace"},
                   "objective": "offline expected frozen-consumer gold-action CE; no test-split reward used"}
        (args.output_dir / f"E_{backend}.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    if args.only_backend:
        print(json.dumps({"trained": args.only_backend, "training": {
            key: value for key, value in training[args.only_backend].items() if key != "trace"}}))
        return
    heldout = [row for row in manifest.values() if row["split"] == "validation"]
    choices = {producer: [choose(*selectors[producer], row, torch) for row in heldout] for producer in BACKENDS}
    heldout_decisions = [{"sample_id": row["sample_id"], "episode_id": row["episode_id"],
                          "selected_event": {producer: choices[producer][index] for producer in BACKENDS}}
                         for index, row in enumerate(heldout)]
    (args.output_dir / "heldout_choices.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in heldout_decisions), encoding="utf-8")
    means = producer_consumer_means(heldout, choices, caches)
    subgroups = {}
    for name, predicate in (
        ("shopping", lambda row: row["sample_id"].startswith("shopping")),
        ("services", lambda row: row["sample_id"].startswith("services")),
        ("two_event_history", lambda row: len(row["history"]) == 2),
        ("three_event_history", lambda row: len(row["history"]) == 3),
    ):
        indices = [index for index, row in enumerate(heldout) if predicate(row)]
        subgroups[name] = ({"n": len(indices), "matrix": producer_consumer_means(
            [heldout[index] for index in indices],
            {producer: [choices[producer][index] for index in indices] for producer in BACKENDS}, caches,
        )} if indices else None)
    baselines = {}
    for strategy, index_of in (("recent_one", lambda row: len(row["history"]) - 1),
                               ("oldest_one", lambda row: 0),
                               ("largest_observed_change", lambda row: max(range(len(row["history"])),
                                key=lambda i: row["history"][i]["outcome"]["observed_rgb_l1"]))):
        baselines[strategy] = {consumer: sum(caches[consumer][row["sample_id"]][index_of(row)]
                              for row in heldout) / len(heldout) for consumer in BACKENDS}
    report = {
        "kind": "storyline_canonical_producer_by_consumer_3x3_offline_gate",
        "metric": "held-out gold-action CE per model token (lower is better); compare within each consumer column",
        "train_n": sum(row["split"] == "train" for row in manifest.values()),
        "heldout_n": len(heldout), "manifest_sha256": digest,
        "interface": "same canonical 224px pre-state/action/post-state transition, one-event memory budget, same current observation",
        "matrix": means,
        "subgroup_matrices": subgroups,
        "matched_advantage_over_transferred": {consumer: sum(means[producer][consumer] for producer in BACKENDS if producer != consumer) / 2
                                                - means[consumer][consumer] for consumer in BACKENDS},
        "episode_bootstrap_matched_advantage": episode_bootstrap_advantage(heldout, choices, caches),
        "baselines": baselines,
        "selection_counts": {producer: dict(Counter(choices[producer])) for producer in BACKENDS},
        "training": {key: {field: value for field, value in metadata.items() if field != "trace"}
                     for key, metadata in training.items()},
        "limitations": ["Logged CoMEM success trajectories have no counterfactual executed outcomes.",
                        "This is an offline action-likelihood gate, not GUI task success.",
                        "Model-native tokenizers and vision towers differ even with identical canonical input images."],
    }
    (args.output_dir / "producer_consumer_3x3.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
