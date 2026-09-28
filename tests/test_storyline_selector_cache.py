import json

import pytest

from examples.train_storyline_selectors import (
    episode_bootstrap_advantage,
    load_cache,
    producer_consumer_means,
)


def test_cache_requires_same_manifest_and_complete_coverage(tmp_path):
    manifest = {"sample-0": {"history": [{"timestamp": 0}, {"timestamp": 1}]}}
    digest = "abc"
    for rank in range(8):
        metadata = {"backend": "qwen25vl", "rank": rank, "world_size": 8, "manifest_sha256": digest}
        (tmp_path / f"rank-{rank}.meta.json").write_text(json.dumps(metadata))
        (tmp_path / f"rank-{rank}.complete.json").write_text(json.dumps({**metadata, "completed_rows": int(rank == 0)}))
        (tmp_path / f"rank-{rank}.errors.jsonl").write_text("")
        records = ([{"sample_id": "sample-0", "candidates": [
            {"event_index": 0, "action_ce": 1.0}, {"event_index": 1, "action_ce": 2.0}]}]
                   if rank == 0 else [])
        (tmp_path / f"rank-{rank}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    assert load_cache(tmp_path, "qwen25vl", digest, manifest) == {"sample-0": [1.0, 2.0]}
    with pytest.raises(ValueError, match="different manifest"):
        load_cache(tmp_path, "qwen25vl", "changed", manifest)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        load_cache(tmp_path, "qwen25vl", digest, {**manifest, "sample-1": manifest["sample-0"]})


def test_episode_bootstrap_preserves_paired_consumer_advantage():
    rows = [{"sample_id": "a", "episode_id": "episode-a"},
            {"sample_id": "b", "episode_id": "episode-b"}]
    choices = {"qwen25vl": [0, 0], "llava_next": [1, 1], "internvl2": [2, 2]}
    caches = {"qwen25vl": {"a": [0, 1, 1], "b": [0, 1, 1]},
              "llava_next": {"a": [1, 0, 1], "b": [1, 0, 1]},
              "internvl2": {"a": [1, 1, 0], "b": [1, 1, 0]}}
    report = episode_bootstrap_advantage(rows, choices, caches, draws=100)
    assert all(value["ci95"] == [1, 1] for value in report.values())
    matrix = producer_consumer_means(rows, choices, caches)
    assert matrix["qwen25vl"]["qwen25vl"] == 0
    assert matrix["qwen25vl"]["llava_next"] == 1
    assert matrix["internvl2"]["internvl2"] == 0
    with pytest.raises(ValueError, match="aligned"):
        producer_consumer_means(rows, {**choices, "qwen25vl": [0]}, caches)
