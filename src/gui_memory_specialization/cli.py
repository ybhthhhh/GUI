from __future__ import annotations

import argparse
import json
from pathlib import Path

from .canonical import RecentEventSelector, TimestampSelector, render_memory
from .baselines import DeterministicRandomSelector, FullHistorySelector, NoHistorySelector, StructuredHeuristicSelector, run_baselines
from .causal_lm import HuggingFaceCausalLMPolicy
from .evaluation import FrozenPolicy, cross_transfer_matrix, matched_minus_transferred
from .qwen25_vl import Qwen25VLPolicy
from .hf_chat_vl import HFChatVLPolicy
from .schema import TrajectorySample
from .specialization import candidate_event_sets, candidate_key


def load_jsonl(path: Path) -> list[TrajectorySample]:
    samples = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    samples.append(TrajectorySample.from_dict(json.loads(line)))
                except (ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
    if not samples:
        raise ValueError("dataset is empty")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("sample_id values must be unique")
    return samples


class ToyFrozenPolicy:
    """Test double: validates evaluator mechanics only, never a research model."""

    def __init__(self, name: str, required_token: str) -> None:
        self.name = name
        self.required_token = required_token

    def score_gold_action(self, sample: TrajectorySample, canonical_memory: str) -> float:
        return float(self.required_token in canonical_memory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "smoke", "policy-smoke", "vl-smoke", "vl-pilot", "hf-vl-smoke", "hf-vl-grid", "qwen-vl-grid", "hf-candidate-cache", "qwen-candidate-cache"))
    parser.add_argument("--consumer-name", default="hf_chat_vl", help="reported name for hf-vl-grid")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-events", type=int, default=2)
    parser.add_argument("--model-path", help="local HF causal-LM directory; required by policy-smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float32")
    parser.add_argument("--shard-index", type=int, default=0, help="zero-based deterministic sample shard")
    parser.add_argument("--num-shards", type=int, default=1, help="number of deterministic sample shards")
    args = parser.parse_args()
    samples = load_jsonl(args.dataset)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    samples = samples[args.shard_index :: args.num_shards]
    if not samples:
        parser.error("selected shard is empty")
    if args.command == "validate":
        print(json.dumps({"valid_samples": len(samples), "status": "ok"}))
        return

    if args.command == "policy-smoke":
        if not args.model_path:
            parser.error("policy-smoke requires --model-path")
        # One row and one producer: an infrastructure check, not a benchmark.
        policy = HuggingFaceCausalLMPolicy("text_policy_smoke", args.model_path, args.device, args.dtype)
        cells = cross_transfer_matrix(samples[:1], [TimestampSelector("recent", 2)], [policy], args.max_events)
        print(json.dumps({"kind": "text_policy_infrastructure_smoke", "cells": [cell.__dict__ for cell in cells]}))
        return

    if args.command == "vl-smoke":
        if not args.model_path:
            parser.error("vl-smoke requires --model-path")
        policy = Qwen25VLPolicy("qwen25vl_smoke", args.model_path, args.device, args.dtype)
        cells = cross_transfer_matrix(samples[:1], [TimestampSelector("recent", 2)], [policy], args.max_events)
        print(json.dumps({"kind": "synthetic_gui_vlm_infrastructure_smoke", "cells": [cell.__dict__ for cell in cells]}))
        return

    if args.command == "hf-vl-smoke":
        if not args.model_path:
            parser.error("hf-vl-smoke requires --model-path")
        policy = HFChatVLPolicy("hf_chat_vl_smoke", args.model_path, args.device, args.dtype)
        cells = cross_transfer_matrix(samples[:1], [TimestampSelector("recent", 2)], [policy], args.max_events)
        print(json.dumps({"kind": "hf_chat_vl_infrastructure_smoke", "cells": [cell.__dict__ for cell in cells]}))
        return

    if args.command == "hf-vl-grid":
        if not args.model_path:
            parser.error("hf-vl-grid requires --model-path")
        policy = HFChatVLPolicy(args.consumer_name, args.model_path, args.device, args.dtype)
        selectors = [RecentEventSelector(), DeterministicRandomSelector(), StructuredHeuristicSelector()]
        results = run_baselines(samples, selectors, [policy], args.max_events)
        print(json.dumps({"kind": "fixed_memory_controls_cross_consumer", "shard": {"index": args.shard_index, "count": args.num_shards}, "results": [item.__dict__ for item in results]}))
        return

    if args.command == "hf-candidate-cache":
        if not args.model_path:
            parser.error("hf-candidate-cache requires --model-path")
        policy = HFChatVLPolicy(args.consumer_name, args.model_path, args.device, args.dtype)
        rows = []
        for sample in samples:
            events_by_candidate = candidate_event_sets(sample, args.max_events)
            rows.append({"sample_id": sample.sample_id, "candidates": [
                {"key": candidate_key(events), "score": policy.score_gold_action(sample, render_memory(events, args.max_events))}
                for events in events_by_candidate
            ]})
        print(json.dumps({"kind": "consumer_candidate_reward_cache", "consumer": policy.name, "shard": {"index": args.shard_index, "count": args.num_shards}, "rows": rows}))
        return

    if args.command == "qwen-vl-grid":
        if not args.model_path:
            parser.error("qwen-vl-grid requires --model-path")
        policy = Qwen25VLPolicy(args.consumer_name, args.model_path, args.device, args.dtype)
        selectors = [RecentEventSelector(), DeterministicRandomSelector(), StructuredHeuristicSelector()]
        results = run_baselines(samples, selectors, [policy], args.max_events)
        print(json.dumps({"kind": "fixed_memory_controls_cross_consumer", "shard": {"index": args.shard_index, "count": args.num_shards}, "results": [item.__dict__ for item in results]}))
        return

    if args.command == "qwen-candidate-cache":
        if not args.model_path:
            parser.error("qwen-candidate-cache requires --model-path")
        policy = Qwen25VLPolicy(args.consumer_name, args.model_path, args.device, args.dtype)
        rows = []
        for sample in samples:
            events_by_candidate = candidate_event_sets(sample, args.max_events)
            rows.append({"sample_id": sample.sample_id, "candidates": [
                {"key": candidate_key(events), "score": policy.score_gold_action(sample, render_memory(events, args.max_events))}
                for events in events_by_candidate
            ]})
        print(json.dumps({"kind": "consumer_candidate_reward_cache", "consumer": policy.name, "shard": {"index": args.shard_index, "count": args.num_shards}, "rows": rows}))
        return

    if args.command == "vl-pilot":
        if not args.model_path:
            parser.error("vl-pilot requires --model-path")
        policy = Qwen25VLPolicy("qwen25vl_pilot", args.model_path, args.device, args.dtype)
        selectors = [
            NoHistorySelector(),
            RecentEventSelector(),
            FullHistorySelector(),
            DeterministicRandomSelector(),
            StructuredHeuristicSelector(),
        ]
        results = run_baselines(samples, selectors, [policy], args.max_events)
        print(json.dumps({"kind": "real_trajectory_single_consumer_pilot", "shard": {"index": args.shard_index, "count": args.num_shards}, "results": [item.__dict__ for item in results]}))
        return

    producers = [TimestampSelector("M1", 0), TimestampSelector("M2", 1), TimestampSelector("M3", 2)]
    consumers: list[FrozenPolicy] = [
        ToyFrozenPolicy("M1", "save_as"),
        ToyFrozenPolicy("M2", "upload"),
        ToyFrozenPolicy("M3", "recover"),
    ]
    cells = cross_transfer_matrix(samples, producers, consumers, args.max_events)
    print(json.dumps({"matrix": [cell.__dict__ for cell in cells], "contrasts": matched_minus_transferred(cells)}, indent=2))


if __name__ == "__main__":
    main()
