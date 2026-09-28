"""Score every fixed-budget canonical memory candidate with one frozen GUI VLM.

One process owns one GPU and an episode-disjoint row shard. No NCCL or model
weight update occurs during caching. The later selector optimization consumes
only training-split candidate rewards; validation targets remain held out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from gui_memory_specialization.memory_efficiency import output_token_window
from gui_memory_specialization.multivl_memw_stage import MultiVLMemWStageOne
from gui_memory_specialization.qwen_memw_stage import QwenMemWStageOne
from gui_memory_specialization.storyline_canonical import candidate_messages


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_ce(logits, labels, torch) -> float:
    if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("gold action and logits do not align")
    total = 0.0
    for start in range(0, labels.shape[1], 32):
        window = slice(start, start + 32)
        total += float(torch.nn.functional.cross_entropy(
            logits[:, window].float().reshape(-1, logits.shape[-1]),
            labels[:, window].reshape(-1), reduction="sum",
        ).item())
    return total / int(labels.numel())


def score(trainer, row: dict, event: dict) -> tuple[float, int]:
    inputs, labels = trainer._prompt_inputs(candidate_messages(row, event), row["target_action"])
    if labels is None or not labels.numel():
        raise ValueError("empty gold action tokens")
    tokens = int(labels.shape[1])
    with trainer.torch.no_grad():
        if isinstance(trainer, QwenMemWStageOne):
            with output_token_window(trainer.model.lm_head, slice(-tokens - 1, -1)):
                logits = trainer.model(**inputs, return_dict=True, use_cache=False).logits
        else:
            head = trainer.model.language_model.get_output_embeddings()
            with output_token_window(head, slice(-tokens - 1, -1)):
                logits = trainer._forward(inputs, hidden=False).logits
        value = action_ce(logits, labels, trainer.torch)
    return value, tokens


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["sample_id"] in ids:
            raise ValueError(f"duplicate cached sample: {row['sample_id']}")
        ids.add(row["sample_id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--backend", choices=("qwen25vl", "llava_next", "internvl2"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0, help="per-rank smoke limit; 0 means the entire shard")
    parser.add_argument("--max-errors", type=int, default=0)
    args = parser.parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size or args.max_rows < 0:
        parser.error("invalid rank or row limit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = file_sha256(args.manifest)
    output = args.output_dir / f"rank-{args.rank}.jsonl"
    errors = args.output_dir / f"rank-{args.rank}.errors.jsonl"
    metadata = args.output_dir / f"rank-{args.rank}.meta.json"
    expected = {
        "kind": "storyline_candidate_cache_shard", "backend": args.backend,
        "rank": args.rank, "world_size": args.world_size, "manifest_sha256": manifest_sha,
        "max_image_pixels": 224 * 224, "images_per_candidate": 3,
        "memory_budget": "one observed pre/action/post event",
    }
    if metadata.exists() and json.loads(metadata.read_text(encoding="utf-8")) != expected:
        raise ValueError("cache resume metadata differs from this run")
    metadata.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    done = completed_ids(output)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for index, row in enumerate(rows) if index % args.world_size == args.rank]
    if args.max_rows:
        selected = selected[: args.max_rows]
    trainer = (QwenMemWStageOne(args.model_path, device="cuda:0", max_image_pixels=224 * 224, memory_efficient=True)
               if args.backend == "qwen25vl" else MultiVLMemWStageOne(
                   args.model_path, backend=args.backend, device="cuda:0", max_image_pixels=224 * 224,
                   memory_efficient=True))
    trainer.compressor.eval()
    print("STORYLINE_CACHE_START", json.dumps({**expected, "rows": len(selected), "resumed": len(done)}), flush=True)
    failures = 0
    with output.open("a", encoding="utf-8") as stream, errors.open("a", encoding="utf-8") as error_stream:
        for index, row in enumerate(selected, 1):
            if row["sample_id"] in done:
                continue
            started = time.monotonic()
            try:
                candidates = []
                for event in row["history"]:
                    ce, tokens = score(trainer, row, event)
                    candidates.append({"event_index": event["timestamp"], "action_ce": ce, "action_tokens": tokens})
                payload = {"sample_id": row["sample_id"], "episode_id": row["episode_id"],
                           "split": row["split"], "candidates": candidates,
                           "elapsed_seconds": round(time.monotonic() - started, 3)}
                stream.write(json.dumps(payload) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                status = "ok"
            except (OSError, RuntimeError, ValueError) as exc:
                error_stream.write(json.dumps({"sample_id": row["sample_id"], "error": repr(exc)}) + "\n")
                error_stream.flush()
                failures += 1
                status = "error"
            print("STORYLINE_CACHE", args.backend, args.rank, index, len(selected), row["sample_id"], status,
                  round(time.monotonic() - started, 2), flush=True)
            if failures > args.max_errors:
                raise RuntimeError(f"candidate cache exceeded {args.max_errors} allowed errors")
    finished = completed_ids(output)
    if len(finished) != len(selected) or failures:
        raise RuntimeError(f"incomplete cache shard: {len(finished)}/{len(selected)}, errors={failures}")
    completion = args.output_dir / f"rank-{args.rank}.complete.json"
    temporary = completion.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({**expected, "completed_rows": len(finished)}) + "\n", encoding="utf-8")
    temporary.replace(completion)
    print("STORYLINE_CACHE_COMPLETE", args.backend, args.rank, len(selected), flush=True)


if __name__ == "__main__":
    main()
