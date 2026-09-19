"""Launch resumable stage-one continuous-memory distillation on CoMEM samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from gui_memory_specialization.qwen_memw_stage import QwenMemWStageOne


def image_message(path: str, text: str) -> dict:
    return {"role": "user", "content": [{"type": "image", "path": path}, {"type": "text", "text": text}]}


def load_records(path: Path, split: str) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in records if row.get("split") == split]
    if not selected:
        raise ValueError(f"no {split} records in {path}")
    for row in selected:
        if (
            not row.get("history")
            or not Path(row["current_image"]).is_file()
            or not all(Path(item.get("image", "")).is_file() for item in row["history"])
        ):
            raise ValueError(f"invalid current image/history for {row.get('sample_id')}")
    return selected


def checkpoint(path: Path, trainer: QwenMemWStageOne, step: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    trainer.torch.save(
        {"step": step, "config": config, "compressor": trainer.compressor.state_dict(), "optimizer": trainer.optimizer.state_dict()},
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--memory-tokens", type=int, default=8)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", type=Path, help="checkpoint written by an earlier run")
    parser.add_argument("--max-errors", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.checkpoint_every <= 0:
        raise ValueError("max-steps and checkpoint-every must be positive")
    records = load_records(args.manifest, "train")
    rng = random.Random(args.seed)
    rng.shuffle(records)
    trainer = QwenMemWStageOne(
        args.model_path,
        device=args.device,
        memory_tokens=args.memory_tokens,
        compressor_heads=16,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "stage1_metrics.jsonl"
    checkpoint_path = args.output / "stage1_last.pt"
    config = {
        "kind": "memw_style_stage_one",
        "manifest": str(args.manifest.resolve()),
        "model_path": args.model_path,
        "memory_tokens": args.memory_tokens,
        "kl_weight": args.kl_weight,
        "learning_rate": args.learning_rate,
        "frozen_backbone": True,
        "stage_two": "requires_online_execution_rewards",
    }
    start_step = 0
    if args.resume:
        state = trainer.torch.load(args.resume, map_location=trainer.device)
        trainer.compressor.load_state_dict(state["compressor"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        print("MEMW_STAGE1_RESUMED", f"step={start_step}", f"checkpoint={args.resume}", flush=True)
    errors = 0
    with metrics_path.open("a", encoding="utf-8") as metrics:
        for step in range(start_step + 1, args.max_steps + 1):
            record = records[(step - 1) % len(records)]
            history = [image_message(item["image"], f"Earlier webpage state. The action taken was: {item['action']}") for item in record["history"]]
            current = [image_message(record["current_image"], f"Task: {record['task']}\nChoose the next webpage action.")]
            try:
                result = trainer.train_step(history_messages=history, current_messages=current, action=record["action"])
            except (OSError, RuntimeError, ValueError) as error:
                errors += 1
                payload = {"step": step, "sample_id": record["sample_id"], "error": str(error)}
                metrics.write(json.dumps(payload) + "\n")
                metrics.flush()
                print("MEMW_STAGE1_SAMPLE_ERROR", json.dumps(payload), flush=True)
                if errors > args.max_errors:
                    raise RuntimeError(f"aborting after {errors} invalid samples") from error
                continue
            payload = {
                "step": step,
                "sample_id": record["sample_id"],
                "action_ce": result.loss.action_cross_entropy,
                "teacher_kl": result.loss.teacher_kl,
                "total": result.loss.total,
                "action_tokens": result.loss.action_tokens,
                "memory_tokens": result.memory_tokens,
            }
            metrics.write(json.dumps(payload) + "\n")
            metrics.flush()
            if step % args.checkpoint_every == 0 or step == args.max_steps:
                checkpoint(checkpoint_path, trainer, step, config)
                print("MEMW_STAGE1_CHECKPOINT", json.dumps(payload), flush=True)
    print("MEMW_STAGE1_TRAIN_COMPLETE", f"steps={args.max_steps}", f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
