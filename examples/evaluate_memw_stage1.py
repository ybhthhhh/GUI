"""Evaluate a stage-one memory checkpoint on fixed held-out CoMEM episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui_memory_specialization.qwen_memw_stage import QwenMemWStageOne


def image_message(path: str, text: str) -> dict:
    return {"role": "user", "content": [{"type": "image", "path": path}, {"type": "text", "text": text}]}


def load_validation(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in records if row.get("split") == "validation"]
    if not selected:
        raise ValueError(f"no validation records in {path}")
    for row in selected:
        if not row.get("history") or not Path(row["current_image"]).is_file():
            raise ValueError(f"invalid held-out record: {row.get('sample_id')}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()
    if args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    records = load_validation(args.manifest)[: args.max_samples]
    trainer = QwenMemWStageOne(args.model_path, device=args.device)
    state = trainer.torch.load(args.checkpoint, map_location=trainer.device)
    trainer.compressor.load_state_dict(state["compressor"])

    rows = []
    for record in records:
        history = [
            image_message(item["image"], f"Earlier webpage state. The action taken was: {item['action']}")
            for item in record["history"]
        ]
        current = [image_message(record["current_image"], f"Task: {record['task']}\nChoose the next webpage action.")]
        result = trainer.evaluate_step(history_messages=history, current_messages=current, action=record["action"])
        rows.append(
            {
                "sample_id": record["sample_id"],
                "action_ce": result.loss.action_cross_entropy,
                "teacher_kl": result.loss.teacher_kl,
                "total": result.loss.total,
                "action_tokens": result.loss.action_tokens,
            }
        )
    aggregate = {
        "kind": "memw_stage_one_held_out",
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "samples": len(rows),
        "mean_action_ce": sum(row["action_ce"] for row in rows) / len(rows),
        "mean_teacher_kl": sum(row["teacher_kl"] for row in rows) / len(rows),
        "mean_total": sum(row["total"] for row in rows) / len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("MEMW_STAGE1_HELD_OUT_OK", json.dumps({key: aggregate[key] for key in aggregate if key != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
