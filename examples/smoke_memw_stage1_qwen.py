"""Run one real Qwen2.5-VL continuous-memory stage-one update."""

from __future__ import annotations

import argparse
from pathlib import Path

from gui_memory_specialization.qwen_memw_stage import QwenMemWStageOne


def image_message(image: Path, text: str) -> dict:
    return {"role": "user", "content": [{"type": "image", "path": str(image)}, {"type": "text", "text": text}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    trainer = QwenMemWStageOne(args.model_path, device=args.device)
    result = trainer.train_step(
        history_messages=[image_message(args.image, "Earlier GUI state. The prior action was opening the upload dialog.")],
        current_messages=[image_message(args.image, "Current GUI state. Choose the next GUI action.")],
        action="click(x=640, y=360)",
    )
    print(
        "MEMW_STAGE1_QWEN_SMOKE_OK",
        f"loss={result.loss.total:.6f}",
        f"action_ce={result.loss.action_cross_entropy:.6f}",
        f"teacher_kl={result.loss.teacher_kl:.6f}",
        f"memory_tokens={result.memory_tokens}",
        f"trainable={result.trainable_parameters}",
    )


if __name__ == "__main__":
    main()
