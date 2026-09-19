"""Frozen Qwen2.5-VL consumer for screenshot-conditioned action scoring.

This is the first real GUI-policy adapter. It shares the exact canonical memory
text with every other consumer; only the frozen model forward pass differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .causal_lm import build_action_prompt
from .runtime import import_torch_then_transformers
from .schema import TrajectorySample


@dataclass
class Qwen25VLPolicy:
    name: str
    model_path: str
    device: str = "cuda:0"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        torch, transformers = import_torch_then_transformers()
        try:
            torch_dtype = getattr(torch, self.dtype)
        except AttributeError as error:
            raise ValueError(f"unsupported torch dtype: {self.dtype}") from error
        self._torch = torch
        self._processor = transformers.AutoProcessor.from_pretrained(self.model_path)
        # Keep real screenshots within a fixed visual-token budget.  This is an
        # observation pre-processing control, not a change to canonical memory.
        self._processor.image_processor.max_pixels = 501_760
        self._model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

    def _prefix_inputs(self, sample: TrajectorySample, canonical_memory: str):
        if not sample.current_screenshot:
            raise ValueError("Qwen25VLPolicy requires sample.current_screenshot")
        image_path = Path(sample.current_screenshot)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image_path)},
                    {"type": "text", "text": build_action_prompt(sample, canonical_memory)},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {name: value.to(self.device) for name, value in inputs.items()}

    def score_gold_action(self, sample: TrajectorySample, canonical_memory: str) -> float:
        torch = self._torch
        inputs = self._prefix_inputs(sample, canonical_memory)
        prefix_ids = inputs["input_ids"]
        action_ids = self._processor.tokenizer(" " + sample.gold_action, add_special_tokens=False, return_tensors="pt").input_ids
        if action_ids.numel() == 0:
            raise ValueError("gold_action tokenized to an empty continuation")
        action_ids = action_ids.to(self.device)
        inputs["input_ids"] = torch.cat((prefix_ids, action_ids), dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        with torch.inference_mode():
            logits = self._model(**inputs).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
        start = prefix_ids.shape[1]
        total = sum(log_probs[start + offset - 1, token].item() for offset, token in enumerate(action_ids[0]))
        score = total / action_ids.shape[1]
        if not torch.isfinite(torch.tensor(score)):
            raise FloatingPointError("non-finite frozen-policy action score")
        return float(score)
