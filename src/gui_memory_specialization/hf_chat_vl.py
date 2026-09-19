"""Standard-Transformers adapter for screenshot-conditioned chat VLMs.

Used for LLaVA-NeXT and Idefics2.  The canonical-memory prompt and the
teacher-forced next-action scoring rule intentionally match the Qwen adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .causal_lm import build_action_prompt
from .runtime import import_torch_then_transformers
from .schema import TrajectorySample


@dataclass
class HFChatVLPolicy:
    name: str
    model_path: str
    device: str = "cuda:0"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        torch, transformers = import_torch_then_transformers()
        self._torch = torch
        try:
            torch_dtype = getattr(torch, self.dtype)
        except AttributeError as error:
            raise ValueError(f"unsupported torch dtype: {self.dtype}") from error
        self._processor = transformers.AutoProcessor.from_pretrained(self.model_path)
        # LLaVA-NeXT's default any-resolution grid may expand a desktop
        # screenshot into several 336px crops.  On the 32 GB consumer GPU this
        # can OOM for a long action continuation.  A single fixed crop gives a
        # deterministic, bounded observation budget across every LLaVA shard.
        image_processor = getattr(self._processor, "image_processor", None)
        if hasattr(image_processor, "image_grid_pinpoints"):
            image_processor.image_grid_pinpoints = [[336, 336]]
        self._model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        if hasattr(image_processor, "image_grid_pinpoints") and hasattr(self._model.config, "image_grid_pinpoints"):
            # LlavaNextModel also derives its image-feature split sizes from this
            # config; keep it identical to the processor setting above.
            self._model.config.image_grid_pinpoints = [[336, 336]]
        self._model = self._model.to(self.device).eval()

    def _prefix_inputs(self, sample: TrajectorySample, canonical_memory: str):
        if not sample.current_screenshot:
            raise ValueError("HFChatVLPolicy requires sample.current_screenshot")
        image_path = Path(sample.current_screenshot)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        messages = [{"role": "user", "content": [
            {"type": "image", "path": str(image_path)},
            {"type": "text", "text": build_action_prompt(sample, canonical_memory)},
        ]}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in inputs.items()}

    def score_gold_action(self, sample: TrajectorySample, canonical_memory: str) -> float:
        torch = self._torch
        inputs = self._prefix_inputs(sample, canonical_memory)
        prefix_ids = inputs["input_ids"]
        action_ids = self._processor.tokenizer(
            " " + sample.gold_action, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        if action_ids.numel() == 0:
            raise ValueError("gold_action tokenized to an empty continuation")
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
