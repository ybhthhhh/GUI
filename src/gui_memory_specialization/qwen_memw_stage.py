"""Frozen-Qwen2.5-VL stage-one training with continuous history memory.

This is intentionally a narrow implementation of the Mem-W first-stage idea:
the policy backbone and visual encoder are frozen; only a Q-Former-like
compressor learns.  The raw-history teacher and compressed-history student are
compared at identical action-token positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .continuous_memory import ContinuousMemoryCompressor, StageOneLoss, stage_one_distillation_loss


def patch_vendor_flash_attention() -> None:
    """Bridge the vendor runtime's legacy rotary symbol before Transformers import."""
    try:
        import flash_attn.layers.rotary as rotary
    except ImportError:
        return
    if not hasattr(rotary, "apply_rotary_emb") and hasattr(rotary, "apply_rotary_emb_func"):
        rotary.apply_rotary_emb = rotary.apply_rotary_emb_func


@dataclass(frozen=True)
class StageOneStep:
    loss: StageOneLoss
    memory_tokens: int
    trainable_parameters: int


class QwenMemWStageOne:
    """A one-example-at-a-time trainer to make multimodal memory mechanics explicit.

    Batching variable numbers of images is deliberately deferred to the data
    collator.  This object is the authoritative implementation of a single
    raw-history teacher/student update and is used by its smoke test before a
    longer distributed run is started.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        memory_tokens: int = 8,
        compressor_layers: int = 8,
        compressor_heads: int = 16,
        compressor_latent_size: int = 1024,
        learning_rate: float = 1e-4,
        kl_weight: float = 0.1,
    ):
        patch_vendor_flash_attention()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("stage-one Qwen training requires CUDA")
        self.torch = torch
        self.device = device
        self.kl_weight = kl_weight
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        hidden_size = int(self.model.config.hidden_size)
        self.compressor_spec = ContinuousMemoryCompressor(
            hidden_size=hidden_size,
            memory_tokens=memory_tokens,
            layers=compressor_layers,
            heads=compressor_heads,
            latent_size=compressor_latent_size,
            share_weights=True,
        )
        self.compressor = self.compressor_spec.module().to(device)
        self.optimizer = torch.optim.AdamW(self.compressor.parameters(), lr=learning_rate)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.compressor.parameters() if parameter.requires_grad)

    def _move(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {name: value.to(self.device) for name, value in inputs.items()}

    def _prompt_inputs(self, messages: list[dict[str, Any]], action: str | None) -> tuple[dict[str, Any], object | None]:
        torch = self.torch
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move(dict(inputs))
        action_ids = None
        if action is not None:
            action_ids = self.processor.tokenizer(
                " " + action, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(self.device)
            inputs["input_ids"] = torch.cat((inputs["input_ids"], action_ids), dim=1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs, action_ids

    def _multimodal_embeds(self, inputs: dict[str, Any]):
        """Embed Qwen image placeholders exactly as the upstream forward does."""
        torch = self.torch
        input_ids = inputs["input_ids"]
        embeds = self.model.model.embed_tokens(input_ids)
        pixels = inputs.get("pixel_values")
        if pixels is not None:
            image_embeds = self.model.visual(pixels.type(self.model.visual.dtype), grid_thw=inputs["image_grid_thw"])
            mask = (input_ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(embeds)
            if int(mask[..., 0].sum()) != image_embeds.shape[0]:
                raise ValueError("Qwen image-token count does not match visual feature count")
            embeds = embeds.masked_scatter(mask, image_embeds.to(embeds.device, embeds.dtype))
        return embeds

    def _prefix_positions(self, memory, current_inputs: dict[str, Any]):
        torch = self.torch
        raw_positions, _ = self.model.get_rope_index(
            current_inputs["input_ids"],
            current_inputs.get("image_grid_thw"),
            None,
            None,
            current_inputs.get("attention_mask"),
        )
        batch, slots, _ = memory.shape
        prefix = torch.arange(slots, device=self.device, dtype=raw_positions.dtype).view(1, 1, slots)
        prefix = prefix.expand(3, batch, slots)
        shifted = raw_positions.clone()
        minimum = shifted.min()
        maximum = prefix.max()
        if minimum <= maximum:
            shifted += maximum + 1 - minimum
        return torch.cat((prefix, shifted), dim=2)

    def _compressed_logits(self, memory, current_inputs: dict[str, Any]):
        embeddings = self._multimodal_embeds(current_inputs)
        joined = self.torch.cat((memory.to(embeddings.dtype), embeddings), dim=1)
        positions = self._prefix_positions(memory, current_inputs)
        outputs = self.model.model(
            input_ids=None,
            inputs_embeds=joined,
            position_ids=positions,
            attention_mask=None,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        return self.model.lm_head(outputs.last_hidden_state)[:, memory.shape[1] :, :]

    def train_step(
        self,
        *,
        history_messages: list[dict[str, Any]],
        current_messages: list[dict[str, Any]],
        action: str,
    ) -> StageOneStep:
        """Run one raw-history-teacher / compressed-history-student update."""
        torch = self.torch
        if not history_messages:
            raise ValueError("stage one requires at least one history message")
        history_inputs, _ = self._prompt_inputs(history_messages, action=None)
        current_inputs, action_ids = self._prompt_inputs(current_messages, action)
        assert action_ids is not None
        teacher_inputs, teacher_actions = self._prompt_inputs(history_messages + current_messages, action)
        assert teacher_actions is not None
        with torch.no_grad():
            history_output = self.model(**history_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            history_states = history_output.hidden_states[-1]
            teacher_logits = self.model(**teacher_inputs, return_dict=True, use_cache=False).logits
        self.compressor.train()
        memory = self.compressor(history_states, history_inputs.get("attention_mask"))
        student_logits = self._compressed_logits(memory, current_inputs)
        token_count = action_ids.shape[1]
        # Logit at t predicts token t+1, so align the final action-token window
        # from the raw teacher and compressed student rather than their unequal
        # context prefixes.
        student_action_logits = student_logits[:, -token_count - 1 : -1, :]
        teacher_action_logits = teacher_logits[:, -token_count - 1 : -1, :]
        total, metrics = stage_one_distillation_loss(
            student_action_logits, teacher_action_logits, action_ids, kl_weight=self.kl_weight
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        return StageOneStep(metrics, memory_tokens=memory.shape[1], trainable_parameters=self.trainable_parameters)

    def evaluate_step(
        self,
        *,
        history_messages: list[dict[str, Any]],
        current_messages: list[dict[str, Any]],
        action: str,
    ) -> StageOneStep:
        """Measure a frozen checkpoint on one held-out trajectory transition.

        This mirrors the teacher/student alignment of :meth:`train_step`, but
        deliberately has no optimizer interaction.  It is therefore safe to
        run against the episode-level validation split after a formal stage-one
        checkpoint has been written.
        """
        torch = self.torch
        if not history_messages:
            raise ValueError("stage one requires at least one history message")
        history_inputs, _ = self._prompt_inputs(history_messages, action=None)
        current_inputs, action_ids = self._prompt_inputs(current_messages, action)
        assert action_ids is not None
        teacher_inputs, teacher_actions = self._prompt_inputs(history_messages + current_messages, action)
        assert teacher_actions is not None
        self.compressor.eval()
        with torch.no_grad():
            history_output = self.model(**history_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            history_states = history_output.hidden_states[-1]
            teacher_logits = self.model(**teacher_inputs, return_dict=True, use_cache=False).logits
            memory = self.compressor(history_states, history_inputs.get("attention_mask"))
            student_logits = self._compressed_logits(memory, current_inputs)
            token_count = action_ids.shape[1]
            student_action_logits = student_logits[:, -token_count - 1 : -1, :]
            teacher_action_logits = teacher_logits[:, -token_count - 1 : -1, :]
            _, metrics = stage_one_distillation_loss(
                student_action_logits, teacher_action_logits, action_ids, kl_weight=self.kl_weight
            )
        return StageOneStep(metrics, memory_tokens=memory.shape[1], trainable_parameters=self.trainable_parameters)
