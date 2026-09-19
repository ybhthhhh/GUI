"""Mem-W-style continuous-memory primitives.

Stage one compresses a variable-length frozen-teacher history into a fixed
number of latent tokens.  Its objective preserves the action distribution of a
teacher that sees the uncompressed history.  Stage two is deliberately kept
separate: it may consume only rewards produced by executing tasks, never the
offline trajectory's asserted success label.
"""

from __future__ import annotations

from dataclasses import dataclass


def _torch():
    import torch
    import torch.nn.functional as functional

    return torch, functional


@dataclass(frozen=True)
class StageOneLoss:
    action_cross_entropy: float
    teacher_kl: float
    total: float
    action_tokens: int


class ContinuousMemoryCompressor:
    """A small Q-Former-like cross-attention compressor with fixed token budget.

    The class constructs its torch module lazily through ``module``.  Keeping
    the public wrapper light means package-level data utilities do not require
    CUDA or Transformers merely to inspect experiment metadata.
    """

    def __init__(self, hidden_size: int, memory_tokens: int = 8, layers: int = 2, heads: int = 8):
        if hidden_size <= 0 or memory_tokens <= 0 or layers <= 0 or heads <= 0:
            raise ValueError("hidden_size, memory_tokens, layers, and heads must be positive")
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.hidden_size = hidden_size
        self.memory_tokens = memory_tokens
        self.layers = layers
        self.heads = heads
        self._module = None

    def module(self):
        """Return the trainable torch module, constructing it on first use."""
        if self._module is not None:
            return self._module
        torch, _ = _torch()
        nn = torch.nn

        class _Compressor(nn.Module):
            def __init__(self, outer: ContinuousMemoryCompressor):
                super().__init__()
                self.latents = nn.Parameter(torch.empty(1, outer.memory_tokens, outer.hidden_size))
                nn.init.trunc_normal_(self.latents, std=0.02)
                self.blocks = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "history_norm": nn.LayerNorm(outer.hidden_size),
                                "latent_norm": nn.LayerNorm(outer.hidden_size),
                                "cross": nn.MultiheadAttention(outer.hidden_size, outer.heads, batch_first=True),
                                "ff_norm": nn.LayerNorm(outer.hidden_size),
                                "ff": nn.Sequential(
                                    nn.Linear(outer.hidden_size, 4 * outer.hidden_size),
                                    nn.GELU(),
                                    nn.Linear(4 * outer.hidden_size, outer.hidden_size),
                                ),
                            }
                        )
                        for _ in range(outer.layers)
                    ]
                )

            def forward(self, history_states, history_mask=None):
                if history_states.ndim != 3:
                    raise ValueError("history_states must have shape [batch, sequence, hidden]")
                if history_states.shape[-1] != self.latents.shape[-1]:
                    raise ValueError("history hidden size does not match compressor")
                if history_mask is not None and history_mask.shape != history_states.shape[:2]:
                    raise ValueError("history_mask must have shape [batch, sequence]")
                output_dtype = history_states.dtype
                # The frozen Qwen backbone emits BF16 states, while the small
                # compressor deliberately keeps its normalization and updates
                # in FP32 for stable distillation.  Cast only at this boundary
                # and restore the backbone dtype for continuous-token injection.
                history_states = history_states.to(self.latents.dtype)
                latents = self.latents.expand(history_states.shape[0], -1, -1)
                padding = ~history_mask.bool() if history_mask is not None else None
                for block in self.blocks:
                    keys = block["history_norm"](history_states)
                    query = block["latent_norm"](latents)
                    update, _ = block["cross"](query, keys, keys, key_padding_mask=padding, need_weights=False)
                    latents = latents + update
                    latents = latents + block["ff"](block["ff_norm"](latents))
                return latents.to(output_dtype)

        self._module = _Compressor(self)
        return self._module


def stage_one_distillation_loss(compressed_logits, teacher_logits, action_labels, *, kl_weight: float) -> tuple[object, StageOneLoss]:
    """Return differentiable loss plus detached metrics for stage-one training.

    ``action_labels`` uses -100 outside assistant/action positions.  The frozen
    teacher distribution is detached, preventing accidental teacher updates.
    """
    torch, functional = _torch()
    if compressed_logits.shape != teacher_logits.shape:
        raise ValueError("compressed and teacher logits must have identical shapes")
    if compressed_logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if action_labels.shape != compressed_logits.shape[:2]:
        raise ValueError("action_labels must have shape [batch, sequence]")
    if kl_weight < 0:
        raise ValueError("kl_weight must be non-negative")
    action_tokens = int((action_labels != -100).sum().item())
    if action_tokens == 0:
        raise ValueError("at least one action token is required")
    action = functional.cross_entropy(
        compressed_logits.reshape(-1, compressed_logits.shape[-1]), action_labels.reshape(-1), ignore_index=-100
    )
    teacher = torch.softmax(teacher_logits.detach().float(), dim=-1)
    kl = functional.kl_div(functional.log_softmax(compressed_logits.float(), dim=-1), teacher, reduction="batchmean")
    total = action + kl_weight * kl
    metrics = StageOneLoss(
        action_cross_entropy=float(action.detach().cpu()),
        teacher_kl=float(kl.detach().cpu()),
        total=float(total.detach().cpu()),
        action_tokens=action_tokens,
    )
    return total, metrics


def validate_online_rewards(rewards) -> None:
    """Reject offline labels masquerading as stage-two executed rewards."""
    if not rewards:
        raise ValueError("stage two requires at least one executed task reward")
    for reward in rewards:
        if not isinstance(reward, dict) or reward.get("source") != "online_execution":
            raise ValueError("stage two accepts only rewards with source='online_execution'")
        value = reward.get("terminal_reward")
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError("online terminal_reward must be in [0, 1]")
