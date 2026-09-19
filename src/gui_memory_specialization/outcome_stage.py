"""Outcome-aware second-stage primitives for the Mem-W-style protocol.

The second stage is intentionally independent from offline data preparation.
It accepts only rewards attached to executed web rollouts and computes a
leave-one-out group baseline (RLOO), which keeps a logged demonstration from
silently becoming a policy-gradient reward.
"""

from __future__ import annotations

from collections import defaultdict


def validate_execution_records(records) -> None:
    if not records:
        raise ValueError("second stage needs executed rollout records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("rollout record must be a dictionary")
        if record.get("source") != "online_execution":
            raise ValueError("second stage rejects non-executed rewards")
        if not record.get("rollout_group"):
            raise ValueError("each rollout needs a rollout_group")
        reward = record.get("terminal_reward")
        if not isinstance(reward, (int, float)) or not 0.0 <= reward <= 1.0:
            raise ValueError("terminal_reward must be a finite value in [0, 1]")


def rloo_advantages(records) -> list[float]:
    """Compute each rollout reward minus the mean of its sibling rollouts."""
    validate_execution_records(records)
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record["rollout_group"])].append((index, float(record["terminal_reward"])))
    advantages = [0.0] * len(records)
    for group, members in groups.items():
        if len(members) < 2:
            raise ValueError(f"RLOO group {group!r} needs at least two rollouts")
        total = sum(reward for _, reward in members)
        for index, reward in members:
            advantages[index] = reward - (total - reward) / (len(members) - 1)
    return advantages


def outcome_policy_loss(action_log_probs, records):
    """Return differentiable negative-RLOO objective for LoRA/Q-Former updates."""
    import torch

    if not torch.is_tensor(action_log_probs) or action_log_probs.ndim != 1:
        raise ValueError("action_log_probs must be a rank-one torch tensor")
    advantages = rloo_advantages(records)
    if action_log_probs.shape[0] != len(advantages):
        raise ValueError("one action log-probability is required per rollout")
    weights = torch.tensor(advantages, dtype=action_log_probs.dtype, device=action_log_probs.device)
    return -(weights * action_log_probs).mean(), advantages
