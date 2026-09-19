from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Protocol, Sequence

from .canonical import EventSelector, compile_memory
from .schema import TrajectorySample


class FrozenPolicy(Protocol):
    """A frozen consumer. Implement this around a local VLM/policy forward pass."""

    name: str

    def score_gold_action(self, sample: TrajectorySample, canonical_memory: str) -> float:
        """Return a comparable scalar, e.g. gold-action log probability."""


@dataclass(frozen=True)
class MatrixCell:
    producer: str
    consumer: str
    mean_score: float
    n: int


def cross_transfer_matrix(
    samples: Sequence[TrajectorySample],
    producers: Sequence[EventSelector],
    consumers: Sequence[FrozenPolicy],
    max_events: int,
) -> list[MatrixCell]:
    if not samples or not producers or not consumers:
        raise ValueError("samples, producers, and consumers must be nonempty")
    rows: list[MatrixCell] = []
    for producer in producers:
        memories = [compile_memory(sample, producer, max_events) for sample in samples]
        for consumer in consumers:
            scores = [consumer.score_gold_action(sample, memory) for sample, memory in zip(samples, memories)]
            rows.append(MatrixCell(producer.name, consumer.name, mean(scores), len(scores)))
    return rows


def matched_minus_transferred(cells: Sequence[MatrixCell]) -> dict[str, float]:
    """Column-wise diagonal contrast; names must match for intended pairs."""
    result: dict[str, float] = {}
    consumers = sorted({cell.consumer for cell in cells})
    for consumer in consumers:
        column = [cell for cell in cells if cell.consumer == consumer]
        matched = [cell.mean_score for cell in column if cell.producer == consumer]
        transferred = [cell.mean_score for cell in column if cell.producer != consumer]
        if len(matched) != 1 or not transferred:
            raise ValueError(f"cannot form matched contrast for {consumer}")
        result[consumer] = matched[0] - mean(transferred)
    return result

