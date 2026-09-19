"""Fixed, auditable history baselines for the stage-one experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from .canonical import EventSelector, compile_memory
from .evaluation import FrozenPolicy
from .schema import Event, TrajectorySample


class NoHistorySelector:
    name = "no_history"

    def select(self, sample: TrajectorySample, max_events: int) -> Sequence[Event]:
        return ()


class FullHistorySelector:
    name = "full_history"

    def select(self, sample: TrajectorySample, max_events: int) -> Sequence[Event]:
        # The runner allocates this baseline its actual history length and reports
        # it separately; it is an upper-context control, not a matched-budget cell.
        return sample.history


class DeterministicRandomSelector:
    name = "random_events"

    def select(self, sample: TrajectorySample, max_events: int) -> Sequence[Event]:
        ranked = sorted(
            sample.history,
            key=lambda event: hashlib.sha256(f"{sample.sample_id}:{event.timestamp}".encode()).hexdigest(),
        )
        return ranked[:max_events]


class StructuredHeuristicSelector:
    name = "structured_heuristic"

    def select(self, sample: TrajectorySample, max_events: int) -> Sequence[Event]:
        def priority(event: Event) -> tuple[int, int]:
            informative = event.outcome not in {"", "unknown", "pending"}
            recovery = event.failure_status not in {"", "none"}
            return (int(recovery) * 4 + int(informative) * 2 + int(bool(event.temporal_links)), event.timestamp)

        return sorted(sample.history, key=priority, reverse=True)[:max_events]


@dataclass(frozen=True)
class BaselineResult:
    baseline: str
    consumer: str
    mean_score: float
    mean_events: float
    n: int


def run_baselines(
    samples: Sequence[TrajectorySample],
    selectors: Sequence[EventSelector],
    consumers: Sequence[FrozenPolicy],
    max_events: int,
) -> list[BaselineResult]:
    if not samples:
        raise ValueError("samples must be nonempty")
    results: list[BaselineResult] = []
    for selector in selectors:
        selected = [tuple(selector.select(sample, max_events)) for sample in samples]
        memories = [compile_memory(sample, selector, max(max_events, len(events))) for sample, events in zip(samples, selected)]
        for consumer in consumers:
            scores = [consumer.score_gold_action(sample, memory) for sample, memory in zip(samples, memories)]
            results.append(BaselineResult(selector.name, consumer.name, mean(scores), mean(map(len, selected)), len(samples)))
    return results

