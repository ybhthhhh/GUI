from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol, Sequence

from .schema import Event, TrajectorySample


class EventSelector(Protocol):
    """Select events without changing the canonical event semantics."""

    name: str

    def select(self, sample: TrajectorySample, max_events: int) -> Sequence[Event]: ...


class RecentEventSelector:
    name = "recent"

    def select(self, sample: TrajectorySample, max_events: int) -> Sequence[Event]:
        return sample.history[-max_events:]


class TimestampSelector:
    """Deterministic stand-in useful only for harness tests, not scientific claims."""

    def __init__(self, name: str, preferred_modulo: int) -> None:
        self.name = name
        self.preferred_modulo = preferred_modulo

    def select(self, sample: TrajectorySample, max_events: int) -> Sequence[Event]:
        ranked = sorted(
            sample.history,
            key=lambda event: (event.timestamp % 3 != self.preferred_modulo, -event.timestamp),
        )
        return ranked[:max_events]


def render_memory(events: Sequence[Event], max_events: int) -> str:
    if not 0 <= len(events) <= max_events:
        raise ValueError("selected events must be within the declared budget")
    slots = []
    for event in sorted(events, key=lambda item: item.timestamp):
        slot = asdict(event)
        slot["visual_evidence"] = list(event.visual_evidence)
        slot["temporal_links"] = list(event.temporal_links)
        slots.append(slot)
    return json.dumps({"schema_version": 1, "events": slots}, ensure_ascii=False, separators=(",", ":"))


def compile_memory(sample: TrajectorySample, selector: EventSelector, max_events: int) -> str:
    return render_memory(selector.select(sample, max_events), max_events)
