from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_EVENT_FIELDS = (
    "timestamp",
    "pre_state",
    "action",
    "outcome",
    "post_state",
    "visual_evidence",
    "subgoal",
    "failure_status",
    "temporal_links",
)


@dataclass(frozen=True)
class Event:
    timestamp: int
    pre_state: str
    action: str
    outcome: str
    post_state: str
    visual_evidence: tuple[str, ...]
    subgoal: str
    failure_status: str
    temporal_links: tuple[int, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Event":
        missing = [field for field in REQUIRED_EVENT_FIELDS if field not in raw]
        if missing:
            raise ValueError(f"event missing fields: {', '.join(missing)}")
        return cls(
            timestamp=int(raw["timestamp"]),
            pre_state=str(raw["pre_state"]),
            action=str(raw["action"]),
            outcome=str(raw["outcome"]),
            post_state=str(raw["post_state"]),
            visual_evidence=tuple(map(str, raw["visual_evidence"])),
            subgoal=str(raw["subgoal"]),
            failure_status=str(raw["failure_status"]),
            temporal_links=tuple(map(int, raw["temporal_links"])),
        )


@dataclass(frozen=True)
class TrajectorySample:
    sample_id: str
    task: str
    current_observation: str
    gold_action: str
    history: tuple[Event, ...]
    current_screenshot: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrajectorySample":
        required = ("sample_id", "task", "current_observation", "gold_action", "history")
        missing = [field for field in required if field not in raw]
        if missing:
            raise ValueError(f"sample missing fields: {', '.join(missing)}")
        history = tuple(Event.from_dict(event) for event in raw["history"])
        if not history:
            raise ValueError("history must contain at least one event")
        timestamps = [event.timestamp for event in history]
        if timestamps != sorted(timestamps):
            raise ValueError("history timestamps must be sorted")
        return cls(
            sample_id=str(raw["sample_id"]),
            task=str(raw["task"]),
            current_observation=str(raw["current_observation"]),
            gold_action=str(raw["gold_action"]),
            history=history,
            current_screenshot=(str(raw["current_screenshot"]) if raw.get("current_screenshot") else None),
        )
