"""Offline, consumer-specific memory-selector training utilities.

The frozen consumer supplies a reward (gold-action log likelihood) for every
budget-valid candidate memory on *training* episodes.  A small deterministic
ranker is then fit per consumer.  The ranker only sees current observation,
history metadata, and visual-state-change descriptors; it never sees a gold
action at selection time.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .schema import Event, TrajectorySample


def candidate_event_sets(sample: TrajectorySample, max_events: int) -> list[tuple[Event, ...]]:
    """Enumerate all fixed-budget event sets in deterministic temporal order."""
    if not 1 <= max_events <= len(sample.history):
        raise ValueError("max_events must be between one and the history length")
    return [tuple(choice) for choice in itertools.combinations(sample.history, max_events)]


def candidate_key(events: Sequence[Event]) -> str:
    return ",".join(str(event.timestamp) for event in events)


def episode_group(sample_id: str) -> str:
    """Group variants of one OSWorld episode without using source/model labels."""
    match = re.search(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", sample_id, flags=re.I)
    return match.group(0).lower() if match else sample_id.rsplit("-step-", 1)[0]


def is_test_group(group: str) -> bool:
    # Stable 80/20 episode split.  This does not inspect rewards or source IDs.
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0


def _click_xy(action: str) -> tuple[float, float, float]:
    match = re.search(r"pyautogui\.click\(([-+0-9.]+),\s*([-+0-9.]+)", action)
    if not match:
        return (0.0, 0.0, 0.0)
    return (1.0, float(match.group(1)) / 1920.0, float(match.group(2)) / 1080.0)


def _image_descriptor(path: str | None) -> tuple[float, float, float]:
    """Small visual-change descriptor; returns zeros if a referenced image is absent."""
    if not path:
        return (0.0, 0.0, 0.0)
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as image:
            image = image.convert("RGB").resize((16, 16))
            stats = ImageStat.Stat(image)
            means = [value / 255.0 for value in stats.mean]
            stddevs = [value / 255.0 for value in stats.stddev]
            return (sum(means) / 3.0, sum(stddevs) / 3.0, image.size[0] / 16.0)
    except (OSError, ValueError, ImportError):
        return (0.0, 0.0, 0.0)


def candidate_features(sample: TrajectorySample, events: Sequence[Event]) -> list[float]:
    """Observable, model-agnostic features for a selected event set.

    Feature layout is deliberately small to reduce overfitting on the 20-episode
    validation set.  Image paths are used only to read pixel statistics, never as
    categorical identifiers.
    """
    current_mean, current_std, _ = _image_descriptor(sample.current_screenshot)
    last_timestamp = max(event.timestamp for event in sample.history)
    ordered = sorted(events, key=lambda event: event.timestamp)
    output: list[float] = []
    for event in ordered:
        exists, x, y = _click_xy(event.action)
        image_path = event.visual_evidence[0] if event.visual_evidence else None
        event_mean, event_std, _ = _image_descriptor(image_path)
        output.extend((
            event.timestamp / max(last_timestamp, 1),
            exists,
            x,
            y,
            abs(current_mean - event_mean),
            abs(current_std - event_std),
        ))
    # Candidate cardinality is fixed by the experiment.  These summary features
    # make the choice invariant to event order while retaining temporal spacing.
    timestamps = [event.timestamp / max(last_timestamp, 1) for event in ordered]
    output.extend((sum(timestamps) / len(timestamps), timestamps[-1] - timestamps[0]))
    return output


@dataclass(frozen=True)
class LinearCandidateRanker:
    name: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature dimension differs from trained ranker")
        return self.bias + sum(
            weight * ((value - location) / spread)
            for value, location, spread, weight in zip(features, self.mean, self.scale, self.weights)
        )

    def select(self, sample: TrajectorySample, max_events: int) -> Sequence[Event]:
        candidates = candidate_event_sets(sample, max_events)
        return max(candidates, key=lambda events: (self.predict(candidate_features(sample, events)), candidate_key(events)))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "weights": list(self.weights),
            "bias": self.bias,
            "feature_schema": "timestamps, click geometry, visual mean/std deltas",
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "LinearCandidateRanker":
        return cls(
            name=str(raw["name"]),
            mean=tuple(map(float, raw["mean"])),
            scale=tuple(map(float, raw["scale"])),
            weights=tuple(map(float, raw["weights"])),
            bias=float(raw["bias"]),
        )


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination for the small ridge system used by this experiment."""
    n = len(vector)
    augmented = [row[:] + [target] for row, target in zip(matrix, vector)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular ranker system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        factor = augmented[column][column]
        augmented[column] = [value / factor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return [row[-1] for row in augmented]


def fit_ranker(name: str, rows: Sequence[tuple[Sequence[float], float]], ridge: float = 10.0) -> LinearCandidateRanker:
    """Fit ridge regression to within-sample centered consumer rewards."""
    if not rows:
        raise ValueError("cannot fit ranker without rows")
    dimensions = len(rows[0][0])
    if any(len(features) != dimensions for features, _ in rows):
        raise ValueError("inconsistent feature dimensions")
    count = len(rows)
    mean = [sum(features[index] for features, _ in rows) / count for index in range(dimensions)]
    scale = []
    for index, location in enumerate(mean):
        variance = sum((features[index] - location) ** 2 for features, _ in rows) / count
        scale.append(max(math.sqrt(variance), 1e-6))
    x = [[(value - mean[index]) / scale[index] for index, value in enumerate(features)] for features, _ in rows]
    y = [target for _, target in rows]
    matrix = [[ridge if row == col else 0.0 for col in range(dimensions)] for row in range(dimensions)]
    vector = [0.0] * dimensions
    for features, target in zip(x, y):
        for row in range(dimensions):
            vector[row] += features[row] * target
            for col in range(dimensions):
                matrix[row][col] += features[row] * features[col]
    weights = _solve(matrix, vector)
    bias = sum(y) / count
    return LinearCandidateRanker(name, tuple(mean), tuple(scale), tuple(weights), bias)
