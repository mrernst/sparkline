"""
Shared utilities: MetricSeries data class and logger setup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("asciitensorboard")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s  %(name)s: %(message)s",
    )
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# MetricSeries — the central data structure
# ---------------------------------------------------------------------------


@dataclass
class MetricSeries:
    """A sequence of (step, value) pairs with optional wall-clock timestamps."""

    steps: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    wall_times: List[Optional[float]] = field(default_factory=list)

    def append(
        self,
        step: int,
        value: float,
        wall_time: Optional[float] = None,
    ) -> None:
        self.steps.append(int(step))
        self.values.append(float(value))
        self.wall_times.append(wall_time)

    def extend(self, other: "MetricSeries") -> None:
        self.steps.extend(other.steps)
        self.values.extend(other.values)
        self.wall_times.extend(other.wall_times)

    def sorted(self) -> "MetricSeries":
        """Return a new MetricSeries sorted by step."""
        if not self.steps:
            return MetricSeries()
        zipped = sorted(zip(self.steps, self.values, self.wall_times), key=lambda t: t[0])
        steps, values, wall_times = zip(*zipped)
        return MetricSeries(list(steps), list(values), list(wall_times))

    def __len__(self) -> int:
        return len(self.steps)

    def __bool__(self) -> bool:
        return bool(self.steps)

    @property
    def min(self) -> float:
        return min(self.values)

    @property
    def max(self) -> float:
        return max(self.values)

    @property
    def last(self) -> float:
        return self.values[-1]

    @property
    def first(self) -> float:
        return self.values[0]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def abbreviate(name: str, max_len: int = 40) -> str:
    """Shorten a long metric name for display."""
    if len(name) <= max_len:
        return name
    return "…" + name[-(max_len - 1):]


def format_value(v: float) -> str:
    """Human-readable formatting for a single float."""
    if abs(v) >= 1e6 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.3e}"
    if abs(v) >= 100:
        return f"{v:.2f}"
    return f"{v:.4f}"
