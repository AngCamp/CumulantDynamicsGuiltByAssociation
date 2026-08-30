from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class TimeInterval:
    """A half-open interval used to search cumulants later."""

    start: float
    stop: float
    label: str = ""

    def contains(self, value: float) -> bool:
        return self.start <= value < self.stop


class TimeIntervalSet:
    """An ordered collection of intervals for cumulant searches."""

    def __init__(self, intervals: Iterable[TimeInterval] | None = None) -> None:
        self._intervals = sorted(intervals or [], key=lambda interval: (interval.start, interval.stop, interval.label))

    @classmethod
    def from_pairs(cls, intervals: Iterable[Sequence[float]]) -> "TimeIntervalSet":
        return cls(TimeInterval(float(start), float(stop)) for start, stop in intervals)

    @property
    def intervals(self) -> tuple[TimeInterval, ...]:
        return tuple(self._intervals)

    def add(self, interval: TimeInterval) -> None:
        self._intervals.append(interval)
        self._intervals.sort(key=lambda item: (item.start, item.stop, item.label))

    def contains(self, value: float) -> bool:
        return any(interval.contains(value) for interval in self._intervals)

    def select(self, values: Iterable[float]) -> list[float]:
        return [value for value in values if self.contains(value)]

    def to_pairs(self) -> list[tuple[float, float]]:
        return [(interval.start, interval.stop) for interval in self._intervals]
