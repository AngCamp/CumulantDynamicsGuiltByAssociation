from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumulant_dynamics.intervals import TimeInterval, TimeIntervalSet


class TimeIntervalSetTest(unittest.TestCase):
    def test_selects_values_inside_any_interval(self) -> None:
        intervals = TimeIntervalSet(
            [
                TimeInterval(0.0, 1.0, "pre"),
                TimeInterval(2.0, 3.0, "task"),
            ]
        )

        selected = intervals.select([-0.1, 0.2, 1.5, 2.4, 4.0])

        self.assertEqual(selected, [0.2, 2.4])
        self.assertTrue(intervals.contains(0.2))
        self.assertFalse(intervals.contains(1.5))
        self.assertEqual(intervals.to_pairs(), [(0.0, 1.0), (2.0, 3.0)])

    def test_from_pairs_sorts_intervals(self) -> None:
        intervals = TimeIntervalSet.from_pairs([(5, 6), (1, 2)])
        self.assertEqual(intervals.to_pairs(), [(1.0, 2.0), (5.0, 6.0)])


if __name__ == "__main__":
    unittest.main()
