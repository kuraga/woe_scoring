import unittest
from typing import Dict, List

from woe_scoring.core.binning.functions import (
    _check_diff_woe,
    _chi2,
    _find_index_of_diff_flag,
    _mono_flags,
)


class TestSimpleBinning(unittest.TestCase):
    def setUp(self):
        self.test_input_dicts: List[Dict] = [
            {
                "event": 0,
                "total": 4,
                "event_rate": 0,
                "woe": 4,
            },
            {
                "event": 1,
                "total": 4,
                "event_rate": 1 / 4,
                "woe": 2.5,
            },
            {
                "event": 2,
                "total": 4,
                "event_rate": 2 / 4,
                "woe": 1,
            },
            {
                "event": 3,
                "total": 4,
                "event_rate": 3 / 4,
                "woe": -1,
            },
        ]
        event: int = sum(event_rate["event"] for event_rate in self.test_input_dicts)
        total: int = sum(event_rate["total"] for event_rate in self.test_input_dicts)
        self.overall_rate: float = event / total
        self.diff_woe_threshold: float = 0.1

        self.non_monotonic_input_dicts: List[Dict] = [
            {
                "event": 3,
                "total": 4,
                "event_rate": 3 / 4,
                "woe": 1.1,
            },
            {
                "event": 2,
                "total": 4,
                "event_rate": 2 / 4,
                "woe": 1.12,
            },
            {
                "event": 3,
                "total": 4,
                "event_rate": 3 / 4,
                "woe": -1,
            },
            {
                "event": 4,
                "total": 4,
                "event_rate": 1,
                "woe": -3,
            },
        ]

    def test__chi2(self):
        """Test the _chi2 function"""
        self.assertEqual(
            3.3333333333333335, _chi2(self.test_input_dicts, self.overall_rate)
        )
        self.assertIsInstance(_chi2(self.test_input_dicts, self.overall_rate), float)

    def test__check_diff_woe(self):
        """Test the _check_diff_woe function"""
        self.assertEqual(
            None, _check_diff_woe(self.test_input_dicts, self.diff_woe_threshold)
        )
        self.assertEqual(
            0, _check_diff_woe(self.non_monotonic_input_dicts, self.diff_woe_threshold)
        )

    def test__mono_flags(self):
        """Test the _mono_flags function"""
        self.assertIsInstance(_mono_flags(self.test_input_dicts), bool)
        self.assertEqual(True, _mono_flags(self.test_input_dicts))
        self.assertEqual(False, _mono_flags(self.non_monotonic_input_dicts))

    def test__find_index_of_diff_flag(self):
        """Test the _find_index_of_diff_flag function"""
        self.assertIsInstance(
            _find_index_of_diff_flag(self.non_monotonic_input_dicts), int
        )
        self.assertEqual(0, _find_index_of_diff_flag(self.non_monotonic_input_dicts))


if __name__ == "__main__":
    unittest.main()
