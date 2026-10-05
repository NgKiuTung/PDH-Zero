"""Unit tests for Mini-PDH structural wins and big-hand patterns."""

from __future__ import annotations

import unittest

from pdh_phase1.config import load_config
from pdh_phase1.mini_pdh.rules import evaluate_win


class RuleTests(unittest.TestCase):
    """Validate exact deterministic pattern recognition."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.patterns = load_config("configs/mini_pdh_v0.yaml")["patterns"]

    def test_triplet_hand_detects_pong_pong_and_closed(self) -> None:
        counts = [0] * 13
        counts[0] = 3
        counts[8] = 3
        counts[9] = 2
        result = evaluate_win(counts, [], self.patterns)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("PONG_PONG", result.patterns)
        self.assertIn("CLOSED", result.patterns)
        self.assertEqual(result.selected_pattern, "TERMINALS_HONORS")
        self.assertEqual(result.multiplier, 4)

    def test_sequence_hand_detects_pure_numbers(self) -> None:
        counts = [0] * 13
        for tile in (0, 1, 2, 3, 4, 5):
            counts[tile] = 1
        counts[6] = 2
        result = evaluate_win(counts, [], self.patterns)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.selected_pattern, "PURE_NUMBERS")
        self.assertEqual(result.multiplier, 3)

    def test_four_identical_tiles_count_as_two_pairs(self) -> None:
        counts = [0] * 13
        counts[0] = 4
        counts[1] = 2
        counts[2] = 2
        result = evaluate_win(counts, [], self.patterns)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("FOUR_PAIRS", result.patterns)
        self.assertEqual(result.multiplier, 4)

    def test_invalid_structure_cannot_win(self) -> None:
        counts = [0] * 13
        counts[0:8] = [1] * 8
        self.assertIsNone(evaluate_win(counts, [], self.patterns))

    def test_open_sequence_hand_without_big_pattern_cannot_win(self) -> None:
        counts = [0] * 13
        counts[0] = counts[1] = counts[2] = 1
        counts[9] = 2
        self.assertIsNone(evaluate_win(counts, [5], self.patterns))


if __name__ == "__main__":
    unittest.main()
