"""混合对手课程与座位分配测试。"""
from __future__ import annotations

import unittest

from pdh_v2.training.curriculum import CurriculumSchedule, SeatMix


class CurriculumTests(unittest.TestCase):
    def test_single_learner_mix_is_deterministic_and_rotates(self) -> None:
        mix = SeatMix((1.0, 0.0, 0.0, 0.0))
        first = [mix.learner_seats(7, 0, hand) for hand in range(32)]
        second = [mix.learner_seats(7, 0, hand) for hand in range(32)]
        self.assertEqual(first, second)
        self.assertTrue(all(len(seats) == 1 for seats in first))
        self.assertEqual(set(seat for seats in first for seat in seats), {0, 1, 2, 3})

    def test_schedule_changes_only_at_declared_thresholds(self) -> None:
        schedule = CurriculumSchedule()
        self.assertEqual(schedule.stage(0).name, "GUIDED_SOLO")
        self.assertEqual(schedule.stage(249_999).name, "GUIDED_SOLO")
        self.assertEqual(schedule.stage(250_000).name, "GUIDED_DUO")
        self.assertEqual(schedule.stage(750_000).name, "GUIDED_TRIO")
        self.assertEqual(schedule.stage(1_500_000).name, "ANCHORED_SELF_PLAY")

    def test_invalid_mix_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SeatMix((0.0, 0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            SeatMix((1.0, -1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
