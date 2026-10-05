"""多路线与资格特征测试。"""
from __future__ import annotations

import unittest

from pdh_v2.strategy import route_potentials, route_state
from pdh_v2.tiles import MeldKind
from tests.helpers import cards, meld


class StrategyFeatureTests(unittest.TestCase):
    def test_seven_pairs_potential_requires_no_declared_group(self) -> None:
        concealed = cards([0, 0, 1, 1, 2, 2, 9, 9, 10, 10, 27, 27, 31, 31])
        closed = route_potentials(concealed, ())
        declared = route_potentials(concealed, (0,))
        self.assertGreater(closed[0], 0.9)
        self.assertEqual(declared[0], -1.0)

    def test_closed_kong_preserves_only_four_concealed_qualification(self) -> None:
        base = route_state(cards([9, 9, 10, 10, 11, 11, 12, 12, 13, 13]), ())
        closed_kong = route_state(
            cards([9, 9, 10, 10, 11, 11, 12, 12, 13, 13]),
            (meld(0, MeldKind.CLOSED_KONG),),
        )
        pong = route_state(
            cards([9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14]),
            (meld(0, MeldKind.PONG),),
        )
        self.assertEqual(base.qualifications, (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(closed_kong.qualifications, (0.0, 0.0, 1.0, 0.0))
        self.assertEqual(pong.qualifications, (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
