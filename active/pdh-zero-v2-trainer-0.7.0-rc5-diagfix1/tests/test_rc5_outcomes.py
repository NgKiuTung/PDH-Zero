"""Candidate-only win accounting must never include its three opponents' combinations."""
import unittest

from pdh_v2.table.outcome_diagnostics import candidate_outcome


def fixture(seat):
    record = dict(method='DISCARD_WIN', audit={'outcome': {'winners': [
        dict(player=2, score=dict(multiplier=4, items=[{'pattern': 'SEVEN_PAIRS'}]))]}})
    result = dict(candidate_seat=seat, win=int(seat == 2), score=12 if seat == 2 else -12,
                  payments=[dict(payer=1, winner=2, multiplier=4, amount=12, horse_hits=2)])
    return record, result


class Rc5OutcomeTests(unittest.TestCase):
    def test_opponent_win_is_not_a_candidate_combination(self):
        record, result = fixture(1)
        own = candidate_outcome(record, result)
        self.assertFalse(own['won'])
        self.assertIsNone(own['combination'])
        self.assertEqual(own['outgoing'], 12)

    def test_candidate_win_has_its_own_pattern_and_income(self):
        record, result = fixture(2)
        own = candidate_outcome(record, result)
        self.assertEqual(own['combination'], 'SEVEN_PAIRS')
        self.assertEqual(own['income'], 12)
        self.assertEqual(own['horse_hits'], [2])

    def test_income_mismatch_is_rejected(self):
        record, result = fixture(2)
        result['score'] = 11
        with self.assertRaisesRegex(ValueError, 'income_reconstruction'):
            candidate_outcome(record, result)