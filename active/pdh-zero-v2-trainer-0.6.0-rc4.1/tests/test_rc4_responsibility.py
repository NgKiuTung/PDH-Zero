"""PDH研究代码：账本金额守恒、责任分母、未知来源与多响去重。"""
import copy
import unittest
from pdh_v2.training.responsibility_metrics import responsibility_row, summarize_responsibility


def row(cause="ordinary_discard", amount=12, seat=0):
    return {"candidate_seat":seat,"score":-amount,"candidate_loss":amount,
        "settlement_audit_verified":True,"wall_seed":1,
        "payments":[{"payer":seat,"winner":1,"amount":amount,"before_horses":amount//3,
                     "horse_uplift":amount-amount//3,"cause":cause}]}


class ResponsibilityTests(unittest.TestCase):
    def test_direct_amount_and_denominator(self):
        report = summarize_responsibility([row()])
        self.assertEqual(report["direct_loss"],12)
        self.assertEqual(report["defense_loss_per_100_seat_hands"],1200)

    def test_self_draw_not_automatically_blame(self):
        report = responsibility_row(row("ordinary_self_draw"))
        self.assertEqual(report["direct_loss"],0)
        self.assertEqual(report["ordinary_self_draw_share"],12)

    def test_multiwin_sums_amount_once_hand(self):
        item = row("discard_multi")
        other = copy.deepcopy(item["payments"][0])
        other["winner"] = 2
        item["payments"].append(other)
        item.update(score=-24,candidate_loss=24)
        report = summarize_responsibility([item])
        self.assertEqual(report["direct_loss"],24)
        self.assertEqual(report["direct_loss_hand"],1)

    def test_rob_kong_direct(self):
        self.assertEqual(responsibility_row(row("rob_kong"))["direct_loss"],12)

    def test_kong_liability_direct(self):
        self.assertEqual(responsibility_row(row("kong_draw_liability"))["direct_loss"],12)

    def test_twelve_liability_direct(self):
        self.assertEqual(responsibility_row(row("twelve_grounded_liability"))["direct_loss"],12)

    def test_unknown_cause_rejected(self):
        with self.assertRaises(ValueError):
            responsibility_row(row("luck"))

    def test_wrong_score_rejected(self):
        item=row()
        item["score"]=-2
        with self.assertRaises(ValueError):
            responsibility_row(item)

    def test_wrong_horse_sum_rejected(self):
        item=row()
        item["payments"][0]["horse_uplift"]=1
        with self.assertRaises(ValueError):
            responsibility_row(item)

    def test_negative_payment_rejected(self):
        with self.assertRaises(ValueError):
            responsibility_row(row(amount=-3))

    def test_boolean_seat_rejected(self):
        with self.assertRaises(ValueError):
            responsibility_row(row(seat=False))

    def test_missing_evidence_rejected(self):
        with self.assertRaises(ValueError):
            responsibility_row({})
