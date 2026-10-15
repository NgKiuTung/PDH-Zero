"""候选不可丢失、尾部配对和真实支付归因回归。"""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

from pdh_v2.ledger import SettlementBook
from pdh_v2.qualifications import WinMethod
from pdh_v2.responsibility import TwelveGroundedLiability
from pdh_v2.settlement import OrdinaryWin, MultiDiscardWin, MultiWinner, DrawRecord
from pdh_v2.training.candidate_store import CandidateStore, file_digest
from pdh_v2.training.loss_attribution import payment_attribution
from pdh_v2.training.paired_metrics import paired_comparison, development_gate
from tests.test_refinement import paired_rows, write_warmup


def source(root):
    path = write_warmup(root)
    path.with_suffix('.pt.sha256').write_text(file_digest(path) + '  warmup.pt\n')
    return path


def terminal(book, receipt, method):
    return SimpleNamespace(audit={'settlement_events': book.events},
                           receipt=receipt, method=method)


class CandidateTests(unittest.TestCase):
    def test_other_candidate_operations_reject_traversal(self):
        with tempfile.TemporaryDirectory() as name:
            store = CandidateStore(Path(name))
            with self.assertRaises(ValueError):
                store.complete('../../elsewhere', {}, {}, 'PASS')
            with self.assertRaises(ValueError):
                store.select('../baseline')

    def test_changed_candidate_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            store = CandidateStore(root)
            saved = store.register(source(root), 'baseline', 0, 0)
            store.complete('baseline', {}, {}, 'BASELINE')
            saved.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                store.select('baseline')

    def test_inconclusive_weight_is_retained_after_latest_changes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            original = source(root)
            store = CandidateStore(root)
            saved = store.register(original, 'd000000265095', 13, 265095)
            before = file_digest(saved)
            store.complete('d000000265095', {'mean_score': .4}, {}, 'INCONCLUSIVE')
            original.write_bytes(b'replaced latest')
            self.assertEqual(file_digest(saved), before)

    def test_candidate_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaises(ValueError):
                CandidateStore(root).register(source(root), '../../bad', 0, 0)

    def test_existing_candidate_not_overwritten(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = source(root)
            store = CandidateStore(root)
            store.register(path, 'baseline', 0, 0)
            with self.assertRaises(FileExistsError):
                store.register(path, 'baseline', 0, 0)

    def test_source_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = source(root)
            path.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                CandidateStore(root).register(path, 'baseline', 0, 0)

    def test_unevaluated_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            store = CandidateStore(root)
            store.register(source(root), 'baseline', 0, 0)
            with self.assertRaises(ValueError):
                store.select('baseline')

    def test_inconclusive_is_selected_but_not_qualified(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            store = CandidateStore(root)
            store.register(source(root), 'baseline', 0, 0)
            store.complete('baseline', {'mean_score': 0}, {}, 'INCONCLUSIVE')
            store.select('baseline')
            record = json.loads((root / 'selected_development.json').read_text())
            self.assertFalse(record['long_training_allowed'])
            with self.assertRaises(ValueError):
                store.select('baseline', qualified=True)

    def test_complete_twice_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            store = CandidateStore(root)
            store.register(source(root), 'baseline', 0, 0)
            store.complete('baseline', {}, {}, 'PASS')
            with self.assertRaises(ValueError):
                store.complete('baseline', {}, {}, 'PASS')


class TailAndPaymentTests(unittest.TestCase):
    def test_tail_bootstrap_constant_shift(self):
        result = paired_comparison(paired_rows(), paired_rows(1), 100)
        self.assertEqual(result['cvar_5']['ci95'], [1., 1.])

    def test_tail_regression_can_reject_positive_mean(self):
        comparison = {'score': {'delta': 1, 'ci95': [.2, 2]},
                      'point_shoot': {'delta': -.01, 'ci95': [-.02, 0]},
                      'cvar_5': {'delta': -5, 'ci95': [-8, -3]}}
        self.assertEqual(development_gate(comparison), 'REGRESSION')

    def test_tail_uncertainty_does_not_claim_safe(self):
        comparison = {'score': {'delta': 1, 'ci95': [.2, 2]},
                      'point_shoot': {'delta': -.01, 'ci95': [-.02, 0]},
                      'cvar_5': {'delta': -1, 'ci95': [-8, 3]}}
        self.assertEqual(development_gate(comparison), 'INCONCLUSIVE')

    def test_ordinary_self_draw_attribution(self):
        book = SettlementBook()
        win = OrdinaryWin('S:H', 'H', 0, 0, 0, 6, 1, WinMethod.SELF_DRAW)
        book.begin(win, 6)
        book.bind('S:H', 'wall', tuple(range(6)))
        receipt = book.pay('S:H')
        result = payment_attribution(terminal(book, receipt, 'SELF_DRAW'), 1)
        self.assertEqual(result['candidate_loss'], -receipt[1])
        self.assertEqual(result['candidate_loss_before_horses'], 6)
        self.assertEqual(result['payments'][0]['cause'], 'ordinary_self_draw')

    def test_twelve_liability_not_counted_as_ordinary_discard(self):
        book = SettlementBook()
        win = OrdinaryWin('S:H', 'H', 0, 0, 0, 6, 1, WinMethod.SELF_DRAW)
        book.begin_twelve_grounded(win, 6, TwelveGroundedLiability('H', 0, 2))
        book.bind('S:H', 'wall', tuple(range(6)))
        receipt = book.pay('S:H')
        result = payment_attribution(terminal(book, receipt, 'SELF_DRAW'), 2)
        self.assertEqual(len(result['payments']), 1)
        self.assertEqual(result['payments'][0]['shares'], 3)
        self.assertEqual(result['payments'][0]['cause'], 'twelve_grounded_liability')

    def test_multiwin_each_transfer_has_its_own_horses(self):
        book = SettlementBook()
        win = MultiDiscardWin('M:H', 'H', 0, 0, 1, 3,
                               (MultiWinner(0, 6), MultiWinner(1, 3)))
        book.begin_multi(win, 6)
        book.bind_multi('M:H', 'wall', tuple(range(6)))
        receipt = book.pay_multi('M:H')
        result = payment_attribution(terminal(book, receipt, 'DISCARD_WIN'), 3)
        self.assertEqual(result['candidate_loss'], -receipt[3])
        self.assertEqual(len(result['payments']), 2)

    def test_draw_has_no_payment_but_is_recorded(self):
        book = SettlementBook()
        book.record_draw(DrawRecord('D:H', 'H', 0, 0))
        result = payment_attribution(terminal(book, (0, 0, 0, 0), 'DRAW'), 2)
        self.assertEqual(result['payments'], [])
        self.assertTrue(result['settlement_audit_verified'])

    def test_bad_receipt_refused(self):
        book = SettlementBook()
        book.record_draw(DrawRecord('D:H', 'H', 0, 0))
        with self.assertRaises(ValueError):
            payment_attribution(terminal(book, (1, -1, 0, 0), 'DRAW'), 0)