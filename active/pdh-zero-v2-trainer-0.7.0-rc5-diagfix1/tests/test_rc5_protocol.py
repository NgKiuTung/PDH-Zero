"""Fixed budgets, source protection and one-use evaluation receipts fail closed."""
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from pdh_v2.table.confirmation import consume_freeze
from pdh_v2.table.exposure import ExposureStore
from pdh_v2.table.process_control import validate_device, RunSignals
from pdh_v2.table.study_contract import StudyConfig, BudgetClock, source_identity, check_identity
from pdh_v2.table.study_evaluation import decision_record


class Rc5ProtocolTests(unittest.TestCase):
    def test_duplicate_seeds_rejected(self):
        with self.assertRaises(ValueError):
            StudyConfig(seeds=(1, 1))

    def test_incomplete_wall_groups_rejected(self):
        with self.assertRaises(ValueError):
            StudyConfig(seeds=(1, 2), evaluation_hands=5)

    def test_smoke_cannot_claim_confirmatory_scale(self):
        with self.assertRaises(ValueError):
            StudyConfig(seeds=(1, 2, 3), smoke_only=False)

    def test_no_negative_worker_budget(self):
        with self.assertRaises(ValueError):
            StudyConfig(seeds=(1, 2), workers=-1)

    def test_smoke_never_promotes_even_with_positive_gate(self):
        result = decision_record(StudyConfig(seeds=(1, 2)), {'x': {'gate': 'PASS'}},
                                 Path('candidate.pt'), {})
        self.assertEqual(result['status'], 'inconclusive')
        self.assertFalse(result['deployment_promoted'])

    def test_development_regression_discards_but_keeps_artifact(self):
        config = StudyConfig(seeds=(1, 2, 3), evaluation_hands=4096, smoke_only=False)
        result = decision_record(config, {'x': {'gate': 'REGRESSION'}}, Path('saved.pt'), {})
        self.assertEqual(result['status'], 'discard')
        self.assertEqual(result['candidate'], 'saved.pt')

    def test_deadline_raises_instead_of_shortening_a_hand(self):
        clock = BudgetClock(10)
        with patch('pdh_v2.table.study_contract.time.monotonic', return_value=clock.started+11):
            with self.assertRaises(TimeoutError):
                clock.check()

    def test_cpp_mutation_invalidates_protected_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'src').mkdir()
            code = root/'src/kernel.cpp'
            code.write_text('original')
            before = source_identity(root)
            code.write_text('modified')
            with self.assertRaisesRegex(RuntimeError, 'protected_source'):
                check_identity(root, before)

    def test_native_root_not_only_src_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'native').mkdir()
            code = root/'native/kernel.cpp'
            code.write_text('original')
            before = source_identity(root)
            code.write_text('modified')
            with self.assertRaisesRegex(RuntimeError, 'protected_source'):
                check_identity(root, before)

    def test_one_use_confirmation_consumed_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with ExposureStore(root/'exposure.sqlite3') as store:
                consume_freeze(store, 'frozen-id', root/'first')
                with self.assertRaises(sqlite3.IntegrityError):
                    consume_freeze(store, 'frozen-id', root/'second')

    def test_empty_registry_cannot_hide_historical_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ExposureStore(Path(tmp)/'exposure.sqlite3') as store:
                with self.assertRaisesRegex(ValueError, 'historical_exposure'):
                    store.require_known_history()

    def test_cuda_unavailable_is_explicit_failure(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'cuda_not_available'):
                validate_device('cuda')

    def test_signal_handlers_restore(self):
        import signal
        before = signal.getsignal(signal.SIGINT)
        with tempfile.TemporaryDirectory() as tmp:
            handler = RunSignals(Path(tmp))
            handler.close()
            self.assertIs(signal.getsignal(signal.SIGINT), before)