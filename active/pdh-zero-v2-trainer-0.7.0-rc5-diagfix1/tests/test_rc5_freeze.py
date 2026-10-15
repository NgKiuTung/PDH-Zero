"""Frozen protocol tests use deliberately synthetic receipts, never scientific evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from pdh_v2.table.data import contract
from pdh_v2.table.freeze import freeze_study, load_freeze, confined_file
from pdh_v2.table.study_contract import source_identity
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.refinement_control import atomic_json


def study_fixture(root, smoke=False, decision='keep_for_independent_confirmation'):
    study = root/'study'
    study.mkdir()
    manifest = dict(config=dict(seeds=[1, 2, 3], smoke_only=smoke,
                                opponents=['heuristic', 'route_mix', 'frozen_rc41']),
                    baseline_sha256=contract()['baseline']['sha256'],
                    protected_source=source_identity(root))
    atomic_json(study/'manifest.json', manifest)
    atomic_json(study/'report.json', dict(status='PASS', completed_arms=9))
    for seed in (1, 2, 3):
        for mode in ('baseline', 'joint'):
            directory = study/f'seed-{seed}/{mode}'
            directory.mkdir(parents=True)
            candidate = directory/'fixture.pt'
            candidate.write_bytes(f'SYNTHETIC FIXTURE {seed} {mode}'.encode())
            atomic_json(directory/'trial.json', dict(checkpoint=str(candidate.relative_to(study)),
                checkpoint_sha256=file_digest(candidate), decision={'status': decision}))
    return study


class Rc5FreezeTests(unittest.TestCase):
    def test_all_seeds_and_both_arms_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = study_fixture(root)
            frozen = freeze_study(root, study, root/'freeze')
            loaded = load_freeze(root, root/'freeze')
            self.assertEqual(len(frozen['candidates']), 6)
            self.assertEqual(frozen, loaded)

    def test_smoke_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = study_fixture(root, smoke=True)
            with self.assertRaisesRegex(ValueError, 'not_eligible'):
                freeze_study(root, study, root/'freeze')
            self.assertFalse((root/'freeze').exists())

    def test_inconclusive_seed_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = study_fixture(root, decision='inconclusive')
            with self.assertRaisesRegex(ValueError, 'not_all_seed'):
                freeze_study(root, study, root/'freeze')

    def test_tampered_frozen_file_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = study_fixture(root)
            frozen = freeze_study(root, study, root/'freeze')
            (root/'freeze'/frozen['candidates'][0]['file']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'candidate_digest'):
                load_freeze(root, root/'freeze')

    def test_tampered_freeze_manifest_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = study_fixture(root)
            freeze_study(root, study, root/'freeze')
            (root/'freeze/freeze.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'freeze_digest'):
                load_freeze(root, root/'freeze')

    def test_candidate_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'study').mkdir()
            (root/'outside.pt').write_bytes(b'outside')
            with self.assertRaisesRegex(ValueError, 'outside_evidence'):
                confined_file(root/'study', '../outside.pt')