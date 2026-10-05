"""离线门禁完整性、未完成和数据篡改测试。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pdh_v2.training.refinement_gate import review_run


def setup_evidence(root, gain=1):
    manifest = dict(scope='solo_stability_refinement_candidate', status='completed',
                    total_learner_decisions=1, accepted_updates=1)
    (root/'manifest.json').write_text(json.dumps(manifest))
    for label, delta in (('baseline', 0), ('final-d000000000001', gain)):
        directory = root/'evaluations'/label
        directory.mkdir(parents=True)
        rows = [dict(wall_seed=seed,candidate_seat=seat,score=delta,win=0,draw=1,point_shoot=0)
                for seed in range(4) for seat in range(4)]
        raw = '\n'.join(json.dumps(row) for row in rows)+'\n'
        (directory/'results.jsonl').write_text(raw)
        (directory/'summary.json').write_text(json.dumps(dict(hands=16,
            results_sha256=hashlib.sha256(raw.encode()).hexdigest())))


class GateTests(unittest.TestCase):
    def test_complete_pairs_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);setup_evidence(root)
            result=review_run(root)
            self.assertEqual(result['gate'],'PASS')
            self.assertFalse(result['long_training_allowed'])

    def test_stopped_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);setup_evidence(root)
            manifest=json.loads((root/'manifest.json').read_text())
            manifest['status']='stopped'
            (root/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(review_run(root)['gate'],'INCOMPLETE')

    def test_tampered_result_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);setup_evidence(root)
            (root/'evaluations/baseline/results.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):
                review_run(root)
