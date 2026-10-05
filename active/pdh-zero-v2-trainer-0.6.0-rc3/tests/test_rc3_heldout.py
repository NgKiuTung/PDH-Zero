"""独立牌墙验证只评冻结模型，拒绝开发集重叠与篡改权重。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pdh_v2.training.heldout_evaluation import (
    evaluate_frozen_pair, load_evaluation_model, reserve_walls, resolve_candidate,
    validate_new_walls,
)
from pdh_v2.training.refinement_artifacts import save_model
from tests.test_refinement import fixture_model, paired_rows


def source_run(root):
    run = root / 'source'
    directory = run / 'evaluations/baseline'
    directory.mkdir(parents=True)
    data = '\n'.join(json.dumps(row) for row in paired_rows()) + '\n'
    (directory / 'results.jsonl').write_text(data)
    (directory / 'summary.json').write_text(json.dumps({
        'hands': 12, 'results_sha256': hashlib.sha256(data.encode()).hexdigest()}))
    (run / 'manifest.json').write_text(json.dumps({'status': 'completed'}))
    model = fixture_model()
    optimizer = torch.optim.AdamW(model.parameters())
    save_model(run / 'checkpoints/checkpoint_initial.pt', model, optimizer, 0, {})
    with torch.no_grad():
        next(model.parameters()).add_(.01)
    candidate = run / 'checkpoints/candidate.pt'
    save_model(candidate, model, optimizer, 1, {})
    return run, candidate


class HeldoutTests(unittest.TestCase):
    def test_reusing_reserved_walls_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            reserve_walls(root, 900, 8, root / 'one')
            with self.assertRaises(ValueError):
                reserve_walls(root, 1000903, 4, root / 'two')

    def test_baseline_fallback_is_not_a_new_candidate(self):
        with tempfile.TemporaryDirectory() as name:
            run, _ = source_run(Path(name))
            (run / 'selected_development.json').write_text('{"label":"baseline"}')
            with self.assertRaises(ValueError):
                resolve_candidate(run, None)

    def test_new_wall_keys_exclude_all_development_groups(self):
        with tempfile.TemporaryDirectory() as name:
            run, _ = source_run(Path(name))
            self.assertEqual(validate_new_walls(run, 900, 8), {900, 1000903})

    def test_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            run, _ = source_run(Path(name))
            with self.assertRaises(ValueError):
                validate_new_walls(run, 1, 4)

    def test_source_path_cannot_escape_checkpoints(self):
        with tempfile.TemporaryDirectory() as name:
            run, _ = source_run(Path(name))
            with self.assertRaises(ValueError):
                resolve_candidate(run, run / '../different.pt')

    def test_tampered_weights_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            _, candidate = source_run(Path(name))
            candidate.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                load_evaluation_model(candidate, torch.device('cpu'))

    def test_complete_cpu_frozen_pair_has_attributed_rows(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run, candidate = source_run(root)
            before = hashlib.sha256(candidate.read_bytes()).hexdigest()
            output = root / 'validation'
            report = evaluate_frozen_pair(run, output, candidate, hands=4, seed=900,
                workers=0, device_name='cpu', max_seconds=120, install_signals=False)
            self.assertEqual(report['status'], 'completed')
            self.assertFalse(report['long_training_allowed'])
            self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest(), before)
            self.assertTrue((output / 'candidate/results.jsonl').is_file())
            row = json.loads((output / 'candidate/results.jsonl').read_text().splitlines()[0])
            self.assertTrue(row['settlement_audit_verified'])