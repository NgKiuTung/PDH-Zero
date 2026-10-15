"""rc4候选输入：真实身份绑定的边界与篡改拒绝。"""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pdh_v2.training.candidate_store import CandidateStore, file_digest
from pdh_v2.training.selection_input import load_selection, resolve_selection
from tests.test_refinement import write_warmup


def fixture(root):
    source = write_warmup(root, update=13)
    source.with_suffix(".pt.sha256").write_text(file_digest(source) + "  warmup.pt\n")
    store = CandidateStore(root)
    store.register(source, "d000000265095", 13, 265095)
    rows = root / "evaluations/d000000265095/results.jsonl"
    rows.parent.mkdir(parents=True)
    rows.write_text('{"score":0}\n')
    store.complete("d000000265095", {"results_sha256": file_digest(rows)}, {}, "INCONCLUSIVE")
    store.select("d000000265095")
    return root / "selected_development.json", rows


class SelectedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selection, self.rows = fixture(self.root)

    def test_model_and_frozen_anchor_load(self):
        model, anchor, report = load_selection(self.selection, torch.device("cpu"))
        self.assertEqual(report["source_update"], 13)
        self.assertFalse(report["source_is_confirmatory"])
        self.assertEqual(report["source_gate"], "INCONCLUSIVE")
        self.assertFalse(any(p.requires_grad for p in anchor.parameters()))
        self.assertEqual(model.config.risk_policy_weight, 0)

    def test_modified_weight_rejected(self):
        source, _ = resolve_selection(self.selection)
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            resolve_selection(self.selection)

    def test_modified_evaluation_rejected(self):
        self.rows.write_text("changed")
        with self.assertRaisesRegex(ValueError, "evaluation.hash"):
            resolve_selection(self.selection)

    def test_pointer_traversal_rejected(self):
        record = json.loads(self.selection.read_text())
        record["checkpoint"] = "warmup.pt"
        self.selection.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "path"):
            resolve_selection(self.selection)

    def test_summary_mismatch_rejected(self):
        record = json.loads(self.selection.read_text())
        record["summary"]["mean_score"] = 99
        self.selection.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "summary"):
            resolve_selection(self.selection)

    def test_unsupported_scope_rejected(self):
        record = json.loads(self.selection.read_text())
        record["long_training_allowed"] = True
        self.selection.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "scope"):
            resolve_selection(self.selection)
