"""冻结规则源、案例追踪和测试引用完整性。"""
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from pdh_v2.config import SOURCE_SHA256

ROOT = Path(__file__).resolve().parents[1]


def test_exists(reference: str) -> bool:
    module_name, class_name, method = reference.rsplit(".", 2)
    module = importlib.import_module(module_name)
    return callable(getattr(getattr(module, class_name), method))


class RuleTraceTests(unittest.TestCase):
    def test_rule_source_hash_matches_frozen_v09(self) -> None:
        raw = (ROOT / "rules/source-v0.9.md").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SOURCE_SHA256)

    def test_case_ids_are_unique_and_preserve_original_range(self) -> None:
        rows = json.loads((ROOT / "rules/cases.json").read_text("utf-8"))["cases"]
        ids = [row["id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({f"T{index:02d}" for index in range(1, 116)} <= set(ids))
        self.assertTrue({f"T{index}" for index in range(116, 131)} <= set(ids))

    def test_test_references_exist(self) -> None:
        rows = json.loads((ROOT / "rules/cases.json").read_text("utf-8"))["cases"]
        refs = [ref for row in rows for ref in row["test_refs"]]
        self.assertTrue(all(map(test_exists, refs)))

    def test_rules_are_frozen_but_training_not_ready(self) -> None:
        decisions = json.loads((ROOT / "rules/open_decisions.json").read_text("utf-8"))
        self.assertIs(decisions["training_ready"], False)
        self.assertIs(decisions["full_rules_frozen"], True)
        self.assertEqual(decisions["items"], [])


if __name__ == "__main__":
    unittest.main()
