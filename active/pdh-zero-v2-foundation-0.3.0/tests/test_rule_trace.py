"""源文档与115项状态追踪完整性；仅基础模块已有测试，不承诺完整引擎。"""
import hashlib
import importlib
import json
from pathlib import Path
import re
import unittest

from pdh_v2.config import SOURCE_SHA256

ROOT = Path(__file__).resolve().parents[1]


def test_exists(reference: str) -> bool:
    """查验所指测试真实存在；不将字符串写进状态就视作已验收。"""
    module_name, class_name, method = reference.rsplit(".",2)
    module = importlib.import_module(module_name)
    return callable(getattr(getattr(module,class_name),method))


class RuleTraceTests(unittest.TestCase):
    """输入源一致，未决项未被默认关闭，旧T编号不丢失。"""

    def test_rule_source_unchanged(self) -> None:
        raw = (ROOT/"rules/source-v0.8.md").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),SOURCE_SHA256)

    def test_all_source_cases_preserved(self) -> None:
        source = (ROOT/"rules/source-v0.8.md").read_text("utf-8")
        rows = re.findall(r"^\| (T\d+) \| (.*?) \| (.*?) \|$",source,re.M)
        trace = json.loads((ROOT/"rules/cases.json").read_text("utf-8"))["cases"]
        actual = [(r["id"],r["condition"],r["expectation"]) for r in trace]
        self.assertEqual(actual,rows)
        self.assertEqual(len(actual),115)

    def test_test_references_exist(self) -> None:
        rows = json.loads((ROOT/"rules/cases.json").read_text("utf-8"))["cases"]
        refs = [ref for row in rows for ref in row["test_refs"]]
        self.assertTrue(all(map(test_exists,refs)))

    def test_no_false_full_training_readiness(self) -> None:
        decisions = json.loads((ROOT/"rules/open_decisions.json").read_text("utf-8"))
        self.assertIs(decisions["training_ready"],False)
        self.assertEqual(len(decisions["items"]),9)
        self.assertEqual({r["status"] for r in decisions["items"]},{"AWAITING_OWNER_RULE"})
