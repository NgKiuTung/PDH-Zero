"""PDH研究代码：容器配额解释与负载选择门禁。"""
from pathlib import Path
import tempfile
import unittest

from pdh_v2.runtime.resources import detect_resources, parse_cpu_max, worker_candidates
from pdh_v2.runtime.plan import choose_trial


def trial(backend, workers, speed, digest="same"):
    return {"status":"PASS","backend":backend,"workers":workers,"tables":64,
            "repeats":[{"environment_decisions_per_second":speed,"digest":digest},
                       {"environment_decisions_per_second":speed,"digest":digest}]}


class ResourceTests(unittest.TestCase):
    def test_uploaded_quota_is_fourteen(self):
        self.assertEqual(parse_cpu_max("1400000 100000"),14)

    def test_unlimited_is_not_128_dedicated(self):
        self.assertIsNone(parse_cpu_max("max 100000"))

    def test_zero_period_rejected(self):
        with self.assertRaises(ValueError):
            parse_cpu_max("1400000 0")

    def test_zero_quota_rejected(self):
        with self.assertRaises(ValueError):
            parse_cpu_max("0 100000")

    def test_visible_ancestor_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"group").mkdir()
            (root/"cpu.max").write_text("1400000 100000")
            (root/"group/cpu.max").write_text("max 100000")
            (root/"memory.max").write_text("64424509440")
            report=detect_resources(root,"0::/group",128)
        self.assertEqual(report["effective_cpu_equivalents"],14)
        self.assertEqual(report["memory_limit_bytes"],60*1024**3)
        self.assertEqual(worker_candidates(report),[2,4,6,8,10,12])

    def test_affinity_is_tighter_than_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"cpu.max").write_text("1400000 100000")
            report=detect_resources(root,"0::/",4)
        self.assertEqual(report["effective_cpu_equivalents"],4)
        self.assertEqual(worker_candidates(report),[2])

    def test_missing_cgroups_use_declared_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=detect_resources(Path(tmp),"",128)
        self.assertFalse(report["quota_known"])
        self.assertEqual(report["effective_cpu_equivalents"],14)

    def test_near_best_chooses_lower_load(self):
        report=choose_trial([trial("python",4,1000),trial("cpp",4,1510),trial("cpp",8,1550)])
        self.assertEqual(report["workers"],4)
        self.assertEqual(report["feature_backend"],"cpp")

    def test_tiny_gain_keeps_python(self):
        report=choose_trial([trial("python",4,1000),trial("cpp",4,1020)])
        self.assertEqual(report["feature_backend"],"python")

    def test_different_outcome_not_a_faster_candidate(self):
        report=choose_trial([trial("python",4,1000),trial("cpp",4,4000,"different")])
        self.assertEqual(report["feature_backend"],"python")
