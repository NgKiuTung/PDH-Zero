"""长训练候选 CPU 门禁：晋级、完整恢复、评估和原子牌谱。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """执行长任务专项测试并输出结构化报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        print(json.dumps({"status": "FAIL", "error": "report exists"}))
        return 2
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_long_training", "-v"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = result.stdout + result.stderr
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "tests_exit_code": result.returncode,
        "long_job_candidate_check": "PASS" if result.returncode == 0 else "FAIL",
        "long_job_entry_ready": False,
        "curriculum_gate_required": True,
        "blocked_reason": "0.6.0-rc1 is a curriculum-tuning candidate",
        "formal_long_run_ready": False,
    }
    if args.report is not None:
        args.report.mkdir(parents=True, exist_ok=False)
        (args.report / "tests.log").write_text(output, encoding="utf-8")
        (args.report / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
