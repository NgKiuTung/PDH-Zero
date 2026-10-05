"""训练候选包 CPU 门禁：观察隔离、批量采样、PPO 与检查点。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """执行包含训练模块的完整测试，不访问 CUDA。"""
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
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
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
        "torch": None,
        "tests_exit_code": result.returncode,
        "training_candidate_check": "PASS" if result.returncode == 0 else "FAIL",
        "engine_ready": True,
        "full_rules_frozen": True,
        "training_entry_ready": result.returncode == 0,
        "formal_long_run_ready": False,
        "required_next_gate": "10-minute CUDA smoke on target RTX 5090",
    }
    try:
        import torch

        report["torch"] = torch.__version__
    except ImportError:
        report["training_candidate_check"] = "FAIL"
        report["training_entry_ready"] = False
    if args.report is not None:
        args.report.mkdir(parents=True, exist_ok=False)
        (args.report / "tests.log").write_text(output, encoding="utf-8")
        (args.report / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["training_candidate_check"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
