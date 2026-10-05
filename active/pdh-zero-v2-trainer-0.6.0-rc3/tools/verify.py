"""CPU基础验收入口；许可由项目所有者决定。不联网、不安装依赖、不加载CUDA。"""
import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    with (ROOT / "rules/open_decisions.json").open(encoding="utf-8") as stream:
        decisions = json.load(stream)
    with (ROOT / "src/pdh_v2/resources/confirmed.json").open(encoding="utf-8") as stream:
        confirmed = json.load(stream)
    return decisions, confirmed


def _run_checks() -> tuple[dict[str, Any], str]:
    decisions, confirmed = _inputs()
    source_hash = hashlib.sha256((ROOT / "rules/source-v0.9.md").read_bytes()).hexdigest()
    cases = json.loads((ROOT / "rules/cases.json").read_text("utf-8"))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    output = result.stdout + result.stderr
    count = re.search(r"Ran (\d+) tests?", output)
    source_hashes = (
        source_hash,
        confirmed["source_sha256"],
        decisions["source_sha256"],
        cases["source_sha256"],
    )
    source_ok = len(set(source_hashes)) == 1
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "scope": "frozen_rules_reference_engine",
        "source_sha256": source_hash,
        "source_matches": source_ok,
        "tests_run": int(count.group(1)) if count else 0,
        "test_exit_code": result.returncode,
        "engine_check": (
            "PASS" if source_ok and result.returncode == 0 and count else "FAIL"
        ),
        "foundation_check": (
            "PASS" if source_ok and result.returncode == 0 and count else "FAIL"
        ),
        "tracked_source_cases": len(cases["cases"]),
        "engine_ready": True,
        "training_ready": bool(decisions["training_ready"]),
        "full_rules_frozen": bool(decisions["full_rules_frozen"]),
        "remaining_rule_ids": [row["id"] for row in decisions["items"]],
        "strict_static_checks": "NOT_RUN_BY_THIS_COMMAND",
    }
    return report, output


def main() -> int:
    """运行测试并可写新报告目录；退出0代表冻结规则参考引擎验收通过，不代表模型训练入口已交付。"""
    messages = json.loads((ROOT / "src/pdh_v2/resources/messages.json").read_text("utf-8"))
    parser = argparse.ArgumentParser(description=messages["verify.description"])
    parser.add_argument("--report-dir", type=Path, help=messages["verify.report_argument"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.report_dir is not None:
        try:
            args.report_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            logging.error(
                json.dumps(
                    {"level": "ERROR", "message": messages["verify.output_exists"]},
                    ensure_ascii=False,
                )
            )
            return 2
    try:
        report, output = _run_checks()
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        logging.error(
            json.dumps(
                {
                    "level": "ERROR",
                    "message": messages["verify.failed"],
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1
    if args.report_dir is not None:
        (args.report_dir / "unit-tests.log").write_text(output, encoding="utf-8")
        report_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        (args.report_dir / "report.json").write_text(
            report_text,
            encoding="utf-8",
        )
    logging.info(json.dumps(report, ensure_ascii=False, indent=2))
    message_key = (
        "verify.ok" if report["engine_check"] == "PASS" else "verify.failed"
    )
    logging.info(messages[message_key])
    return 0 if report["engine_check"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
