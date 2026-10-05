"""PDH-Zero v2 长训练任务控制入口。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdh_v2.training.job_control import (  # noqa: E402
    job_status,
    launch_new,
    resume_job,
    stop_job,
)
from pdh_v2.training.long_config import LongRunConfig  # noqa: E402
from pdh_v2.training.long_runner import run_long  # noqa: E402
from pdh_v2.training.long_runtime import read_json  # noqa: E402
from pdh_v2.training.promotion import validate_smoke  # noqa: E402


def _pdh_home(value: str | None) -> Path:
    return Path(value or os.environ.get("PDH_HOME", "/root/rivermind-data/pdh-zero"))


def _job_dir(args: argparse.Namespace) -> Path:
    if args.job_dir is not None:
        return args.job_dir.resolve()
    pointer = _pdh_home(args.pdh_home) / "state" / "current_v2_long_job.txt"
    return Path(pointer.read_text(encoding="utf-8").strip()).resolve()


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-smoke")
    validate.add_argument("--smoke-dir", required=True, type=Path)

    start = subparsers.add_parser("start")
    start.add_argument("--smoke-dir", required=True, type=Path)
    start.add_argument("--pdh-home")
    start.add_argument("--max-hours", type=float, default=48.0)
    start.add_argument("--target-decisions", type=_positive, default=50_000_000)
    start.add_argument("--environments", type=_positive, default=96)
    start.add_argument("--hands-per-update", type=_positive, default=192)
    start.add_argument("--evaluation-hands", type=_positive, default=2048)
    start.add_argument("--evaluation-interval-decisions", type=_positive, default=5_000_000)
    start.add_argument("--snapshot-interval-decisions", type=_positive, default=5_000_000)
    start.add_argument("--min-free-gib", type=float, default=8.0)

    for name in ("status", "stop", "resume"):
        child = subparsers.add_parser(name)
        child.add_argument("--job-dir", type=Path)
        child.add_argument("--pdh-home")

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--job-dir", required=True, type=Path)
    worker.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    """解析命令并返回结构化结果。"""
    args = _parser().parse_args()
    script = Path(__file__).resolve()
    try:
        if args.command == "validate-smoke":
            result = validate_smoke(args.smoke_dir)
            print(json.dumps(asdict(result), default=str, ensure_ascii=False, indent=2))
            return 0
        if args.command == "start":
            if args.max_hours <= 0 or args.min_free_gib <= 0:
                raise ValueError("max-hours and min-free-gib must be positive")
            overrides = {
                "max_wall_seconds": int(args.max_hours * 3600),
                "target_decisions": args.target_decisions,
                "environments": args.environments,
                "hands_per_update": args.hands_per_update,
                "evaluation_hands": args.evaluation_hands,
                "evaluation_interval_decisions": args.evaluation_interval_decisions,
                "snapshot_interval_decisions": args.snapshot_interval_decisions,
                "min_free_bytes": int(args.min_free_gib * 1024**3),
            }
            result = launch_new(
                script,
                _pdh_home(args.pdh_home),
                args.smoke_dir,
                overrides,
            )
        elif args.command == "status":
            result = job_status(_job_dir(args))
        elif args.command == "stop":
            result = stop_job(_job_dir(args))
        elif args.command == "resume":
            result = resume_job(script, _pdh_home(args.pdh_home), _job_dir(args))
        else:
            config = LongRunConfig.from_dict(read_json(args.job_dir / "config.json"))
            return run_long(config, resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        logging.exception(
            json.dumps(
                {
                    "level": "ERROR",
                    "command": args.command,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
