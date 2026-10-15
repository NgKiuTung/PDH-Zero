"""限时训练 Smoke 入口；用于 CPU/GPU 正确性与吞吐门禁。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdh_v2.training.runner import RunnerConfig, run  # noqa: E402


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    """解析参数并运行不覆盖既有目录的训练 Smoke。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-seconds", type=_positive, default=600)
    parser.add_argument("--environments", type=_positive, default=64)
    parser.add_argument("--hands-per-update", type=_positive, default=128)
    parser.add_argument("--hidden-size", type=_positive, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--reward-scale", type=float, default=32.0)
    parser.add_argument("--ppo-epochs", type=_positive, default=4)
    parser.add_argument("--minibatch-decisions", type=_positive, default=4096)
    parser.add_argument("--warmup-hands", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=_positive, default=2)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.output.exists():
        logging.error(
            json.dumps(
                {
                    "level": "ERROR",
                    "message": "输出目录已存在，拒绝覆盖。",
                    "output": str(args.output),
                },
                ensure_ascii=False,
            )
        )
        return 2
    config = RunnerConfig(
        output=args.output,
        duration_seconds=args.duration_seconds,
        environments=args.environments,
        hands_per_update=args.hands_per_update,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        reward_scale=args.reward_scale,
        ppo_epochs=args.ppo_epochs,
        minibatch_decisions=args.minibatch_decisions,
        warmup_hands=args.warmup_hands,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        device=args.device,
        allow_shared_gpu=args.allow_shared_gpu,
    )
    try:
        return run(config)
    except BaseException as exc:
        logging.exception(
            json.dumps(
                {
                    "level": "ERROR",
                    "message": "训练 Smoke 失败。",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
