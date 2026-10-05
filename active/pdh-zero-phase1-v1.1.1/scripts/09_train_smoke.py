#!/usr/bin/env python3
"""Run bounded, resumable Phase 1 recurrent PPO training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from script_common import artifact_dir
from pdh_phase1.rl.trainer import Phase1Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_smoke.yaml")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "rocm"))
    parser.add_argument("--resume")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = artifact_dir(args.output, child="training")
    source = Path(args.config)
    resolved = json.loads(source.read_text(encoding="utf-8"))
    if args.seed is not None:
        resolved["seed"] = args.seed
        resolved["config_id"] = f"{resolved['config_id']}_seed{args.seed}"
    if args.device is not None:
        resolved["device"] = args.device
    resolved_path = out / "resolved_config.json"
    resolved_path.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    trainer = Phase1Trainer(resolved_path, out)
    if args.resume:
        trainer.resume(args.resume)
    summary = trainer.train(args.updates)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
