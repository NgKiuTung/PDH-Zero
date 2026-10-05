#!/usr/bin/env python3
"""Benchmark deterministic Mini-PDH environment throughput and invariants."""

from __future__ import annotations

import argparse
import time

from script_common import artifact_dir, write_json
from pdh_phase1.config import load_config
from pdh_phase1.mini_pdh import MiniPDHEnv
from pdh_phase1.mini_pdh.bots import GoalDistanceBot, RandomLegalBot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", default="configs/mini_pdh_v0.yaml")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--bot", choices=("random", "heuristic"), default="random")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("games must be positive")
    rules = load_config(args.rules)
    decisions = 0
    wins = 0
    draws = 0
    started = time.perf_counter()
    for game in range(args.games):
        seed = args.seed + game
        env = MiniPDHEnv(rules)
        env.reset(seed)
        bot = RandomLegalBot(seed ^ 0x1234) if args.bot == "random" else GoalDistanceBot(seed ^ 0x1234)
        while not env.done:
            env.step(bot.choose_action(env))
            decisions += 1
        env.assert_invariants()
        wins += int(env.outcome.get("type") != "DRAW")
        draws += int(env.outcome.get("type") == "DRAW")
    elapsed = time.perf_counter() - started
    summary = {
        "status": "PASS",
        "ruleset_id": rules["ruleset_id"],
        "bot": args.bot,
        "games": args.games,
        "decisions": decisions,
        "elapsed_seconds": elapsed,
        "games_per_second": args.games / elapsed,
        "decisions_per_second": decisions / elapsed,
        "mean_decisions_per_game": decisions / args.games,
        "win_rate": wins / args.games,
        "draw_rate": draws / args.games,
        "invariant_failures": 0,
    }
    write_json(artifact_dir(args.output) / "engine_benchmark.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
