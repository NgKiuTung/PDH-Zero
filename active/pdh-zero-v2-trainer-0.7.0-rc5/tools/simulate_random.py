"""固定随机流运行合法整局，输出牌实体守恒与终局类型摘要。"""
import argparse
import json
import logging
from pathlib import Path
import random
from typing import Any

from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.engine import MahjongHand
from pdh_v2.game.state import Phase

LOGGER = logging.getLogger("pdh_v2.simulate_random")


def _choose_action(hand: MahjongHand, rng: random.Random) -> Action:
    """优先合法胡；其余动作以固定概率覆盖碰杠和普通弃牌。"""
    player = hand.current_player if hand.phase is Phase.ACTION else hand.response.queue[0]
    actions = list(hand.legal_actions(player))
    wins = [action for action in actions if action.kind is ActionKind.WIN]
    if wins:
        return wins[0]
    if hand.phase is not Phase.ACTION:
        claims = [
            action
            for action in actions
            if action.kind in (ActionKind.PONG, ActionKind.OPEN_KONG)
        ]
        if claims and rng.random() < 0.18:
            return rng.choice(claims)
        return next(action for action in actions if action.kind is ActionKind.PASS)
    kongs = [
        action
        for action in actions
        if action.kind in (ActionKind.CLOSED_KONG, ActionKind.ADDED_KONG)
    ]
    if kongs and rng.random() < 0.15:
        return rng.choice(kongs)
    discards = [action for action in actions if action.kind is ActionKind.DISCARD]
    return rng.choice(discards)


def simulate(hands: int, seed: int, max_steps: int) -> dict[str, Any]:
    """运行指定手数；任何非法状态、超步数或牌实体漂移都会直接失败。"""
    rng = random.Random(seed)
    outcomes: dict[str, int] = {}
    decisions = 0
    for index in range(hands):
        wall = list(range(136))
        rng.shuffle(wall)
        hand = MahjongHand(f"SIM-{index:08d}", index % 4, tuple(wall))
        hand.validate_invariants()
        for _ in range(max_steps):
            if hand.phase is Phase.TERMINAL:
                break
            hand.step(_choose_action(hand, rng))
            decisions += 1
            hand.validate_invariants()
        if hand.outcome is None:
            raise RuntimeError(f"hand exceeded max_steps: {hand.hand_id}")
        outcomes[hand.outcome.method] = outcomes.get(hand.outcome.method, 0) + 1
    return {
        "schema_version": 1,
        "hands": hands,
        "seed": seed,
        "max_steps": max_steps,
        "decisions": decisions,
        "outcomes": outcomes,
        "status": "PASS",
    }


def main() -> int:
    """CLI入口；可选输出文件必须为新路径，防止覆盖既有证据。"""
    parser = argparse.ArgumentParser(description="运行PDH-Zero v2随机合法整局压力验收。")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.hands <= 0 or args.max_steps <= 0:
        parser.error("--hands和--max-steps必须为正整数")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = simulate(args.hands, args.seed, args.max_steps)
        raw = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(raw)
        LOGGER.info(raw.rstrip())
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error(
            json.dumps(
                {"status": "FAIL", "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
