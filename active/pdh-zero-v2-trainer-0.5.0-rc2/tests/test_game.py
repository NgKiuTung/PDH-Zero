"""136张发牌、响应窗口、责任自动生成与确定性牌谱。"""
from collections import Counter
import random
import unittest

from pdh_v2.contracts import RuleError
from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.engine import MahjongHand
from pdh_v2.game.state import MeldRecord, Phase
from pdh_v2.game.wall import Wall
from pdh_v2.replay import HandTranscript, replay_hand
from pdh_v2.tiles import Meld, MeldKind, tile_type


def allocate(kind_lists: list[list[int]]) -> list[list[int]]:
    used: Counter[int] = Counter()
    result: list[list[int]] = []
    for kinds in kind_lists:
        hand: list[int] = []
        for kind in kinds:
            offset = used[kind]
            if offset >= 4:
                raise ValueError(kind)
            used[kind] += 1
            hand.append(kind * 4 + offset)
        result.append(hand)
    return result


def wall_for_hands(
    dealer: int,
    hands: list[list[int]],
    back_tile: int | None = None,
) -> tuple[int, ...]:
    """按正式发牌时序把目标起手牌映射到牌墙前缀。"""
    expected = [13] * 4
    expected[dealer] = 14
    if [len(hand) for hand in hands] != expected:
        raise ValueError("hand sizes")
    cursor = [0] * 4
    prefix: list[int] = []
    seats = [(dealer + offset) % 4 for offset in range(4)]
    for _ in range(3):
        for seat in seats:
            prefix.extend(hands[seat][cursor[seat] : cursor[seat] + 4])
            cursor[seat] += 4
    for index, seat in enumerate(seats):
        count = 2 if index == 0 else 1
        prefix.extend(hands[seat][cursor[seat] : cursor[seat] + count])
        cursor[seat] += count
    remaining = [card for card in range(136) if card not in set(prefix)]
    if back_tile is not None:
        remaining.remove(back_tile)
        remaining.append(back_tile)
    return tuple(prefix + remaining)


def pass_current(hand: MahjongHand) -> None:
    actions = hand.legal_actions(hand.response.queue[0])
    hand.step(next(action for action in actions if action.kind is ActionKind.PASS))


class WallTests(unittest.TestCase):
    def test_traditional_deal_sizes_and_order(self) -> None:
        wall = Wall(tuple(range(136)))
        hands = wall.deal(0)
        self.assertEqual([len(hand) for hand in hands], [14, 13, 13, 13])
        self.assertEqual(hands[0][:4], (0, 1, 2, 3))
        self.assertEqual(hands[1][:4], (4, 5, 6, 7))
        self.assertEqual(hands[0][-2:], (48, 49))
        self.assertEqual(hands[1][-1], 50)
        self.assertEqual(wall.remaining, 83)

    def test_front_and_back_have_independent_successors(self) -> None:
        wall = Wall(tuple(range(136)))
        front = wall.draw_front(0)
        back = wall.draw_back(1)
        self.assertEqual((front.tile, back.tile), (0, 135))
        self.assertEqual(wall.available_after("FRONT")[:2], (1, 2))
        self.assertEqual(wall.available_after("BACK")[:2], (134, 133))

    def test_kong_actions_require_a_real_supplement_tile(self) -> None:
        hands = allocate([
            [0, 0, 0, 0] + [1, 2, 3, 4, 5, 6, 7, 10, 11, 12],
            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30, 31, 32, 33, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        ])
        hand = MahjongHand("no-supplement", 0, wall_for_hands(0, hands))
        hand.wall.front = hand.wall.back + 1

        actions = hand.legal_actions(0)

        self.assertFalse(any(action.kind is ActionKind.CLOSED_KONG for action in actions))


class ResponseTests(unittest.TestCase):
    def test_player_with_three_matching_tiles_may_choose_pong(self) -> None:
        hands = allocate([
            [31] + [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15],
            [31, 31, 31] + [16, 17, 19, 20, 21, 22, 23, 24, 25, 28],
            [18, 26, 27, 30, 32, 33, 8, 9, 0, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        ])
        hand = MahjongHand("priority", 0, wall_for_hands(0, hands))

        discard = next(
            action
            for action in hand.legal_actions(0)
            if action.kind is ActionKind.DISCARD and tile_type(action.tiles[0]) == 31
        )
        hand.step(discard)
        choices = hand.legal_actions(1)
        self.assertTrue(any(a.kind is ActionKind.PONG for a in choices))
        self.assertTrue(any(a.kind is ActionKind.OPEN_KONG for a in choices))
        hand.step(next(a for a in choices if a.kind is ActionKind.PONG))
        pass_current(hand)
        pass_current(hand)
        self.assertEqual(hand.current_player, 1)
        self.assertEqual(hand.players[1].melds[0].meld.kind, MeldKind.PONG)
        self.assertEqual(hand.phase, Phase.ACTION)

    def test_discard_can_have_two_winners(self) -> None:
        orphan = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32]
        hands = allocate([
            [33] + [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15],
            orphan + [0],
            orphan + [8],
            [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15],
        ])
        hand = MahjongHand("multi", 0, wall_for_hands(0, hands))

        discard = next(
            action
            for action in hand.legal_actions(0)
            if action.kind is ActionKind.DISCARD and tile_type(action.tiles[0]) == 33
        )
        hand.step(discard)
        hand.step(next(a for a in hand.legal_actions(1) if a.kind is ActionKind.WIN))
        hand.step(next(a for a in hand.legal_actions(2) if a.kind is ActionKind.WIN))
        pass_current(hand)
        self.assertIsNotNone(hand.outcome)
        self.assertEqual(tuple(item.player for item in hand.outcome.winners), (1, 2))
        self.assertEqual(hand.outcome.method, "DISCARD_WIN")

    def test_passed_win_is_locked_until_own_draw(self) -> None:
        orphan = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32]
        hands = allocate([
            [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15],
            orphan + [0],
            [33] + [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16],
            [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15],
        ])
        wall = list(wall_for_hands(2, hands))
        # NOTE: p2弃第一张白板后，p3下一摸再得到另一张白板。
        next_front = 53

        used_cards = {card for values in hands for card in values}
        white = next(
            card for card in range(33 * 4, 34 * 4) if card not in used_cards
        )
        old_index = wall.index(white)
        wall[next_front], wall[old_index] = wall[old_index], wall[next_front]
        hand = MahjongHand("pass-lock", 2, tuple(wall))

        discard = next(
            action
            for action in hand.legal_actions(2)
            if action.kind is ActionKind.DISCARD and tile_type(action.tiles[0]) == 33
        )
        hand.step(discard)
        pass_current(hand)  # p3
        pass_current(hand)  # p0
        self.assertTrue(any(a.kind is ActionKind.WIN for a in hand.legal_actions(1)))
        pass_current(hand)  # p1 passes a legal win
        self.assertTrue(hand.pass_win_locked[1])
        # p3 has drawn white and discards it before p1 has drawn.

        discard2 = next(
            action
            for action in hand.legal_actions(3)
            if action.kind is ActionKind.DISCARD and tile_type(action.tiles[0]) == 33
        )
        hand.step(discard2)
        pass_current(hand)  # p0
        self.assertFalse(any(a.kind is ActionKind.WIN for a in hand.legal_actions(1)))


class AddedKongResponseTests(unittest.TestCase):

    def test_added_kong_can_be_robbed_by_multiple_players(self) -> None:
        orphan_without_white = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32]
        hands = allocate([
            [33, 33, 33, 33] + [1, 2, 3, 4, 5, 6, 7, 10, 11, 12],
            orphan_without_white + [0],
            orphan_without_white + [8],
            [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15],
        ])
        hand = MahjongHand("rob-added", 0, wall_for_hands(0, hands))
        whites = [card for card in hand.players[0].concealed if tile_type(card) == 33]
        for card in whites[:3]:
            hand.players[0].concealed.remove(card)
        hand.players[0].melds.append(
            MeldRecord(Meld(MeldKind.PONG, tuple(whites[:3])), provider=3)
        )
        action = next(
            item for item in hand.legal_actions(0)
            if item.kind is ActionKind.ADDED_KONG
        )

        hand.step(action)
        hand.step(next(item for item in hand.legal_actions(1) if item.kind is ActionKind.WIN))
        hand.step(next(item for item in hand.legal_actions(2) if item.kind is ActionKind.WIN))
        pass_current(hand)

        self.assertEqual(hand.outcome.method, "ROB_KONG")
        self.assertEqual(tuple(item.player for item in hand.outcome.winners), (1, 2))
        self.assertEqual(hand.outcome.discarder, 0)
        self.assertEqual(hand.players[0].melds[0].meld.kind, MeldKind.PONG)
        self.assertIn(whites[3], hand.players[0].concealed)



class ResponsibilityEngineTests(unittest.TestCase):
    def test_fourth_pong_locks_provider(self) -> None:
        hands = allocate([
            [3] + [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        ])
        hand = MahjongHand("grounded", 0, wall_for_hands(0, hands))
        p1 = hand.players[1]
        for provider, kind in zip((2, 3, 0), (0, 1, 2)):
            tiles = tuple(card for card in p1.concealed if tile_type(card) == kind)
            for card in tiles:
                p1.concealed.remove(card)
            p1.melds.append(MeldRecord(Meld(MeldKind.PONG, tiles), provider))

        discard = next(
            action
            for action in hand.legal_actions(0)
            if action.kind is ActionKind.DISCARD and tile_type(action.tiles[0]) == 3
        )
        hand.step(discard)
        hand.step(next(a for a in hand.legal_actions(1) if a.kind is ActionKind.PONG))
        pass_current(hand)
        pass_current(hand)
        liability = hand.twelve_liabilities[1]
        self.assertEqual((liability.claimant, liability.responsible), (1, 0))

    def test_open_kong_direct_draw_win_records_provider(self) -> None:
        hands = allocate([
            [27] + [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [27, 27, 27, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 8],
        ])
        used = {card for values in hands for card in values}
        winning = next(card for card in range(4) if card not in used)
        hand = MahjongHand("kong-win", 0, wall_for_hands(0, hands, winning))
        # NOTE: 同时存在十二张责任时，直接点杠是更近的付款原因。
        from pdh_v2.responsibility import TwelveGroundedLiability
        hand.twelve_liabilities[1] = TwelveGroundedLiability("kong-win", 1, 2)

        discard = next(
            action
            for action in hand.legal_actions(0)
            if action.kind is ActionKind.DISCARD and tile_type(action.tiles[0]) == 27
        )
        hand.step(discard)
        hand.step(next(a for a in hand.legal_actions(1) if a.kind is ActionKind.OPEN_KONG))
        pass_current(hand)
        pass_current(hand)
        self.assertEqual(hand.last_draw.tile, winning)
        hand.step(next(a for a in hand.legal_actions(1) if a.kind is ActionKind.WIN))
        self.assertEqual(hand.outcome.kong_provider, 0)
        self.assertIsNone(hand.outcome.twelve_grounded)


class ReplayTests(unittest.TestCase):
    def test_action_stream_replays_exact_state(self) -> None:
        hands = allocate([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            [27, 28, 29, 30, 31, 32, 33, 0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        ])
        hand = MahjongHand("replay", 0, wall_for_hands(0, hands))
        hand.step(next(a for a in hand.legal_actions(0) if a.kind is ActionKind.DISCARD))
        pass_current(hand)
        pass_current(hand)
        pass_current(hand)
        hand.step(next(a for a in hand.legal_actions(1) if a.kind is ActionKind.DISCARD))
        transcript = HandTranscript.from_hand(hand)
        loaded = HandTranscript.from_json(transcript.to_json())
        replayed = replay_hand(loaded)
        self.assertEqual(replayed.state_digest(), hand.state_digest())

    def test_transcript_rejects_wrong_rule_hash_and_state_hash(self) -> None:
        import json

        hands = allocate([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            [27, 28, 29, 30, 31, 32, 33, 0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        ])
        hand = MahjongHand("tamper", 0, wall_for_hands(0, hands))
        transcript = HandTranscript.from_hand(hand)
        payload = json.loads(transcript.to_json())
        payload["rule_source_sha256"] = "0" * 64
        with self.assertRaises(RuleError):
            HandTranscript.from_json(json.dumps(payload).encode())
        payload = json.loads(transcript.to_json())
        payload["state_sha256"] = "f" * 64
        changed = HandTranscript.from_json(json.dumps(payload).encode())
        with self.assertRaises(RuleError):
            replay_hand(changed)


class RandomLegalStressTests(unittest.TestCase):
    """固定随机流覆盖响应、碰杠、抢杠和流局，不作为牌型概率测试。"""

    def test_random_legal_hands_preserve_all_tile_entities(self) -> None:
        rng = random.Random(20260930)
        for hand_index in range(80):
            wall = list(range(136))
            rng.shuffle(wall)
            hand = MahjongHand(f"stress-{hand_index}", hand_index % 4, tuple(wall))
            hand.validate_invariants()
            for _ in range(1000):
                if hand.phase is Phase.TERMINAL:
                    break
                player = (
                    hand.current_player
                    if hand.phase is Phase.ACTION
                    else hand.response.queue[0]
                )
                actions = list(hand.legal_actions(player))
                wins = [item for item in actions if item.kind is ActionKind.WIN]
                if wins:
                    chosen = wins[0]
                elif hand.phase is not Phase.ACTION:
                    claims = [
                        item for item in actions
                        if item.kind in (ActionKind.PONG, ActionKind.OPEN_KONG)
                    ]
                    chosen = rng.choice(claims) if claims and rng.random() < 0.18 else next(
                        item for item in actions if item.kind is ActionKind.PASS
                    )
                else:
                    kongs = [
                        item for item in actions
                        if item.kind in (ActionKind.CLOSED_KONG, ActionKind.ADDED_KONG)
                    ]
                    discards = [item for item in actions if item.kind is ActionKind.DISCARD]

                    chosen = (
                        rng.choice(kongs)
                        if kongs and rng.random() < 0.15
                        else rng.choice(discards)
                    )
                hand.step(chosen)
                hand.validate_invariants()
            self.assertIsNotNone(hand.outcome, hand.hand_id)



if __name__ == "__main__":
    unittest.main()
