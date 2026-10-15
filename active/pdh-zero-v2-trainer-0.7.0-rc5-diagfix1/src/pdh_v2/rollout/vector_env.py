"""单进程多桌参考采样器。

该实现优先保证规则语义与批量推理接口正确。它把多个独立牌桌的决策合并为
一个 GPU 批次，但环境推进仍是 Python 参考引擎；正式性能版本需在相同固定
牌墙与动作流上做差分后再引入多进程。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from pdh_v2.policy.heuristic import choose_action

from pdh_v2.game.actions import Action
from pdh_v2.game.session import GameSession
from pdh_v2.observation import DecisionView, decision_view
from pdh_v2.replay import HandTranscript


@dataclass(frozen=True, slots=True)
class EnvDecision:
    """一张牌桌当前需要模型选择的合法动作。"""

    env_index: int
    hand_id: str
    view: DecisionView


@dataclass(frozen=True, slots=True)
class TerminalHand:
    """一手已完成抓马与付款后的训练事实。"""

    env_index: int
    hand_id: str
    receipt: tuple[int, ...]
    draw: bool
    method: str
    winners: tuple[int, ...]
    discarder: int | None
    transcript: HandTranscript
    audit: dict | None = None


class BatchedSessions:
    """维护若干连续会话，并提供同步批量决策接口。"""

    def __init__(self, count: int, seed: int, seeds: tuple[int, ...] | None = None) -> None:
        if count < 1:
            raise ValueError("count must be positive")
        if seeds is not None and len(seeds) != count:
            raise ValueError("seed count mismatch")
        keys = seeds or tuple(seed + index * 1_000_003 for index in range(count))
        self.sessions = [GameSession(key) for key in keys]
        self._views: dict[int, EnvDecision] = {}
        self._supply_chains = {i: [] for i in range(count)}
        self.active = [True] * count

    @property
    def count(self) -> int:
        """返回牌桌数量。"""
        return len(self.sessions)

    def reactivate_all(self) -> None:
        """在一次 PPO 更新后继续已经发好的下一手。"""
        self.active = [True] * self.count

    def requests(self) -> tuple[EnvDecision, ...]:
        """返回所有活跃牌桌当前唯一决策。"""
        from pdh_v2.native import backend_name
        if backend_name() == "cpp":
            from pdh_v2.native.bridge import encode_many
            from pdh_v2.observation import acting_player
            missing = [i for i in range(self.count) if self.active[i] and i not in self._views]
            if missing:
                sessions = [self.sessions[i] for i in missing]
                players = [acting_player(s) for s in sessions]
                groups = [s.current_hand.legal_actions(p) for s, p in zip(sessions, players)]
                views = encode_many(sessions, players, groups)
                for i, view in zip(missing, views):
                    self._views[i] = EnvDecision(i, self.sessions[i].current_hand.hand_id, view)
        result: list[EnvDecision] = []
        for index, session in enumerate(self.sessions):
            if not self.active[index]:
                continue
            hand = session.current_hand
            if index not in self._views:
                self._views[index] = EnvDecision(index, hand.hand_id, decision_view(session))
            result.append(self._views[index])
        return tuple(result)

    def step(
        self,
        choices: dict[int, int],
        drain: bool = False,
    ) -> tuple[TerminalHand, ...]:
        """每个活跃牌桌执行一个动作；终局时完成结算。

        `drain=True` 时，完成当前手的牌桌暂停在下一手起点，确保 PPO 更新不会
        混合两个策略版本的同一手牌。
        """
        terminals: list[TerminalHand] = []
        for index, action_index in choices.items():
            if not self.active[index]:
                raise RuntimeError("inactive environment received an action")
            session = self.sessions[index]
            hand = session.current_hand
            request = self._views.get(index)
            view = request.view if request is not None else decision_view(session)
            if type(action_index) is not int or not 0 <= action_index < len(view.actions):
                raise ValueError("action index out of range")
            try:
                action: Action = view.actions[action_index]
            except IndexError as exc:
                raise ValueError("action index out of range") from exc
            from pdh_v2.replay.supply import before_meld_transition, capture_transition
            before_meld = before_meld_transition(session, action)
            outcome = hand.step(action)
            capture_transition(session, before_meld, self._supply_chains[index])
            self._views.pop(index, None)
            if outcome is None:
                continue
            transcript = HandTranscript.from_hand(hand)
            book_cursor = session.book.event_count
            session_cursor = len(session.events)
            before_dealer = asdict(session.dealer_state)
            initial_revealed = [sorted(p.revealed_retained) for p in hand.players]
            resolved = session.resolve(continue_play=True)
            audit = {
                "session_seed": session.seed,
                "supply_chain": self._supply_chains[index],
                "before_dealer": before_dealer,
                "after_dealer": asdict(session.dealer_state),
                "initial_revealed_retained": initial_revealed,
                "settlement_events": session.book.events_since(book_cursor),
                "session_events": session.events[session_cursor:],
                "outcome": hand.state_digest()["outcome"],
            }
            terminals.append(
                TerminalHand(
                    index,
                    outcome.hand_id,
                    resolved.receipt,
                    outcome.draw,
                    outcome.method,
                    tuple(item.player for item in outcome.winners),
                    outcome.discarder,
                    transcript,
                    audit,
                )
            )
            self._supply_chains[index] = []
            if drain:
                self.active[index] = False
        return tuple(terminals)

    def all_inactive(self) -> bool:
        """返回本次采样是否已排空所有进行中的手牌。"""
        return not any(self.active)

    def heuristic_actions(self, requests: tuple[EnvDecision, ...]) -> dict[int, int]:
        """仅对指定请求运行公开信息启发式；供串并行后端共用。"""
        return {
            request.env_index: (max(range(len(request.view.actions)),
                                    key=request.view.teacher_scores.__getitem__)
                                if request.view.teacher_scores is not None else
                                choose_action(self.sessions[request.env_index],
                                              request.view.actions))
            for request in requests
        }

    def close(self) -> None:
        """串行后端没有外部进程；提供与并行后端相同生命周期。"""

    def reset(self, seeds: tuple[int, ...]) -> None:
        """受控评估新牌墙；不用于把训练中的进行中牌局偷偷重开。"""
        if len(seeds) != self.count or not self.all_inactive():
            raise ValueError("reset requires inactive environments and matching seeds")
        self.sessions = [GameSession(seed) for seed in seeds]
        self.active = [True] * self.count
        self._views = {}
        self._supply_chains = {i: [] for i in range(self.count)}
