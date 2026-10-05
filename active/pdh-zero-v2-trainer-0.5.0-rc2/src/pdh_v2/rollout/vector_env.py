"""单进程多桌参考采样器。

该实现优先保证规则语义与批量推理接口正确。它把多个独立牌桌的决策合并为
一个 GPU 批次，但环境推进仍是 Python 参考引擎；正式性能版本需在相同固定
牌墙与动作流上做差分后再引入多进程。
"""
from __future__ import annotations

from dataclasses import dataclass

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


class BatchedSessions:
    """维护若干连续会话，并提供同步批量决策接口。"""

    def __init__(self, count: int, seed: int) -> None:
        if count < 1:
            raise ValueError("count must be positive")
        self.sessions = [GameSession(seed + index * 1_000_003) for index in range(count)]
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
        result: list[EnvDecision] = []
        for index, session in enumerate(self.sessions):
            if not self.active[index]:
                continue
            hand = session.current_hand
            result.append(EnvDecision(index, hand.hand_id, decision_view(session)))
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
            view = decision_view(session)
            try:
                action: Action = view.actions[action_index]
            except IndexError as exc:
                raise ValueError("action index out of range") from exc
            outcome = hand.step(action)
            if outcome is None:
                continue
            transcript = HandTranscript.from_hand(hand)
            resolved = session.resolve(continue_play=True)
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
                )
            )
            if drain:
                self.active[index] = False
        return tuple(terminals)

    def all_inactive(self) -> bool:
        """返回本次采样是否已排空所有进行中的手牌。"""
        return not any(self.active)
