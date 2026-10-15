"""启发式暖启动、事件写入和限时运行器测试。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.session import GameSession
from pdh_v2.policy import CandidateActorCritic, ModelConfig, choose_action
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.events import JsonlWriter, ReplayWriter
from pdh_v2.training.imitation import (
    behavior_clone,
    collect_teacher_episodes,
    split_teacher_episodes,
)
from pdh_v2.training.runner import RunnerConfig, run


class TrainingRunnerTests(unittest.TestCase):
    def test_heuristic_prefers_legal_win_candidate(self) -> None:
        session = GameSession(41)
        player = session.current_hand.current_player
        actions = (
            Action(player, ActionKind.PASS),
            Action(player, ActionKind.WIN),
        )
        self.assertEqual(choose_action(session, actions), 1)

    def test_teacher_clone_and_replay_writers(self) -> None:
        device = torch.device("cpu")
        environments = BatchedSessions(1, 43)
        episodes, hands = collect_teacher_episodes(environments, 1)
        self.assertGreaterEqual(hands, 1)
        model = CandidateActorCritic(ModelConfig(hidden_size=16)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = behavior_clone(
            model,
            optimizer,
            episodes,
            device,
            epochs=1,
            minibatch_decisions=2048,
            seed=47,
        )
        self.assertGreater(metrics.decisions, 0)
        self.assertGreaterEqual(metrics.accuracy, 0.0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = JsonlWriter(root / "events.jsonl")
            events.write({"event": "test", "value": 1})
            self.assertEqual(json.loads((root / "events.jsonl").read_text())["value"], 1)

            second = BatchedSessions(1, 53)
            terminal = ()
            while not terminal:
                request = second.requests()[0]
                choice = choose_action(second.sessions[0], request.view.actions)
                terminal = second.step({0: choice})
            writer = ReplayWriter(root / "hands.jsonl.gz")
            writer.write_many(terminal, update=1)
            with gzip.open(root / "hands.jsonl.gz", "rt", encoding="utf-8") as stream:
                payload = json.loads(stream.readline())
            self.assertEqual(payload["update"], 1)

    def test_teacher_split_keeps_all_seats_of_one_hand_together(self) -> None:
        environments = BatchedSessions(2, 59)
        episodes, _ = collect_teacher_episodes(environments, 4)
        training, validation = split_teacher_episodes(episodes, 0.5, 61)
        training_groups = {episode.group_id for episode in training}
        validation_groups = {episode.group_id for episode in validation}
        self.assertTrue(training_groups)
        self.assertTrue(validation_groups)
        self.assertFalse(training_groups & validation_groups)

    def test_runner_completes_small_cpu_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = run(
                RunnerConfig(
                    output=output,
                    duration_seconds=1,
                    environments=1,
                    hands_per_update=1,
                    hidden_size=16,
                    ppo_epochs=1,
                    minibatch_decisions=512,
                    warmup_hands=0,
                    device="cpu",
                )
            )
            self.assertEqual(result, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue((output / "checkpoints/checkpoint_latest.pt").is_file())
            self.assertTrue((output / "replays/hands.jsonl.gz").is_file())


if __name__ == "__main__":
    unittest.main()
