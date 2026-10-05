"""批量采样、PPO 与检查点 CPU Smoke。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.checkpoint import load_checkpoint, save_checkpoint
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.ppo import PPOConfig, update_ppo


class TrainingCandidateTests(unittest.TestCase):
    def test_collect_update_and_checkpoint(self) -> None:
        torch.manual_seed(7)
        device = torch.device("cpu")
        model = CandidateActorCritic(ModelConfig(hidden_size=32)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        environments = BatchedSessions(2, 101)
        collector = RolloutCollector(
            environments,
            model,
            device,
            CollectionConfig(minimum_hands=2),
        )
        result = collector.collect()
        self.assertGreaterEqual(result.hands, 2)
        self.assertGreater(result.decisions, 0)
        self.assertEqual(sum(result.raw_score_sum), 0)
        metrics = update_ppo(
            model,
            optimizer,
            result.episodes,
            device,
            PPOConfig(epochs=1, minibatch_decisions=512),
            seed=11,
        )
        self.assertGreater(metrics.decisions, 0)
        self.assertGreater(metrics.minibatches, 0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 3, {"hands": result.hands})
            restored = CandidateActorCritic(ModelConfig(hidden_size=32)).to(device)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            update, metadata = load_checkpoint(
                path, restored, restored_optimizer, device
            )
            self.assertEqual(update, 3)
            self.assertEqual(metadata["hands"], result.hands)
            self.assertEqual(asdict(restored.config), asdict(model.config))
            for left, right in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
