"""One-update CPU integration test for self-play, PPO, and checkpoint output."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pdh_phase1.rl.trainer import Phase1Trainer


class TrainingSmokeTests(unittest.TestCase):
    """Prove the bounded training path creates reusable evidence."""

    def test_one_update_completes_and_resumes(self) -> None:
        base = json.loads(Path("configs/ppo_smoke.yaml").read_text(encoding="utf-8"))
        base.update(
            {
                "device": "cpu",
                "updates": 1,
                "games_per_update": 4,
                "ppo_epochs": 1,
                "sequence_minibatch_size": 4,
                "evaluation_every_updates": 0,
            }
        )
        base["ruleset"] = str(Path("configs/mini_pdh_v0.yaml").resolve())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(base), encoding="utf-8")
            first = Phase1Trainer(config_path, root / "first")
            first_summary = first.train()
            checkpoint = Path(first_summary.checkpoint_path)
            self.assertTrue(checkpoint.is_file())
            second = Phase1Trainer(config_path, root / "second")
            second.resume(checkpoint)
            second_summary = second.train(updates_override=1)
            self.assertEqual(second_summary.completed_updates, 2)
            self.assertTrue((root / "second" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
