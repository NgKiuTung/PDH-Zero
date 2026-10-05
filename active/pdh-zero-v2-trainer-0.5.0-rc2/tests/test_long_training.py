"""长训练晋级、恢复、评估和原子牌谱测试。"""
from __future__ import annotations

from dataclasses import asdict
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import torch

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.rollout import BatchedSessions
from pdh_v2.rollout.snapshot import restore_batched, snapshot_batched
from pdh_v2.training.checkpoint import save_checkpoint
from pdh_v2.training.evaluation import EvaluationConfig, evaluate_vs_heuristic
from pdh_v2.training.job_control import launch_new
from pdh_v2.training.long_artifacts import _rebuild_live_metrics
from pdh_v2.training.long_checkpoint import load_long_checkpoint, save_long_checkpoint
from pdh_v2.training.long_config import LongRunConfig
from pdh_v2.training.long_runner import run_long
from pdh_v2.training.promotion import validate_smoke
from pdh_v2.training.replay_updates import reconcile_replays, replay_path, write_replay_update


def _fake_smoke(root: Path, hidden_size: int = 16) -> Path:
    smoke = root / "smoke"
    (smoke / "checkpoints").mkdir(parents=True)
    (smoke / "replays").mkdir(parents=True)
    model = CandidateActorCritic(ModelConfig(hidden_size=hidden_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    save_checkpoint(
        smoke / "checkpoints" / "checkpoint_latest.pt",
        model,
        optimizer,
        3,
        {"elapsed_seconds": 600.0, "total_hands": 12, "total_decisions": 144},
    )
    manifest = {
        "schema_version": 1,
        "scope": "training_smoke_candidate",
        "status": "completed",
        "rule_source_sha256": SOURCE_SHA256,
        "config": {
            "duration_seconds": 600,
            "hidden_size": hidden_size,
            "seed": 17,
        },
        "updates": 3,
        "total_hands": 12,
        "total_decisions": 144,
        "elapsed_seconds": 600.0,
    }
    (smoke / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metric = {
        "update": 3,
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 0.3,
        "raw_score_sum": [1, -1, 0, 0],
    }
    (smoke / "metrics.jsonl").write_text(json.dumps(metric) + "\n", encoding="utf-8")
    with gzip.open(smoke / "replays" / "hands.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema_version": 1, "hand": 1}) + "\n")
    return smoke


class LongTrainingTests(unittest.TestCase):
    def test_smoke_gate_accepts_complete_zero_sum_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = validate_smoke(_fake_smoke(Path(directory)))
            self.assertEqual(result.update, 3)
            self.assertEqual(result.total_hands, 12)
            self.assertEqual(result.hidden_size, 16)

    def test_batched_snapshot_round_trip(self) -> None:
        environments = BatchedSessions(2, 23)
        environments.active = [False, False]
        restored = restore_batched(snapshot_batched(environments))
        self.assertTrue(restored.all_inactive())
        for left, right in zip(environments.sessions, restored.sessions):
            self.assertEqual(left.dealer_state, right.dealer_state)
            self.assertEqual(left.current_hand.state_digest(), right.current_hand.state_digest())
            self.assertEqual(left.wall_rng.getstate(), right.wall_rng.getstate())

    def test_long_checkpoint_restores_environment_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = torch.device("cpu")
            model = CandidateActorCritic(ModelConfig(hidden_size=16)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            environments = BatchedSessions(2, 29)
            environments.active = [False, False]
            identity = {"seed": 29}
            path = root / "checkpoint.pt"
            save_long_checkpoint(
                path,
                model,
                optimizer,
                environments,
                7,
                {"long_decisions": 99},
                identity,
            )
            expected = [parameter.detach().clone() for parameter in model.parameters()]
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)
            restored_envs, update, counters = load_long_checkpoint(
                path,
                model,
                optimizer,
                device,
                identity,
            )
            self.assertEqual(update, 7)
            self.assertEqual(counters["long_decisions"], 99)
            self.assertTrue(restored_envs.all_inactive())
            for actual, wanted in zip(model.parameters(), expected):
                self.assertTrue(torch.equal(actual, wanted))

    def test_reserved_lock_is_treated_as_active_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            job_dir = home / "evidence" / "v2-long-runs" / "starting"
            lock = home / "state" / "v2-long-job.lock.json"
            lock.parent.mkdir(parents=True)
            lock.write_text(
                json.dumps(
                    {
                        "status": "reserved",
                        "pid": 456,
                        "job_dir": str(job_dir),
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "pdh_v2.training.job_control.pid_alive",
                return_value=True,
            ):
                result = launch_new(
                    Path("missing-script.py"),
                    home,
                    home / "missing-smoke",
                )
            self.assertEqual(result["status"], "already_starting")
            self.assertEqual(result["pid"], 456)

    def test_fixed_evaluation_returns_finite_metrics(self) -> None:
        device = torch.device("cpu")
        model = CandidateActorCritic(ModelConfig(hidden_size=16)).to(device)
        result = evaluate_vs_heuristic(
            model,
            device,
            EvaluationConfig(hands=4, environments=4, seed=31),
        )
        self.assertGreaterEqual(result.hands, 4)
        self.assertGreaterEqual(result.draw_rate, 0.0)
        self.assertLessEqual(result.draw_rate, 1.0)

    def test_live_metrics_are_rebuilt_only_from_committed_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            updates = root / "metrics" / "updates"
            updates.mkdir(parents=True)
            for update in (1, 2):
                payload = {"update": update, "value": float(update)}
                (updates / f"update-{update:09d}.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            (root / "metrics.jsonl").write_text(
                '{"update":1}\n{"update":2}\n{"update":2}\n',
                encoding="utf-8",
            )
            count = _rebuild_live_metrics(root, committed_update=1)
            records = [
                json.loads(line)
                for line in (root / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(count, 1)
            self.assertEqual(records, [{"update": 1, "value": 1.0}])

    def test_duplicate_start_returns_existing_job_before_smoke_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            job_dir = home / "evidence" / "v2-long-runs" / "existing"
            lock = home / "state" / "v2-long-job.lock.json"
            lock.parent.mkdir(parents=True)
            lock.write_text(
                json.dumps({"pid": 123, "job_dir": str(job_dir)}),
                encoding="utf-8",
            )
            with mock.patch(
                "pdh_v2.training.job_control.job_process_alive",
                return_value=True,
            ):
                result = launch_new(
                    Path("missing-script.py"),
                    home,
                    home / "missing-smoke",
                )
            self.assertEqual(result["status"], "already_running")
            self.assertEqual(result["job_dir"], str(job_dir))
            self.assertEqual(result["pid"], 123)

    def test_replay_updates_are_atomic_and_orphans_are_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "replays"
            environments = BatchedSessions(1, 37)
            terminals = ()
            while not terminals:
                request = environments.requests()[0]
                terminals = environments.step({0: 0})
                if not terminals and not request.view.actions:
                    self.fail("active environment has no action")
            record = write_replay_update(root, terminals, 4)
            self.assertTrue((root.parent / str(record["path"])).is_file())
            moved = reconcile_replays(root, committed_update=3)
            self.assertEqual(len(moved), 1)
            self.assertFalse(replay_path(root, 4).exists())

    def test_tiny_long_run_warm_starts_from_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = _fake_smoke(root)
            output = root / "long"
            config = LongRunConfig(
                output=output,
                smoke_dir=smoke,
                max_wall_seconds=30,
                target_decisions=1,
                environments=1,
                hands_per_update=1,
                hidden_size=16,
                ppo_epochs=1,
                minibatch_decisions=512,
                device="cpu",
                evaluation_hands=4,
                evaluation_environments=4,
                evaluation_interval_decisions=1,
                snapshot_interval_decisions=1,
                min_free_bytes=1,
            )
            self.assertEqual(run_long(config), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertGreater(state["long_decisions"], 0)
            self.assertTrue((output / "checkpoints/checkpoint_final.pt").is_file())
            self.assertTrue(tuple((output / "replays").glob("*/*.jsonl.gz")))

    def test_resume_path_uses_complete_update_boundary_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = _fake_smoke(root)
            output = root / "resume-long"
            output.mkdir()
            config = LongRunConfig(
                output=output,
                smoke_dir=smoke,
                max_wall_seconds=30,
                target_decisions=1,
                environments=1,
                hands_per_update=1,
                hidden_size=16,
                ppo_epochs=1,
                minibatch_decisions=512,
                device="cpu",
                evaluation_hands=4,
                evaluation_environments=4,
                evaluation_interval_decisions=1,
                snapshot_interval_decisions=1,
                min_free_bytes=1,
            )
            (output / "config.json").write_text(
                json.dumps(config.to_dict()), encoding="utf-8"
            )
            manifest = {
                "schema_version": 1,
                "scope": "full_rules_long_baseline_candidate",
                "status": "stopped",
                "rule_source_sha256": SOURCE_SHA256,
                "config": config.to_dict(),
                "source_smoke": {"path": str(smoke)},
            }
            (output / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            device = torch.device("cpu")
            model = CandidateActorCritic(ModelConfig(hidden_size=16)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            environments = BatchedSessions(1, config.seed ^ 0x51A71E)
            environments.active = [False]
            counters = {
                "source_smoke_metadata": {},
                "long_updates": 0,
                "long_hands": 0,
                "long_decisions": 0,
                "consumed_training_seconds": 0.0,
                "next_evaluation_decisions": 0,
                "next_snapshot_decisions": 1,
                "best_evaluation_mean_score": None,
                "attempts": 1,
            }
            save_long_checkpoint(
                output / "checkpoints/checkpoint_latest.pt",
                model,
                optimizer,
                environments,
                3,
                counters,
                config.checkpoint_identity(),
            )
            shutil.rmtree(smoke)
            self.assertEqual(run_long(config, resume=True), 0)
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(state["attempts"], 2)
            self.assertGreater(state["long_decisions"], 0)


if __name__ == "__main__":
    unittest.main()
