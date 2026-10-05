"""Bounded, resumable Phase 1 recurrent PPO training orchestration."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pdh_phase1.checkpoint import load_checkpoint, save_checkpoint
from pdh_phase1.config import load_config, require_keys
from pdh_phase1.device import DeviceInfo, resolve_device, seed_everything
from pdh_phase1.evaluation import evaluate_model
from pdh_phase1.io_utils import append_jsonl, utc_now_iso, write_json
from pdh_phase1.mini_pdh import ACTOR_OBSERVATION_DIM, PRIVILEGED_OBSERVATION_DIM

from .model import RecurrentActorCritic
from .ppo import ppo_update
from .self_play import clone_frozen_model, collect_episodes


@dataclass(frozen=True)
class TrainingSummary:
    """Final bounded training status written to ``summary.json``."""

    completed_updates: int
    total_environment_decisions: int
    elapsed_seconds: float
    checkpoint_path: str
    device_backend: str


class Phase1Trainer:
    """Own model, optimizer, frozen opponent snapshot, metrics, and checkpoints."""

    def __init__(self, config_path: str | Path, output_dir: str | Path):
        """Load immutable experiment configuration and initialize runtime state."""
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        require_keys(
            self.config,
            (
                "ruleset",
                "seed",
                "device",
                "privileged_critic",
                "hidden_size",
                "updates",
                "games_per_update",
            ),
            "training config",
        )
        self.rules_path = Path(self.config["ruleset"])
        self.rules_config = load_config(self.rules_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_info: DeviceInfo = resolve_device(str(self.config["device"]))
        seed_everything(int(self.config["seed"]), bool(self.config.get("deterministic_torch", False)))
        critic_dim = (
            PRIVILEGED_OBSERVATION_DIM
            if bool(self.config["privileged_critic"])
            else ACTOR_OBSERVATION_DIM
        )
        self.model = RecurrentActorCritic(
            ACTOR_OBSERVATION_DIM,
            critic_dim,
            int(self.config["hidden_size"]),
        ).to(self.device_info.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
        )
        self.frozen_model = clone_frozen_model(self.model, self.device_info.device)
        self.rng = random.Random(int(self.config["seed"]) ^ 0x5A17)
        self.start_update = 0
        self.total_environment_decisions = 0

    def resume(self, checkpoint_path: str | Path) -> None:
        """Restore model, optimizer, update counters, and frozen-opponent parameters."""
        metadata = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.device_info.device,
            restore_rng=True,
        )
        if metadata.get("rules_hash") != self._rules_hash():
            raise RuntimeError("Checkpoint rules hash does not match the current ruleset")
        self.start_update = int(metadata["next_update"])
        self.total_environment_decisions = int(metadata["total_environment_decisions"])
        frozen_state = metadata.get("frozen_model_state")
        if frozen_state is not None:
            self.frozen_model.load_state_dict(frozen_state)
        self.rng.setstate(metadata["trainer_rng_state"])

    def train(self, updates_override: int | None = None) -> TrainingSummary:
        """Run a bounded number of updates and persist fresh evidence after each update."""
        requested_updates = int(updates_override or self.config["updates"])
        end_update = self.start_update + requested_updates
        start_time = time.perf_counter()
        last_checkpoint = self.output_dir / "checkpoint_latest.pt"
        for update in range(self.start_update, end_update):
            collection_start = time.perf_counter()
            episodes, collection = collect_episodes(
                self.model,
                self.frozen_model,
                self.rules_config,
                self.config,
                self.device_info.device,
                update,
            )
            metrics = ppo_update(
                self.model,
                self.optimizer,
                episodes,
                self.config,
                self.device_info.device,
                self.rng,
            )
            self.total_environment_decisions += collection.environment_decisions
            if (update + 1) % int(self.config["opponent_sync_updates"]) == 0:
                self.frozen_model.load_state_dict(self.model.state_dict())
            row: dict[str, Any] = {
                "timestamp": utc_now_iso(),
                "update": update + 1,
                "episodes": collection.episodes,
                "attempts": collection.attempts,
                "environment_decisions": collection.environment_decisions,
                "total_environment_decisions": self.total_environment_decisions,
                "mean_score": collection.mean_score,
                "win_rate": collection.win_rate,
                "draw_rate": collection.draw_rate,
                "collection_seconds": collection.elapsed_seconds,
                "update_seconds": time.perf_counter() - collection_start,
                "policy_loss": metrics.policy_loss,
                "value_loss": metrics.value_loss,
                "entropy": metrics.entropy,
                "approximate_kl": metrics.approximate_kl,
                "gradient_norm": metrics.gradient_norm,
            }
            evaluation_every = int(self.config.get("evaluation_every_updates", 0))
            if evaluation_every and (update + 1) % evaluation_every == 0:
                evaluation_seeds = [int(self.config["seed"]) + 900_000 + index for index in range(16)]
                row["evaluation"] = evaluate_model(
                    self.model,
                    self.rules_config,
                    self.device_info.device,
                    evaluation_seeds,
                    opponent_kind="heuristic",
                )
            append_jsonl(self.output_dir / "metrics.jsonl", row)
            if (update + 1) % int(self.config["checkpoint_every_updates"]) == 0:
                self._save(last_checkpoint, update + 1)
        self._save(last_checkpoint, end_update)
        summary = TrainingSummary(
            completed_updates=end_update,
            total_environment_decisions=self.total_environment_decisions,
            elapsed_seconds=time.perf_counter() - start_time,
            checkpoint_path=str(last_checkpoint),
            device_backend=self.device_info.backend,
        )
        write_json(self.output_dir / "summary.json", summary.__dict__)
        return summary

    def _save(self, path: Path, next_update: int) -> None:
        metadata = {
            "schema_version": "1.0",
            "next_update": int(next_update),
            "total_environment_decisions": self.total_environment_decisions,
            "rules_hash": self._rules_hash(),
            "config": copy.deepcopy(self.config),
            "frozen_model_state": {
                name: tensor.detach().cpu() for name, tensor in self.frozen_model.state_dict().items()
            },
            "trainer_rng_state": self.rng.getstate(),
        }
        save_checkpoint(path, self.model, self.optimizer, metadata)

    def _rules_hash(self) -> str:
        encoded = json.dumps(self.rules_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
