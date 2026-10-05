"""Balanced, resumable Gate B long-campaign helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import load_config
from .device import DeviceInfo
from .evaluation import evaluate_model, evaluate_rule_bot
from .io_utils import utc_now_iso, write_json
from .mini_pdh import ACTOR_OBSERVATION_DIM, PRIVILEGED_OBSERVATION_DIM
from .rl.model import RecurrentActorCritic

METHOD_CONFIGS = {
    "plain": "configs/ppo_phase1_plain_critic.yaml",
    "privileged": "configs/ppo_phase1.yaml",
}


@dataclass(frozen=True)
class ParticipantSpec:
    """One method/seed arm in the balanced Gate B campaign."""

    method: str
    seed: int
    config_path: str

    @property
    def participant_id(self) -> str:
        """Return a stable filesystem-safe participant identifier."""
        return f"{self.method}_seed{self.seed}"


@dataclass(frozen=True)
class CampaignSettings:
    """Immutable settings recorded in the campaign manifest."""

    training_seconds: float
    chunk_updates: int
    evaluation_wall_seeds: int
    evaluation_seed_start: int
    opponent: str
    device: str
    snapshot_seconds: float
    telemetry_interval_seconds: int
    min_free_bytes: int


def parse_csv_ints(raw: str) -> list[int]:
    """Parse a comma-separated integer list while rejecting duplicates."""
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one seed is required")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate seeds are not allowed: {values}")
    return values


def parse_csv_methods(raw: str) -> list[str]:
    """Parse and validate campaign method names."""
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one method is required")
    unknown = sorted(set(values) - set(METHOD_CONFIGS))
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate methods are not allowed: {values}")
    return values


def build_participants(methods: Iterable[str], seeds: Iterable[int]) -> list[ParticipantSpec]:
    """Build a seed-paired round-robin participant order."""
    participants = []
    for seed in seeds:
        for method in methods:
            participants.append(ParticipantSpec(method, int(seed), METHOD_CONFIGS[method]))
    return participants


def write_resolved_config(
    project_root: Path,
    job_dir: Path,
    participant: ParticipantSpec,
    device: str,
) -> Path:
    """Create the immutable per-participant training configuration."""
    source = project_root / participant.config_path
    config = load_config(source)
    config.update(
        {
            "config_id": f"gate_b_long_{participant.participant_id}",
            "seed": participant.seed,
            "device": device,
            # NOTE: Campaign-level evaluation uses a common fixed-wall protocol.
            "evaluation_every_updates": 0,
        }
    )
    target = job_dir / "participants" / participant.participant_id / "config.json"
    write_json(target, config)
    return target


def initial_state(
    job_id: str,
    settings: CampaignSettings,
    participants: Iterable[ParticipantSpec],
) -> dict[str, Any]:
    """Construct a fresh resumable campaign state document."""
    now = utc_now_iso()
    return {
        "schema_version": "1.0",
        "job_id": job_id,
        "status": "created",
        "created_at_utc": now,
        "updated_at_utc": now,
        "stop_reason": None,
        "settings": asdict(settings),
        "consumed_training_seconds": 0.0,
        "completed_chunks": 0,
        "completed_rounds": 0,
        "next_participant_index": 0,
        "participants": {
            item.participant_id: {
                "method": item.method,
                "seed": item.seed,
                "completed_updates": 0,
                "total_environment_decisions": 0,
                "chunks": 0,
                "last_chunk_seconds": None,
                "checkpoint": None,
                "last_metrics": None,
            }
            for item in participants
        },
    }


def load_state(path: Path) -> dict[str, Any]:
    """Load a campaign state document."""
    return json.loads(path.read_text(encoding="utf-8"))


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append one durable structured campaign event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": utc_now_iso(), **event}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def latest_jsonl_row(path: Path) -> dict[str, Any] | None:
    """Return the final non-empty JSONL row without loading the whole file."""
    if not path.is_file():
        return None
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(rows[-1]) if rows else None


def available_bytes(path: Path) -> int:
    """Return available filesystem bytes for a campaign path."""
    return shutil.disk_usage(path).free


def gpu_compute_pids() -> set[int]:
    """Return active NVIDIA compute-process PIDs; empty on non-NVIDIA hosts."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    return {
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    }


def evaluate_checkpoint(
    checkpoint: Path,
    device_info: DeviceInfo,
    wall_seeds: int,
    seed_start: int,
    opponent: str,
) -> dict[str, Any]:
    """Evaluate one checkpoint on fixed walls and all four seats."""
    payload = torch.load(checkpoint, map_location=torch.device("cpu"), weights_only=False)
    config = payload["metadata"]["config"]
    rules = load_config(config["ruleset"])
    critic_dim = (
        PRIVILEGED_OBSERVATION_DIM
        if bool(config["privileged_critic"])
        else ACTOR_OBSERVATION_DIM
    )
    model = RecurrentActorCritic(
        ACTOR_OBSERVATION_DIM,
        critic_dim,
        int(config["hidden_size"]),
    ).to(device_info.device)
    model.load_state_dict(payload["model"])
    seeds = [seed_start + index for index in range(wall_seeds)]
    result = evaluate_model(model, rules, device_info.device, seeds, opponent)
    del model, payload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_rule_baselines(
    rules_path: Path,
    wall_seeds: int,
    seed_start: int,
    opponent: str,
) -> dict[str, Any]:
    """Compute common fixed-wall rule-bot baselines once per campaign."""
    rules = load_config(rules_path)
    seeds = [seed_start + index for index in range(wall_seeds)]
    return {
        "random_vs_opponent": evaluate_rule_bot(rules, seeds, "random", opponent),
        "heuristic_vs_random": evaluate_rule_bot(rules, seeds, "heuristic", "random"),
    }


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path) -> Path:
    """Write deterministic relative checksums for finalized campaign evidence."""
    target = root / "checksums.sha256"
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path == target:
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return target
