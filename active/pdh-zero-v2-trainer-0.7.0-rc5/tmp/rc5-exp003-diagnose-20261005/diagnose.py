from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import torch

from pdh_v2.table.checkpoints import load_candidate, tensor_digest
from pdh_v2.table.data import load_baseline


BASE = Path("/root/rivermind-data/pdh-zero")
RC5 = BASE / "active/pdh-zero-v2-trainer-0.7.0-rc5"
EXP3 = Path(os.environ["EXP3"])
STUDY = EXP3 / "study"

BASELINE_PATH = (
    BASE
    / "evidence/v2-refinement-rc4.1"
    / "REFINE_20261003T131918636050Z"
    / "checkpoints/candidates/d000000266137/model.pt"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def effective_candidate(arm: Path) -> Path:
    report = read_json(arm / "report.json")
    return arm / report["effective_checkpoint"]


def legacy_state(candidate: Path, legacy_keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    model, _ = load_candidate(candidate, torch.device("cpu"))
    state = model.state_dict()
    return {key: state[key] for key in legacy_keys}


def compare_states(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> dict:
    differences = []

    for key in left:
        a = left[key]
        b = right[key]

        if torch.equal(a, b):
            continue

        delta = (a.to(torch.float64) - b.to(torch.float64)).abs()
        differences.append(
            {
                "tensor": key,
                "shape": list(a.shape),
                "different_elements": int((a != b).sum().item()),
                "max_abs_difference": float(delta.max().item()),
                "mean_abs_difference": float(delta.mean().item()),
            }
        )

    return {
        "exact": not differences,
        "different_tensor_count": len(differences),
        "differences": differences,
    }


def evaluation_identity(row: dict) -> tuple[int, int]:
    return row["wall_seed"], row["candidate_seat"]


def compare_evaluation(left_path: Path, right_path: Path) -> dict:
    left_rows = read_jsonl(left_path)
    right_rows = read_jsonl(right_path)

    left = {
        evaluation_identity(row): (
            row["state_sha256"],
            row["score"],
            row["action_events"],
        )
        for row in left_rows
    }

    right = {
        evaluation_identity(row): (
            row["state_sha256"],
            row["score"],
            row["action_events"],
        )
        for row in right_rows
    }

    keys = sorted(set(left) | set(right))
    mismatches = []

    for key in keys:
        if left.get(key) != right.get(key):
            mismatches.append(
                {
                    "wall_seed": key[0],
                    "candidate_seat": key[1],
                    "baseline": left.get(key),
                    "diagnostic": right.get(key),
                }
            )

    return {
        "exact": left == right,
        "baseline_rows": len(left_rows),
        "diagnostic_rows": len(right_rows),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:8],
    }


def replay_projection(path: Path) -> list[dict]:
    result = []

    for row in read_gzip_jsonl(path):
        transcript = row["transcript"]
        result.append(
            {
                "update": row["update"],
                "receipt": row["receipt"],
                "draw": row["draw"],
                "method": row["method"],
                "winners": row["winners"],
                "discarder": row["discarder"],
                "state_sha256": transcript["state_sha256"],
                "actions": transcript["actions"],
            }
        )

    return result


manifest = read_json(STUDY / "manifest.json")
opponents = manifest["config"]["opponents"]

baseline_model = load_baseline(BASELINE_PATH, torch.device("cpu"))
legacy_keys = tuple(baseline_model.state_dict().keys())

analysis = {
    "experiment": str(EXP3),
    "failure": read_json(STUDY / "failure.json"),
    "seeds": [],
}

for seed_dir in sorted(STUDY.glob("seed-*")):
    baseline_arm = seed_dir / "baseline"
    diagnostic_arm = seed_dir / "diagnostic"

    if not (
        (baseline_arm / "report.json").is_file()
        and (diagnostic_arm / "report.json").is_file()
    ):
        continue

    seed_result = {
        "seed": seed_dir.name.removeprefix("seed-"),
        "checkpoints": {},
        "training_replay": {},
        "evaluation": {},
    }

    # Compare Actor tensors at initialization and every matching proposal.
    candidate_names = ["candidates/initial.pt"]

    baseline_candidates = {
        path.relative_to(baseline_arm).as_posix()
        for path in (baseline_arm / "candidates").glob("*.pt")
    }
    diagnostic_candidates = {
        path.relative_to(diagnostic_arm).as_posix()
        for path in (diagnostic_arm / "candidates").glob("*.pt")
    }

    candidate_names.extend(
        sorted(
            (baseline_candidates & diagnostic_candidates)
            - {"candidates/initial.pt"}
        )
    )

    for relative in candidate_names:
        left_path = baseline_arm / relative
        right_path = diagnostic_arm / relative

        left = legacy_state(left_path, legacy_keys)
        right = legacy_state(right_path, legacy_keys)

        seed_result["checkpoints"][relative] = {
            "baseline_tensor_digest": tensor_digest(left),
            "diagnostic_tensor_digest": tensor_digest(right),
            **compare_states(left, right),
        }

    # Compare actually selected effective checkpoint.
    left_effective = effective_candidate(baseline_arm)
    right_effective = effective_candidate(diagnostic_arm)

    left_state = legacy_state(left_effective, legacy_keys)
    right_state = legacy_state(right_effective, legacy_keys)

    seed_result["effective_checkpoint"] = {
        "baseline": str(left_effective.relative_to(seed_dir)),
        "diagnostic": str(right_effective.relative_to(seed_dir)),
        "baseline_tensor_digest": tensor_digest(left_state),
        "diagnostic_tensor_digest": tensor_digest(right_state),
        **compare_states(left_state, right_state),
    }

    # Compare complete natural-hand TRAIN trajectories.
    left_replay = replay_projection(baseline_arm / "hands.jsonl.gz")
    right_replay = replay_projection(diagnostic_arm / "hands.jsonl.gz")

    seed_result["training_replay"] = {
        "exact": left_replay == right_replay,
        "baseline_hands": len(left_replay),
        "diagnostic_hands": len(right_replay),
    }

    # Compare development trajectories for all three opponent families.
    for opponent in opponents:
        seed_result["evaluation"][opponent] = compare_evaluation(
            baseline_arm / "evaluation" / opponent / "results.jsonl",
            diagnostic_arm / "evaluation" / opponent / "results.jsonl",
        )

    analysis["seeds"].append(seed_result)

out = EXP3 / "diagnostic-control-analysis.json"
out.write_text(json.dumps(analysis, indent=2) + "\n")

print(json.dumps(analysis, indent=2))
print()
print(f"RESULT={out}")
