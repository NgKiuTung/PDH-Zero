#!/usr/bin/env python3
"""Run a balanced, resumable, wall-clock-bounded Gate B campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from script_common import PROJECT_ROOT
from pdh_phase1 import __version__
from pdh_phase1.campaign import (
    CampaignSettings,
    append_event,
    available_bytes,
    build_participants,
    evaluate_checkpoint,
    evaluate_rule_baselines,
    gpu_compute_pids,
    initial_state,
    latest_jsonl_row,
    load_state,
    parse_csv_ints,
    parse_csv_methods,
    write_checksums,
    write_resolved_config,
)
from pdh_phase1.device import resolve_device
from pdh_phase1.io_utils import utc_now_iso, write_json
from pdh_phase1.rl.trainer import Phase1Trainer

STOP_REQUESTED = False


def request_stop(signum: int, _frame: object) -> None:
    """Request a checkpoint-safe stop after the current training chunk."""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Stop requested by signal {signum}; finishing the current chunk.", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse long-campaign settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--training-hours", type=float, default=7.5)
    parser.add_argument("--chunk-updates", type=int, default=20)
    parser.add_argument("--methods", default="plain,privileged")
    parser.add_argument("--seeds", default="20260929,3407,9102")
    parser.add_argument("--evaluation-wall-seeds", type=int, default=5000)
    parser.add_argument("--evaluation-seed-start", type=int, default=950000)
    parser.add_argument("--opponent", choices=("random", "heuristic"), default="heuristic")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "rocm"), default="cuda")
    parser.add_argument("--snapshot-minutes", type=float, default=60.0)
    parser.add_argument("--telemetry-interval-seconds", type=int, default=30)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.training_hours <= 0 or args.chunk_updates <= 0:
        parser.error("training-hours and chunk-updates must be positive")
    if args.evaluation_wall_seeds <= 0 or args.telemetry_interval_seconds <= 0:
        parser.error("evaluation and telemetry settings must be positive")
    return args


def start_telemetry(job_dir: Path, interval: int, attempt: int) -> tuple[subprocess.Popen[bytes] | None, Any, Any]:
    """Start one telemetry stream for this campaign attempt."""
    telemetry_dir = job_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    stdout = (telemetry_dir / f"attempt_{attempt:02d}.csv").open("ab")
    stderr = (telemetry_dir / f"attempt_{attempt:02d}.stderr").open("ab")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu,"
        "utilization.memory,memory.used,memory.total,clocks.sm,clocks.mem",
        "--format=csv",
        "-l",
        str(interval),
    ]
    try:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
    except OSError:
        stdout.close()
        stderr.close()
        return None, None, None
    return process, stdout, stderr


def stop_telemetry(process: subprocess.Popen[bytes] | None, stdout: Any, stderr: Any) -> None:
    """Stop only the telemetry process started by this campaign."""
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if stdout is not None:
        stdout.close()
    if stderr is not None:
        stderr.close()


def snapshot_checkpoint(participant_dir: Path, completed_updates: int) -> str:
    """Copy the latest checkpoint to a numbered immutable snapshot."""
    source = participant_dir / "training" / "checkpoint_latest.pt"
    target = participant_dir / "snapshots" / f"checkpoint_update_{completed_updates:08d}.pt"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return str(target)


def run_chunk(
    job_dir: Path,
    participant: Any,
    config_path: Path,
    chunk_updates: int,
) -> dict[str, Any]:
    """Run one resumable participant chunk and return measured progress."""
    participant_dir = job_dir / "participants" / participant.participant_id
    training_dir = participant_dir / "training"
    checkpoint = training_dir / "checkpoint_latest.pt"
    trainer = Phase1Trainer(config_path, training_dir)
    if checkpoint.is_file():
        trainer.resume(checkpoint)
    started = time.perf_counter()
    summary = trainer.train(chunk_updates)
    elapsed = time.perf_counter() - started
    latest = latest_jsonl_row(training_dir / "metrics.jsonl")
    del trainer
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "elapsed_seconds": elapsed,
        "completed_updates": summary.completed_updates,
        "total_environment_decisions": summary.total_environment_decisions,
        "checkpoint": summary.checkpoint_path,
        "last_metrics": latest,
    }


def main() -> int:
    """Run training chunks round-robin, then fixed-wall evaluation."""
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    job_dir = Path(args.job_dir).resolve()
    state_path = job_dir / "state.json"
    events_path = job_dir / "events.jsonl"
    manifest_path = job_dir / "manifest.json"
    participants = build_participants(parse_csv_methods(args.methods), parse_csv_ints(args.seeds))
    settings = CampaignSettings(
        training_seconds=args.training_hours * 3600.0,
        chunk_updates=args.chunk_updates,
        evaluation_wall_seeds=args.evaluation_wall_seeds,
        evaluation_seed_start=args.evaluation_seed_start,
        opponent=args.opponent,
        device=args.device,
        snapshot_seconds=args.snapshot_minutes * 60.0,
        telemetry_interval_seconds=args.telemetry_interval_seconds,
        min_free_bytes=int(args.min_free_gib * 1024**3),
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    if state_path.exists() and not args.resume:
        raise RuntimeError(f"Existing campaign requires --resume: {job_dir}")
    if args.resume:
        state = load_state(state_path)
        if state["settings"] != settings.__dict__:
            raise RuntimeError("Resume settings do not match the existing campaign")
        (job_dir / "checksums.sha256").unlink(missing_ok=True)
    else:
        state = initial_state(job_dir.name, settings, participants)
        write_json(state_path, state)
    attempt = int(state.get("attempts", 0)) + 1
    state["attempts"] = attempt
    state["status"] = "running"
    state["updated_at_utc"] = utc_now_iso()
    write_json(state_path, state)
    manifest = {
        "schema_version": "1.0",
        "job_id": job_dir.name,
        "package_version": __version__,
        "project_root": str(PROJECT_ROOT),
        "job_dir": str(job_dir),
        "pid": os.getpid(),
        "attempt": attempt,
        "status": "running",
        "started_at_utc": utc_now_iso(),
        "settings": settings.__dict__,
        "participants": [item.__dict__ for item in participants],
    }
    write_json(manifest_path, manifest)
    append_event(events_path, {"event": "campaign_started", "attempt": attempt})
    info = resolve_device(args.device)
    manifest.update({"device_backend": info.backend, "device_name": info.name})
    write_json(manifest_path, manifest)
    telemetry, telemetry_stdout, telemetry_stderr = start_telemetry(
        job_dir, args.telemetry_interval_seconds, attempt
    )
    config_paths = {
        item.participant_id: write_resolved_config(PROJECT_ROOT, job_dir, item, args.device)
        for item in participants
    }
    next_snapshot_at = float(state.get("consumed_training_seconds", 0.0)) + settings.snapshot_seconds
    try:
        while True:
            if STOP_REQUESTED:
                break
            if (
                float(state["consumed_training_seconds"]) >= settings.training_seconds
                and int(state.get("next_participant_index", 0)) == 0
            ):
                break
            participant_index = int(state.get("next_participant_index", 0))
            participant = participants[participant_index]
            if available_bytes(job_dir) < settings.min_free_bytes:
                raise RuntimeError("Free disk space fell below the configured safety floor")
            other_pids = gpu_compute_pids() - {os.getpid()}
            if other_pids:
                raise RuntimeError(f"Shared GPU detected; foreign compute PIDs: {sorted(other_pids)}")
            result = run_chunk(
                job_dir,
                participant,
                config_paths[participant.participant_id],
                settings.chunk_updates,
            )
            participant_state = state["participants"][participant.participant_id]
            participant_state.update(
                {
                    "completed_updates": result["completed_updates"],
                    "total_environment_decisions": result["total_environment_decisions"],
                    "chunks": int(participant_state["chunks"]) + 1,
                    "last_chunk_seconds": result["elapsed_seconds"],
                    "checkpoint": result["checkpoint"],
                    "last_metrics": result["last_metrics"],
                }
            )
            next_index = (participant_index + 1) % len(participants)
            state["next_participant_index"] = next_index
            if next_index == 0:
                state["completed_rounds"] = int(state["completed_rounds"]) + 1
            state["consumed_training_seconds"] = float(
                state["consumed_training_seconds"]
            ) + float(result["elapsed_seconds"])
            state["completed_chunks"] = int(state["completed_chunks"]) + 1
            state["updated_at_utc"] = utc_now_iso()
            write_json(state_path, state)
            append_event(
                events_path,
                {
                    "event": "chunk_completed",
                    "participant_id": participant.participant_id,
                    **result,
                    "consumed_training_seconds": state["consumed_training_seconds"],
                    "next_participant_index": next_index,
                },
            )
            if float(state["consumed_training_seconds"]) >= next_snapshot_at:
                snapshot = snapshot_checkpoint(
                    job_dir / "participants" / participant.participant_id,
                    int(result["completed_updates"]),
                )
                append_event(events_path, {"event": "checkpoint_snapshot", "path": snapshot})
                next_snapshot_at += settings.snapshot_seconds
        if STOP_REQUESTED:
            state["status"] = "stopped"
            state["stop_reason"] = "signal_requested"
        else:
            state["status"] = "evaluating"
            write_json(state_path, state)
            evaluations_dir = job_dir / "evaluations"
            evaluations_dir.mkdir(parents=True, exist_ok=True)
            baseline_path = evaluations_dir / "rule_baselines.json"
            if not baseline_path.exists():
                first_config = json.loads(
                    config_paths[participants[0].participant_id].read_text(encoding="utf-8")
                )
                write_json(
                    baseline_path,
                    evaluate_rule_baselines(
                        PROJECT_ROOT / first_config["ruleset"],
                        settings.evaluation_wall_seeds,
                        settings.evaluation_seed_start,
                        settings.opponent,
                    ),
                )
            for participant in participants:
                output = evaluations_dir / f"{participant.participant_id}.json"
                if output.exists():
                    continue
                checkpoint = Path(state["participants"][participant.participant_id]["checkpoint"])
                result = evaluate_checkpoint(
                    checkpoint,
                    info,
                    settings.evaluation_wall_seeds,
                    settings.evaluation_seed_start,
                    settings.opponent,
                )
                write_json(output, result)
                append_event(
                    events_path,
                    {
                        "event": "evaluation_completed",
                        "participant_id": participant.participant_id,
                        "mean_score": result["mean_score"],
                        "win_rate": result["win_rate"],
                    },
                )
            state["status"] = "completed"
            state["stop_reason"] = "target_training_time_reached"
        state["updated_at_utc"] = utc_now_iso()
        write_json(state_path, state)
        manifest.update({"status": state["status"], "ended_at_utc": utc_now_iso()})
        write_json(manifest_path, manifest)
        append_event(events_path, {"event": "campaign_finished", "status": state["status"]})
        stop_telemetry(telemetry, telemetry_stdout, telemetry_stderr)
        telemetry = telemetry_stdout = telemetry_stderr = None
        write_checksums(job_dir)
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["stop_reason"] = f"{type(exc).__name__}: {exc}"
        state["updated_at_utc"] = utc_now_iso()
        write_json(state_path, state)
        manifest.update({"status": "failed", "ended_at_utc": utc_now_iso()})
        write_json(manifest_path, manifest)
        write_json(
            job_dir / "failure.json",
            {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        append_event(events_path, {"event": "campaign_failed", "error": str(exc)})
        stop_telemetry(telemetry, telemetry_stdout, telemetry_stderr)
        telemetry = telemetry_stdout = telemetry_stderr = None
        write_checksums(job_dir)
        raise
    finally:
        stop_telemetry(telemetry, telemetry_stdout, telemetry_stderr)


if __name__ == "__main__":
    raise SystemExit(main())
