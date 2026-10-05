"""Device detection and reproducibility helpers for CPU, CUDA, and ROCm."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved Torch device and backend metadata."""

    device: torch.device
    backend: str
    name: str
    total_memory_bytes: int | None


def resolve_device(requested: str = "auto") -> DeviceInfo:
    """Resolve a user request without silently falling back from an explicit GPU request."""
    normalized = requested.lower()
    if normalized not in {"auto", "cpu", "cuda", "rocm"}:
        raise ValueError(f"Unsupported device request: {requested}")
    if normalized == "cpu":
        return DeviceInfo(torch.device("cpu"), "cpu", "CPU", None)
    if normalized in {"cuda", "rocm"} and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {normalized}, but torch.cuda.is_available() is false")
    if normalized == "rocm" and torch.version.hip is None:
        raise RuntimeError("Requested ROCm, but torch.version.hip is empty")
    if normalized == "cuda" and torch.version.hip is not None:
        raise RuntimeError("Requested CUDA, but the current Torch build uses ROCm/HIP")
    if normalized == "auto" and not torch.cuda.is_available():
        return DeviceInfo(torch.device("cpu"), "cpu", "CPU", None)
    props = torch.cuda.get_device_properties(0)
    backend = "rocm" if torch.version.hip is not None else "cuda"
    return DeviceInfo(
        torch.device("cuda:0"),
        backend,
        torch.cuda.get_device_name(0),
        int(props.total_memory),
    )


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python and Torch RNGs for a bounded, repeatable experiment."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, CPU Torch, and accelerator RNG state."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_accelerator"] = torch.cuda.get_rng_state_all()
    return state


def _normalize_rng_tensor(value: Any, label: str) -> torch.Tensor:
    """Return one contiguous CPU ByteTensor accepted by Torch RNG APIs."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} RNG state must be a torch.Tensor")
    if value.dtype != torch.uint8:
        raise TypeError(f"{label} RNG state must use torch.uint8")
    return value.detach().to(device="cpu").contiguous()


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state returned by capture_rng_state."""
    random.setstate(state["python"])
    torch.set_rng_state(_normalize_rng_tensor(state["torch_cpu"], "CPU"))
    if torch.cuda.is_available() and "torch_accelerator" in state:
        accelerator_states = state["torch_accelerator"]
        if not isinstance(accelerator_states, (list, tuple)):
            raise TypeError("Accelerator RNG state must be a list or tuple")
        torch.cuda.set_rng_state_all(
            [
                _normalize_rng_tensor(value, f"Accelerator {index}")
                for index, value in enumerate(accelerator_states)
            ]
        )
