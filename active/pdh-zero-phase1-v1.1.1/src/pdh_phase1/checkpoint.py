"""Checkpoint persistence shared by smoke tests and PPO training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .device import capture_rng_state, restore_rng_state


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
) -> None:
    """Atomically save model, optimizer, metadata, and RNG state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": metadata,
        "rng_state": capture_rng_state(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint on the requested backend and optionally restore RNG state.

    The caller must construct ``model`` and ``optimizer`` on ``device`` before loading.
    """
    # NOTE: The payload contains CPU RNG state. Mapping the whole checkpoint to
    # CUDA/ROCm converts that ByteTensor into an accelerator tensor, which
    # ``torch.set_rng_state`` rejects. Stage on CPU; state_dict loaders copy
    # model and optimizer tensors to the devices of their destination objects.
    payload = torch.load(
        Path(path),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return dict(payload["metadata"])
