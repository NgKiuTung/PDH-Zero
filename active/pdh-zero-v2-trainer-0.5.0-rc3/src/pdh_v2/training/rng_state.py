"""检查点随机数状态的设备无关恢复工具。"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def as_cpu_byte_tensor(value: Any, label: str) -> torch.Tensor:
    """将序列化 RNG 状态规范化为 PyTorch 期望的 CPU ByteTensor。"""
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{label} must be a torch.Tensor")
    state = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if state.ndim != 1:
        raise RuntimeError(f"{label} must be one-dimensional")
    return state


def restore_rng_states(torch_rng: Any, cuda_rng: Any) -> None:
    """恢复 CPU/CUDA RNG；避免 map_location 将 RNG 张量迁移到 CUDA。"""
    torch.set_rng_state(as_cpu_byte_tensor(torch_rng, "torch_rng"))
    if cuda_rng is None or not torch.cuda.is_available():
        return
    if isinstance(cuda_rng, torch.Tensor) or not isinstance(cuda_rng, Iterable):
        raise RuntimeError("cuda_rng must be an iterable of tensors")
    states = [
        as_cpu_byte_tensor(state, f"cuda_rng[{index}]")
        for index, state in enumerate(cuda_rng)
    ]
    if not states:
        raise RuntimeError("cuda_rng cannot be empty when CUDA is available")
    torch.cuda.set_rng_state_all(states)
