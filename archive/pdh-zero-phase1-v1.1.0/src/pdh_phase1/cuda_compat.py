
"""CUDA architecture compatibility rules used by cloud preflight checks."""

from __future__ import annotations

import re
from collections.abc import Sequence

_MIN_BLACKWELL_CUDA = (12, 8)


def parse_cuda_version(value: str | None) -> tuple[int, int] | None:
    """Parse the major and minor components from a CUDA runtime version string."""
    if value is None:
        return None
    match = re.match(r"\s*(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"Unsupported CUDA version string: {value!r}")
    return int(match.group(1)), int(match.group(2))


def architecture_tag(capability: tuple[int, int]) -> str:
    """Return the canonical Torch architecture tag for a compute capability."""
    major, minor = capability
    return f"sm_{major}{minor}"


def validate_cuda_architecture(
    cuda_version: str | None,
    capability: tuple[int, int],
    arch_list: Sequence[str],
) -> list[str]:
    """Return blocking compatibility errors for the selected CUDA device.

    Consumer Blackwell GPUs, including RTX 50-series devices with compute
    capability 12.0, require CUDA 12.8 or newer and a Torch build containing
    the matching ``sm_120`` kernels.
    """
    errors: list[str] = []
    parsed = parse_cuda_version(cuda_version)
    required_tag = architecture_tag(capability)

    if capability[0] >= 12:
        if parsed is None or parsed < _MIN_BLACKWELL_CUDA:
            errors.append(
                "Blackwell compute capability requires a Torch build with CUDA 12.8 or newer"
            )
        if not any(item == required_tag or item.startswith(f"{required_tag}a") for item in arch_list):
            errors.append(f"Torch architecture list does not contain {required_tag}")
    return errors
