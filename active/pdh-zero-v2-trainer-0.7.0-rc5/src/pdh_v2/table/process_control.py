"""Owned-process receipts and safe signal cancellation; never kills unrelated jobs."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess

import torch

from pdh_v2.training.refinement_control import StopRequested, atomic_json, utc


class RunSignals:
    """Install main-process handlers for a bounded run, restoring previous handlers afterward."""
    def __init__(self, output: Path):
        atomic_json(output/'process.json', dict(pid=os.getpid(), pgid=os.getpgid(0),
                                               started_at=utc(), only_owned_processes=True))
        self.previous = {}
        for name in (signal.SIGINT, signal.SIGTERM):
            self.previous[name] = signal.signal(name, self._stop)

    @staticmethod
    def _stop(signum, frame):
        raise StopRequested(f'signal_{signum}')

    def close(self):
        """Restore handlers even when fitting, collection, or evaluation raises."""
        for name, handler in self.previous.items():
            signal.signal(name, handler)


def validate_device(name: str) -> torch.device:
    """Explicit CUDA request fails closed if unavailable or another GPU job is present."""
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('rc5.cuda_not_available')
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True,
            check=True, timeout=10)
        if result.stdout.strip():
            raise RuntimeError('rc5.gpu_occupied_no_process_was_killed')
    return device