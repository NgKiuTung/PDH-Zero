"""Atomic model/optimizer/RNG checkpoint tests."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from pdh_phase1.checkpoint import load_checkpoint, save_checkpoint
from pdh_phase1.device import restore_rng_state, seed_everything


class CheckpointTests(unittest.TestCase):
    """Verify continuation is exactly reproducible on CPU."""

    def test_checkpoint_restores_rng_and_optimizer(self) -> None:
        seed_everything(101, deterministic=True)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self._step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, {"step": 1})
            expected_loss = self._step(model, optimizer)
            expected = [parameter.detach().clone() for parameter in model.parameters()]
            restored = nn.Linear(4, 2)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            metadata = load_checkpoint(path, restored, restored_optimizer, torch.device("cpu"))
            actual_loss = self._step(restored, restored_optimizer)
            self.assertEqual(metadata["step"], 1)
            self.assertAlmostEqual(expected_loss, actual_loss, places=7)
            for left, right in zip(expected, restored.parameters()):
                self.assertTrue(torch.equal(left, right.detach()))

    def test_checkpoint_payload_is_staged_on_cpu_before_target_load(self) -> None:
        """Keep CPU RNG tensors on CPU even when the target model is an accelerator."""
        model = nn.Linear(4, 2)
        payload = {
            "model": model.state_dict(),
            "optimizer": {},
            "metadata": {"step": 1},
            "rng_state": {},
        }
        with mock.patch(
            "pdh_phase1.checkpoint.torch.load",
            return_value=payload,
        ) as mocked_load:
            metadata = load_checkpoint(
                "unused.pt",
                model,
                optimizer=None,
                device=torch.device("cuda:0"),
                restore_rng=False,
            )

        self.assertEqual(metadata["step"], 1)
        self.assertEqual(mocked_load.call_args.kwargs["map_location"], torch.device("cpu"))

    def test_restore_rng_state_normalizes_byte_tensors_for_torch_apis(self) -> None:
        """Normalize RNG tensors before passing them to CPU or accelerator generators."""
        noncontiguous = torch.arange(32, dtype=torch.uint8)[::2]
        state = {
            "python": random.getstate(),
            "torch_cpu": noncontiguous,
            "torch_accelerator": [noncontiguous],
        }
        with (
            mock.patch("pdh_phase1.device.torch.set_rng_state") as cpu_setter,
            mock.patch("pdh_phase1.device.torch.cuda.is_available", return_value=True),
            mock.patch("pdh_phase1.device.torch.cuda.set_rng_state_all") as accelerator_setter,
        ):
            restore_rng_state(state)

        cpu_state = cpu_setter.call_args.args[0]
        accelerator_state = accelerator_setter.call_args.args[0][0]
        for restored in (cpu_state, accelerator_state):
            self.assertEqual(restored.device.type, "cpu")
            self.assertEqual(restored.dtype, torch.uint8)
            self.assertTrue(restored.is_contiguous())

    @staticmethod
    def _step(model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        inputs = torch.randn(16, 4)
        targets = torch.randint(0, 2, (16,))
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())


if __name__ == "__main__":
    unittest.main()
