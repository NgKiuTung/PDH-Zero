"""检查点 RNG 状态跨设备恢复回归测试。"""
from __future__ import annotations

import unittest
from unittest import mock

import torch

from pdh_v2.training.rng_state import as_cpu_byte_tensor, restore_rng_states


class RngStateTests(unittest.TestCase):
    def test_rng_tensor_is_normalized_to_cpu_uint8(self) -> None:
        source = torch.arange(16, dtype=torch.int64)
        normalized = as_cpu_byte_tensor(source, "test_rng")
        self.assertEqual(normalized.device.type, "cpu")
        self.assertEqual(normalized.dtype, torch.uint8)
        self.assertTrue(normalized.is_contiguous())
        self.assertEqual(normalized.ndim, 1)

    def test_cuda_restore_receives_cpu_byte_tensors(self) -> None:
        original = torch.get_rng_state()
        cuda_state = torch.arange(32, dtype=torch.int64)
        try:
            with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
                "torch.cuda.set_rng_state_all"
            ) as setter:
                restore_rng_states(original.clone(), [cuda_state])
            restored = setter.call_args.args[0]
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0].device.type, "cpu")
            self.assertEqual(restored[0].dtype, torch.uint8)
        finally:
            torch.set_rng_state(original)

    def test_invalid_cuda_rng_container_is_rejected(self) -> None:
        original = torch.get_rng_state()
        with mock.patch("torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "iterable"):
                restore_rng_states(original, torch.arange(8, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
