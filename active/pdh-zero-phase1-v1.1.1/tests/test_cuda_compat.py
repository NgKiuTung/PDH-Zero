
"""Unit tests for CUDA and Blackwell image compatibility rules."""

from __future__ import annotations

import unittest

from pdh_phase1.cuda_compat import parse_cuda_version, validate_cuda_architecture


class CudaCompatibilityTests(unittest.TestCase):
    """Verify that incompatible RTX 50-series images are rejected."""

    def test_parse_cuda_version(self) -> None:
        self.assertEqual(parse_cuda_version("13.0"), (13, 0))
        self.assertEqual(parse_cuda_version("12.8.1"), (12, 8))
        self.assertIsNone(parse_cuda_version(None))

    def test_blackwell_rejects_cuda_12_1(self) -> None:
        errors = validate_cuda_architecture("12.1", (12, 0), ["sm_120"])
        self.assertIn("CUDA 12.8 or newer", errors[0])

    def test_blackwell_rejects_missing_sm_120(self) -> None:
        errors = validate_cuda_architecture("13.0", (12, 0), ["sm_90", "sm_100"])
        self.assertEqual(errors, ["Torch architecture list does not contain sm_120"])

    def test_blackwell_accepts_cuda_12_8_and_sm_120(self) -> None:
        errors = validate_cuda_architecture("12.8", (12, 0), ["sm_90", "sm_120"])
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
