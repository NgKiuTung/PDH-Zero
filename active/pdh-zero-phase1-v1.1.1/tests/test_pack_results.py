"""Portable result-archive tests for cloud images without a system zip binary."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACK_SCRIPT = PROJECT_ROOT / "scripts" / "12_pack_results.sh"


class PackResultsTests(unittest.TestCase):
    """Verify result packaging uses only Bash and the active Python runtime."""

    def test_pack_results_works_when_system_zip_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._create_run_dir(root)
            environment = self._python_only_path(root)

            result = subprocess.run(
                ["/bin/bash", str(PACK_SCRIPT), str(run_dir)],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            archive = Path(f"{run_dir}.zip")
            checksum = Path(f"{archive}.sha256")
            self.assertTrue(archive.is_file())
            self.assertTrue(checksum.is_file())
            with zipfile.ZipFile(archive) as packaged:
                self.assertIsNone(packaged.testzip())
                self.assertIn(f"{run_dir.name}/manifest.json", packaged.namelist())
                self.assertIn(f"{run_dir.name}/artifacts/metrics.json", packaged.namelist())
            expected = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(checksum.read_text(encoding="utf-8"), f"{expected}  {archive.name}\n")

    def test_pack_results_refuses_to_overwrite_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._create_run_dir(root)
            environment = self._python_only_path(root)

            first = subprocess.run(
                ["/bin/bash", str(PACK_SCRIPT), str(run_dir)],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            second = subprocess.run(
                ["/bin/bash", str(PACK_SCRIPT), str(run_dir)],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Refusing to overwrite", second.stderr)

    @staticmethod
    def _create_run_dir(root: Path) -> Path:
        """Create a minimal evidence directory for one packaging test."""
        run_dir = root / "A01_system_info_20260929T022006Z"
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True)
        (run_dir / "manifest.json").write_text('{"status":"success"}\n', encoding="utf-8")
        (run_dir / "stdout.log").write_text("complete\n", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        (artifacts / "metrics.json").write_text('{"value":1}\n', encoding="utf-8")
        return run_dir

    @staticmethod
    def _python_only_path(root: Path) -> dict[str, str]:
        """Return an environment whose PATH intentionally contains no zip executable."""
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "python3").symlink_to(sys.executable)
        environment = os.environ.copy()
        environment["PATH"] = str(bin_dir)
        return environment


if __name__ == "__main__":
    unittest.main()
