#!/usr/bin/env python3
"""Create an immutable ZIP archive and SHA-256 digest for one result directory."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import uuid
import zipfile
from pathlib import Path


class PackagingError(RuntimeError):
    """Raised when a result directory cannot be packaged safely."""


def parse_args() -> argparse.Namespace:
    """Parse the result directory to archive."""
    parser = argparse.ArgumentParser(
        description="Package one result directory without requiring the system zip command."
    )
    parser.add_argument("run_directory", help="Existing immutable result directory")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest for one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_members(run_dir: Path) -> list[Path]:
    """Return sorted archive members and reject symbolic links."""
    members = sorted(run_dir.rglob("*"), key=lambda item: item.as_posix())
    symlinks = [item for item in members if item.is_symlink()]
    if symlinks:
        relative = ", ".join(str(item.relative_to(run_dir)) for item in symlinks[:5])
        raise PackagingError(f"Refusing to archive symbolic links: {relative}")
    return members


def create_archive(run_dir: Path) -> tuple[Path, Path]:
    """Create and validate a ZIP archive plus its checksum sidecar."""
    run_dir = run_dir.expanduser().resolve(strict=True)
    if not run_dir.is_dir():
        raise PackagingError(f"Not a directory: {run_dir}")

    archive = run_dir.parent / f"{run_dir.name}.zip"
    checksum = Path(f"{archive}.sha256")
    if archive.exists() or checksum.exists():
        existing = archive if archive.exists() else checksum
        raise PackagingError(f"Refusing to overwrite existing output: {existing}")

    token = uuid.uuid4().hex
    temporary_archive = run_dir.parent / f".{run_dir.name}.{token}.zip.tmp"
    temporary_checksum = run_dir.parent / f".{run_dir.name}.{token}.sha256.tmp"
    members = archive_members(run_dir)

    try:
        with zipfile.ZipFile(
            temporary_archive,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as output:
            root_name = Path(run_dir.name)
            output.writestr(f"{root_name.as_posix()}/", b"")
            for item in members:
                relative = item.relative_to(run_dir)
                archive_name = root_name / relative
                if item.is_dir():
                    output.writestr(f"{archive_name.as_posix()}/", b"")
                elif item.is_file():
                    output.write(item, archive_name.as_posix())
                else:
                    raise PackagingError(f"Unsupported filesystem entry: {item}")

        with zipfile.ZipFile(temporary_archive, mode="r") as packaged:
            damaged_member = packaged.testzip()
            if damaged_member is not None:
                raise PackagingError(f"ZIP integrity check failed at: {damaged_member}")

        digest = sha256(temporary_archive)
        temporary_checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        os.replace(temporary_archive, archive)
        os.replace(temporary_checksum, checksum)
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        temporary_checksum.unlink(missing_ok=True)
        raise

    return archive, checksum


def main() -> int:
    """Package the requested result directory and print generated paths."""
    args = parse_args()
    try:
        archive, checksum = create_archive(Path(args.run_directory))
    except (OSError, PackagingError, zipfile.BadZipFile) as exc:
        print(f"pack_results: {exc}", file=sys.stderr)
        return 1
    print(archive)
    print(checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
