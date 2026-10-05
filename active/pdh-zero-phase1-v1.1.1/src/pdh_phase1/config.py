"""Load and validate JSON-compatible YAML configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML file using only the Python standard library."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON-compatible YAML: {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be an object: {config_path}")
    return data


def require_keys(data: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    """Raise ConfigError when required keys are absent."""
    missing = [key for key in keys if key not in data]
    if missing:
        raise ConfigError(f"Missing keys in {context}: {', '.join(missing)}")
