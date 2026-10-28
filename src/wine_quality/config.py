"""Project paths and configuration loading."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration file.
    Args:
        path: Path to the YAML file. Defaults to ``config.yaml`` at the project root.
    Returns:
        The configuration as a nested dictionary.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative: str | Path) -> Path:
    """Turn a path written in the config into an absolute path.
    Args:
        relative: A path relative to the project root (an absolute path is kept).
    Returns:
        The absolute path.
    """
    return PROJECT_ROOT / relative
