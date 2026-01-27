from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

PathLike = Union[str, Path]


def load_settings(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """
    Load a YAML settings file into a Python dictionary.

    Parameters
    ----------
    path:
        Path to a YAML file. If None, loads the default example configuration
        shipped with the package (settings_example.yaml next to this module).

    Returns
    -------
    dict
        Parsed settings structure.

    Notes
    -----
    - Uses `yaml.safe_load` to avoid executing arbitrary YAML tags.
    - Raises `FileNotFoundError` if the file does not exist.
    - Raises `yaml.YAMLError` if the YAML is invalid.
    """
    if path is None:
        path = Path(__file__).parent / "settings_example.yaml"
    else:
        path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    if settings is None:
        return {}

    if not isinstance(settings, dict):
        raise ValueError(f"Settings YAML must parse to a dict at top level, got: {type(settings)}")

    return settings