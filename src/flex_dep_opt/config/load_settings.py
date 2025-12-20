import yaml
from pathlib import Path

def load_settings(path: str | Path = None) -> dict:
    """
    Load settings from YAML file into Python dictionary.
    """
    if path is None:
        path = Path(__file__).parent / "settings_example.yaml"

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)