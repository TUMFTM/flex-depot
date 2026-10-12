"""The configuration reference in README.md must match the settings model."""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_config_reference.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_config_reference", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_readme_config_reference_is_up_to_date():
    script = _load_script_module()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    _, current, _ = script.split_readme(readme)

    assert current == script.render(), "Run: python scripts/generate_config_reference.py"
