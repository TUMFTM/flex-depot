"""Golden-file test: the VB-bounds generator reproduces the bundled inputs.

Guards the reproducibility of the two example flexibility bands. If the
aggregation logic in ``scripts/generate_example_vb_bounds.py`` or the bundled
fleet-3 source data ever drift, this test fails instead of the mismatch going
unnoticed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_example_vb_bounds.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_example_vb_bounds", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", ["uni", "bidi"])
def test_generator_reproduces_bundled_band(case):
    generator = _load_generator()
    target = generator.CASE_OUTPUT[case]

    generated_text = generator.generate_case(case).to_csv(index=False, lineterminator="\n")
    expected_text = target.read_text().replace("\r\n", "\n")

    assert generated_text == expected_text, (
        f"{SCRIPT_PATH.name} no longer reproduces {target.name} exactly. "
        f"Regenerate with `python scripts/generate_example_vb_bounds.py --case {case}` "
        f"or investigate the drift."
    )
