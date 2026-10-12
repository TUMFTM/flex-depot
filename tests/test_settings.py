"""Fast tests for building Settings from a TOML file, from a dict, and from a batch default plus override."""

import tomllib
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from flex_dep_opt.cli import _load_merged
from flex_dep_opt.config.settings import Settings

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_TOML = REPO_ROOT / "src/flex_dep_opt/config/settings_quickstart.toml"


def _example_dict() -> dict:
    with open(EXAMPLE_TOML, "rb") as f:
        return tomllib.load(f)


def test_from_dict_matches_load():
    from_file = Settings.from_toml(EXAMPLE_TOML)
    from_dict = Settings.from_dict(_example_dict())

    assert from_dict.model_dump() == from_file.model_dump()
    assert from_file.get_source_path() == EXAMPLE_TOML
    assert from_dict.get_source_path() is None


def test_from_dict_uses_the_given_values():
    data = _example_dict()
    data["simulation"]["name"] = "from_python"

    assert Settings.from_dict(data).simulation.name == "from_python"


def test_unknown_top_level_section_is_rejected():
    with pytest.raises(ValidationError):
        Settings.from_dict({**_example_dict(), "bogus": {"x": 1}})


def test_batch_override_replaces_only_the_given_keys(tmp_path):
    override = tmp_path / "run_a.toml"
    override.write_text('[simulation]\nname = "run_a"\n')

    merged = _load_merged(str(override), str(EXAMPLE_TOML))
    base = Settings.from_toml(EXAMPLE_TOML)

    assert merged.simulation.name == "run_a"
    assert merged.simulation.model_dump(exclude={"name"}) == base.simulation.model_dump(exclude={"name"})
    assert merged.optimization == base.optimization
    assert merged.get_source_path() == override


def _fields_without_description(model: type[BaseModel], prefix: str = "") -> list[str]:
    missing = []
    for name, field in model.model_fields.items():
        if not field.description:
            missing.append(prefix + name)
        if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
            missing += _fields_without_description(field.annotation, f"{prefix}{name}.")
    return missing


def test_every_setting_has_a_description():
    assert _fields_without_description(Settings) == []
