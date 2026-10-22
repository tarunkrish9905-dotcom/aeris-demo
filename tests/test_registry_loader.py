"""Tests for AERIS Parameter Registry YAML Loader."""

import pathlib

import pytest

from aeris.core.registry import (
    DuplicateAliasError,
    DuplicateParameterError,
    InvalidParameterDefinitionError,
    load_registry_from_yaml,
)


@pytest.fixture
def registry_fixtures_dir(fixtures_dir: pathlib.Path) -> pathlib.Path:
    """Return the registry fixtures directory."""
    return fixtures_dir / "registry"


def test_load_valid_yaml(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify that a well-formed YAML file populates the registry correctly."""
    yaml_path = registry_fixtures_dir / "valid_registry.yaml"
    registry = load_registry_from_yaml(yaml_path)

    assert len(registry.list()) == 2

    speed = registry.get("test_speed")
    assert speed.display_name == "Test Speed"
    assert speed.data_type == "FLOAT"
    assert speed.canonical_unit == "kts"
    assert set(speed.aliases) == {"ts", "spd"}

    # Check alias resolution
    assert registry.resolve_alias("spd") == "test_speed"

    alt = registry.get("test_altitude")
    assert alt.display_name == "Test Altitude"


def test_load_duplicate_id(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify loading fails on duplicate IDs."""
    yaml_path = registry_fixtures_dir / "duplicate_id.yaml"
    with pytest.raises(DuplicateParameterError):
        load_registry_from_yaml(yaml_path)


def test_load_duplicate_alias(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify loading fails on duplicate aliases across parameters."""
    yaml_path = registry_fixtures_dir / "duplicate_alias.yaml"
    with pytest.raises(DuplicateAliasError):
        load_registry_from_yaml(yaml_path)


def test_load_alias_collision(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify loading fails if an alias collides with a registered ID."""
    yaml_path = registry_fixtures_dir / "alias_collision.yaml"
    with pytest.raises(DuplicateAliasError):
        load_registry_from_yaml(yaml_path)


def test_load_missing_required(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify loading fails if a parameter is missing required schema fields."""
    yaml_path = registry_fixtures_dir / "missing_required.yaml"
    with pytest.raises(InvalidParameterDefinitionError, match="Validation failed"):
        load_registry_from_yaml(yaml_path)


def test_load_invalid_data_type(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify loading fails if data_type is not a recognized enum value."""
    yaml_path = registry_fixtures_dir / "invalid_data_type.yaml"
    with pytest.raises(InvalidParameterDefinitionError, match="Validation failed"):
        load_registry_from_yaml(yaml_path)


def test_load_malformed_yaml(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify loader traps standard YAML parsing errors."""
    yaml_path = registry_fixtures_dir / "malformed.yaml"
    with pytest.raises(InvalidParameterDefinitionError, match="Malformed YAML syntax"):
        load_registry_from_yaml(yaml_path)


def test_load_missing_parameters_key(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify loader requires the root 'parameters' key."""
    yaml_path = registry_fixtures_dir / "missing_parameters_key.yaml"
    with pytest.raises(
        InvalidParameterDefinitionError, match="missing the required top-level 'parameters' key"
    ):
        load_registry_from_yaml(yaml_path)


def test_load_non_list_parameters(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify loader expects a list of dictionaries."""
    yaml_path = registry_fixtures_dir / "non_list_parameters.yaml"
    with pytest.raises(InvalidParameterDefinitionError, match="must contain a list"):
        load_registry_from_yaml(yaml_path)


def test_load_atomic_failure(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify atomic failure: one bad parameter aborts the entire load operation,
    leaving no partial state (the function simply throws and returns nothing)."""
    yaml_path = registry_fixtures_dir / "atomic_failure.yaml"
    with pytest.raises(
        InvalidParameterDefinitionError, match="Validation failed for parameter 'test_altitude'"
    ):
        load_registry_from_yaml(yaml_path)


def test_loaded_definition_immutability(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify that parameters loaded from YAML remain completely immutable."""
    from pydantic import ValidationError

    yaml_path = registry_fixtures_dir / "valid_registry.yaml"
    registry = load_registry_from_yaml(yaml_path)

    speed = registry.get("test_speed")
    with pytest.raises(ValidationError):
        speed.display_name = "Mutated via code"  # type: ignore


def test_deterministic_equivalence(registry_fixtures_dir: pathlib.Path) -> None:
    """Verify that repeatedly loading the same file yields equivalent registries."""
    yaml_path = registry_fixtures_dir / "valid_registry.yaml"
    registry1 = load_registry_from_yaml(yaml_path)
    registry2 = load_registry_from_yaml(yaml_path)

    # Internal collections should match functionally
    assert registry1.list() == registry2.list()
