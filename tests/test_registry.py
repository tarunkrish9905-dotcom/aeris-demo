"""Tests for AERIS Parameter Registry."""

import pandas as pd
import pytest

from aeris.core.registry import (
    DuplicateAliasError,
    DuplicateParameterError,
    InvalidParameterDefinitionError,
    ParameterDataType,
    ParameterDefinition,
    ParameterNotFoundError,
    ParameterRegistry,
)


def create_mock_def(
    parameter_id: str,
    aliases: list[str] | None = None,
    canonical_unit: str | None = None,
) -> ParameterDefinition:
    """Helper to quickly spin up valid parameter definitions."""
    return ParameterDefinition(
        parameter_id=parameter_id,
        display_name=f"{parameter_id.capitalize()} Display",
        description="A mock parameter definition.",
        physical_quantity="mock_quantity",
        data_type=ParameterDataType.FLOAT,
        canonical_unit=canonical_unit,
        aliases=aliases or [],
    )


def test_registry_register_and_get() -> None:
    """Verify standard registration and exact retrieval."""
    registry = ParameterRegistry()
    pdef = create_mock_def("test_param")

    registry.register(pdef)

    assert registry.contains("test_param")
    retrieved = registry.get("test_param")
    assert retrieved == pdef


def test_registry_list_parameters() -> None:
    """Verify registry listing functionality."""
    registry = ParameterRegistry()
    registry.register(create_mock_def("param1"))
    registry.register(create_mock_def("param2"))

    all_params = registry.list()
    assert len(all_params) == 2
    ids = {p.parameter_id for p in all_params}
    assert ids == {"param1", "param2"}


def test_registry_resolve_valid_alias() -> None:
    """Verify deterministic resolution of a valid alias."""
    registry = ParameterRegistry()
    registry.register(create_mock_def("primary", aliases=["alias1", "alias2"]))

    assert registry.resolve_alias("primary") == "primary"
    assert registry.resolve_alias("alias1") == "primary"
    assert registry.resolve_alias("alias2") == "primary"


def test_registry_unknown_parameter() -> None:
    """Verify structured exceptions for unknown parameters."""
    registry = ParameterRegistry()

    assert not registry.contains("does_not_exist")

    with pytest.raises(ParameterNotFoundError, match="not found"):
        registry.get("does_not_exist")

    with pytest.raises(ParameterNotFoundError, match="could not be resolved"):
        registry.resolve_alias("does_not_exist")


def test_registry_duplicate_parameter_id() -> None:
    """Verify duplicate parameter registration throws."""
    registry = ParameterRegistry()
    registry.register(create_mock_def("conflict"))

    with pytest.raises(DuplicateParameterError):
        registry.register(create_mock_def("conflict"))


def test_registry_duplicate_alias_collision_with_other_alias() -> None:
    """Verify an alias cannot be claimed if another parameter already owns it."""
    registry = ParameterRegistry()
    registry.register(create_mock_def("param_a", aliases=["shared_alias"]))

    with pytest.raises(DuplicateAliasError, match="is already claimed by 'param_a'"):
        registry.register(create_mock_def("param_b", aliases=["shared_alias"]))


def test_registry_duplicate_alias_collision_with_id() -> None:
    """Verify an alias cannot conflict with a registered primary ID."""
    registry = ParameterRegistry()
    registry.register(create_mock_def("param_x"))

    with pytest.raises(DuplicateAliasError, match="conflicts with an existing parameter ID"):
        registry.register(create_mock_def("param_y", aliases=["param_x"]))


def test_registry_id_precedence_over_alias() -> None:
    """Verify ID precedence and alias-to-ID block logic."""
    registry = ParameterRegistry()
    registry.register(create_mock_def("alpha", aliases=["beta"]))

    # 'beta' is an alias of 'alpha', so it can't be used as an ID
    with pytest.raises(DuplicateAliasError, match="already claimed as an alias by 'alpha'"):
        registry.register(create_mock_def("beta"))


def test_registry_invalid_alias_matches_id() -> None:
    """Verify parameter cannot declare its own ID as an alias."""
    registry = ParameterRegistry()
    with pytest.raises(InvalidParameterDefinitionError, match="matches the parameter ID itself"):
        registry.register(create_mock_def("same", aliases=["same"]))


def test_immutable_parameter_definition() -> None:
    """Verify definition model is strictly immutable."""
    from pydantic import ValidationError

    pdef = create_mock_def("frozen")
    with pytest.raises(ValidationError):
        pdef.display_name = "Mutated"  # type: ignore


def test_canonical_unit_as_metadata() -> None:
    """Verify unit is purely metadata without behavioral conversion execution attached."""
    pdef = create_mock_def("speed", canonical_unit="kts")
    assert pdef.canonical_unit == "kts"
    assert "kts" in pdef.model_dump()["canonical_unit"]


def test_no_automatic_dataframe_mutation() -> None:
    """NEGATIVE TEST: Verify that the registry never mutates a DataFrame implicitly."""
    registry = ParameterRegistry()
    # Register that 'speed' is an alias of 'airspeed'
    registry.register(create_mock_def("airspeed", aliases=["speed"]))

    # Create a DataFrame
    df = pd.DataFrame({"speed": [1.0, 2.0, 3.0]})

    # Simply passing the DataFrame, or interacting with the registry,
    # should NOT mutate the DataFrame's column name.
    # The registry is passive storage.
    alias_resolution = registry.resolve_alias("speed")
    assert alias_resolution == "airspeed"

    # Prove the DataFrame is strictly unaltered
    assert "speed" in df.columns
    assert "airspeed" not in df.columns

    # In a real pipeline, the Normalization Engine must be explicitly told to do:
    # df.rename(columns={"speed": alias_resolution})


def test_registry_unregister() -> None:
    """Verify deterministic unregistration of ID and aliases."""
    registry = ParameterRegistry()
    registry.register(create_mock_def("temp", aliases=["t1", "t2"]))

    assert registry.contains("temp")
    assert registry.resolve_alias("t1") == "temp"

    registry.unregister("temp")

    assert not registry.contains("temp")
    with pytest.raises(ParameterNotFoundError):
        registry.resolve_alias("t1")
