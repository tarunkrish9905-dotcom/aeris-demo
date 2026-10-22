"""Tests for Registry ↔ Normalization integration."""

import pandas as pd
import pytest

from aeris.core.normalization.engine import (
    DataNormalizationEngine,
    NormalizationRequest,
)
from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry
from aeris.core.results.types import ResultStatus


@pytest.fixture
def test_registry() -> ParameterRegistry:
    """Fixture to provide a populated registry for testing."""
    registry = ParameterRegistry()
    registry.register(
        ParameterDefinition(
            parameter_id="airspeed",
            display_name="Airspeed",
            description="True airspeed",
            physical_quantity="velocity",
            data_type=ParameterDataType.FLOAT,
            canonical_unit="knots",
            aliases=["speed", "tas", "airspeed_kts"],
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="timestamp",
            display_name="Timestamp",
            description="UTC timestamp",
            physical_quantity="time",
            data_type=ParameterDataType.TIMESTAMP,
            canonical_unit="iso8601",
            aliases=["time", "ts"],
        )
    )
    return registry


def test_registry_resolve_by_parameter_id_and_alias(test_registry: ParameterRegistry) -> None:
    """Verify explicit rename using registry-resolved parameter (both ID and alias)."""
    df = pd.DataFrame(
        {"airspeed_kts": [100.0, 150.0], "time": ["2026-10-06T10:00:00Z", "2026-10-06T10:00:01Z"]}
    )
    req = NormalizationRequest()
    req.resolve_and_rename("airspeed_kts")  # alias -> airspeed
    req.resolve_and_rename("time")  # alias -> timestamp

    engine = DataNormalizationEngine(registry=test_registry)
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.SUCCESS
    assert res.data is not None
    assert "airspeed" in res.data.columns
    assert "timestamp" in res.data.columns
    assert "airspeed_kts" not in res.data.columns
    assert "time" not in res.data.columns


def test_unresolved_parameter_failure(test_registry: ParameterRegistry) -> None:
    """Verify registry raises structured error if the requested parameter does not exist."""
    df = pd.DataFrame({"unknown_col": [1.0]})
    req = NormalizationRequest()
    req.resolve_and_rename("unknown_col")

    engine = DataNormalizationEngine(registry=test_registry)
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_NOT_FOUND" for i in res.issues)


def test_missing_source_column_failure(test_registry: ParameterRegistry) -> None:
    """Verify registry resolves fine but DataFrame is missing the column."""
    df = pd.DataFrame({"other_col": [1.0]})
    req = NormalizationRequest()
    req.resolve_and_rename("speed")  # valid alias, but "speed" is not in DF!

    engine = DataNormalizationEngine(registry=test_registry)
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_MISSING_SOURCE_COLUMN" for i in res.issues)


def test_explicit_numeric_conversion(test_registry: ParameterRegistry) -> None:
    """Verify explicit numeric conversion using registry-resolved parameter."""
    df = pd.DataFrame({"speed": ["100.5", "150.2"]})
    req = NormalizationRequest()
    req.resolve_and_rename("speed")
    req.resolve_and_convert_type("speed", float)

    engine = DataNormalizationEngine(registry=test_registry)
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.SUCCESS
    assert res.data is not None
    assert pd.api.types.is_float_dtype(res.data["airspeed"])


def test_explicit_timestamp_conversion(test_registry: ParameterRegistry) -> None:
    """Verify explicit timestamp conversion using registry-resolved parameter."""
    df = pd.DataFrame({"time": ["2026-10-06T10:00:00Z"]})
    req = NormalizationRequest()
    req.resolve_and_rename("time")
    req.resolve_and_standardize_timestamp("time")

    engine = DataNormalizationEngine(registry=test_registry)
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.SUCCESS
    assert res.data is not None
    assert pd.api.types.is_datetime64_any_dtype(res.data["timestamp"])


def test_incompatible_transformation_failure(test_registry: ParameterRegistry) -> None:
    """Verify engine validates transformation against registry semantics."""
    df = pd.DataFrame({"speed": ["100.5"]})
    req = NormalizationRequest()
    # Speed is FLOAT, but we mistakenly request timestamp standardization
    req.resolve_and_standardize_timestamp("speed")

    engine = DataNormalizationEngine(registry=test_registry)
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_INCOMPATIBLE_TRANSFORMATION" for i in res.issues)


def test_no_automatic_transformation(test_registry: ParameterRegistry) -> None:
    """NEGATIVE TEST: Ensure no automatic transformation happens just because a registry exists."""
    df = pd.DataFrame({"speed": [100.0]})
    req = NormalizationRequest()  # Empty request

    engine = DataNormalizationEngine(registry=test_registry)
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.SUCCESS
    assert res.data is not None
    assert "speed" in res.data.columns
    assert "airspeed" not in res.data.columns


def test_no_automatic_unit_conversion(test_registry: ParameterRegistry) -> None:
    """IMPORTANT TEST: Verify NO automatic unit conversion occurs."""
    # Data is provided in 'speed' (which maps to 'airspeed' with canonical_unit='knots')
    # Let's say the data is 100.0, we just request a rename.
    # The system must NOT magically convert it. It must stay 100.0.
    df = pd.DataFrame({"speed": [100.0]})
    req = NormalizationRequest()
    req.resolve_and_rename("speed")

    engine = DataNormalizationEngine(registry=test_registry)
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data["airspeed"].iloc[0] == 100.0


def test_dataframe_immutability(test_registry: ParameterRegistry) -> None:
    """Verify input DataFrame remains unchanged."""
    df = pd.DataFrame({"speed": ["100.5"]})
    df_copy = df.copy(deep=True)
    req = NormalizationRequest()
    req.resolve_and_rename("speed")
    req.resolve_and_convert_type("speed", float)

    engine = DataNormalizationEngine(registry=test_registry)
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.SUCCESS
    pd.testing.assert_frame_equal(df, df_copy)


def test_backward_compatibility_without_registry() -> None:
    """Verify legacy requests still work without a registry."""
    df = pd.DataFrame({"airspeed_kts": ["100.5"]})
    req = NormalizationRequest()
    req.rename_column("airspeed_kts", "airspeed")
    req.convert_type("airspeed", float)

    engine = DataNormalizationEngine()  # No registry
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.SUCCESS
    assert res.data is not None
    assert "airspeed" in res.data.columns
    assert pd.api.types.is_float_dtype(res.data["airspeed"])


def test_registry_required_for_resolved_requests() -> None:
    """Verify an explicit registry request fails gracefully if no registry is provided."""
    df = pd.DataFrame({"speed": ["100.5"]})
    req = NormalizationRequest()
    req.resolve_and_rename("speed")

    engine = DataNormalizationEngine()  # No registry
    res = engine.normalize(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_REGISTRY_NOT_AVAILABLE" for i in res.issues)
