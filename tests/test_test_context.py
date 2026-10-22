"""Tests for TestContext domain model and boundaries."""

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry
from aeris.core.vehicle.test_context import TestContext


def test_valid_minimal_context() -> None:
    ctx = TestContext(context_id="CTX-001", vehicle_id="AC-123", configuration_id="CONF-A")
    assert ctx.context_id == "CTX-001"
    assert ctx.vehicle_id == "AC-123"
    assert ctx.configuration_id == "CONF-A"
    assert ctx.name is None
    assert ctx.description is None
    assert ctx.metadata == {}


def test_valid_populated_context() -> None:
    ctx = TestContext(
        context_id="CTX-001",
        vehicle_id="AC-123",
        configuration_id="CONF-A",
        name="High Alpha Test",
        description="Testing high alpha flight regime.",
        metadata={"weather": "clear", "pilot": "Maverick"},
    )
    assert ctx.context_id == "CTX-001"
    assert ctx.vehicle_id == "AC-123"
    assert ctx.configuration_id == "CONF-A"
    assert ctx.name == "High Alpha Test"
    assert ctx.description == "Testing high alpha flight regime."
    assert ctx.metadata == {"weather": "clear", "pilot": "Maverick"}


def test_missing_context_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestContext(vehicle_id="AC-123", configuration_id="CONF-A")  # type: ignore[call-arg]
    assert "context_id" in str(exc_info.value)


def test_missing_vehicle_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestContext(context_id="CTX-001", configuration_id="CONF-A")  # type: ignore[call-arg]
    assert "vehicle_id" in str(exc_info.value)


def test_missing_configuration_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestContext(context_id="CTX-001", vehicle_id="AC-123")  # type: ignore[call-arg]
    assert "configuration_id" in str(exc_info.value)


def test_empty_identifiers() -> None:
    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        TestContext(context_id="", vehicle_id="AC-123", configuration_id="CONF-A")

    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        TestContext(context_id="CTX-001", vehicle_id="   ", configuration_id="CONF-A")

    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        TestContext(context_id="CTX-001", vehicle_id="AC-123", configuration_id="   ")


def test_invalid_identifiers() -> None:
    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        TestContext(context_id="CTX 001", vehicle_id="AC-123", configuration_id="CONF-A")

    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        TestContext(context_id="CTX-001", vehicle_id="AC!123", configuration_id="CONF-A")

    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        TestContext(context_id="CTX-001", vehicle_id="AC-123", configuration_id="CONF A")


def test_immutable_context() -> None:
    ctx = TestContext(context_id="CTX-001", vehicle_id="AC-123", configuration_id="CONF-A")
    with pytest.raises(ValidationError) as exc_info:
        ctx.name = "New Name"  # type: ignore[misc]
    assert "Instance is frozen" in str(exc_info.value)


def test_deterministic_serialization() -> None:
    ctx = TestContext(
        context_id="CTX-001",
        vehicle_id="AC-123",
        configuration_id="CONF-A",
        name="Test 1",
    )
    data = ctx.model_dump()
    assert data == {
        "context_id": "CTX-001",
        "vehicle_id": "AC-123",
        "configuration_id": "CONF-A",
        "name": "Test 1",
        "description": None,
        "metadata": {},
    }

    json_str = ctx.model_dump_json()
    assert '"context_id":"CTX-001"' in json_str.replace(" ", "")
    assert '"vehicle_id":"AC-123"' in json_str.replace(" ", "")


def test_repeated_construction_produces_equivalent_output() -> None:
    ctx1 = TestContext(context_id="CTX-1", vehicle_id="V-1", configuration_id="C-1")
    ctx2 = TestContext(context_id="CTX-1", vehicle_id="V-1", configuration_id="C-1")

    assert ctx1 == ctx2
    assert ctx1.model_dump() == ctx2.model_dump()


def test_distinct_contexts_remain_distinct() -> None:
    ctx1 = TestContext(context_id="CTX-1", vehicle_id="V-1", configuration_id="C-1")
    ctx2 = TestContext(context_id="CTX-2", vehicle_id="V-1", configuration_id="C-1")
    ctx3 = TestContext(context_id="CTX-1", vehicle_id="V-2", configuration_id="C-1")

    assert ctx1 != ctx2
    assert ctx1 != ctx3
    assert ctx1.context_id == "CTX-1"
    assert ctx2.context_id == "CTX-2"


def test_metadata_preservation() -> None:
    ctx = TestContext(
        context_id="CTX-1",
        vehicle_id="V-1",
        configuration_id="C-1",
        metadata={"nested": {"key": "value"}},
    )
    assert ctx.metadata["nested"]["key"] == "value"


def test_no_pandas_dependency() -> None:
    """IMPORTANT NEGATIVE TEST: DataFrame inside metadata must not influence context."""
    df = pd.DataFrame({"altitude_ft": [1000, 2000]})
    ctx = TestContext(
        context_id="CTX-1",
        vehicle_id="V-1",
        configuration_id="C-1",
        metadata={"data": df},
    )

    assert not hasattr(ctx, "max_altitude")
    pd.testing.assert_frame_equal(df, ctx.metadata["data"])


def test_no_parameter_registry_mutation() -> None:
    """IMPORTANT NEGATIVE TEST: Context creation does not touch ParameterRegistry."""
    registry = ParameterRegistry()
    registry.register(
        ParameterDefinition(
            parameter_id="engine_rpm",
            display_name="Engine RPM",
            description="Engine RPM",
            physical_quantity="angular_velocity",
            data_type=ParameterDataType.FLOAT,
        )
    )

    _ctx = TestContext(context_id="CTX-1", vehicle_id="V-1", configuration_id="C-1")

    # Registry must remain exactly as it was
    params = registry.list()
    assert len(params) == 1
    assert params[0].parameter_id == "engine_rpm"


def test_rejection_of_performance_limit_fields() -> None:
    """IMPORTANT NEGATIVE TEST: Structural performance fields are forbidden at this level."""
    with pytest.raises(ValidationError) as exc_info:
        TestContext(
            context_id="CTX-1",
            vehicle_id="V-1",
            configuration_id="C-1",
            max_speed_kts=300.0,  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)
    assert "max_speed_kts" in str(exc_info.value)


def test_rejection_of_telemetry_fields() -> None:
    """IMPORTANT NEGATIVE TEST: Telemetry fields cannot be directly assigned to context."""
    with pytest.raises(ValidationError) as exc_info:
        TestContext(
            context_id="CTX-1",
            vehicle_id="V-1",
            configuration_id="C-1",
            altitude_ft=[1000, 2000],  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)
    assert "altitude_ft" in str(exc_info.value)
