"""Tests for Vehicle domain models and boundary constraints."""

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry
from aeris.core.vehicle.models import Vehicle


def test_valid_minimal_vehicle() -> None:
    v = Vehicle(vehicle_id="AC-123")
    assert v.vehicle_id == "AC-123"
    assert v.name is None
    assert v.manufacturer is None
    assert v.model is None
    assert v.variant is None
    assert v.configuration_identifier is None
    assert v.description is None
    assert v.metadata == {}


def test_valid_fully_populated_vehicle() -> None:
    v = Vehicle(
        vehicle_id="N12345",
        name="Testbed Alpha",
        manufacturer="Aeris Aerospace",
        model="X-1",
        variant="Block 10",
        configuration_identifier="CONFIG-A-HEAVY",
        description="Primary test asset for high-alpha flight.",
        metadata={"tail_color": "red", "operator": "Flight Test Dept"},
    )
    assert v.vehicle_id == "N12345"
    assert v.name == "Testbed Alpha"
    assert v.manufacturer == "Aeris Aerospace"
    assert v.model == "X-1"
    assert v.variant == "Block 10"
    assert v.configuration_identifier == "CONFIG-A-HEAVY"
    assert v.description == "Primary test asset for high-alpha flight."
    assert v.metadata == {"tail_color": "red", "operator": "Flight Test Dept"}


def test_missing_vehicle_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Vehicle(name="Missing ID")  # type: ignore[call-arg]
    assert "Field required" in str(exc_info.value) or "vehicle_id" in str(exc_info.value)


def test_empty_vehicle_id() -> None:
    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        Vehicle(vehicle_id="")

    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        Vehicle(vehicle_id="   ")


def test_invalid_vehicle_id() -> None:
    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        Vehicle(vehicle_id="AC 123")  # Contains space

    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        Vehicle(vehicle_id="AC!123")  # Contains special char


def test_immutable_vehicle() -> None:
    v = Vehicle(vehicle_id="AC-123")
    with pytest.raises(ValidationError) as exc_info:
        v.name = "New Name"  # type: ignore[misc]
    assert "Instance is frozen" in str(exc_info.value)


def test_deterministic_serialization() -> None:
    v = Vehicle(vehicle_id="AC-123", manufacturer="Aeris")
    data = v.model_dump()
    assert data == {
        "vehicle_id": "AC-123",
        "name": None,
        "manufacturer": "Aeris",
        "model": None,
        "variant": None,
        "configuration_identifier": None,
        "description": None,
        "metadata": {},
    }

    json_str = v.model_dump_json()
    assert '"vehicle_id":"AC-123"' in json_str.replace(" ", "")
    assert '"manufacturer":"Aeris"' in json_str.replace(" ", "")


def test_distinct_vehicles_remain_distinct() -> None:
    v1 = Vehicle(vehicle_id="AC-001", configuration_identifier="CLEAN")
    v2 = Vehicle(vehicle_id="AC-002", configuration_identifier="CLEAN")
    v3 = Vehicle(vehicle_id="AC-001", configuration_identifier="STORE-1")

    assert v1 != v2
    assert v1 != v3
    assert v1.vehicle_id == "AC-001"
    assert v2.vehicle_id == "AC-002"


def test_configuration_identifier_preserved() -> None:
    v = Vehicle(vehicle_id="AC-123", configuration_identifier="STORES-DRAG-PROFILE-B")
    assert v.configuration_identifier == "STORES-DRAG-PROFILE-B"


def test_no_dataframe_dependency_and_no_telemetry_inference() -> None:
    """IMPORTANT NEGATIVE TEST: Passing a DataFrame to Vehicle fails or prevents inference."""
    df = pd.DataFrame(
        {
            "altitude_ft": [1000, 2000],
            "airspeed_kts": [100, 150],
            "engine_rpm": [2400, 2500],
        }
    )

    # 1. Attempting to pass as an attribute must fail (extra=forbid)
    with pytest.raises(ValidationError) as exc_info:
        Vehicle(vehicle_id="AC-123", df=df)  # type: ignore[call-arg]
    assert "Extra inputs are not permitted" in str(exc_info.value)

    # 2. Attempting to pass in metadata does not inspect or infer
    v = Vehicle(vehicle_id="AC-123", metadata={"telemetry": df})
    # Vehicle must not have created any "altitude" or "max_speed" limits
    assert not hasattr(v, "max_speed")
    assert not hasattr(v, "max_altitude")

    # DataFrame must remain completely untouched
    pd.testing.assert_frame_equal(df, v.metadata["telemetry"])


def test_no_parameter_registry_mutation() -> None:
    """IMPORTANT NEGATIVE TEST: Vehicle does not create or alter ParameterDefinitions."""
    registry = ParameterRegistry()
    registry.register(
        ParameterDefinition(
            parameter_id="airspeed",
            display_name="Airspeed",
            description="Airspeed",
            physical_quantity="velocity",
            data_type=ParameterDataType.FLOAT,
        )
    )

    # Vehicle creation does not touch the registry
    _v = Vehicle(vehicle_id="AC-123")

    # Registry must still have exactly 1 parameter
    params = registry.list()
    assert len(params) == 1
    assert params[0].parameter_id == "airspeed"


def test_no_performance_limits() -> None:
    """IMPORTANT NEGATIVE TEST: Do not allow speculative performance limits."""
    with pytest.raises(ValidationError) as exc_info:
        Vehicle(
            vehicle_id="AC-123",
            max_speed_kts=250.0,  # type: ignore[call-arg]
            service_ceiling=40000.0,  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_deterministic_repeated_construction() -> None:
    v1 = Vehicle(vehicle_id="AC-123", model="X-1")
    v2 = Vehicle(vehicle_id="AC-123", model="X-1")

    # Pydantic equality
    assert v1 == v2
    assert v1.model_dump() == v2.model_dump()
