"""Tests for VehicleConfiguration domain models and boundaries."""

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry
from aeris.core.vehicle.configuration import VehicleConfiguration
from aeris.core.vehicle.models import Vehicle


def test_valid_minimal_configuration() -> None:
    c = VehicleConfiguration(configuration_id="CLEAN-01")
    assert c.configuration_id == "CLEAN-01"
    assert c.vehicle_id is None
    assert c.name is None
    assert c.description is None
    assert c.metadata == {}


def test_valid_populated_configuration() -> None:
    c = VehicleConfiguration(
        configuration_id="STORES-DRAG-B",
        vehicle_id="AC-123",
        name="Stores Config B",
        description="Configuration with heavy external wing stores.",
        metadata={"cg_station": 45.2, "weight_class": "heavy"},
    )
    assert c.configuration_id == "STORES-DRAG-B"
    assert c.vehicle_id == "AC-123"
    assert c.name == "Stores Config B"
    assert c.description == "Configuration with heavy external wing stores."
    assert c.metadata == {"cg_station": 45.2, "weight_class": "heavy"}


def test_missing_configuration_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        VehicleConfiguration(name="Missing ID")  # type: ignore[call-arg]
    assert "Field required" in str(exc_info.value) or "configuration_id" in str(exc_info.value)


def test_empty_configuration_id() -> None:
    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        VehicleConfiguration(configuration_id="")

    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        VehicleConfiguration(configuration_id="   ")


def test_invalid_configuration_id() -> None:
    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        VehicleConfiguration(configuration_id="CLEAN 01")

    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        VehicleConfiguration(configuration_id="CLEAN!01")


def test_immutable_configuration() -> None:
    c = VehicleConfiguration(configuration_id="CLEAN-01")
    with pytest.raises(ValidationError) as exc_info:
        c.name = "New Name"  # type: ignore[misc]
    assert "Instance is frozen" in str(exc_info.value)


def test_deterministic_serialization() -> None:
    c = VehicleConfiguration(configuration_id="CLEAN-01", name="Clean")
    data = c.model_dump()
    assert data == {
        "configuration_id": "CLEAN-01",
        "vehicle_id": None,
        "name": "Clean",
        "description": None,
        "metadata": {},
    }

    json_str = c.model_dump_json()
    assert '"configuration_id":"CLEAN-01"' in json_str.replace(" ", "")
    assert '"name":"Clean"' in json_str.replace(" ", "")


def test_distinct_configurations_remain_distinct() -> None:
    c1 = VehicleConfiguration(configuration_id="CONF-A", description="Alpha")
    c2 = VehicleConfiguration(configuration_id="CONF-B", description="Alpha")
    c3 = VehicleConfiguration(configuration_id="CONF-A", description="Beta")

    assert c1 != c2
    assert c1 != c3
    assert c1.configuration_id == "CONF-A"
    assert c2.configuration_id == "CONF-B"


def test_metadata_preservation() -> None:
    c = VehicleConfiguration(configuration_id="CONF-A", metadata={"nested": {"key": "value"}})
    assert c.metadata["nested"]["key"] == "value"


def test_telemetry_independence_and_dataframe_rejection() -> None:
    """IMPORTANT NEGATIVE TEST: DataFrame must not mutate or define the config."""
    df = pd.DataFrame(
        {
            "altitude_ft": [1000, 2000],
            "airspeed_kts": [100, 150],
        }
    )

    # 1. Reject df passed directly as an attribute due to strict boundary (extra=forbid)
    with pytest.raises(ValidationError) as exc_info:
        VehicleConfiguration(configuration_id="CONF-A", df=df)  # type: ignore[call-arg]
    assert "Extra inputs are not permitted" in str(exc_info.value)

    # 2. DataFrame inside metadata must remain uninspected and unmutated
    c = VehicleConfiguration(configuration_id="CONF-A", metadata={"data": df})

    # Must not infer "max_altitude" or "stall_speed"
    assert not hasattr(c, "max_altitude")
    assert not hasattr(c, "stall_speed_kts")

    # DataFrame contents are completely undisturbed
    pd.testing.assert_frame_equal(df, c.metadata["data"])


def test_parameter_registry_independence() -> None:
    """IMPORTANT NEGATIVE TEST: Config creation does not touch ParameterRegistry."""
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

    _c = VehicleConfiguration(configuration_id="CONF-A")

    # Registry must remain exactly exactly as it was
    params = registry.list()
    assert len(params) == 1
    assert params[0].parameter_id == "engine_rpm"


def test_rejection_of_performance_limit_fields() -> None:
    """IMPORTANT NEGATIVE TEST: Structural performance fields are forbidden at this level."""
    with pytest.raises(ValidationError) as exc_info:
        VehicleConfiguration(
            configuration_id="CONF-A",
            max_speed_kts=300.0,  # type: ignore[call-arg]
            service_ceiling_ft=50000.0,  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)
    assert "max_speed_kts" in str(exc_info.value)


def test_deterministic_repeated_construction() -> None:
    c1 = VehicleConfiguration(configuration_id="CONF-A", name="A")
    c2 = VehicleConfiguration(configuration_id="CONF-A", name="A")

    assert c1 == c2
    assert c1.model_dump() == c2.model_dump()


def test_configuration_independent_of_vehicle_object() -> None:
    """Test Vehicle and VehicleConfiguration exist independently without embedding."""
    v = Vehicle(vehicle_id="AC-1")
    c = VehicleConfiguration(configuration_id="CONF-A", vehicle_id="AC-1")

    # Configuration has explicit reference via string, not object embedding
    assert c.vehicle_id == v.vehicle_id
    assert not hasattr(c, "vehicle")  # No auto-injected object graph
    assert not hasattr(v, "configurations")  # No auto-injected object graph
