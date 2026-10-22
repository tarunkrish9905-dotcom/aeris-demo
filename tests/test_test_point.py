"""Tests for the TestPoint domain model and boundaries."""

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry
from aeris.core.test_operations.test_point import TestPoint


def test_valid_minimal_test_point() -> None:
    tp = TestPoint(test_point_id="TP-001", context_id="CTX-001", name="Engine Run")
    assert tp.test_point_id == "TP-001"
    assert tp.context_id == "CTX-001"
    assert tp.name == "Engine Run"
    assert tp.description is None
    assert tp.point_type is None
    assert tp.metadata == {}


def test_valid_populated_test_point() -> None:
    tp = TestPoint(
        test_point_id="TP-001",
        context_id="CTX-001",
        name="High Alpha Trim",
        description="Trim condition at 15 degrees AOA",
        point_type="handling_qualities",
        metadata={"target_airspeed": "150kts", "priority": "high"},
    )
    assert tp.test_point_id == "TP-001"
    assert tp.context_id == "CTX-001"
    assert tp.name == "High Alpha Trim"
    assert tp.description == "Trim condition at 15 degrees AOA"
    assert tp.point_type == "handling_qualities"
    assert tp.metadata == {"target_airspeed": "150kts", "priority": "high"}


def test_missing_test_point_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestPoint(context_id="CTX-001", name="Engine Run")  # type: ignore[call-arg]
    assert "test_point_id" in str(exc_info.value)


def test_missing_context_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestPoint(test_point_id="TP-001", name="Engine Run")  # type: ignore[call-arg]
    assert "context_id" in str(exc_info.value)


def test_missing_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestPoint(test_point_id="TP-001", context_id="CTX-001")  # type: ignore[call-arg]
    assert "name" in str(exc_info.value)


def test_empty_invalid_identifiers() -> None:
    # Empty
    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        TestPoint(test_point_id="", context_id="CTX-001", name="A")
    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        TestPoint(test_point_id="TP-001", context_id="   ", name="A")

    # Invalid characters
    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        TestPoint(test_point_id="TP 001", context_id="CTX-001", name="A")
    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        TestPoint(test_point_id="TP-001", context_id="CTX!001", name="A")


def test_immutable_test_point() -> None:
    tp = TestPoint(test_point_id="TP-001", context_id="CTX-001", name="Engine Run")
    with pytest.raises(ValidationError) as exc_info:
        tp.name = "New Name"  # type: ignore[misc]
    assert "Instance is frozen" in str(exc_info.value)


def test_deterministic_serialization() -> None:
    tp = TestPoint(test_point_id="TP-001", context_id="CTX-001", name="Test")
    data = tp.model_dump()
    assert data == {
        "test_point_id": "TP-001",
        "context_id": "CTX-001",
        "name": "Test",
        "description": None,
        "point_type": None,
        "metadata": {},
    }

    json_str = tp.model_dump_json()
    assert '"test_point_id":"TP-001"' in json_str.replace(" ", "")
    assert '"context_id":"CTX-001"' in json_str.replace(" ", "")


def test_repeated_construction_produces_equivalent_output() -> None:
    tp1 = TestPoint(test_point_id="TP-1", context_id="C-1", name="A")
    tp2 = TestPoint(test_point_id="TP-1", context_id="C-1", name="A")

    assert tp1 == tp2
    assert tp1.model_dump() == tp2.model_dump()


def test_distinct_test_points_remain_distinct() -> None:
    tp1 = TestPoint(test_point_id="TP-1", context_id="C-1", name="A")
    tp2 = TestPoint(test_point_id="TP-2", context_id="C-1", name="A")
    tp3 = TestPoint(test_point_id="TP-1", context_id="C-2", name="A")

    assert tp1 != tp2
    assert tp1 != tp3
    assert tp1.test_point_id == "TP-1"
    assert tp2.test_point_id == "TP-2"


def test_metadata_preservation() -> None:
    tp = TestPoint(
        test_point_id="TP-1",
        context_id="C-1",
        name="A",
        metadata={"nested": {"key": "value"}},
    )
    assert tp.metadata["nested"]["key"] == "value"


def test_no_vehicle_or_configuration_dependency() -> None:
    """TestPoint requires no object graphs for contexts/vehicles."""
    # We do not have to pass a Vehicle or VehicleConfiguration to create a TestPoint.
    tp = TestPoint(test_point_id="TP-1", context_id="C-1", name="A")
    assert not hasattr(tp, "vehicle")
    assert not hasattr(tp, "configuration")


def test_no_parameter_registry_dependency() -> None:
    """TestPoint requires no ParameterRegistry and modifies none."""
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

    _tp = TestPoint(test_point_id="TP-1", context_id="C-1", name="A")

    # Registry must remain exactly as it was
    params = registry.list()
    assert len(params) == 1
    assert params[0].parameter_id == "engine_rpm"
    # TestPoint has no parameter references internally
    assert not hasattr(_tp, "parameter_registry")
    assert not hasattr(_tp, "parameter_ids")


def test_no_pandas_dependency() -> None:
    """IMPORTANT NEGATIVE TEST: DataFrame inside metadata must not influence point."""
    df = pd.DataFrame({"altitude_ft": [1000, 2000]})
    tp = TestPoint(
        test_point_id="TP-1",
        context_id="C-1",
        name="A",
        metadata={"data": df},
    )

    assert not hasattr(tp, "telemetry")
    pd.testing.assert_frame_equal(df, tp.metadata["data"])


def test_rejection_of_telemetry_fields() -> None:
    """IMPORTANT NEGATIVE TEST: Telemetry fields cannot be directly assigned."""
    with pytest.raises(ValidationError) as exc_info:
        TestPoint(
            test_point_id="TP-1",
            context_id="C-1",
            name="A",
            altitude_ft=[1000, 2000],  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)
    assert "altitude_ft" in str(exc_info.value)


def test_rejection_of_measured_value_fields() -> None:
    """IMPORTANT NEGATIVE TEST: Measured values cannot be directly assigned."""
    with pytest.raises(ValidationError) as exc_info:
        TestPoint(
            test_point_id="TP-1",
            context_id="C-1",
            name="A",
            measured_speed=150.0,  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)
    assert "measured_speed" in str(exc_info.value)


def test_rejection_of_engineering_limit_fields() -> None:
    """IMPORTANT NEGATIVE TEST: Engineering limit fields are forbidden."""
    with pytest.raises(ValidationError) as exc_info:
        TestPoint(
            test_point_id="TP-1",
            context_id="C-1",
            name="A",
            max_altitude=50000.0,  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)
    assert "max_altitude" in str(exc_info.value)


def test_rejection_of_analysis_result_fields() -> None:
    """IMPORTANT NEGATIVE TEST: Analysis results fields are forbidden."""
    with pytest.raises(ValidationError) as exc_info:
        TestPoint(
            test_point_id="TP-1",
            context_id="C-1",
            name="A",
            anomaly_score=0.9,  # type: ignore[call-arg]
            pass_fail="PASS",  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)
    assert "anomaly_score" in str(exc_info.value)
