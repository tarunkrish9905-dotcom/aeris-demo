"""Tests for the TestCard domain model and boundaries."""

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry
from aeris.core.test_operations.test_card import TestCard


def test_valid_minimal_test_card() -> None:
    tc = TestCard(test_card_id="TC-001", context_id="CTX-001", name="Basic Flight")
    assert tc.test_card_id == "TC-001"
    assert tc.context_id == "CTX-001"
    assert tc.name == "Basic Flight"
    assert tc.description is None
    assert tc.objective is None
    assert tc.test_point_ids == ()
    assert tc.metadata == {}


def test_valid_populated_test_card() -> None:
    tc = TestCard(
        test_card_id="TC-001",
        context_id="CTX-001",
        name="High Alpha Block",
        description="Executing maneuvers at 15-25 deg AOA.",
        objective="Validate stability margins.",
        test_point_ids=("TP-01", "TP-02"),
        metadata={"priority": "high"},
    )
    assert tc.test_card_id == "TC-001"
    assert tc.context_id == "CTX-001"
    assert tc.name == "High Alpha Block"
    assert tc.description == "Executing maneuvers at 15-25 deg AOA."
    assert tc.objective == "Validate stability margins."
    assert tc.test_point_ids == ("TP-01", "TP-02")
    assert tc.metadata == {"priority": "high"}


def test_missing_test_card_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestCard(context_id="CTX-001", name="Name")  # type: ignore[call-arg]
    assert "test_card_id" in str(exc_info.value)


def test_missing_context_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestCard(test_card_id="TC-001", name="Name")  # type: ignore[call-arg]
    assert "context_id" in str(exc_info.value)


def test_missing_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestCard(test_card_id="TC-001", context_id="CTX-001")  # type: ignore[call-arg]
    assert "name" in str(exc_info.value)


def test_invalid_empty_identifiers() -> None:
    # Empty strings
    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        TestCard(test_card_id="", context_id="CTX-001", name="A")
    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        TestCard(test_card_id="TC-001", context_id="   ", name="A")
    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        TestCard(test_card_id="TC-001", context_id="CTX-001", name="   ")

    # Invalid characters in ID
    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        TestCard(test_card_id="TC 001", context_id="CTX-001", name="A")


def test_immutable_test_card() -> None:
    tc = TestCard(test_card_id="TC-001", context_id="CTX-001", name="A")
    with pytest.raises(ValidationError) as exc_info:
        tc.name = "New Name"  # type: ignore[misc]
    assert "Instance is frozen" in str(exc_info.value)


def test_deterministic_serialization() -> None:
    tc = TestCard(
        test_card_id="TC-001",
        context_id="CTX-001",
        name="A",
        test_point_ids=("TP-1", "TP-2"),
    )
    data = tc.model_dump()
    assert data == {
        "test_card_id": "TC-001",
        "context_id": "CTX-001",
        "name": "A",
        "description": None,
        "objective": None,
        "test_point_ids": ("TP-1", "TP-2"),
        "metadata": {},
    }

    json_str = tc.model_dump_json()
    assert '"test_card_id":"TC-001"' in json_str.replace(" ", "")
    assert '"TP-1","TP-2"' in json_str.replace(" ", "")


def test_repeated_construction_equivalence() -> None:
    tc1 = TestCard(test_card_id="TC-1", context_id="C-1", name="A")
    tc2 = TestCard(test_card_id="TC-1", context_id="C-1", name="A")

    assert tc1 == tc2
    assert tc1.model_dump() == tc2.model_dump()


def test_distinct_test_cards_remain_distinct() -> None:
    tc1 = TestCard(test_card_id="TC-1", context_id="C-1", name="A")
    tc2 = TestCard(test_card_id="TC-2", context_id="C-1", name="A")
    tc3 = TestCard(test_card_id="TC-1", context_id="C-2", name="A")

    assert tc1 != tc2
    assert tc1 != tc3


def test_metadata_preservation() -> None:
    tc = TestCard(
        test_card_id="TC-1",
        context_id="C-1",
        name="A",
        metadata={"nested": {"key": "value"}},
    )
    assert tc.metadata["nested"]["key"] == "value"


def test_empty_test_point_ids_allowed() -> None:
    tc = TestCard(test_card_id="TC-1", context_id="C-1", name="A")
    assert tc.test_point_ids == ()


def test_ordered_test_point_ids_preserved() -> None:
    tc = TestCard(
        test_card_id="TC-1",
        context_id="C-1",
        name="A",
        test_point_ids=("TP-3", "TP-1", "TP-2"),
    )
    assert tc.test_point_ids == ("TP-3", "TP-1", "TP-2")


def test_duplicate_test_point_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate test_point_id found: TP-1"):
        TestCard(
            test_card_id="TC-1",
            context_id="C-1",
            name="A",
            test_point_ids=("TP-1", "TP-2", "TP-1"),
        )


def test_invalid_test_point_id_rejected() -> None:
    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        TestCard(
            test_card_id="TC-1",
            context_id="C-1",
            name="A",
            test_point_ids=("TP!1",),
        )


def test_no_vehicle_or_configuration_dependency() -> None:
    tc = TestCard(test_card_id="TC-1", context_id="C-1", name="A")
    assert not hasattr(tc, "vehicle")
    assert not hasattr(tc, "configuration")


def test_no_test_context_dependency() -> None:
    # We do not have to pass a TestContext object to create a TestCard.
    tc = TestCard(test_card_id="TC-1", context_id="C-1", name="A")
    assert not hasattr(tc, "context")


def test_no_parameter_registry_dependency() -> None:
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

    _tc = TestCard(test_card_id="TC-1", context_id="C-1", name="A")

    params = registry.list()
    assert len(params) == 1
    assert params[0].parameter_id == "engine_rpm"
    assert not hasattr(_tc, "parameter_registry")


def test_no_pandas_dependency() -> None:
    df = pd.DataFrame({"altitude_ft": [1000, 2000]})
    tc = TestCard(
        test_card_id="TC-1",
        context_id="C-1",
        name="A",
        metadata={"data": df},
    )

    assert not hasattr(tc, "telemetry")
    pd.testing.assert_frame_equal(df, tc.metadata["data"])


def test_rejection_of_telemetry_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestCard(
            test_card_id="TC-1",
            context_id="C-1",
            name="A",
            altitude_ft=[1000, 2000],  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_rejection_of_timestamp_temporal_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestCard(
            test_card_id="TC-1",
            context_id="C-1",
            name="A",
            start_timestamp="2026-10-06T10:00:00Z",  # type: ignore[call-arg]
            end_timestamp="2026-10-06T11:00:00Z",  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_rejection_of_measured_value_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestCard(
            test_card_id="TC-1",
            context_id="C-1",
            name="A",
            measured_speed=150.0,  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_rejection_of_engineering_limit_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestCard(
            test_card_id="TC-1",
            context_id="C-1",
            name="A",
            max_altitude=50000.0,  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_rejection_of_analysis_result_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TestCard(
            test_card_id="TC-1",
            context_id="C-1",
            name="A",
            pass_fail="PASS",  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)
