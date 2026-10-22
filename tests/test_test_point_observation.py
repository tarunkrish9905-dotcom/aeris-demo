"""Tests for the TestPointObservation domain model and boundaries."""

import datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry
from aeris.core.test_operations.test_point_observation import TestPointObservation


def test_valid_observation() -> None:
    t_start = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 12, 5, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-001",
        test_point_id="TP-001",
        source_reference="clean_flight.csv",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    assert obs.observation_id == "OBS-001"
    assert obs.test_point_id == "TP-001"
    assert obs.source_reference == "clean_flight.csv"
    assert obs.start_timestamp == t_start
    assert obs.end_timestamp == t_end


def test_valid_utc_timestamps() -> None:
    # Explicitly UTC
    t_start = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 11, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    assert obs.start_timestamp.tzinfo == datetime.UTC


def test_equal_start_end_timestamp_accepted() -> None:
    t_stamp = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_stamp,
        end_timestamp=t_stamp,
    )
    assert obs.start_timestamp == obs.end_timestamp


def test_start_timestamp_before_end_accepted() -> None:
    t_start = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 11, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    assert obs.start_timestamp < obs.end_timestamp


def test_start_after_end_rejected() -> None:
    t_start = datetime.datetime(2026, 1, 1, 11, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    with pytest.raises(
        ValidationError, match="start_timestamp must be less than or equal to end_timestamp"
    ):
        TestPointObservation(
            observation_id="OBS-1",
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_start,
            end_timestamp=t_end,
        )


def test_naive_start_timestamp_rejected() -> None:
    t_naive = datetime.datetime(2026, 1, 1, 10)  # No tzinfo
    t_end = datetime.datetime(2026, 1, 1, 11, tzinfo=datetime.UTC)
    with pytest.raises(ValidationError, match="start_timestamp must be timezone-aware"):
        TestPointObservation(
            observation_id="OBS-1",
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_naive,
            end_timestamp=t_end,
        )


def test_naive_end_timestamp_rejected() -> None:
    t_start = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    t_naive = datetime.datetime(2026, 1, 1, 11)  # No tzinfo
    with pytest.raises(ValidationError, match="end_timestamp must be timezone-aware"):
        TestPointObservation(
            observation_id="OBS-1",
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_start,
            end_timestamp=t_naive,
        )


def test_non_utc_timestamp_policy() -> None:
    # We strictly reject non-UTC timezones (e.g., UTC+1)
    tz_offset = datetime.timezone(datetime.timedelta(hours=1))
    t_non_utc = datetime.datetime(2026, 1, 1, 10, tzinfo=tz_offset)
    t_end = datetime.datetime(2026, 1, 1, 11, tzinfo=datetime.UTC)

    with pytest.raises(ValidationError, match="must explicitly use UTC timezone"):
        TestPointObservation(
            observation_id="OBS-1",
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_non_utc,
            end_timestamp=t_end,
        )


def test_missing_fields() -> None:
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    with pytest.raises(ValidationError, match="observation_id"):
        TestPointObservation(
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_val,
            end_timestamp=t_val,
        )  # type: ignore[call-arg]

    with pytest.raises(ValidationError, match="test_point_id"):
        TestPointObservation(
            observation_id="OBS-1",
            source_reference="src",
            start_timestamp=t_val,
            end_timestamp=t_val,
        )  # type: ignore[call-arg]

    with pytest.raises(ValidationError, match="source_reference"):
        TestPointObservation(
            observation_id="OBS-1",
            test_point_id="TP-1",
            start_timestamp=t_val,
            end_timestamp=t_val,
        )  # type: ignore[call-arg]


def test_invalid_and_empty_identifiers() -> None:
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)

    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        TestPointObservation(
            observation_id="",
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_val,
            end_timestamp=t_val,
        )

    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        TestPointObservation(
            observation_id="OBS 001",
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_val,
            end_timestamp=t_val,
        )


def test_immutable_observation() -> None:
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_val,
        end_timestamp=t_val,
    )
    with pytest.raises(ValidationError, match="Instance is frozen"):
        obs.source_reference = "new_src"  # type: ignore[misc]


def test_deterministic_serialization() -> None:
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_val,
        end_timestamp=t_val,
    )
    data = obs.model_dump()
    assert data["observation_id"] == "OBS-1"
    assert "start_timestamp" in data

    json_str = obs.model_dump_json()
    assert '"observation_id":"OBS-1"' in json_str.replace(" ", "")
    assert "2026-01-01T10:00:00Z" in json_str


def test_repeated_construction_equivalence() -> None:
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    obs1 = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_val,
        end_timestamp=t_val,
    )
    obs2 = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_val,
        end_timestamp=t_val,
    )
    assert obs1 == obs2


def test_no_pandas_dependency_and_no_dataframe_acceptance() -> None:
    df = pd.DataFrame({"time": [1, 2]})
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TestPointObservation(
            observation_id="OBS-1",
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_val,
            end_timestamp=t_val,
            telemetry_data=df,  # type: ignore[call-arg]
        )


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
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    _obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_val,
        end_timestamp=t_val,
    )
    assert len(registry.list()) == 1
    assert not hasattr(_obs, "parameter_registry")


def test_rejection_of_measured_value_fields() -> None:
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TestPointObservation(
            observation_id="OBS-1",
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_val,
            end_timestamp=t_val,
            measured_altitude=5000,  # type: ignore[call-arg]
        )


def test_rejection_of_engineering_limit_fields() -> None:
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TestPointObservation(
            observation_id="OBS-1",
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_val,
            end_timestamp=t_val,
            max_temperature=120.0,  # type: ignore[call-arg]
        )


def test_rejection_of_analysis_result_fields() -> None:
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TestPointObservation(
            observation_id="OBS-1",
            test_point_id="TP-1",
            source_reference="src",
            start_timestamp=t_val,
            end_timestamp=t_val,
            pass_fail="PASS",  # type: ignore[call-arg]
        )


def test_source_reference_is_opaque_and_no_filesystem_access() -> None:
    """The source_reference field does not try to validate existence on disk."""
    t_val = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="non_existent_file.csv",
        start_timestamp=t_val,
        end_timestamp=t_val,
    )
    assert obs.source_reference == "non_existent_file.csv"


def test_temporal_interval_semantics_are_explicitly_tested() -> None:
    """The interval is a closed bound [start, end]."""
    t_start = datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 11, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    # Testing that it physically models the inclusive range by validating the object
    # does not reduce or open the bounds on construction.
    assert obs.start_timestamp == t_start
    assert obs.end_timestamp == t_end
