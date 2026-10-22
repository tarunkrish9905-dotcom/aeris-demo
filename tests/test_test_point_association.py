"""Tests for Test Point Association Engine."""

import datetime
from uuid import uuid4

import pandas as pd
import pytest

from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import ResultStatus
from aeris.core.test_operations.association import (
    TestPointAssociationEngine,
)
from aeris.core.test_operations.test_point_observation import TestPointObservation


@pytest.fixture
def sample_data() -> pd.DataFrame:
    # 5 rows, 1 second apart, starting at 10:00:00 UTC
    times = pd.date_range(start="2026-01-01T10:00:00", periods=5, freq="s", tz="UTC")
    df = pd.DataFrame({"timestamp": times, "value": [10, 20, 30, 40, 50]})
    return df


@pytest.fixture
def engine() -> TestPointAssociationEngine:
    return TestPointAssociationEngine()


@pytest.fixture
def base_obs() -> TestPointObservation:
    t_start = datetime.datetime(2026, 1, 1, 10, 0, 1, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 10, 0, 3, tzinfo=datetime.UTC)
    return TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="clean_flight.csv",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )


def test_successful_extraction_and_result_status(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    res = engine.extract(sample_data, base_obs, "timestamp")
    assert res.status == ResultStatus.SUCCESS
    assert len(res.issues) == 0
    assert res.test_point_id == "TP-1"
    assert res.observation_id == "OBS-1"
    assert res.timestamp_column == "timestamp"

    # 10:00:01, 10:00:02, 10:00:03 -> 3 rows
    assert res.row_count == 3
    assert res.data is not None
    assert len(res.data) == 3


def test_exact_start_boundary_included(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine
) -> None:
    t_start = datetime.datetime(2026, 1, 1, 10, 0, 0, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 10, 0, 0, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    res = engine.extract(sample_data, obs, "timestamp")
    assert res.row_count == 1
    assert res.data is not None
    assert res.data["value"].iloc[0] == 10


def test_exact_end_boundary_included(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine
) -> None:
    t_start = datetime.datetime(2026, 1, 1, 10, 0, 4, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 10, 0, 4, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    res = engine.extract(sample_data, obs, "timestamp")
    assert res.row_count == 1
    assert res.data is not None
    assert res.data["value"].iloc[0] == 50


def test_both_boundaries_included(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    res = engine.extract(sample_data, base_obs, "timestamp")
    assert res.data is not None
    times = res.data["timestamp"].tolist()
    assert pd.Timestamp("2026-01-01T10:00:01Z") in times
    assert pd.Timestamp("2026-01-01T10:00:03Z") in times


def test_no_matching_rows(sample_data: pd.DataFrame, engine: TestPointAssociationEngine) -> None:
    t_start = datetime.datetime(2026, 1, 1, 11, 0, 0, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 11, 0, 5, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    res = engine.extract(sample_data, obs, "timestamp")
    assert res.status == ResultStatus.NOT_AVAILABLE
    assert res.row_count == 0
    assert res.data is None
    assert res.issues[0].issue_code == "ERR_NO_OBSERVED_ROWS_IN_INTERVAL"


def test_only_start_boundary_matched(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine
) -> None:
    t_start = datetime.datetime(2026, 1, 1, 10, 0, 4, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 11, 0, 0, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    res = engine.extract(sample_data, obs, "timestamp")
    assert res.row_count == 1  # only 10:00:04 matches


def test_only_end_boundary_matched(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine
) -> None:
    t_start = datetime.datetime(2026, 1, 1, 9, 0, 0, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 10, 0, 0, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    res = engine.extract(sample_data, obs, "timestamp")
    assert res.row_count == 1  # only 10:00:00 matches


def test_multiple_matching_rows(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    res = engine.extract(sample_data, base_obs, "timestamp")
    assert res.row_count > 1


def test_full_dataset_interval(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine
) -> None:
    t_start = datetime.datetime(2026, 1, 1, 9, 0, 0, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 11, 0, 0, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="src",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    res = engine.extract(sample_data, obs, "timestamp")
    assert res.row_count == 5


def test_unordered_source_preserves_order(
    engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    times = pd.to_datetime(
        [
            "2026-01-01T10:00:03Z",
            "2026-01-01T10:00:01Z",
            "2026-01-01T10:00:02Z",
        ],
        utc=True,
    )
    df = pd.DataFrame({"timestamp": times, "value": [30, 10, 20]})
    res = engine.extract(df, base_obs, "timestamp")
    assert res.data is not None
    # Order should be preserved exactly as input
    assert res.data["value"].tolist() == [30, 10, 20]


def test_duplicate_timestamps_preserved(
    engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    times = pd.to_datetime(
        [
            "2026-01-01T10:00:02Z",
            "2026-01-01T10:00:02Z",
        ],
        utc=True,
    )
    df = pd.DataFrame({"timestamp": times, "value": [20, 21]})
    res = engine.extract(df, base_obs, "timestamp")
    assert res.data is not None
    assert res.row_count == 2
    assert res.data["value"].tolist() == [20, 21]


def test_missing_timestamp_column(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    res = engine.extract(sample_data, base_obs, "wrong_column")
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_TIMESTAMP_COLUMN_MISSING"


def test_non_datetime_timestamp_column(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    sample_data["timestamp_str"] = sample_data["timestamp"].astype(str)
    res = engine.extract(sample_data, base_obs, "timestamp_str")
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_TIMESTAMP_COLUMN_INVALID"


def test_naive_timestamp_column(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    sample_data["naive_ts"] = sample_data["timestamp"].dt.tz_localize(None)
    res = engine.extract(sample_data, base_obs, "naive_ts")
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_TIMESTAMP_NAIVE"


def test_non_utc_timestamp_column(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    sample_data["offset_ts"] = sample_data["timestamp"].dt.tz_convert("Asia/Tokyo")
    res = engine.extract(sample_data, base_obs, "offset_ts")
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_TIMESTAMP_NON_UTC"


def test_empty_dataframe(
    engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    df = pd.DataFrame()
    res = engine.extract(df, base_obs, "timestamp")
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_NO_DATA"


def test_source_dataframe_immutability(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    original_copy = sample_data.copy(deep=True)
    res = engine.extract(sample_data, base_obs, "timestamp")
    # Mutate the result
    assert res.data is not None
    res.data.loc[:, "value"] = 999

    # Source must remain unchanged
    pd.testing.assert_frame_equal(sample_data, original_copy)
    assert (sample_data["value"] != 999).all()


def test_extracted_dataframe_independence(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    res = engine.extract(sample_data, base_obs, "timestamp")
    assert res.data is not None

    # Mutating source must not affect result
    sample_data.loc[:, "value"] = 888
    assert (res.data["value"] != 888).all()


def test_explicit_timestamp_column_requirement(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    # Must pass string explicitly
    with pytest.raises(TypeError):
        engine.extract(sample_data, base_obs)  # type: ignore[call-arg]


def test_provenance_exists_and_is_calculated(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    parent_prov = ProvenanceRecord(
        record_id=str(uuid4()),
        epistemic_level=EpistemicLevel.OBSERVED,
        activity_name="normalization",
        source_uri="clean_flight.csv",
    )
    res = engine.extract(sample_data, base_obs, "timestamp", parent_provenance=parent_prov)

    assert res.provenance is not None
    assert res.provenance.epistemic_level == EpistemicLevel.CALCULATED
    assert res.provenance.activity_name == "test_point_association"
    assert parent_prov.record_id in res.provenance.parent_record_ids

    # records extraction info
    assert res.provenance.parameters["observation_id"] == "OBS-1"
    assert res.provenance.parameters["extracted_row_count"] == 3


def test_no_engineering_calculations_or_registry(
    sample_data: pd.DataFrame, engine: TestPointAssociationEngine, base_obs: TestPointObservation
) -> None:
    res = engine.extract(sample_data, base_obs, "timestamp")
    # Pure slicing, no engineering fields appended
    assert "mean" not in res.data.columns
    assert "anomaly" not in res.data.columns
    assert "pass_fail" not in dir(res)
    assert not hasattr(engine, "parameter_registry")
