import datetime
import pathlib

import pytest

from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.pipeline.orchestrator import TestPointAssociationRequest, run_csv_pipeline
from aeris.core.results.types import ResultStatus
from aeris.core.statistics.models import MetricType
from aeris.core.statistics.time_series.models import (
    TimeSeriesAnalysisRequest,
    WindowConfiguration,
    WindowType,
)
from aeris.core.test_operations.test_point_observation import TestPointObservation


@pytest.fixture
def sample_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    csv_file = tmp_path / "ts_test.csv"
    data = (
        "timestamp,altitude_ft,airspeed_kts,pitch_deg,roll_deg,engine_rpm,target_val\n"
        "2026-01-01T12:00:00Z,1000,100,1,1,2000,10.0\n"
        "2026-01-01T12:00:10Z,1010,110,2,2,2000,20.0\n"
        "2026-01-01T12:00:20Z,1020,120,3,3,2000,\n"
        "2026-01-01T12:00:30Z,1030,130,4,4,2000,40.0\n"
        "2026-01-01T12:00:40Z,1040,140,5,5,2000,50.0\n"
    )
    csv_file.write_text(data)
    return csv_file


def test_time_series_request_absent(sample_csv):
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req)
    assert res.time_series is None
    assert res.status == ResultStatus.WARNING


def test_time_series_request_present_raw_ingestion(sample_csv):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts_req_1",
        target_column="target_val",
        timestamp_column="timestamp",
        window_configuration=WindowConfiguration(
            window_type=WindowType.OBSERVATION_COUNT, window_size_observations=2
        ),
        requested_metrics=[MetricType.COUNT, MetricType.MEAN],
    )
    # The pipeline leaves timestamps as strings without normalization
    res = run_csv_pipeline(sample_csv, time_series_request=req)

    assert res.time_series is not None
    assert res.time_series.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_INVALID_TIMESTAMP_DTYPE" for i in res.time_series.issues)
    assert res.ingestion.provenance.record_id in res.time_series.provenance.parent_record_ids


def test_time_series_request_with_normalization(sample_csv):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts_req_1",
        target_column="NormTarget",
        timestamp_column="timestamp",
        requested_metrics=[MetricType.COUNT],
        window_configuration=WindowConfiguration(
            window_type=WindowType.DURATION, window_size_seconds=20.0
        ),
    )
    norm_req = NormalizationRequest()
    norm_req.rename_column("target_val", "NormTarget")
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, time_series_request=req)

    assert res.time_series is not None
    assert res.time_series.status == ResultStatus.SUCCESS
    # Should point to normalization output
    assert res.normalization.provenance.record_id in res.time_series.provenance.parent_record_ids


def test_time_series_request_with_association(sample_csv):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts_req_1",
        target_column="target_val",
        timestamp_column="timestamp",
        requested_metrics=[MetricType.MEAN],
    )
    obs = TestPointObservation(
        observation_id="obs1",
        test_point_id="tp1",
        source_reference="some_ref",
        start_timestamp=datetime.datetime(2026, 1, 1, 12, 0, 15, tzinfo=datetime.UTC),
        end_timestamp=datetime.datetime(2026, 1, 1, 12, 0, 35, tzinfo=datetime.UTC),
    )
    assoc_req = TestPointAssociationRequest(
        observation=obs,
        timestamp_column="timestamp",
    )
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(
        sample_csv,
        normalization_request=norm_req,
        association_request=assoc_req,
        time_series_request=req,
    )

    assert res.time_series is not None
    # time_series engine operates on association output (which contains 2 rows: 12:00:20 and 12:00:30)
    assert res.time_series.observation_count == 2
    assert res.association.provenance.record_id in res.time_series.provenance.parent_record_ids


def test_time_series_engine_error_preservation(sample_csv):
    # Pass an invalid column to trigger an internal engine error (ERR_MISSING_TARGET_COLUMN)
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts_req_1",
        target_column="nonexistent",
        timestamp_column="timestamp",
    )
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, time_series_request=req)

    assert res.time_series is not None
    assert res.time_series.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_MISSING_TARGET_COLUMN" for i in res.time_series.issues)
    assert res.status == ResultStatus.ERROR


def test_no_silent_fallback_after_normalization_failure(sample_csv):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts_req_1",
        target_column="target_val",
        timestamp_column="timestamp",
    )
    # This will fail normalization
    norm_req = NormalizationRequest()
    norm_req.rename_column("nonexistent", "NormTarget")
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, time_series_request=req)

    assert res.normalization.status == ResultStatus.ERROR
    # Time-series should fail because input is unavailable
    assert res.time_series is not None
    assert res.time_series.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_TIME_SERIES_DATA_UNAVAILABLE" for i in res.time_series.issues)


def test_no_silent_fallback_after_association_failure(sample_csv):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts_req_1",
        target_column="target_val",
        timestamp_column="timestamp",
    )
    # This will fail association (missing timestamp column)
    obs = TestPointObservation(
        observation_id="obs1",
        test_point_id="tp1",
        source_reference="some_ref",
        start_timestamp=datetime.datetime(2026, 1, 1, 12, 0, 15, tzinfo=datetime.UTC),
        end_timestamp=datetime.datetime(2026, 1, 1, 12, 0, 35, tzinfo=datetime.UTC),
    )
    assoc_req = TestPointAssociationRequest(
        observation=obs,
        timestamp_column="wrong_time_col",
    )
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(
        sample_csv,
        normalization_request=norm_req,
        association_request=assoc_req,
        time_series_request=req,
    )

    assert res.association.status == ResultStatus.ERROR
    assert res.time_series is not None
    assert res.time_series.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_TIME_SERIES_DATA_UNAVAILABLE" for i in res.time_series.issues)


def test_input_immutability_integration(sample_csv):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts_req_1",
        target_column="target_val",
        timestamp_column="timestamp",
        requested_metrics=[MetricType.COUNT],
        window_configuration=WindowConfiguration(
            window_type=WindowType.OBSERVATION_COUNT, window_size_observations=2
        ),
    )
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, time_series_request=req)

    df = res.ingestion.data
    assert list(df.columns) == [
        "timestamp",
        "altitude_ft",
        "airspeed_kts",
        "pitch_deg",
        "roll_deg",
        "engine_rpm",
        "target_val",
    ]
    assert res.time_series is not None
    assert res.time_series.observation_count == 5
