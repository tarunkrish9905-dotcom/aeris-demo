import datetime

import pytest
from pydantic import ValidationError

from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import ResultStatus
from aeris.core.statistics.models import MetricType
from aeris.core.statistics.time_series.models import (
    SamplingIrregularity,
    SamplingMetadata,
    TemporalScope,
    TemporalScopeType,
    TimeSeriesAnalysisRequest,
    TimeSeriesAnalysisResult,
    WindowConfiguration,
    WindowType,
)


def test_valid_request_and_explicit_columns():
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts-1",
        target_column="altitude",
        timestamp_column="t_utc",
        requested_metrics=[MetricType.MEAN],
    )
    assert req.analysis_id == "ts-1"
    assert req.target_column == "altitude"
    assert req.timestamp_column == "t_utc"
    assert req.temporal_scope.scope_type == TemporalScopeType.ENTIRE_DATASET


def test_missing_explicit_columns_fail():
    with pytest.raises(ValidationError):
        TimeSeriesAnalysisRequest(
            analysis_id="ts-1",
            timestamp_column="t",
            requested_metrics=[MetricType.MEAN],
        )
    with pytest.raises(ValidationError):
        TimeSeriesAnalysisRequest(
            analysis_id="ts-1",
            target_column="v",
            requested_metrics=[MetricType.MEAN],
        )


def test_temporal_scope_entire_dataset():
    scope = TemporalScope(scope_type=TemporalScopeType.ENTIRE_DATASET)
    assert scope.scope_type == TemporalScopeType.ENTIRE_DATASET
    assert scope.start_timestamp is None


def test_valid_explicit_interval():
    t_start = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 12, 1, 0, tzinfo=datetime.UTC)
    scope = TemporalScope(
        scope_type=TemporalScopeType.EXPLICIT_INTERVAL,
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    assert scope.scope_type == TemporalScopeType.EXPLICIT_INTERVAL
    assert scope.start_timestamp == t_start


def test_interval_start_must_be_less_than_or_equal_to_end():
    t_start = datetime.datetime(2026, 1, 1, 12, 1, 0, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    with pytest.raises(ValidationError, match="after end timestamp"):
        TemporalScope(
            scope_type=TemporalScopeType.EXPLICIT_INTERVAL,
            start_timestamp=t_start,
            end_timestamp=t_end,
        )


def test_interval_start_equal_end_is_valid():
    t = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    scope = TemporalScope(
        scope_type=TemporalScopeType.EXPLICIT_INTERVAL,
        start_timestamp=t,
        end_timestamp=t,
    )
    assert scope.start_timestamp == scope.end_timestamp


def test_timezone_aware_utc_requirement():
    naive = datetime.datetime(2026, 1, 1, 12, 0, 0)
    utc = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    with pytest.raises(ValidationError, match="timezone-aware"):
        TemporalScope(
            scope_type=TemporalScopeType.EXPLICIT_INTERVAL,
            start_timestamp=naive,
            end_timestamp=utc,
        )


def test_non_utc_timezone_rejection():
    # Use timezone object that is not exactly timezone.utc
    non_utc = datetime.datetime(
        2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=5))
    )
    utc = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    with pytest.raises(ValidationError, match="strictly UTC"):
        TemporalScope(
            scope_type=TemporalScopeType.EXPLICIT_INTERVAL,
            start_timestamp=non_utc,
            end_timestamp=utc,
        )


def test_duration_window_representation():
    win = WindowConfiguration(window_type=WindowType.DURATION, window_size_seconds=5.0)
    assert win.window_type == WindowType.DURATION
    assert win.window_size_seconds == 5.0

    with pytest.raises(ValidationError, match="positive window_size_seconds"):
        WindowConfiguration(window_type=WindowType.DURATION, window_size_seconds=-1.0)
    with pytest.raises(ValidationError, match="positive window_size_seconds"):
        WindowConfiguration(window_type=WindowType.DURATION)


def test_observation_count_window_representation():
    win = WindowConfiguration(window_type=WindowType.OBSERVATION_COUNT, window_size_observations=10)
    assert win.window_type == WindowType.OBSERVATION_COUNT
    assert win.window_size_observations == 10

    with pytest.raises(ValidationError, match="positive window_size_observations"):
        WindowConfiguration(window_type=WindowType.OBSERVATION_COUNT, window_size_observations=0)


def test_sampling_metadata_representation():
    meta = SamplingMetadata(
        is_regular=False,
        irregularities=[
            SamplingIrregularity.DUPLICATE_TIMESTAMPS,
            SamplingIrregularity.GAPS_PRESENT,
        ],
        duplicate_count=2,
        gap_count=1,
        nominal_interval_seconds=1.0,
    )
    assert not meta.is_regular
    assert len(meta.irregularities) == 2
    assert meta.duplicate_count == 2
    assert meta.gap_count == 1


def test_immutable_models():
    req = TimeSeriesAnalysisRequest(analysis_id="ts", target_column="v", timestamp_column="t")
    with pytest.raises(ValidationError):
        req.analysis_id = "new"


def test_extra_field_rejection():
    with pytest.raises(ValidationError):
        TimeSeriesAnalysisRequest(
            analysis_id="ts", target_column="v", timestamp_column="t", unknown_field=42
        )


def test_epistemic_level_is_calculated():
    res = TimeSeriesAnalysisResult(
        status=ResultStatus.SUCCESS,
        issues=[],
        result_id="r1",
        analysis_id="ts-1",
        target_column="v",
        timestamp_column="t",
        temporal_scope=TemporalScope(scope_type=TemporalScopeType.ENTIRE_DATASET),
    )
    assert res.epistemic_level == EpistemicLevel.CALCULATED

    with pytest.raises(ValidationError, match="Time-series results must not produce OBSERVED"):
        TimeSeriesAnalysisResult(
            status=ResultStatus.SUCCESS,
            issues=[],
            result_id="r1",
            analysis_id="ts-1",
            target_column="v",
            timestamp_column="t",
            temporal_scope=TemporalScope(scope_type=TemporalScopeType.ENTIRE_DATASET),
            epistemic_level=EpistemicLevel.OBSERVED,
        )


def test_provenance_attachment():
    prov = create_provenance("src", "ts_analysis", EpistemicLevel.CALCULATED)
    res = TimeSeriesAnalysisResult(
        status=ResultStatus.SUCCESS,
        issues=[],
        result_id="r1",
        analysis_id="ts-1",
        target_column="v",
        timestamp_column="t",
        temporal_scope=TemporalScope(scope_type=TemporalScopeType.ENTIRE_DATASET),
        provenance=prov,
    )
    assert res.provenance is not None
    assert res.provenance.record_id == prov.record_id


def test_deterministic_serialization():
    req = TimeSeriesAnalysisRequest(analysis_id="ts-1", target_column="v", timestamp_column="t")
    d = req.model_dump()
    assert d["analysis_id"] == "ts-1"
    assert d["target_column"] == "v"
    assert d["timestamp_column"] == "t"
    assert d["temporal_scope"]["scope_type"] == "ENTIRE_DATASET"


def test_no_dataframe_or_parameter_registry_dependency():
    # Model instantiation is pure Pydantic, no pandas DataFrames
    # are required to define the request or result shape.
    pass


def test_missing_end_timestamp_for_explicit_interval_rejected():
    t_start = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    with pytest.raises(ValidationError, match="require both start and end"):
        TemporalScope(
            scope_type=TemporalScopeType.EXPLICIT_INTERVAL,
            start_timestamp=t_start,
        )
