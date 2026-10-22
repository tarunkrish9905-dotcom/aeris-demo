import datetime

import numpy as np
import pandas as pd
import pytest

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus
from aeris.core.statistics.models import MetricType
from aeris.core.statistics.time_series.engine import TimeSeriesStatisticalAnalysisEngine
from aeris.core.statistics.time_series.models import (
    SamplingIrregularity,
    TemporalScope,
    TemporalScopeType,
    TimeSeriesAnalysisRequest,
    WindowConfiguration,
    WindowType,
)


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 10, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 20, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 30, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 40, tzinfo=datetime.UTC),
            ],
            "target": [10.0, 20.0, np.nan, 40.0, 50.0],
            "other": ["a", "b", "c", "d", "e"],
        }
    )


def test_missing_timestamp_column(base_df):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1", target_column="target", timestamp_column="missing"
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)
    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_MISSING_TIMESTAMP_COLUMN" for i in res.issues)


def test_missing_target_column(base_df):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1", target_column="missing", timestamp_column="ts"
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)
    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_MISSING_TARGET_COLUMN" for i in res.issues)


def test_naive_timestamps_rejected(base_df):
    base_df["ts"] = base_df["ts"].dt.tz_localize(None)
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1", target_column="target", timestamp_column="ts"
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)
    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_NON_UTC_TIMESTAMP" for i in res.issues)


def test_unordered_timestamps_rejected(base_df):
    base_df.loc[0, "ts"], base_df.loc[1, "ts"] = base_df.loc[1, "ts"], base_df.loc[0, "ts"]
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1", target_column="target", timestamp_column="ts"
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)
    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_UNORDERED_TIMESTAMPS" for i in res.issues)


def test_observation_count_rolling_metrics(base_df):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        window_configuration=WindowConfiguration(
            window_type=WindowType.OBSERVATION_COUNT,
            window_size_observations=2,
            min_observations=1,
        ),
        requested_metrics=[
            MetricType.COUNT,
            MetricType.MEAN,
            MetricType.MINIMUM,
            MetricType.MAXIMUM,
            MetricType.VARIANCE,
            MetricType.STANDARD_DEVIATION,
        ],
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)

    assert res.status == ResultStatus.PARTIAL  # contains NaNs

    counts = next(m.values for m in res.metrics if m.metric_type == MetricType.COUNT)
    assert counts == (1.0, 2.0, 1.0, 1.0, 2.0)

    means = next(m.values for m in res.metrics if m.metric_type == MetricType.MEAN)
    assert means == (10.0, 15.0, 20.0, 40.0, 45.0)

    variances = next(m.values for m in res.metrics if m.metric_type == MetricType.VARIANCE)
    # var(10, 20) = 50
    # var(40, 50) = 50
    assert variances[1] == 50.0
    assert variances[2] is None  # var(20, nan) -> N < 2


def test_duration_based_rolling_behavior(base_df):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        window_configuration=WindowConfiguration(
            window_type=WindowType.DURATION,
            window_size_seconds=20.0,
            min_observations=1,
        ),
        requested_metrics=[MetricType.COUNT, MetricType.MEAN],
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)

    counts = next(m.values for m in res.metrics if m.metric_type == MetricType.COUNT)
    # windows: [0, 0] (10.0) -> count=1
    # [0, 10] (10.0, 20.0) -> count=2
    # [0, 20] (10.0, 20.0, nan) -> count=2
    # [10, 30] (20.0, nan, 40.0) -> count=2
    # [20, 40] (nan, 40.0, 50.0) -> count=2
    assert counts == (1.0, 2.0, 2.0, 2.0, 2.0)


def test_minimum_observation_behavior_and_all_missing(base_df):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        window_configuration=WindowConfiguration(
            window_type=WindowType.OBSERVATION_COUNT,
            window_size_observations=2,
            min_observations=2,
        ),
        requested_metrics=[MetricType.MEAN],
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)

    means = next(m.values for m in res.metrics if m.metric_type == MetricType.MEAN)
    # [10] -> N=1 < min=2 -> None
    # [10, 20] -> N=2 -> 15.0
    # [20, nan] -> N=1 < min=2 -> None
    # [nan, 40] -> N=1 < min=2 -> None
    # [40, 50] -> N=2 -> 45.0
    assert means[0] is None
    assert means[1] == 15.0
    assert means[2] is None
    assert means[3] is None
    assert means[4] == 45.0


def test_duplicate_timestamps():
    df = pd.DataFrame(
        {
            "ts": [
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            ],
            "target": [10.0, 20.0],
        }
    )

    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        window_configuration=WindowConfiguration(
            window_type=WindowType.DURATION,
            window_size_seconds=10.0,
        ),
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, df)
    # Duration window with duplicates is rejected per prompt
    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_DURATION_WINDOW_DUPLICATES" for i in res.issues)

    req2 = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        window_configuration=WindowConfiguration(
            window_type=WindowType.OBSERVATION_COUNT,
            window_size_observations=2,
        ),
        requested_metrics=[MetricType.MEAN],
    )
    res2 = engine.execute(req2, df)
    # Observation count window succeeds
    assert res2.status == ResultStatus.SUCCESS
    assert SamplingIrregularity.DUPLICATE_TIMESTAMPS in res2.sampling_metadata.irregularities


def test_temporal_gaps(base_df):
    base_df.loc[3, "ts"] = datetime.datetime(2026, 1, 1, 12, 10, 0, tzinfo=datetime.UTC)
    base_df.loc[4, "ts"] = datetime.datetime(2026, 1, 1, 12, 10, 10, tzinfo=datetime.UTC)

    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        window_configuration=WindowConfiguration(
            window_type=WindowType.DURATION,
            window_size_seconds=20.0,
        ),
        requested_metrics=[MetricType.COUNT],
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)

    assert SamplingIrregularity.GAPS_PRESENT in res.sampling_metadata.irregularities
    counts = next(m.values for m in res.metrics if m.metric_type == MetricType.COUNT)
    # At index 3 (12:10:00), window is [12:09:40, 12:10:00], only contains index 3 (40.0) -> count 1
    assert counts[3] == 1.0


def test_explicit_temporal_interval_and_empty_scope(base_df):
    t_start = datetime.datetime(2026, 1, 1, 12, 0, 15, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 12, 0, 35, tzinfo=datetime.UTC)
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        temporal_scope=TemporalScope(
            scope_type=TemporalScopeType.EXPLICIT_INTERVAL,
            start_timestamp=t_start,
            end_timestamp=t_end,
        ),
        window_configuration=WindowConfiguration(
            window_type=WindowType.OBSERVATION_COUNT,
            window_size_observations=1,
        ),
        requested_metrics=[MetricType.MEAN],
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)

    assert res.observation_count == 2  # rows 2 and 3
    assert len(res.metrics[0].values) == 2

    # Empty scope
    req_empty = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        temporal_scope=TemporalScope(
            scope_type=TemporalScopeType.EXPLICIT_INTERVAL,
            start_timestamp=datetime.datetime(2027, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            end_timestamp=datetime.datetime(2027, 1, 1, 12, 1, 0, tzinfo=datetime.UTC),
        ),
    )
    res_empty = engine.execute(req_empty, base_df)
    assert res_empty.status == ResultStatus.NOT_AVAILABLE


def test_non_numeric_target(base_df):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="other",
        timestamp_column="ts",
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)
    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_NON_NUMERIC_TARGET" for i in res.issues)


def test_input_immutability(base_df):
    df_copy = base_df.copy(deep=True)
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        window_configuration=WindowConfiguration(
            window_type=WindowType.OBSERVATION_COUNT,
            window_size_observations=2,
        ),
        requested_metrics=[MetricType.MEAN],
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    engine.execute(req, base_df)
    pd.testing.assert_frame_equal(base_df, df_copy)


def test_provenance_and_epistemic_level(base_df):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        window_configuration=WindowConfiguration(
            window_type=WindowType.OBSERVATION_COUNT,
            window_size_observations=2,
        ),
        requested_metrics=[MetricType.MEAN],
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df, parent_record_ids=["par-1"])
    assert res.epistemic_level == EpistemicLevel.CALCULATED
    assert res.provenance is not None
    assert "par-1" in res.provenance.parent_record_ids


def test_unsupported_metric(base_df):
    req = TimeSeriesAnalysisRequest(
        analysis_id="ts1",
        target_column="target",
        timestamp_column="ts",
        window_configuration=WindowConfiguration(
            window_type=WindowType.OBSERVATION_COUNT,
            window_size_observations=2,
        ),
        requested_metrics=[MetricType.MEDIAN],  # Not supported yet per prompt
    )
    engine = TimeSeriesStatisticalAnalysisEngine()
    res = engine.execute(req, base_df)
    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_UNSUPPORTED_METRIC" for i in res.issues)
