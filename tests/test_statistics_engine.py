import numpy as np
import pandas as pd
import pytest

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus
from aeris.core.statistics.engine import StatisticalAnalysisEngine
from aeris.core.statistics.models import (
    MetricType,
    StatisticalAnalysisRequest,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0],
            "string_col": ["a", "b", "c", "d", "e", "f", "g"],
            "one_valid": [np.nan, np.nan, 42.0, np.nan, np.nan, np.nan, np.nan],
            "all_missing": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "mixed_col": [1.0, "two", 3.0, np.nan, "five", 6.0, 7.0],
        }
    )


def test_count_and_missing_count(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-1",
        target_column="numeric_col",
        requested_metrics=[MetricType.COUNT, MetricType.MISSING_COUNT],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.SUCCESS
    assert len(res.metrics) == 2

    c = next(m for m in res.metrics if m.metric_type == MetricType.COUNT)
    mc = next(m for m in res.metrics if m.metric_type == MetricType.MISSING_COUNT)

    assert c.value == 6.0
    assert mc.value == 1.0


def test_basic_metrics(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-2",
        target_column="numeric_col",
        requested_metrics=[
            MetricType.MEAN,
            MetricType.MEDIAN,
            MetricType.MINIMUM,
            MetricType.MAXIMUM,
            MetricType.VARIANCE,
            MetricType.STANDARD_DEVIATION,
        ],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.SUCCESS
    metrics = {m.metric_type: m.value for m in res.metrics}

    assert metrics[MetricType.MEAN] == pytest.approx(22.0 / 6.0)  # (1+2+3+4+5+7)/6
    assert metrics[MetricType.MEDIAN] == 3.5
    assert metrics[MetricType.MINIMUM] == 1.0
    assert metrics[MetricType.MAXIMUM] == 7.0

    # var = sum((x - mean)^2) / 5
    mean = 22.0 / 6.0
    sq_diff = sum((x - mean) ** 2 for x in [1, 2, 3, 4, 5, 7])
    assert metrics[MetricType.VARIANCE] == pytest.approx(sq_diff / 5)
    assert metrics[MetricType.STANDARD_DEVIATION] == pytest.approx((sq_diff / 5) ** 0.5)


def test_percentile_explicit(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-3",
        target_column="numeric_col",
        requested_metrics=[MetricType.PERCENTILE],
        requested_percentiles=[50.0, 90.0],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.SUCCESS
    assert len(res.metrics) == 2

    p50 = next(m for m in res.metrics if m.parameters["percentile"] == 50.0)
    p90 = next(m for m in res.metrics if m.parameters["percentile"] == 90.0)

    assert p50.value == 3.5
    assert p90.value == pytest.approx(6.0)  # numpy default percentile interpolation


def test_missing_target_column(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-4",
        target_column="does_not_exist",
        requested_metrics=[MetricType.COUNT],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_TARGET_COLUMN_MISSING" for i in res.issues)


def test_non_numeric_target(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-5",
        target_column="string_col",
        requested_metrics=[MetricType.MEAN],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_TARGET_DATA_TYPE_INVALID" for i in res.issues)


def test_empty_dataframe() -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-6",
        target_column="empty_col",
        requested_metrics=[MetricType.COUNT, MetricType.MEAN],
    )
    engine = StatisticalAnalysisEngine()
    df = pd.DataFrame({"empty_col": pd.Series(dtype=float)})
    res = engine.execute(req, df)

    # 0 total elements. COUNT = 0, MEAN = None. NOT_AVAILABLE for MEAN.
    # Overall status should be PARTIAL/WARNING or NOT_AVAILABLE depending on logic.
    assert res.status in [ResultStatus.WARNING, ResultStatus.NOT_AVAILABLE]
    c = next(m for m in res.metrics if m.metric_type == MetricType.COUNT)
    assert c.value == 0.0


def test_all_missing(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-7",
        target_column="all_missing",
        requested_metrics=[MetricType.MEAN, MetricType.MISSING_COUNT],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.WARNING  # Since MISSING_COUNT is available but MEAN is not
    m = next(m for m in res.metrics if m.metric_type == MetricType.MEAN)
    assert m.value is None

    mc = next(m for m in res.metrics if m.metric_type == MetricType.MISSING_COUNT)
    assert mc.value == 7.0


def test_one_valid_observation(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-8",
        target_column="one_valid",
        requested_metrics=[MetricType.MEAN, MetricType.VARIANCE],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.WARNING

    mean = next(m for m in res.metrics if m.metric_type == MetricType.MEAN)
    var = next(m for m in res.metrics if m.metric_type == MetricType.VARIANCE)

    assert mean.value == 42.0
    assert var.value is None
    assert any(i.issue_code == "ERR_INSUFFICIENT_DATA" for i in res.issues)


def test_input_dataframe_immutability(sample_df: pd.DataFrame) -> None:
    original_df = sample_df.copy(deep=True)
    req = StatisticalAnalysisRequest(
        analysis_id="stat-9",
        target_column="numeric_col",
        requested_metrics=[MetricType.MEAN, MetricType.MAXIMUM],
    )
    engine = StatisticalAnalysisEngine()
    engine.execute(req, sample_df)

    pd.testing.assert_frame_equal(sample_df, original_df)


def test_provenance_and_epistemic_level(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-10",
        target_column="numeric_col",
        requested_metrics=[MetricType.COUNT],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df, source_uri="test_source")

    assert res.epistemic_level == EpistemicLevel.CALCULATED
    assert res.provenance is not None
    assert res.provenance.activity_name == "statistical_analysis"
    assert res.provenance.source_uri == "test_source"


def test_invalid_percentile(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-11",
        target_column="numeric_col",
        requested_metrics=[MetricType.PERCENTILE],
        requested_percentiles=[150.0],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_INVALID_PERCENTILE_CONFIGURATION" for i in res.issues)


def test_missing_percentile_config(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-12",
        target_column="numeric_col",
        requested_metrics=[MetricType.PERCENTILE],
        requested_percentiles=[],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_INVALID_PERCENTILE_CONFIGURATION" for i in res.issues)


def test_mixed_invalid_values(sample_df: pd.DataFrame) -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-13",
        target_column="mixed_col",
        requested_metrics=[MetricType.MEAN],
    )
    engine = StatisticalAnalysisEngine()
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_TARGET_DATA_TYPE_INVALID" for i in res.issues)
