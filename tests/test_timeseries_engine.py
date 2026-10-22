import datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.anomaly.models import (
    AnomalyClassification,
    DetectionMethod,
)
from aeris.core.anomaly.timeseries import (
    TimeSeriesAnomalyDetectionEngine,
    TimeSeriesAnomalyDetectionRequest,
)
from aeris.core.results.types import ResultStatus


@pytest.fixture
def engine() -> TimeSeriesAnomalyDetectionEngine:
    return TimeSeriesAnomalyDetectionEngine()


@pytest.fixture
def manual_test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 1, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 2, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 3, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 4, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 5, tzinfo=datetime.UTC),
            ],
            "target": [100.0, 101.0, 99.0, 100.0, 102.0, 150.0],
        }
    )


def test_manual_test_case_rolling_z_score(
    engine: TimeSeriesAnomalyDetectionEngine, manual_test_df: pd.DataFrame
) -> None:
    # At t5, target is 150. Window size is 5.
    # Ref data: [100.0, 101.0, 99.0, 100.0, 102.0]
    # mean = 100.4, std (ddof=1) = 1.140175425
    # z = (150 - 100.4) / 1.140175425 = 43.502
    req = TimeSeriesAnomalyDetectionRequest(
        analysis_id="a1",
        target_column="target",
        timestamp_column="timestamp",
        window_size=5,
        z_threshold=5.0,
    )
    res = engine.execute(req, manual_test_df)

    assert (
        res.status == ResultStatus.PARTIAL
    )  # The first 2 observations do not have minimum 2 prior obs!
    assert len(res.anomalies) == 1

    a = res.anomalies[0]
    assert a.observed_value == 150.0
    assert a.deviation_metric == "rolling_z_score"
    assert a.classification == AnomalyClassification.CONTEXTUAL_ANOMALY
    assert a.detection_method == DetectionMethod.TIME_SERIES
    assert a.reference_mean == pytest.approx(100.4)
    assert a.reference_standard_deviation == pytest.approx(1.140175425)
    assert a.deviation_value == pytest.approx(43.502, abs=1e-3)
    assert a.evidence_reference is not None
    assert a.evidence_reference.row_index == 5


def test_zero_variance_negative_case(engine: TimeSeriesAnomalyDetectionEngine) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 1, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 2, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 3, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 4, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 5, tzinfo=datetime.UTC),
            ],
            "target": [100.0, 100.0, 100.0, 100.0, 100.0, 150.0],
        }
    )
    req = TimeSeriesAnomalyDetectionRequest(
        analysis_id="a1",
        target_column="target",
        timestamp_column="timestamp",
        window_size=5,
        z_threshold=2.0,
    )
    res = engine.execute(req, df)
    # The last row has rolling std == 0, so it's not evaluable.
    # The other rows also have std == 0.
    # Result should be NOT_AVAILABLE
    assert res.status == ResultStatus.NOT_AVAILABLE
    assert len(res.anomalies) == 0


def test_missing_timestamp_column(
    engine: TimeSeriesAnomalyDetectionEngine, manual_test_df: pd.DataFrame
) -> None:
    req = TimeSeriesAnomalyDetectionRequest(
        analysis_id="a1",
        target_column="target",
        timestamp_column="missing",
        window_size=5,
        z_threshold=2.0,
    )
    res = engine.execute(req, manual_test_df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_MISSING_TIMESTAMP"


def test_unordered_timestamps(engine: TimeSeriesAnomalyDetectionEngine) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 1, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            ],
            "target": [100.0, 100.0],
        }
    )
    req = TimeSeriesAnomalyDetectionRequest(
        analysis_id="a1",
        target_column="target",
        timestamp_column="timestamp",
        window_size=5,
        z_threshold=2.0,
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_TIMESTAMP_UNORDERED"


def test_duplicate_timestamps(engine: TimeSeriesAnomalyDetectionEngine) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            ],
            "target": [100.0, 100.0],
        }
    )
    req = TimeSeriesAnomalyDetectionRequest(
        analysis_id="a1",
        target_column="target",
        timestamp_column="timestamp",
        window_size=5,
        z_threshold=2.0,
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_DUPLICATE_TIMESTAMPS"


def test_naive_timestamp(engine: TimeSeriesAnomalyDetectionEngine) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 0),
            ],
            "target": [100.0],
        }
    )
    req = TimeSeriesAnomalyDetectionRequest(
        analysis_id="a1",
        target_column="target",
        timestamp_column="timestamp",
        window_size=5,
        z_threshold=2.0,
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_NAIVE_TIMESTAMP"


def test_invalid_requests() -> None:
    with pytest.raises(ValidationError):
        TimeSeriesAnomalyDetectionRequest(
            analysis_id="a1",
            target_column="target",
            timestamp_column="timestamp",
            window_size=-1,
            z_threshold=2.0,
        )
    with pytest.raises(ValidationError):
        TimeSeriesAnomalyDetectionRequest(
            analysis_id="a1",
            target_column="target",
            timestamp_column="timestamp",
            window_size=5,
            z_threshold=-1.0,
        )


def test_input_immutability(
    engine: TimeSeriesAnomalyDetectionEngine, manual_test_df: pd.DataFrame
) -> None:
    original = manual_test_df.copy(deep=True)
    req = TimeSeriesAnomalyDetectionRequest(
        analysis_id="a1",
        target_column="target",
        timestamp_column="timestamp",
        window_size=5,
        z_threshold=2.0,
    )
    engine.execute(req, manual_test_df)
    pd.testing.assert_frame_equal(manual_test_df, original)


def test_missing_values_ignored(engine: TimeSeriesAnomalyDetectionEngine) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 1, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 2, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 3, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 4, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 5, tzinfo=datetime.UTC),
            ],
            "target": pd.Series([100.0, 101.0, np.nan, 100.0, 102.0, 150.0], dtype="float64"),
        }
    )
    req = TimeSeriesAnomalyDetectionRequest(
        analysis_id="a1",
        target_column="target",
        timestamp_column="timestamp",
        window_size=5,
        z_threshold=3.0,
    )
    res = engine.execute(req, df)
    assert len(res.anomalies) == 1
    # Missing value is excluded. Evaluable obs is 4. mean of [100.0, 101.0, 100.0, 102.0] = 100.75
    # std = 0.9574
    # z = (150 - 100.75) / 0.9574 = 51.4
    a = res.anomalies[0]
    assert a.reference_mean == 100.75
    assert a.deviation_value == pytest.approx(51.439, abs=1e-2)
