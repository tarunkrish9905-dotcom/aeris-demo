import datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.anomaly.models import (
    AnomalyClassification,
    AnomalySeverity,
    DetectionMethod,
)
from aeris.core.anomaly.statistical import (
    StatisticalAnomalyDetectionEngine,
    StatisticalAnomalyDetectionRequest,
)
from aeris.core.results.types import ResultStatus


@pytest.fixture
def engine() -> StatisticalAnomalyDetectionEngine:
    return StatisticalAnomalyDetectionEngine()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 10, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 20, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 30, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 40, tzinfo=datetime.UTC),
            ],
            # Values: 10, 10, 10, 10, 100 -> mean=28, std=40.249223595
            # z-scores:
            # (10-28)/40.249 = -0.447
            # (100-28)/40.249 = 1.788
            "target": [10.0, 10.0, 10.0, 10.0, 100.0],
        }
    )


def test_valid_statistical_detection(
    engine: StatisticalAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.5, timestamp_column="timestamp"
    )
    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.SUCCESS
    assert len(res.anomalies) == 1

    a = res.anomalies[0]
    assert a.observed_value == 100.0
    assert a.deviation_metric == "z_score"
    assert a.deviation_value == pytest.approx(1.78885, abs=1e-4)
    assert a.reference_mean == 28.0
    assert a.reference_standard_deviation == pytest.approx(40.24922, abs=1e-4)
    assert a.evidence_reference is not None
    assert a.evidence_reference.row_index == 4


def test_no_anomalies_returns_success(
    engine: StatisticalAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=2.0
    )
    res = engine.execute(req, sample_df)
    assert res.status == ResultStatus.SUCCESS
    assert len(res.anomalies) == 0


def test_multiple_anomalies(engine: StatisticalAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": [-100.0, 10.0, 10.0, 10.0, 100.0]})
    # mean: 6.0, std: 71.27
    # -100 -> z = -1.48
    # 100 -> z = 1.31
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.3
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.SUCCESS
    assert len(res.anomalies) == 2
    assert res.anomalies[0].observed_value == -100.0
    assert res.anomalies[1].observed_value == 100.0


def test_missing_target_column(engine: StatisticalAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"other": [1, 2, 3]})
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.5
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_MISSING_TARGET"


def test_non_numeric_target(engine: StatisticalAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": ["a", "b", "c"]})
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.5
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_NON_NUMERIC_TARGET"


def test_boolean_rejection(engine: StatisticalAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": [True, False, True]})
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.5
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_BOOLEAN_TARGET"


def test_missing_values_excluded(engine: StatisticalAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": pd.Series([10.0, np.nan, 10.0, 10.0, 100.0], dtype="float64")})
    # Valid obs: 4. mean: 32.5, std: 45.0
    # 100 -> z = (100 - 32.5) / 45.0 = 1.5
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.4
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.PARTIAL
    assert res.metadata["evaluable_observations"] == 4
    assert len(res.anomalies) == 1
    assert res.anomalies[0].observed_value == 100.0
    assert res.anomalies[0].reference_mean == 32.5
    assert res.anomalies[0].reference_standard_deviation == 45.0


def test_all_values_missing(engine: StatisticalAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": pd.Series([np.nan, np.nan], dtype="float64")})
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.5
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.NOT_AVAILABLE
    assert res.issues[0].issue_code == "ERR_INSUFFICIENT_OBSERVATIONS"


def test_insufficient_observations(engine: StatisticalAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": [10.0]})
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.5, minimum_observations=2
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.NOT_AVAILABLE
    assert res.issues[0].issue_code == "ERR_INSUFFICIENT_OBSERVATIONS"


def test_zero_standard_deviation(engine: StatisticalAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": [10.0, 10.0, 10.0]})
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.5
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.NOT_AVAILABLE
    assert res.issues[0].issue_code == "ERR_ZERO_STANDARD_DEVIATION"


def test_invalid_z_thresholds() -> None:
    with pytest.raises(ValidationError):
        StatisticalAnomalyDetectionRequest(
            analysis_id="a1", target_column="target", z_threshold=0.0
        )
    with pytest.raises(ValidationError):
        StatisticalAnomalyDetectionRequest(
            analysis_id="a1", target_column="target", z_threshold=-1.0
        )
    with pytest.raises(ValidationError):
        StatisticalAnomalyDetectionRequest(
            analysis_id="a1", target_column="target", z_threshold=float("nan")
        )
    with pytest.raises(ValidationError):
        StatisticalAnomalyDetectionRequest(
            analysis_id="a1", target_column="target", z_threshold=float("inf")
        )


def test_exact_z_threshold_boundary_is_not_anomalous(
    engine: StatisticalAnomalyDetectionEngine,
) -> None:
    df = pd.DataFrame({"target": [-100.0, 10.0, 10.0, 10.0, 100.0]})
    # z for 100 is 1.3188
    z_exact = 1.3253905234916354
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=z_exact
    )
    res = engine.execute(req, df)
    # The anomaly is strictly > z_threshold
    assert len(res.anomalies) == 1
    assert res.anomalies[0].observed_value == -100.0  # z = -1.48


def test_naive_timestamp_rejected(engine: StatisticalAnomalyDetectionEngine) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 0),
                datetime.datetime(2026, 1, 1, 12, 0, 1),
            ],
            "target": [10.0, 20.0],
        }
    )
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.5, timestamp_column="timestamp"
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_NAIVE_TIMESTAMP"


def test_duplicate_timestamps_allowed(engine: StatisticalAnomalyDetectionEngine) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            ],
            "target": [10.0, 10.0, 100.0],
        }
    )
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.1, timestamp_column="timestamp"
    )
    res = engine.execute(req, df)
    assert len(res.anomalies) == 1
    assert res.anomalies[0].observed_value == 100.0


def test_input_immutability(
    engine: StatisticalAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    original = sample_df.copy(deep=True)
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1", target_column="target", z_threshold=1.5
    )
    engine.execute(req, sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


def test_anomaly_properties(
    engine: StatisticalAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    req = StatisticalAnomalyDetectionRequest(
        analysis_id="a1",
        target_column="target",
        z_threshold=1.5,
        severity=AnomalySeverity.HIGH,
        classification=AnomalyClassification.POINT_ANOMALY,
    )
    res = engine.execute(
        req, sample_df, parent_record_ids=["prov_1"], source_uri="file:///test.csv"
    )

    a = res.anomalies[0]
    assert a.analysis_id == "a1"
    assert a.target_column == "target"
    assert a.classification == AnomalyClassification.POINT_ANOMALY
    assert a.severity == AnomalySeverity.HIGH
    assert a.detection_method == DetectionMethod.STATISTICAL
    assert a.provenance.epistemic_level == "CALCULATED"
    assert a.evidence_reference is not None
    assert a.evidence_reference.source_uri == "file:///test.csv"
    assert a.evidence_reference.parent_record_ids == ["prov_1"]
    assert res.metadata["z_threshold"] == 1.5
