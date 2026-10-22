import datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.anomaly.models import (
    AnomalyClassification,
    AnomalySeverity,
    DetectionMethod,
)
from aeris.core.anomaly.threshold import (
    ThresholdAnomalyDetectionEngine,
    ThresholdDetectionRequest,
    ThresholdOperator,
    ThresholdRule,
)
from aeris.core.results.types import ResultStatus


@pytest.fixture
def engine() -> ThresholdAnomalyDetectionEngine:
    return ThresholdAnomalyDetectionEngine()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 10, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 20, tzinfo=datetime.UTC),
            ],
            "target": [249.0, 250.0, 251.0],
            "other": [1, 2, 3],
        }
    )


def test_greater_than_boundaries(
    engine: ThresholdAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(
        analysis_id="a1", target_column="target", rule=rule, timestamp_column="timestamp"
    )

    res = engine.execute(req, sample_df)

    assert res.status == ResultStatus.SUCCESS
    assert len(res.anomalies) == 1
    assert res.anomalies[0].observed_value == 251.0
    assert res.anomalies[0].deviation_value == 1.0


def test_greater_than_or_equal_boundaries(
    engine: ThresholdAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN_OR_EQUAL,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(
        analysis_id="a1", target_column="target", rule=rule, timestamp_column="timestamp"
    )

    res = engine.execute(req, sample_df)
    assert len(res.anomalies) == 2
    assert res.anomalies[0].observed_value == 250.0
    assert res.anomalies[1].observed_value == 251.0


def test_less_than_boundaries(
    engine: ThresholdAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.LESS_THAN,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(
        analysis_id="a1", target_column="target", rule=rule, timestamp_column="timestamp"
    )
    res = engine.execute(req, sample_df)
    assert len(res.anomalies) == 1
    assert res.anomalies[0].observed_value == 249.0
    assert res.anomalies[0].deviation_value == 1.0


def test_less_than_or_equal_boundaries(
    engine: ThresholdAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.LESS_THAN_OR_EQUAL,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(
        analysis_id="a1", target_column="target", rule=rule, timestamp_column="timestamp"
    )
    res = engine.execute(req, sample_df)
    assert len(res.anomalies) == 2
    assert res.anomalies[0].observed_value == 249.0
    assert res.anomalies[1].observed_value == 250.0


def test_equal_boundaries(engine: ThresholdAnomalyDetectionEngine, sample_df: pd.DataFrame) -> None:
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.EQUAL,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(
        analysis_id="a1", target_column="target", rule=rule, timestamp_column="timestamp"
    )
    res = engine.execute(req, sample_df)
    assert len(res.anomalies) == 1
    assert res.anomalies[0].observed_value == 250.0


def test_not_equal_boundaries(
    engine: ThresholdAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.NOT_EQUAL,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(
        analysis_id="a1", target_column="target", rule=rule, timestamp_column="timestamp"
    )
    res = engine.execute(req, sample_df)
    assert len(res.anomalies) == 2
    assert res.anomalies[0].observed_value == 249.0
    assert res.anomalies[1].observed_value == 251.0


def test_between_boundaries(engine: ThresholdAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": [99.0, 100.0, 150.0, 200.0, 201.0]})
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.BETWEEN,
        threshold_lower=100.0,
        threshold_upper=200.0,
    )
    req = ThresholdDetectionRequest(analysis_id="a1", target_column="target", rule=rule)
    res = engine.execute(req, df)
    assert len(res.anomalies) == 2
    assert res.anomalies[0].observed_value == 99.0
    assert res.anomalies[0].deviation_value == 1.0  # 100 - 99
    assert res.anomalies[1].observed_value == 201.0
    assert res.anomalies[1].deviation_value == 1.0  # 201 - 200


def test_invalid_between_bounds() -> None:
    with pytest.raises(ValidationError):
        ThresholdRule(
            rule_id="r1",
            target_column="target",
            operator=ThresholdOperator.BETWEEN,
            threshold_lower=200.0,
            threshold_upper=100.0,
        )


def test_missing_target_column(
    engine: ThresholdAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    rule = ThresholdRule(
        rule_id="r1",
        target_column="missing",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(analysis_id="a1", target_column="missing", rule=rule)
    res = engine.execute(req, sample_df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_MISSING_TARGET"


def test_non_numeric_target(engine: ThresholdAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": ["a", "b", "c"]})
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(analysis_id="a1", target_column="target", rule=rule)
    res = engine.execute(req, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_NON_NUMERIC_TARGET"


def test_boolean_target_rejected(engine: ThresholdAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": [True, False, True]})
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=0.5,
    )
    req = ThresholdDetectionRequest(analysis_id="a1", target_column="target", rule=rule)
    res = engine.execute(req, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_BOOLEAN_TARGET"


def test_missing_target_values(engine: ThresholdAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": [251.0, None, 252.0]})
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(analysis_id="a1", target_column="target", rule=rule)
    res = engine.execute(req, df)
    assert res.status == ResultStatus.PARTIAL
    assert len(res.anomalies) == 2


def test_all_target_values_missing(engine: ThresholdAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": pd.Series([None, None, None], dtype="float64")})
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(analysis_id="a1", target_column="target", rule=rule)
    res = engine.execute(req, df)
    assert res.status == ResultStatus.NOT_AVAILABLE


def test_zero_anomalies_returns_success(engine: ThresholdAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"target": [10.0, 20.0, 30.0]})
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=100.0,
    )
    req = ThresholdDetectionRequest(analysis_id="a1", target_column="target", rule=rule)
    res = engine.execute(req, df)
    assert res.status == ResultStatus.SUCCESS
    assert len(res.anomalies) == 0


def test_duplicate_timestamps(engine: ThresholdAnomalyDetectionEngine) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            ],
            "target": [251.0, 252.0],
        }
    )
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(
        analysis_id="a1", target_column="target", rule=rule, timestamp_column="timestamp"
    )
    res = engine.execute(req, df)
    assert len(res.anomalies) == 2


def test_naive_timestamp_rejected(engine: ThresholdAnomalyDetectionEngine) -> None:
    df = pd.DataFrame({"timestamp": [datetime.datetime(2026, 1, 1, 12, 0, 0)], "target": [251.0]})
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(
        analysis_id="a1", target_column="target", rule=rule, timestamp_column="timestamp"
    )
    res = engine.execute(req, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_NAIVE_TIMESTAMP"


def test_input_immutability(
    engine: ThresholdAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    original = sample_df.copy(deep=True)
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=250.0,
    )
    req = ThresholdDetectionRequest(
        analysis_id="a1", target_column="target", rule=rule, timestamp_column="timestamp"
    )
    engine.execute(req, sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


def test_explicit_target_column_mismatch() -> None:
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target1",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=250.0,
    )
    with pytest.raises(ValidationError):
        ThresholdDetectionRequest(analysis_id="a1", target_column="target2", rule=rule)


def test_anomaly_record_properties(
    engine: ThresholdAnomalyDetectionEngine, sample_df: pd.DataFrame
) -> None:
    rule = ThresholdRule(
        rule_id="r1",
        target_column="target",
        operator=ThresholdOperator.GREATER_THAN,
        threshold_value=250.0,
        severity=AnomalySeverity.HIGH,
    )
    req = ThresholdDetectionRequest(
        analysis_id="a1",
        target_column="target",
        rule=rule,
        timestamp_column="timestamp",
        classification=AnomalyClassification.POINT_ANOMALY,
    )
    res = engine.execute(
        req, sample_df, source_uri="file:///test.csv", parent_record_ids=["prov_1"]
    )

    a = res.anomalies[0]
    assert a.analysis_id == "a1"
    assert a.target_column == "target"
    assert a.classification == AnomalyClassification.POINT_ANOMALY
    assert a.severity == AnomalySeverity.HIGH
    assert a.detection_method == DetectionMethod.RULE_BASED
    assert a.evidence_reference is not None
    assert a.evidence_reference.source_uri == "file:///test.csv"
    assert a.evidence_reference.row_index == 2
    assert a.provenance.epistemic_level == "CALCULATED"
