import datetime

import pytest
from pydantic import ValidationError

from aeris.core.anomaly.models import (
    AnomalyClassification,
    AnomalyRecord,
    AnomalyResult,
    AnomalySeverity,
    DetectionMethod,
    EvidenceReference,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


@pytest.fixture
def base_provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        record_id="prov_123",
        source_uri="file:///test.csv",
        activity_name="anomaly_detection",
        epistemic_level=EpistemicLevel.CALCULATED,
    )


def test_valid_point_anomaly(base_provenance: ProvenanceRecord) -> None:
    record = AnomalyRecord(
        anomaly_id="anom_1",
        analysis_id="analysis_1",
        target_column="altitude_ft",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.LOW,
        detection_method=DetectionMethod.RULE_BASED,
        timestamp=datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
        observed_value=5000.0,
        expected_value=1000.0,
        deviation_value=4000.0,
        deviation_metric="absolute_difference",
        provenance=base_provenance,
    )
    assert record.anomaly_id == "anom_1"
    assert record.classification == AnomalyClassification.POINT_ANOMALY
    assert record.epistemic_level == EpistemicLevel.CALCULATED


def test_valid_contextual_anomaly(base_provenance: ProvenanceRecord) -> None:
    record = AnomalyRecord(
        anomaly_id="anom_2",
        analysis_id="analysis_1",
        target_column="airspeed_kts",
        classification=AnomalyClassification.CONTEXTUAL_ANOMALY,
        severity=AnomalySeverity.MEDIUM,
        detection_method=DetectionMethod.STATISTICAL,
        start_timestamp=datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
        end_timestamp=datetime.datetime(2026, 1, 1, 12, 1, 0, tzinfo=datetime.UTC),
        explanation="Airspeed variance unusual during climb phase context.",
        provenance=base_provenance,
    )
    assert record.classification == AnomalyClassification.CONTEXTUAL_ANOMALY
    assert record.start_timestamp is not None
    assert record.end_timestamp is not None


def test_valid_collective_anomaly(base_provenance: ProvenanceRecord) -> None:
    record = AnomalyRecord(
        anomaly_id="anom_3",
        analysis_id="analysis_1",
        target_column="pitch_deg",
        classification=AnomalyClassification.COLLECTIVE_ANOMALY,
        severity=AnomalySeverity.HIGH,
        detection_method=DetectionMethod.TIME_SERIES,
        evidence_reference=EvidenceReference(source_uri="file:///data.csv", row_index=15),
        provenance=base_provenance,
    )
    assert record.classification == AnomalyClassification.COLLECTIVE_ANOMALY
    assert record.evidence_reference is not None
    assert record.evidence_reference.row_index == 15


def test_epistemic_level_rejection(base_provenance: ProvenanceRecord) -> None:
    # Anomaly records cannot be INFERRED, HYPOTHESIS, or RECOMMENDATION
    with pytest.raises(ValidationError) as exc_info:
        AnomalyRecord(
            anomaly_id="anom_4",
            analysis_id="analysis_1",
            target_column="target",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.INFO,
            detection_method=DetectionMethod.OTHER,
            provenance=base_provenance,
            epistemic_level=EpistemicLevel.RECOMMENDATION,
        )
    assert "must be CALCULATED" in str(exc_info.value)

    with pytest.raises(ValidationError):
        AnomalyRecord(
            anomaly_id="anom_4",
            analysis_id="analysis_1",
            target_column="target",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.INFO,
            detection_method=DetectionMethod.OTHER,
            provenance=base_provenance,
            epistemic_level=EpistemicLevel.HYPOTHESIS,
        )

    with pytest.raises(ValidationError):
        AnomalyRecord(
            anomaly_id="anom_4",
            analysis_id="analysis_1",
            target_column="target",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.INFO,
            detection_method=DetectionMethod.OTHER,
            provenance=base_provenance,
            epistemic_level=EpistemicLevel.INFERRED,
        )


def test_naive_timestamp_rejection(base_provenance: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError) as exc_info:
        AnomalyRecord(
            anomaly_id="anom_5",
            analysis_id="analysis_1",
            target_column="target",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.INFO,
            detection_method=DetectionMethod.OTHER,
            timestamp=datetime.datetime(2026, 1, 1, 12, 0, 0),  # Naive
            provenance=base_provenance,
        )
    assert "timezone-aware UTC" in str(exc_info.value)


def test_zero_anomaly_successful_result(base_provenance: ProvenanceRecord) -> None:
    # A valid analysis can produce zero anomalies and be SUCCESS
    result = AnomalyResult(
        analysis_id="analysis_1",
        target_column="altitude_ft",
        detection_method=DetectionMethod.STATISTICAL,
        status=ResultStatus.SUCCESS,
        anomalies=(),
        provenance=base_provenance,
    )
    assert result.status == ResultStatus.SUCCESS
    assert len(result.anomalies) == 0


def test_not_available_result(base_provenance: ProvenanceRecord) -> None:
    result = AnomalyResult(
        analysis_id="analysis_2",
        target_column="airspeed_kts",
        detection_method=DetectionMethod.TIME_SERIES,
        status=ResultStatus.NOT_AVAILABLE,
        anomalies=(),
        issues=[
            Issue(
                issue_code="ERR_NO_DATA",
                severity=IssueSeverity.WARNING,
                message="No data in scope.",
            )
        ],
        provenance=base_provenance,
    )
    assert result.status == ResultStatus.NOT_AVAILABLE


def test_partial_result_with_anomalies(base_provenance: ProvenanceRecord) -> None:
    record = AnomalyRecord(
        anomaly_id="anom_1",
        analysis_id="analysis_3",
        target_column="pitch_deg",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.LOW,
        detection_method=DetectionMethod.RULE_BASED,
        provenance=base_provenance,
    )
    result = AnomalyResult(
        analysis_id="analysis_3",
        target_column="pitch_deg",
        detection_method=DetectionMethod.RULE_BASED,
        status=ResultStatus.PARTIAL,
        anomalies=(record,),
        issues=[
            Issue(
                issue_code="WARN_MISSING_DATA",
                severity=IssueSeverity.WARNING,
                message="Some rows skipped.",
            )
        ],
        provenance=base_provenance,
    )
    assert result.status == ResultStatus.PARTIAL
    assert len(result.anomalies) == 1


def test_immutable_anomaly_model(base_provenance: ProvenanceRecord) -> None:
    record = AnomalyRecord(
        anomaly_id="anom_1",
        analysis_id="analysis_1",
        target_column="target",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.INFO,
        detection_method=DetectionMethod.OTHER,
        provenance=base_provenance,
    )
    with pytest.raises(ValidationError):
        record.observed_value = 100.0  # type: ignore


def test_extra_fields_rejected(base_provenance: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        AnomalyRecord(
            anomaly_id="anom_1",
            analysis_id="analysis_1",
            target_column="target",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.INFO,
            detection_method=DetectionMethod.OTHER,
            provenance=base_provenance,
            root_cause="Sensor failure",  # type: ignore
        )

    with pytest.raises(ValidationError):
        AnomalyRecord(
            anomaly_id="anom_1",
            analysis_id="analysis_1",
            target_column="target",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.INFO,
            detection_method=DetectionMethod.OTHER,
            provenance=base_provenance,
            recommendation="Replace sensor",  # type: ignore
        )


def test_missing_required_fields_rejected(base_provenance: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        # Missing analysis_id
        AnomalyRecord(
            anomaly_id="anom_1",  # type: ignore
            target_column="target",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.INFO,
            detection_method=DetectionMethod.OTHER,
            provenance=base_provenance,
        )


def test_independent_result_instances(base_provenance: ProvenanceRecord) -> None:
    record1 = AnomalyRecord(
        anomaly_id="anom_1",
        analysis_id="a1",
        target_column="target1",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.LOW,
        detection_method=DetectionMethod.RULE_BASED,
        provenance=base_provenance,
    )
    record2 = AnomalyRecord(
        anomaly_id="anom_2",
        analysis_id="a2",
        target_column="target2",
        classification=AnomalyClassification.CONTEXTUAL_ANOMALY,
        severity=AnomalySeverity.HIGH,
        detection_method=DetectionMethod.STATISTICAL,
        provenance=base_provenance,
    )
    res1 = AnomalyResult(
        analysis_id="a1",
        target_column="target1",
        detection_method=DetectionMethod.RULE_BASED,
        status=ResultStatus.SUCCESS,
        anomalies=(record1,),
    )
    res2 = AnomalyResult(
        analysis_id="a2",
        target_column="target2",
        detection_method=DetectionMethod.STATISTICAL,
        status=ResultStatus.SUCCESS,
        anomalies=(record2,),
    )
    assert res1.anomalies[0].anomaly_id == "anom_1"
    assert res2.anomalies[0].anomaly_id == "anom_2"
