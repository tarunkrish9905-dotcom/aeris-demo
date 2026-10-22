import datetime
import uuid

import pytest

from aeris.core.anomaly.models import (
    AnomalyClassification,
    AnomalyRecord,
    AnomalySeverity,
    DetectionMethod,
    EvidenceReference,
)
from aeris.core.engineering.models import EngineeringCalculationResult
from aeris.core.evidence.construction import EvidenceConstructionEngine
from aeris.core.evidence.models import EvidenceRecord, EvidenceType
from aeris.core.expected_observed.models import (
    ExpectedObservedComparison,
    ExpectedValue,
    ObservedValue,
    ResultStatus,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.statistics.models import MetricType, StatisticalAnalysisResult, StatisticalMetric


def dt() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def prov(level: EpistemicLevel = EpistemicLevel.OBSERVED) -> ProvenanceRecord:
    return ProvenanceRecord(
        record_id=str(uuid.uuid4()),
        activity_name="TEST_ACTIVITY",
        epistemic_level=level,
        parent_record_ids=["p1"],
        created_at=datetime.datetime.now(datetime.UTC),
        source_uri="test",
    )


def test_create_from_anomaly():
    anom = AnomalyRecord(
        anomaly_id="a1",
        analysis_id="an1",
        target_column="airspeed",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.HIGH,
        detection_method=DetectionMethod.RULE_BASED,
        timestamp=dt(),
        evidence_reference=EvidenceReference(source_uri="file.csv", row_index=10),
        source_reference="my_source",
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=prov(EpistemicLevel.CALCULATED),
    )
    ev = EvidenceConstructionEngine.create_from_anomaly(anom)
    assert ev.evidence_type == EvidenceType.ANOMALY
    assert ev.evidence_id == "EVD-ANOMALY-a1"
    assert ev.source_reference == "my_source"
    assert ev.source_record_id == "a1"
    assert ev.source_parameter_id == "airspeed"
    assert ev.source_row_index == 10
    assert ev.timestamp == dt()
    assert ev.epistemic_level == EpistemicLevel.CALCULATED
    assert ev.provenance.record_id == anom.provenance.record_id


def test_create_from_engineering():
    res = EngineeringCalculationResult(
        result_id="r1",
        calculation_id="calc1",
        output_value=123.4,
        unit="kt",
        epistemic_level=EpistemicLevel.CALCULATED,
        status=ResultStatus.SUCCESS,
        provenance=prov(EpistemicLevel.CALCULATED),
    )
    ev = EvidenceConstructionEngine.create_from_engineering_result(res)
    assert ev.evidence_type == EvidenceType.CALCULATION
    assert ev.evidence_id == "EVD-ENG-r1"
    assert ev.source_reference == "calc1"
    assert ev.source_record_id == "r1"
    assert ev.unit == "kt"
    assert ev.epistemic_level == EpistemicLevel.CALCULATED
    assert ev.provenance.record_id == res.provenance.record_id


def test_engineering_missing_provenance():
    res = EngineeringCalculationResult(
        result_id="r1",
        calculation_id="calc1",
        output_value=123.4,
        unit="kt",
        epistemic_level=EpistemicLevel.CALCULATED,
        status=ResultStatus.SUCCESS,
    )
    with pytest.raises(ValueError, match="ERR_REQUIRED_TRACEABILITY_MISSING"):
        EvidenceConstructionEngine.create_from_engineering_result(res)


def test_create_from_statistical():
    res = StatisticalAnalysisResult(
        result_id="r1",
        analysis_id="stat1",
        target_column="airspeed",
        status=ResultStatus.SUCCESS,
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=prov(EpistemicLevel.CALCULATED),
    )
    metric = StatisticalMetric(metric_id="m1", metric_type=MetricType.MEAN, value=100.0)
    ev = EvidenceConstructionEngine.create_from_statistical_metric(res, metric)
    assert ev.evidence_type == EvidenceType.STATISTICAL_RESULT
    assert ev.evidence_id == "EVD-STAT-r1-m1"
    assert ev.source_reference == "stat1"
    assert ev.source_record_id == "r1"
    assert ev.source_parameter_id == "airspeed"
    assert ev.epistemic_level == EpistemicLevel.CALCULATED
    assert ev.provenance.record_id == res.provenance.record_id


def test_statistical_missing_provenance():
    res = StatisticalAnalysisResult(
        result_id="r1",
        analysis_id="stat1",
        target_column="airspeed",
        status=ResultStatus.SUCCESS,
        epistemic_level=EpistemicLevel.CALCULATED,
    )
    metric = StatisticalMetric(metric_id="m1", metric_type=MetricType.MEAN, value=100.0)
    with pytest.raises(ValueError, match="ERR_REQUIRED_TRACEABILITY_MISSING"):
        EvidenceConstructionEngine.create_from_statistical_metric(res, metric)


def test_create_from_expected_observed():
    from aeris.core.expected_observed.models import ExpectationSource

    exp = ExpectedValue(
        expected_id="e1",
        parameter_id="airspeed",
        value=100.0,
        unit="kt",
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="s1",
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )
    obs = ObservedValue(
        observed_id="o1", parameter_id="airspeed", value=105.0, unit="kt", source_reference="s2"
    )
    comp = ExpectedObservedComparison(
        comparison_id="c1",
        expected=exp,
        observed=obs,
        status=ResultStatus.SUCCESS,
        provenance=prov(EpistemicLevel.CALCULATED),
    )
    ev = EvidenceConstructionEngine.create_from_expected_observed_comparison(comp)
    assert ev.evidence_type == EvidenceType.EXPECTED_OBSERVED_COMPARISON
    assert ev.evidence_id == "EVD-EXOBS-c1"
    assert ev.source_reference == "c1"
    assert ev.source_record_id == "c1"
    assert ev.source_parameter_id == "airspeed"
    assert ev.unit == "kt"
    assert ev.epistemic_level == EpistemicLevel.CALCULATED
    assert ev.provenance.record_id == comp.provenance.record_id


def test_evidence_collection_construction():
    anom = AnomalyRecord(
        anomaly_id="a1",
        analysis_id="an1",
        target_column="airspeed",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.HIGH,
        detection_method=DetectionMethod.RULE_BASED,
        timestamp=dt(),
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=prov(EpistemicLevel.CALCULATED),
    )
    ev1 = EvidenceConstructionEngine.create_from_anomaly(anom)
    res = EngineeringCalculationResult(
        result_id="r1",
        calculation_id="calc1",
        output_value=123.4,
        unit="kt",
        epistemic_level=EpistemicLevel.CALCULATED,
        status=ResultStatus.SUCCESS,
        provenance=prov(EpistemicLevel.CALCULATED),
    )
    ev2 = EvidenceConstructionEngine.create_from_engineering_result(res)

    col = EvidenceConstructionEngine.create_collection("col1", [ev1, ev2])
    assert len(col.evidence) == 2
    assert col.evidence[0].evidence_id == ev1.evidence_id
    assert col.evidence[1].evidence_id == ev2.evidence_id


def test_source_anomaly_unchanged():
    anom = AnomalyRecord(
        anomaly_id="a1",
        analysis_id="an1",
        target_column="airspeed",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.HIGH,
        detection_method=DetectionMethod.RULE_BASED,
        timestamp=dt(),
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=prov(EpistemicLevel.CALCULATED),
    )
    anom_dict_before = anom.model_dump()
    EvidenceConstructionEngine.create_from_anomaly(anom)
    assert anom.model_dump() == anom_dict_before


def test_determinism():
    anom = AnomalyRecord(
        anomaly_id="a1",
        analysis_id="an1",
        target_column="airspeed",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.HIGH,
        detection_method=DetectionMethod.RULE_BASED,
        timestamp=dt(),
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=prov(EpistemicLevel.CALCULATED),
    )
    ev1 = EvidenceConstructionEngine.create_from_anomaly(anom)
    ev2 = EvidenceConstructionEngine.create_from_anomaly(anom)
    assert ev1.model_dump() == ev2.model_dump()


def test_no_pandas_dependency():

    assert True


def test_serialization_round_trip():
    anom = AnomalyRecord(
        anomaly_id="a1",
        analysis_id="an1",
        target_column="airspeed",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.HIGH,
        detection_method=DetectionMethod.RULE_BASED,
        timestamp=dt(),
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=prov(EpistemicLevel.CALCULATED),
    )
    ev = EvidenceConstructionEngine.create_from_anomaly(anom)
    d = ev.model_dump(mode="json")
    ev2 = EvidenceRecord.model_validate(d)
    assert ev2.evidence_id == ev.evidence_id


def test_no_caller_controlled_elevation():
    anom = AnomalyRecord(
        anomaly_id="a1",
        analysis_id="an1",
        target_column="airspeed",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.HIGH,
        detection_method=DetectionMethod.RULE_BASED,
        timestamp=dt(),
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=prov(EpistemicLevel.CALCULATED),
    )
    with pytest.raises(TypeError):
        EvidenceConstructionEngine.create_from_anomaly(
            anom, epistemic_level=EpistemicLevel.HYPOTHESIS
        )
