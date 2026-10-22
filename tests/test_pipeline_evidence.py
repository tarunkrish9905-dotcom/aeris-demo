from aeris.core.anomaly.models import AnomalySeverity
from aeris.core.anomaly.threshold import ThresholdDetectionRequest, ThresholdOperator, ThresholdRule
from aeris.core.engineering.models import EngineeringCalculationDefinition
from aeris.core.evidence.pipeline import (
    EvidenceSourceSelection,
    PipelineEvidenceRequest,
    StatisticalMetricSelection,
)
from aeris.core.expected_observed.models import (
    ComparisonMethod,
    ExpectationSource,
    ExpectedValue,
    PipelineExpectedObservedRequest,
    ScalarComparisonRequest,
)
from aeris.core.pipeline.orchestrator import (
    DataPipelineOrchestrator,
    EngineeringCalculationRequest,
    PipelineAnomalyDetectionRequest,
)
from aeris.core.results.types import ResultStatus
from aeris.core.statistics.models import MetricType, StatisticalAnalysisRequest


def get_test_csv(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text(
        "timestamp,altitude_ft,airspeed_kts,pitch_deg,roll_deg,engine_rpm\n2026-01-01T12:00:00Z,5000,100,0,0,2000\n2026-01-01T12:00:10Z,5100,120,5,0,2100\n"
    )
    return p


def test_pipeline_evidence_optionality(tmp_path):
    p = get_test_csv(tmp_path)
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(p)
    assert res.status == ResultStatus.SUCCESS
    assert res.evidence is None


def test_pipeline_evidence_explicit_anomaly(tmp_path):
    p = get_test_csv(tmp_path)
    orch = DataPipelineOrchestrator()

    anom_req = PipelineAnomalyDetectionRequest(
        threshold_requests=(
            ThresholdDetectionRequest(
                analysis_id="th1",
                target_column="airspeed_kts",
                rule=ThresholdRule(
                    rule_id="r1",
                    target_column="airspeed_kts",
                    operator=ThresholdOperator.GREATER_THAN,
                    threshold_value=110.0,
                    severity=AnomalySeverity.HIGH,
                ),
            ),
        )
    )
    # First run without evidence to get the anomaly ID
    res1 = orch.run_csv_pipeline(p, anomaly_request=anom_req)
    anom_id = res1.anomaly.detector_results[0].anomalies[0].anomaly_id

    # Run again with evidence request
    ev_req = PipelineEvidenceRequest(
        collection_id="col1", selections=EvidenceSourceSelection(anomaly_ids=[anom_id])
    )
    res2 = orch.run_csv_pipeline(p, anomaly_request=anom_req, evidence_request=ev_req)

    assert res2.evidence is not None
    assert res2.evidence.status == ResultStatus.SUCCESS
    assert len(res2.evidence.collection.evidence) == 1
    ev = res2.evidence.collection.evidence[0]
    assert ev.source_record_id == anom_id


def test_pipeline_evidence_missing_anomaly(tmp_path):
    p = get_test_csv(tmp_path)
    orch = DataPipelineOrchestrator()
    ev_req = PipelineEvidenceRequest(
        collection_id="col1", selections=EvidenceSourceSelection(anomaly_ids=["MISSING_ID"])
    )
    res = orch.run_csv_pipeline(p, evidence_request=ev_req)
    assert res.evidence.status == ResultStatus.NOT_AVAILABLE
    assert len(res.evidence.issues) == 1


def test_pipeline_evidence_engineering(tmp_path):
    p = get_test_csv(tmp_path)
    orch = DataPipelineOrchestrator()
    eng_req = EngineeringCalculationRequest(
        calculation_definition=EngineeringCalculationDefinition(
            calculation_id="altitude_rate",
            name="A",
            description="A",
            output_parameter_id="A",
            output_unit="ft/s",
            input_parameter_ids=["altitude_ft"],
            metadata={
                "altitude_column": "altitude_ft",
                "timestamp_column": "timestamp",
                "altitude_unit": "ft",
            },
        )
    )
    from aeris.core.normalization.engine import NormalizationRequest

    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    orch.run_csv_pipeline(p, engineering_request=eng_req, normalization_request=norm)
    eng_id = eng_req.calculation_definition.calculation_id

    ev_req = PipelineEvidenceRequest(
        collection_id="col1", selections=EvidenceSourceSelection(engineering_result_ids=[eng_id])
    )
    res2 = orch.run_csv_pipeline(
        p, engineering_request=eng_req, normalization_request=norm, evidence_request=ev_req
    )
    assert res2.evidence.status == ResultStatus.SUCCESS
    assert len(res2.evidence.collection.evidence) == 1
    assert res2.evidence.collection.evidence[0].source_record_id.startswith("res_altitude_rate")


def test_pipeline_evidence_statistics(tmp_path):
    p = get_test_csv(tmp_path)
    orch = DataPipelineOrchestrator()
    stat_req = StatisticalAnalysisRequest(
        analysis_id="stat1", target_column="airspeed_kts", requested_metrics=[MetricType.MEAN]
    )
    res1 = orch.run_csv_pipeline(p, statistics_request=stat_req)

    ev_req = PipelineEvidenceRequest(
        collection_id="col1",
        selections=EvidenceSourceSelection(
            statistical_metrics=[
                StatisticalMetricSelection(
                    analysis_id="stat1", metric_id=res1.statistics.metrics[0].metric_id
                )
            ]
        ),
    )
    res2 = orch.run_csv_pipeline(p, statistics_request=stat_req, evidence_request=ev_req)
    assert res2.evidence.status == ResultStatus.SUCCESS
    assert len(res2.evidence.collection.evidence) == 1


def test_pipeline_evidence_expected_observed(tmp_path):
    p = get_test_csv(tmp_path)
    orch = DataPipelineOrchestrator()

    from aeris.core.provenance.tracker import EpistemicLevel

    exp = ExpectedValue(
        expected_id="e1",
        parameter_id="airspeed_kts",
        value=100.0,
        unit="kt",
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="s1",
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )
    req = PipelineExpectedObservedRequest(
        scalar_requests=[
            ScalarComparisonRequest(
                comparison_id="comp1",
                expected=exp,
                observed_parameter_id="airspeed_kts",
                observed_row_index=0,
                method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
            )
        ]
    )

    ev_req = PipelineEvidenceRequest(
        collection_id="col1",
        selections=EvidenceSourceSelection(expected_observed_comparison_ids=["comp1"]),
    )
    res = orch.run_csv_pipeline(p, expected_observed_request=req, evidence_request=ev_req)
    assert res.evidence.status == ResultStatus.SUCCESS
    assert len(res.evidence.collection.evidence) == 1
    assert res.evidence.collection.evidence[0].source_record_id == "comp1"


def test_pipeline_evidence_partial_failure(tmp_path):
    p = get_test_csv(tmp_path)
    orch = DataPipelineOrchestrator()
    stat_req = StatisticalAnalysisRequest(
        analysis_id="stat1", target_column="airspeed_kts", requested_metrics=[MetricType.MEAN]
    )
    res1 = orch.run_csv_pipeline(p, statistics_request=stat_req)

    ev_req = PipelineEvidenceRequest(
        collection_id="col1",
        selections=EvidenceSourceSelection(
            statistical_metrics=[
                StatisticalMetricSelection(
                    analysis_id="stat1", metric_id=res1.statistics.metrics[0].metric_id
                ),
                StatisticalMetricSelection(analysis_id="stat1", metric_id="NON_EXISTENT"),
            ]
        ),
    )
    res = orch.run_csv_pipeline(p, statistics_request=stat_req, evidence_request=ev_req)
    assert res.evidence.status == ResultStatus.PARTIAL
    assert len(res.evidence.issues) == 1
    assert len(res.evidence.collection.evidence) == 1
