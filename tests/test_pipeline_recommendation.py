"""
AERIS Controlled Recommendation Generation Pipeline Integration Test Suite.

Milestone 12, Step 12i Verification Suite.
Validates the controlled integration of RecommendationGenerationEngine into
DataPipelineOrchestrator as an explicit, optional pipeline stage.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.evidence.models import (
    EvidenceCollection,
    EvidenceRecord,
    EvidenceType,
)
from aeris.core.hypothesis.assessment_models import (
    HypothesisAssessmentConfiguration,
    HypothesisAssessmentResult,
)
from aeris.core.hypothesis.models import Hypothesis, HypothesisAssessmentState
from aeris.core.pipeline.orchestrator import (
    DataPipelineOrchestrator,
    run_csv_pipeline,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.recommendation.generation_models import (
    RecommendationGenerationPolicy,
    RecommendationGenerationRequest,
    RecommendationGenerationResult,
)
from aeris.core.recommendation.models import (
    Recommendation,
    RecommendationType,
)
from aeris.core.recommendation.pipeline import PipelineRecommendationRequest
from aeris.core.results.types import (
    QualityResult,
    ResultStatus,
    ValidationResult,
)


@pytest.fixture
def sample_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    csv_file = tmp_path / "valid_flight_data.csv"
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T12:00:00Z",
                "2026-01-01T12:00:01Z",
                "2026-01-01T12:00:02Z",
                "2026-01-01T12:00:03Z",
                "2026-01-01T12:00:04Z",
            ],
            "altitude_ft": [10000.0, 10020.0, 10040.0, 10060.0, 10080.0],
            "airspeed_kts": [250.0, 252.0, 251.0, 253.0, 250.0],
            "pitch_deg": [2.5, 2.6, 2.5, 2.4, 2.5],
            "roll_deg": [0.0, 0.1, -0.1, 0.0, 0.0],
            "engine_rpm": [2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def mock_prov() -> ProvenanceRecord:
    return ProvenanceRecord(
        source_uri="urn:aeris:test:fixture",
        activity_name="TestFixturePreparation",
        epistemic_level=EpistemicLevel.CALCULATED,
    )


@pytest.fixture
def sample_hypothesis(mock_prov: ProvenanceRecord) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="hyp-fcs-01",
        investigation_id="inv-alpha",
        title="Suspected Pitch Oscillations",
        statement="Elevator servo oscillation suspected during climb segment.",
        provenance=mock_prov,
    )


@pytest.fixture
def sample_assessment_unresolved(mock_prov: ProvenanceRecord) -> HypothesisAssessmentResult:
    return HypothesisAssessmentResult(
        hypothesis_id="hyp-fcs-01",
        assessment_state=HypothesisAssessmentState.UNRESOLVED,
        supporting_evidence_ids=("ev-01",),
        contradicting_evidence_ids=("ev-02",),
        configuration_used=HypothesisAssessmentConfiguration(),
        provenance=mock_prov,
        status=ResultStatus.SUCCESS,
    )


@pytest.fixture
def sample_evidence_collection(mock_prov: ProvenanceRecord) -> EvidenceCollection:
    ev_item = EvidenceRecord(
        evidence_id="ev-anom-01",
        evidence_type=EvidenceType.ANOMALY,
        description="Pitch oscillation detected beyond 2 Hz threshold",
        source_reference="flight_test_01",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=mock_prov,
    )
    return EvidenceCollection(
        collection_id="ev-coll-001",
        evidence=(ev_item,),
    )


# ==============================================================================
# Category A: Optional Stage Behavior & Execution Control
# ==============================================================================


def test_01_stage_skipped_when_request_is_none(sample_csv: pathlib.Path) -> None:
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(sample_csv)
    assert res.recommendation is None
    assert res.recommendation_result is None


def test_02_stage_executes_when_request_is_provided(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    orch = DataPipelineOrchestrator()
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = orch.run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert isinstance(res.recommendation, RecommendationGenerationResult)
    assert res.recommendation.status == ResultStatus.SUCCESS


def test_03_backward_compatibility_run_csv_pipeline_default_call(
    sample_csv: pathlib.Path,
) -> None:
    res = run_csv_pipeline(sample_csv)
    assert res.recommendation is None
    assert res.status in (ResultStatus.SUCCESS, ResultStatus.WARNING)


def test_04_convenience_function_run_csv_pipeline_supports_recommendations(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert res.recommendation.status == ResultStatus.SUCCESS
    assert len(res.recommendation.recommendations) > 0


def test_05_pipeline_result_has_recommendation_and_recommendation_result_property(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is res.recommendation_result


def test_06_upstream_stages_unaffected_when_recommendation_skipped(
    sample_csv: pathlib.Path,
) -> None:
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(sample_csv)
    assert res.ingestion is not None
    assert res.validation is not None
    assert res.quality is not None
    assert res.recommendation is None


def test_07_upstream_stages_normal_when_recommendation_executed(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.ingestion is not None
    assert res.ingestion.status == ResultStatus.SUCCESS
    assert res.validation is not None
    assert res.quality is not None
    assert res.recommendation is not None


def test_08_both_orchestrator_instance_and_function_yield_consistent_result(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    orch = DataPipelineOrchestrator()
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res1 = orch.run_csv_pipeline(sample_csv, recommendation_request=req)
    res2 = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res1.recommendation is not None
    assert res2.recommendation is not None
    assert len(res1.recommendation.recommendations) == len(res2.recommendation.recommendations)
    assert res1.recommendation.status == res2.recommendation.status


# ==============================================================================
# Category B: Input Binding & Upstream Stage Linking
# ==============================================================================


def test_09_explicit_hypothesis_assessment_input(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    rec_res = res.recommendation
    assert rec_res is not None
    assert any(
        r.recommendation_type == RecommendationType.REVIEW_EVIDENCE for r in rec_res.recommendations
    )


def test_10_explicit_recommendation_generation_request_direct_pass_through(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    direct_req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        investigation_id="direct-inv-01",
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=direct_req)
    assert res.recommendation is not None
    assert res.recommendation.recommendation_collection.investigation_id == "direct-inv-01"


def test_11_pipeline_validation_binding_clean_data(sample_csv: pathlib.Path) -> None:
    req = PipelineRecommendationRequest(use_pipeline_validation=True)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    # Clean validation should not trigger VERIFY_DATA
    assert not any(
        r.recommendation_type == RecommendationType.VERIFY_DATA
        for r in res.recommendation.recommendations
    )


def test_12_pipeline_validation_binding_defective_data_triggers_rule_a(
    tmp_path: pathlib.Path,
) -> None:
    bad_csv = tmp_path / "bad_timestamps.csv"
    pd.DataFrame(
        {
            "timestamp": ["bad-date-1", "bad-date-2"],
            "altitude_ft": [1000.0, 2000.0],
        }
    ).to_csv(bad_csv, index=False)

    req = PipelineRecommendationRequest(use_pipeline_validation=True)
    res = run_csv_pipeline(bad_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert any(
        r.recommendation_type == RecommendationType.VERIFY_DATA
        for r in res.recommendation.recommendations
    )


def test_13_pipeline_quality_binding_defective_data_triggers_rule_a(
    tmp_path: pathlib.Path,
) -> None:
    nulls_csv = tmp_path / "nulls_quality.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T12:00:00Z",
                "2026-01-01T12:00:01Z",
                "2026-01-01T12:00:02Z",
            ],
            "pitch_deg": [None, None, 1.0],
        }
    ).to_csv(nulls_csv, index=False)

    req = PipelineRecommendationRequest(use_pipeline_quality=True)
    res = run_csv_pipeline(nulls_csv, recommendation_request=req)
    assert res.recommendation is not None
    # Quality issues should be detected and evaluated
    assert "RULE_A_VERIFY_DATA" in res.recommendation.rules_evaluated


def test_14_pipeline_evidence_binding_with_evidence_collection(
    sample_csv: pathlib.Path,
    sample_evidence_collection: EvidenceCollection,
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        evidence_collection=sample_evidence_collection,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert any(
        r.recommendation_type == RecommendationType.REVIEW_EVIDENCE
        for r in res.recommendation.recommendations
    )


def test_15_explicit_validation_result_takes_precedence_over_pipeline_flag(
    sample_csv: pathlib.Path, mock_prov: ProvenanceRecord
) -> None:
    explicit_val = ValidationResult(
        is_valid=False,
        violation_count=3,
        validated_columns=["altitude_ft"],
        status=ResultStatus.WARNING,
        provenance=mock_prov,
    )
    req = PipelineRecommendationRequest(
        use_pipeline_validation=True,  # requested flag
        validation_result=explicit_val,  # but explicit supplied
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    # Explicit defective val triggers VERIFY_DATA
    assert any(
        r.recommendation_type == RecommendationType.VERIFY_DATA
        for r in res.recommendation.recommendations
    )


def test_16_explicit_quality_result_takes_precedence_over_pipeline_flag(
    sample_csv: pathlib.Path, mock_prov: ProvenanceRecord
) -> None:
    explicit_qual = QualityResult(
        quality_score=0.75,
        summary="Frequent sensor dropout in pitch channel.",
        status=ResultStatus.WARNING,
        provenance=mock_prov,
    )
    req = PipelineRecommendationRequest(
        use_pipeline_quality=True,
        quality_result=explicit_qual,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert any(
        r.recommendation_type == RecommendationType.VERIFY_DATA
        for r in res.recommendation.recommendations
    )


def test_17_combined_hypothesis_and_pipeline_validation_bindings(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        use_pipeline_validation=True,
        use_pipeline_quality=True,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert res.recommendation.status == ResultStatus.SUCCESS


def test_18_pipeline_evidence_binding_when_evidence_stage_was_skipped(
    sample_csv: pathlib.Path,
) -> None:
    req = PipelineRecommendationRequest(
        use_pipeline_evidence=True,
        use_pipeline_validation=True,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    # Evidence collection remained None, validation ran cleanly
    assert res.recommendation.status == ResultStatus.SUCCESS


def test_19_pipeline_validation_binding_when_validation_succeeded_cleanly(
    sample_csv: pathlib.Path,
) -> None:
    req = PipelineRecommendationRequest(use_pipeline_validation=True)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.validation is not None
    assert res.validation.is_valid is True
    assert res.recommendation is not None
    assert len(res.recommendation.recommendations) == 0


# ==============================================================================
# Category C: Stage Execution Ordering & Immutability Isolation
# ==============================================================================


def test_20_recommendation_stage_runs_after_all_prior_stages(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        use_pipeline_validation=True,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.ingestion is not None
    assert res.validation is not None
    assert res.quality is not None
    assert res.recommendation is not None


def test_21_recommendation_generation_does_not_mutate_ingestion_result(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    res_without = run_csv_pipeline(sample_csv)
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res_with = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res_without.ingestion.row_count == res_with.ingestion.row_count
    assert res_without.ingestion.column_names == res_with.ingestion.column_names


def test_22_recommendation_generation_does_not_mutate_validation_result(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    res_without = run_csv_pipeline(sample_csv)
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res_with = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res_without.validation is not None
    assert res_with.validation is not None
    assert res_without.validation.is_valid == res_with.validation.is_valid
    assert res_without.validation.violation_count == res_with.validation.violation_count


def test_23_recommendation_generation_does_not_mutate_quality_result(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    res_without = run_csv_pipeline(sample_csv)
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res_with = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res_without.quality is not None
    assert res_with.quality is not None
    assert res_without.quality.quality_score == res_with.quality.quality_score


def test_24_recommendation_generation_does_not_mutate_contributor_result(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    res = run_csv_pipeline(
        sample_csv,
        recommendation_request=PipelineRecommendationRequest(
            hypothesis_assessment=sample_assessment_unresolved
        ),
    )
    assert res.contributor_analysis is None


def test_25_recommendation_stage_does_not_retrigger_upstream_execution(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        use_pipeline_validation=True,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    # If retriggered, validation violations or row counts would duplicate
    assert res.validation is not None
    assert res.validation.violation_count == 0


def test_26_data_pipeline_result_is_immutable(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    with pytest.raises(ValidationError):
        res.recommendation.status = ResultStatus.ERROR  # type: ignore[misc]


def test_27_pipeline_recommendation_request_is_immutable(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    with pytest.raises(ValidationError):
        req.use_pipeline_validation = True  # type: ignore[misc]


# ==============================================================================
# Category D: Status Roll-up & Issue Aggregation
# ==============================================================================


def test_28_successful_recommendations_maintain_pipeline_success(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert res.recommendation.status == ResultStatus.SUCCESS
    assert res.status == ResultStatus.SUCCESS


def test_29_warning_recommendations_propagate_warning_status(
    sample_csv: pathlib.Path,
    sample_assessment_unresolved: HypothesisAssessmentResult,
    mock_prov: ProvenanceRecord,
) -> None:
    # Trigger 2 recommendations: Rule B from unresolved assessment + Rule A from invalid validation
    policy = RecommendationGenerationPolicy(max_recommendations=1)
    failed_val = ValidationResult(
        is_valid=False,
        violation_count=2,
        validated_columns=["pitch_deg"],
        provenance=mock_prov,
        status=ResultStatus.WARNING,
    )
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        validation_result=failed_val,
        policy=policy,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert res.recommendation.status == ResultStatus.WARNING
    assert res.status == ResultStatus.WARNING


def test_30_truncated_recommendations_propagate_warning_status(
    sample_csv: pathlib.Path,
    sample_assessment_unresolved: HypothesisAssessmentResult,
    mock_prov: ProvenanceRecord,
) -> None:
    policy = RecommendationGenerationPolicy(max_recommendations=1)
    failed_val = ValidationResult(
        is_valid=False,
        violation_count=2,
        validated_columns=["pitch_deg"],
        provenance=mock_prov,
        status=ResultStatus.WARNING,
    )
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        validation_result=failed_val,
        policy=policy,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert any(i.issue_code == "WARN_MAX_RECOMMENDATIONS_EXCEEDED" for i in res.issues)


def test_31_error_in_recommendation_engine_propagates_error_status(
    sample_csv: pathlib.Path,
) -> None:
    # Pass an invalid object to recommendation_request to trigger error handling
    res = run_csv_pipeline(sample_csv, recommendation_request="invalid_type")  # type: ignore[arg-type]
    assert res.status == ResultStatus.ERROR
    assert res.recommendation is not None
    assert res.recommendation.status == ResultStatus.ERROR


def test_32_recommendation_issues_aggregated_into_pipeline_issues(
    sample_csv: pathlib.Path,
    sample_assessment_unresolved: HypothesisAssessmentResult,
    mock_prov: ProvenanceRecord,
) -> None:
    policy = RecommendationGenerationPolicy(max_recommendations=1)
    failed_val = ValidationResult(
        is_valid=False,
        violation_count=2,
        validated_columns=["pitch_deg"],
        provenance=mock_prov,
        status=ResultStatus.WARNING,
    )
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        validation_result=failed_val,
        policy=policy,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert any(i.issue_code == "WARN_MAX_RECOMMENDATIONS_EXCEEDED" for i in res.issues)


def test_33_upstream_issues_preserved_alongside_recommendation_issues(
    tmp_path: pathlib.Path,
    sample_assessment_unresolved: HypothesisAssessmentResult,
    mock_prov: ProvenanceRecord,
) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["invalid"], "altitude_ft": [100.0]}).to_csv(bad_csv, index=False)
    policy = RecommendationGenerationPolicy(max_recommendations=1)
    failed_val = ValidationResult(
        is_valid=False,
        violation_count=2,
        validated_columns=["pitch_deg"],
        provenance=mock_prov,
        status=ResultStatus.WARNING,
    )
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        validation_result=failed_val,
        policy=policy,
        use_pipeline_validation=False,
    )
    res = run_csv_pipeline(bad_csv, recommendation_request=req)
    codes = [i.issue_code for i in res.issues]
    assert "WARN_MAX_RECOMMENDATIONS_EXCEEDED" in codes
    assert res.status == ResultStatus.ERROR


def test_34_skipped_recommendation_does_not_inject_issues(sample_csv: pathlib.Path) -> None:
    res = run_csv_pipeline(sample_csv, recommendation_request=None)
    assert not any(i.issue_code.startswith("REC_") for i in res.issues)


def test_35_graceful_handling_of_invalid_recommendation_request_type(
    sample_csv: pathlib.Path,
) -> None:
    res = run_csv_pipeline(sample_csv, recommendation_request=12345)  # type: ignore[arg-type]
    assert res.status == ResultStatus.ERROR
    assert res.recommendation is not None
    assert res.recommendation.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_RECOMMENDATION_STAGE_FAILED" for i in res.issues)


def test_36_failed_recommendation_stage_contains_informative_error_issue(
    sample_csv: pathlib.Path,
) -> None:
    res = run_csv_pipeline(sample_csv, recommendation_request=object())  # type: ignore[arg-type]
    err_issues = [i for i in res.issues if i.issue_code == "ERR_RECOMMENDATION_STAGE_FAILED"]
    assert len(err_issues) > 0
    assert "Unsupported recommendation_request type" in err_issues[0].message


def test_36a_recommendation_error_cannot_turn_pipeline_error_into_success(
    tmp_path: pathlib.Path,
) -> None:
    bad_csv = tmp_path / "corrupt.csv"
    bad_csv.write_text("invalid,header\n")
    # Pipeline fails upstream with ERROR. Pass invalid recommendation request type.
    res = run_csv_pipeline(bad_csv, recommendation_request="invalid_type")  # type: ignore[arg-type]
    assert res.status == ResultStatus.ERROR
    assert res.recommendation is not None
    assert res.recommendation.status == ResultStatus.ERROR


def test_36b_pipeline_warning_upgraded_to_partial_by_partial_recommendation(
    sample_csv: pathlib.Path,
    sample_assessment_unresolved: HypothesisAssessmentResult,
    mock_prov: ProvenanceRecord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aeris.core.recommendation.generation_engine import RecommendationGenerationEngine
    from aeris.core.recommendation.models import RecommendationCollection

    warn_qual = QualityResult(
        quality_score=0.80,
        summary="Slight dropout",
        status=ResultStatus.WARNING,
        provenance=mock_prov,
    )
    partial_rec_res = RecommendationGenerationResult(
        recommendations=(),
        recommendation_collection=RecommendationCollection(
            collection_id="rec-coll-partial",
            recommendations=(),
            provenance=mock_prov,
        ),
        rules_evaluated=(),
        rules_triggered=(),
        policy_used=RecommendationGenerationPolicy(),
        provenance=mock_prov,
        status=ResultStatus.PARTIAL,
        issues=[],
    )
    monkeypatch.setattr(RecommendationGenerationEngine, "generate", lambda req: partial_rec_res)

    res = run_csv_pipeline(
        sample_csv,
        recommendation_request=PipelineRecommendationRequest(
            quality_result=warn_qual,
            use_pipeline_validation=True,
        ),
    )
    assert res.status == ResultStatus.PARTIAL


def test_36c_errors_do_not_produce_empty_successful_collections(
    sample_csv: pathlib.Path,
) -> None:
    res = run_csv_pipeline(sample_csv, recommendation_request=object())  # type: ignore[arg-type]
    assert res.recommendation is not None
    assert res.recommendation.status == ResultStatus.ERROR
    assert res.recommendation.status != ResultStatus.SUCCESS
    assert len(res.recommendation.recommendations) == 0


def test_36d_issue_aggregation_does_not_duplicate_stage_issues(
    sample_csv: pathlib.Path,
) -> None:
    res = run_csv_pipeline(sample_csv, recommendation_request="bad_request")  # type: ignore[arg-type]
    stage_issues = res.recommendation.issues if res.recommendation else []
    pipeline_err_issues = [
        i for i in res.issues if i.issue_code == "ERR_RECOMMENDATION_STAGE_FAILED"
    ]
    # Verify exactly 1 instance in res.issues
    assert len(pipeline_err_issues) == 1
    assert len(stage_issues) == 1
    assert pipeline_err_issues[0] == stage_issues[0]


# ==============================================================================
# Category E: Epistemic Integrity & Provenance Chaining
# ==============================================================================


def test_37_recommendation_epistemic_level_is_strictly_recommendation(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    rec_res = res.recommendation
    assert rec_res is not None
    assert rec_res.provenance.epistemic_level == EpistemicLevel.RECOMMENDATION
    for rec in rec_res.recommendations:
        assert rec.provenance.epistemic_level == EpistemicLevel.RECOMMENDATION


def test_38_recommendation_provenance_chains_parent_records(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    parent_ids = res.recommendation.provenance.parent_record_ids
    assert len(parent_ids) > 0


def test_39_pipeline_provenance_contains_recommendation_record_id(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert res.provenance is not None
    assert res.recommendation.provenance.record_id in res.provenance.parent_record_ids


def test_40_epistemic_level_never_claims_fact_or_calculated(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert res.recommendation.provenance.epistemic_level != EpistemicLevel.OBSERVED
    assert res.recommendation.provenance.epistemic_level != EpistemicLevel.CALCULATED


def test_41_recommendation_collection_provenance_chains_to_each_recommendation(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    coll = res.recommendation.recommendation_collection
    rec_ids = [r.provenance.record_id for r in res.recommendation.recommendations]
    for r_id in rec_ids:
        assert r_id in coll.provenance.parent_record_ids


def test_42_provenance_source_uri_recorded_correctly(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert res.recommendation.provenance.source_uri == "urn:aeris:engine:recommendation:execution"


def test_43_provenance_parameters_record_rule_and_recommendation_counts(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    params = res.recommendation.provenance.parameters
    assert "rules_evaluated_count" in params
    assert "recommendation_count" in params


def test_44_epistemic_level_validation_failure_cannot_bypass(
    mock_prov: ProvenanceRecord,
) -> None:
    with pytest.raises(ValidationError, match="ERR_EPISTEMIC_MISMATCH"):
        Recommendation(
            recommendation_id="rec-invalid",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Invalid",
            description="Invalid level",
            rationale="Test rationale",
            epistemic_level=EpistemicLevel.OBSERVED,  # Must be RECOMMENDATION
            provenance=mock_prov,
        )


# ==============================================================================
# Category F: Policy Configuration & Rule Selectivity
# ==============================================================================


def test_45_pipeline_custom_policy_disables_rule_a(
    tmp_path: pathlib.Path,
) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["bad"]}).to_csv(bad_csv, index=False)
    policy = RecommendationGenerationPolicy(enable_data_verification_rule=False)
    req = PipelineRecommendationRequest(
        use_pipeline_validation=True,
        policy=policy,
    )
    res = run_csv_pipeline(bad_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert not any(
        r.recommendation_type == RecommendationType.VERIFY_DATA
        for r in res.recommendation.recommendations
    )


def test_46_pipeline_custom_policy_enables_repeat_analysis(
    sample_csv: pathlib.Path, mock_prov: ProvenanceRecord
) -> None:
    partial_asmt = HypothesisAssessmentResult(
        hypothesis_id="hyp-partial-99",
        assessment_state=HypothesisAssessmentState.INSUFFICIENT_EVIDENCE,
        missing_evidence_ids=("ev-missing-1",),
        configuration_used=HypothesisAssessmentConfiguration(),
        provenance=mock_prov,
        status=ResultStatus.PARTIAL,
    )
    policy = RecommendationGenerationPolicy(
        enable_repeat_analysis_rule=True,
        enable_additional_evidence_rule=False,
    )
    req = PipelineRecommendationRequest(
        hypothesis_assessment=partial_asmt,
        policy=policy,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert any(
        r.recommendation_type == RecommendationType.REPEAT_ANALYSIS
        for r in res.recommendation.recommendations
    )


def test_47_pipeline_custom_policy_enables_configuration_review(
    sample_csv: pathlib.Path, mock_prov: ProvenanceRecord
) -> None:
    policy = RecommendationGenerationPolicy(enable_configuration_review_rule=True)
    hyp_cfg = Hypothesis(
        hypothesis_id="hyp-cfg-01",
        investigation_id="inv-cfg",
        title="Configuration Mismatch",
        statement="Suspected vehicle configuration flap index error.",
        provenance=mock_prov,
    )
    req = PipelineRecommendationRequest(
        hypothesis=hyp_cfg,
        policy=policy,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert "RULE_F_CHECK_CONFIGURATION_RECORD" in res.recommendation.rules_evaluated


def test_48_pipeline_custom_policy_max_recommendations_limit(
    sample_csv: pathlib.Path,
    sample_assessment_unresolved: HypothesisAssessmentResult,
    mock_prov: ProvenanceRecord,
) -> None:
    policy = RecommendationGenerationPolicy(max_recommendations=1)
    failed_val = ValidationResult(
        is_valid=False,
        violation_count=2,
        validated_columns=["pitch_deg"],
        provenance=mock_prov,
        status=ResultStatus.WARNING,
    )
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        validation_result=failed_val,
        policy=policy,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert len(res.recommendation.recommendations) == 1
    assert res.recommendation.status == ResultStatus.WARNING


def test_49_pipeline_policy_all_rules_disabled_yields_empty_recommendations(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    policy = RecommendationGenerationPolicy(
        enable_data_verification_rule=False,
        enable_evidence_review_rule=False,
        enable_additional_evidence_rule=False,
        enable_engineering_review_rule=False,
        enable_repeat_analysis_rule=False,
        enable_configuration_review_rule=False,
    )
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        policy=policy,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert len(res.recommendation.recommendations) == 0
    assert res.recommendation.status == ResultStatus.SUCCESS


def test_50_pipeline_policy_metadata_recorded(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    policy = RecommendationGenerationPolicy(metadata={"policy_tag": "flt_test_v1"})
    req = PipelineRecommendationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        policy=policy,
    )
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert res.recommendation.policy_used.metadata["policy_tag"] == "flt_test_v1"


def test_51_rules_evaluated_matches_policy_toggles(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    assert "RULE_A_VERIFY_DATA" in res.recommendation.rules_evaluated
    assert "RULE_B_REVIEW_EVIDENCE" in res.recommendation.rules_evaluated


def test_52_rules_triggered_deterministic_in_pipeline(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res1 = run_csv_pipeline(sample_csv, recommendation_request=req)
    res2 = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res1.recommendation is not None
    assert res2.recommendation is not None
    assert res1.recommendation.rules_triggered == res2.recommendation.rules_triggered


# ==============================================================================
# Category G: Negative Scenarios, Edge Cases & Safety Safeguards
# ==============================================================================


def test_53_empty_pipeline_recommendation_request_raises_validation_error() -> None:
    with pytest.raises(ValueError, match="ERR_EMPTY_REQUEST"):
        PipelineRecommendationRequest()


def test_54_empty_investigation_id_raises_validation_error(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    with pytest.raises(ValidationError):
        PipelineRecommendationRequest(
            hypothesis_assessment=sample_assessment_unresolved,
            investigation_id="",
        )


def test_55_whitespace_investigation_id_raises_validation_error(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    with pytest.raises(ValidationError):
        PipelineRecommendationRequest(
            hypothesis_assessment=sample_assessment_unresolved,
            investigation_id="   ",
        )


def test_56_safeguard_disallows_root_cause_claim_in_metadata(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    with pytest.raises(ValidationError, match="ERR_METADATA_SAFEGUARD"):
        PipelineRecommendationRequest(
            hypothesis_assessment=sample_assessment_unresolved,
            metadata={"root_cause": "Hydraulic failure confirmed"},
        )


def test_57_safeguard_disallows_actionable_commands_in_metadata(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    with pytest.raises(ValidationError, match="ERR_METADATA_SAFEGUARD"):
        PipelineRecommendationRequest(
            hypothesis_assessment=sample_assessment_unresolved,
            metadata={"control_command": "reboot_avionics"},
        )


def test_58_nonexistent_csv_file_with_recommendation_requested(
    tmp_path: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(tmp_path / "nonexistent.csv", recommendation_request=req)
    assert res.status == ResultStatus.ERROR
    assert res.ingestion.status == ResultStatus.ERROR


def test_59_unsupported_request_type_handled_without_crash(sample_csv: pathlib.Path) -> None:
    res = run_csv_pipeline(sample_csv, recommendation_request={"not": "valid"})  # type: ignore[arg-type]
    assert res.status == ResultStatus.ERROR
    assert res.recommendation is not None
    assert res.recommendation.status == ResultStatus.ERROR


def test_60_recommendation_ids_are_unique_and_deterministic(
    sample_csv: pathlib.Path, sample_assessment_unresolved: HypothesisAssessmentResult
) -> None:
    req = PipelineRecommendationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = run_csv_pipeline(sample_csv, recommendation_request=req)
    assert res.recommendation is not None
    rec_ids = [r.recommendation_id for r in res.recommendation.recommendations]
    assert len(rec_ids) == len(set(rec_ids))
