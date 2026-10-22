"""
Unit tests for Milestone 12, Step 12h: Controlled Engineering Recommendation Generation Engine.

Validates deterministic rule evaluation, policy gating, source traceability,
provenance preservation, safety safeguards, and immutability.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aeris.core.hypothesis.assessment_models import (
    HypothesisAssessmentConfiguration,
    HypothesisAssessmentResult,
)
from aeris.core.hypothesis.models import HypothesisAssessmentState
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.recommendation import (
    RecommendationGenerationEngine,
    RecommendationGenerationPolicy,
    RecommendationGenerationRequest,
    RecommendationStatus,
    RecommendationType,
)
from aeris.core.results.types import QualityResult, ResultStatus, ValidationResult


@pytest.fixture
def mock_prov() -> ProvenanceRecord:
    return ProvenanceRecord(
        source_uri="urn:aeris:test:upstream",
        activity_name="TestActivity",
        epistemic_level=EpistemicLevel.CALCULATED,
    )


@pytest.fixture
def sample_assessment_unresolved(mock_prov: ProvenanceRecord) -> HypothesisAssessmentResult:
    return HypothesisAssessmentResult(
        hypothesis_id="hyp-flt-401",
        assessment_state=HypothesisAssessmentState.UNRESOLVED,
        supporting_evidence_ids=("ev-01",),
        contradicting_evidence_ids=("ev-02",),
        configuration_used=HypothesisAssessmentConfiguration(),
        provenance=mock_prov,
        status=ResultStatus.SUCCESS,
    )


@pytest.fixture
def sample_assessment_insufficient(mock_prov: ProvenanceRecord) -> HypothesisAssessmentResult:
    return HypothesisAssessmentResult(
        hypothesis_id="hyp-flt-402",
        assessment_state=HypothesisAssessmentState.INSUFFICIENT_EVIDENCE,
        supporting_evidence_ids=(),
        contradicting_evidence_ids=(),
        configuration_used=HypothesisAssessmentConfiguration(minimum_total_evidence=2),
        provenance=mock_prov,
        status=ResultStatus.SUCCESS,
    )


@pytest.fixture
def sample_assessment_inconclusive(mock_prov: ProvenanceRecord) -> HypothesisAssessmentResult:
    return HypothesisAssessmentResult(
        hypothesis_id="hyp-flt-403",
        assessment_state=HypothesisAssessmentState.INCONCLUSIVE,
        supporting_evidence_ids=(),
        contradicting_evidence_ids=(),
        relevant_neutral_evidence_ids=("ev-10",),
        configuration_used=HypothesisAssessmentConfiguration(),
        provenance=mock_prov,
        status=ResultStatus.SUCCESS,
    )


@pytest.fixture
def sample_assessment_supported(mock_prov: ProvenanceRecord) -> HypothesisAssessmentResult:
    return HypothesisAssessmentResult(
        hypothesis_id="hyp-flt-404",
        assessment_state=HypothesisAssessmentState.SUPPORTED,
        supporting_evidence_ids=("ev-20", "ev-21"),
        contradicting_evidence_ids=(),
        configuration_used=HypothesisAssessmentConfiguration(),
        provenance=mock_prov,
        status=ResultStatus.SUCCESS,
    )


@pytest.fixture
def sample_validation_failed(mock_prov: ProvenanceRecord) -> ValidationResult:
    return ValidationResult(
        is_valid=False,
        violation_count=3,
        validated_row_count=100,
        validated_columns=["time", "pitch", "roll"],
        provenance=mock_prov,
        status=ResultStatus.WARNING,
    )


@pytest.fixture
def sample_quality_degraded(mock_prov: ProvenanceRecord) -> QualityResult:
    return QualityResult(
        quality_score=0.75,
        summary="Frequent sensor dropout in pitch rate channel.",
        provenance=mock_prov,
        status=ResultStatus.WARNING,
    )


# ---------------------------------------------------------------------------
# 1. Request Contract & Input Validation
# ---------------------------------------------------------------------------


def test_01_empty_request_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        RecommendationGenerationRequest()
    assert "ERR_EMPTY_REQUEST" in str(exc.value)


def test_02_valid_request_with_assessment(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        investigation_id="inv-101",
    )
    assert req.hypothesis_assessment is not None
    assert req.investigation_id == "inv-101"


def test_03_valid_request_with_validation(sample_validation_failed: ValidationResult) -> None:
    req = RecommendationGenerationRequest(validation_result=sample_validation_failed)
    assert req.validation_result is not None


def test_04_valid_request_with_quality(sample_quality_degraded: QualityResult) -> None:
    req = RecommendationGenerationRequest(quality_result=sample_quality_degraded)
    assert req.quality_result is not None


def test_05_request_extra_fields_forbidden(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationRequest(
            hypothesis_assessment=sample_assessment_unresolved,
            unexpected_field="disallowed",  # type: ignore[call-arg]
        )


def test_06_request_invalid_metadata_rejected(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationRequest(
            hypothesis_assessment=sample_assessment_unresolved,
            metadata={"root_cause": "true"},
        )


# ---------------------------------------------------------------------------
# 2. Rule Evaluation: Rule A (VERIFY_DATA)
# ---------------------------------------------------------------------------


def test_07_rule_a_fires_on_failed_validation(
    sample_validation_failed: ValidationResult,
) -> None:
    req = RecommendationGenerationRequest(validation_result=sample_validation_failed)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.VERIFY_DATA
    assert "Verify Data Integrity" in rec.title
    assert rec.other_assessment_ids == (sample_validation_failed.provenance.record_id,)
    assert RecommendationGenerationEngine.RULE_A_VERIFY_DATA in res.rules_triggered


def test_08_rule_a_fires_on_degraded_quality(
    sample_quality_degraded: QualityResult,
) -> None:
    req = RecommendationGenerationRequest(quality_result=sample_quality_degraded)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.VERIFY_DATA
    assert "Signal Quality" in rec.title
    assert rec.other_assessment_ids == (sample_quality_degraded.provenance.record_id,)


def test_09_rule_a_does_not_fire_when_disabled(
    sample_validation_failed: ValidationResult,
) -> None:
    policy = RecommendationGenerationPolicy(enable_data_verification_rule=False)
    req = RecommendationGenerationRequest(validation_result=sample_validation_failed, policy=policy)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 0
    assert RecommendationGenerationEngine.RULE_A_VERIFY_DATA not in res.rules_triggered


def test_10_rule_a_does_not_fire_on_clean_validation(mock_prov: ProvenanceRecord) -> None:
    clean_val = ValidationResult(
        is_valid=True,
        violation_count=0,
        validated_row_count=100,
        provenance=mock_prov,
        status=ResultStatus.SUCCESS,
    )
    req = RecommendationGenerationRequest(validation_result=clean_val)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 0


# ---------------------------------------------------------------------------
# 3. Rule Evaluation: Rule B (REVIEW_EVIDENCE)
# ---------------------------------------------------------------------------


def test_11_rule_b_fires_on_unresolved_assessment(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.REVIEW_EVIDENCE
    assert "hyp-flt-401" in rec.title
    assert "ev-01" in rec.evidence_ids
    assert "ev-02" in rec.evidence_ids
    assert rec.hypothesis_ids == ("hyp-flt-401",)
    assert RecommendationGenerationEngine.RULE_B_REVIEW_EVIDENCE in res.rules_triggered


def test_12_rule_b_does_not_fire_when_disabled(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    policy = RecommendationGenerationPolicy(enable_evidence_review_rule=False)
    req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_unresolved, policy=policy
    )
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 0


def test_13_rule_b_does_not_fire_on_supported_assessment(
    sample_assessment_supported: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_supported)
    res = RecommendationGenerationEngine.generate(req)
    # Supported hypothesis with no contradiction does not need conflicting evidence review
    assert not any(
        r.recommendation_type == RecommendationType.REVIEW_EVIDENCE for r in res.recommendations
    )


# ---------------------------------------------------------------------------
# 4. Rule Evaluation: Rule C (COLLECT_ADDITIONAL_EVIDENCE)
# ---------------------------------------------------------------------------


def test_14_rule_c_fires_on_insufficient_evidence(
    sample_assessment_insufficient: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_insufficient)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.COLLECT_ADDITIONAL_EVIDENCE
    assert "Collect Additional Evidence" in rec.title
    assert rec.hypothesis_ids == ("hyp-flt-402",)
    assert RecommendationGenerationEngine.RULE_C_COLLECT_ADDITIONAL_EVIDENCE in res.rules_triggered


def test_15_rule_c_does_not_fire_when_disabled(
    sample_assessment_insufficient: HypothesisAssessmentResult,
) -> None:
    policy = RecommendationGenerationPolicy(enable_additional_evidence_rule=False)
    req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_insufficient, policy=policy
    )
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 0


# ---------------------------------------------------------------------------
# 5. Rule Evaluation: Rule D (REQUEST_ENGINEERING_REVIEW)
# ---------------------------------------------------------------------------


def test_16_rule_d_fires_on_inconclusive_assessment(
    sample_assessment_inconclusive: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_inconclusive)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.REQUEST_ENGINEERING_REVIEW
    assert "Disciplinary Engineering Review" in rec.title
    assert RecommendationGenerationEngine.RULE_D_REQUEST_ENGINEERING_REVIEW in res.rules_triggered


def test_17_rule_d_does_not_fire_when_disabled(
    sample_assessment_inconclusive: HypothesisAssessmentResult,
) -> None:
    policy = RecommendationGenerationPolicy(enable_engineering_review_rule=False)
    req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_inconclusive, policy=policy
    )
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 0


# ---------------------------------------------------------------------------
# 6. Rule Evaluation: Rule E (REPEAT_ANALYSIS) [Gated / Deferred]
# ---------------------------------------------------------------------------


def test_18_rule_e_fires_when_enabled_on_partial_assessment(mock_prov: ProvenanceRecord) -> None:
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
        enable_additional_evidence_rule=False,  # Isolate Rule E
    )
    req = RecommendationGenerationRequest(hypothesis_assessment=partial_asmt, policy=policy)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.REPEAT_ANALYSIS
    assert "Repeat Assessment Analysis" in rec.title


def test_19_rule_e_does_not_fire_when_disabled_default(mock_prov: ProvenanceRecord) -> None:
    partial_asmt = HypothesisAssessmentResult(
        hypothesis_id="hyp-partial-99",
        assessment_state=HypothesisAssessmentState.INSUFFICIENT_EVIDENCE,
        missing_evidence_ids=("ev-missing-1",),
        configuration_used=HypothesisAssessmentConfiguration(),
        provenance=mock_prov,
        status=ResultStatus.PARTIAL,
    )
    policy = RecommendationGenerationPolicy(enable_additional_evidence_rule=False)
    assert policy.enable_repeat_analysis_rule is False
    req = RecommendationGenerationRequest(hypothesis_assessment=partial_asmt, policy=policy)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 0


# ---------------------------------------------------------------------------
# 7. Policy Limit & Truncation
# ---------------------------------------------------------------------------


def test_20_policy_max_recommendations_limit_applied(
    sample_assessment_unresolved: HypothesisAssessmentResult,
    sample_validation_failed: ValidationResult,
    sample_quality_degraded: QualityResult,
) -> None:
    policy = RecommendationGenerationPolicy(max_recommendations=1)
    req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        validation_result=sample_validation_failed,
        quality_result=sample_quality_degraded,
        policy=policy,
    )
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 1
    assert res.status == ResultStatus.WARNING
    assert any(i.issue_code == "WARN_MAX_RECOMMENDATIONS_EXCEEDED" for i in res.issues)


# ---------------------------------------------------------------------------
# 8. Determinism & Immutability
# ---------------------------------------------------------------------------


def test_21_deterministic_repeated_execution(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    r1 = RecommendationGenerationEngine.generate(req)
    r2 = RecommendationGenerationEngine.generate(req)
    assert len(r1.recommendations) == len(r2.recommendations)
    assert r1.recommendations[0].recommendation_id == r2.recommendations[0].recommendation_id
    assert r1.recommendations[0].title == r2.recommendations[0].title
    assert r1.recommendations[0].rationale == r2.recommendations[0].rationale


def test_22_policy_immutability() -> None:
    policy = RecommendationGenerationPolicy()
    with pytest.raises((ValidationError, TypeError)):
        policy.enable_data_verification_rule = False  # type: ignore[misc]


def test_23_request_immutability(sample_assessment_unresolved: HypothesisAssessmentResult) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    with pytest.raises((ValidationError, TypeError)):
        req.hypothesis_assessment = None  # type: ignore[misc]


def test_24_result_immutability(sample_assessment_unresolved: HypothesisAssessmentResult) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    with pytest.raises((ValidationError, TypeError)):
        res.recommendations = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 9. Provenance & Epistemic Integrity
# ---------------------------------------------------------------------------


def test_25_provenance_parent_linkage_preserved(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    rec = res.recommendations[0]
    assert sample_assessment_unresolved.provenance.record_id in rec.provenance.parent_record_ids
    assert rec.provenance.epistemic_level == EpistemicLevel.RECOMMENDATION
    assert res.provenance.epistemic_level == EpistemicLevel.RECOMMENDATION


def test_26_investigation_id_propagated(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        investigation_id="inv-alpha-77",
    )
    res = RecommendationGenerationEngine.generate(req)
    rec = res.recommendations[0]
    assert rec.investigation_id == "inv-alpha-77"
    assert res.recommendation_collection.investigation_id == "inv-alpha-77"


# ---------------------------------------------------------------------------
# 10. Safety Boundaries & Safeguards
# ---------------------------------------------------------------------------


def test_27_no_root_cause_in_generated_recommendations(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    rec = res.recommendations[0]
    dump = str(rec.model_dump())
    assert "root_cause" not in dump


def test_28_no_confidence_or_probability_in_result(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    dump = str(res.model_dump())
    assert "probability" not in dump
    assert "confidence" not in dump


def test_29_recommendation_status_is_strictly_proposed(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    for rec in res.recommendations:
        assert rec.status == RecommendationStatus.PROPOSED


def test_30_rules_evaluated_and_triggered_tracked(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.rules_evaluated) >= 4
    assert RecommendationGenerationEngine.RULE_B_REVIEW_EVIDENCE in res.rules_triggered


# ---------------------------------------------------------------------------
# 11. Additional Comprehensive Boundary Tests (Tests 31-55)
# ---------------------------------------------------------------------------


def test_31_multiple_rules_fire_simultaneously(
    sample_assessment_unresolved: HypothesisAssessmentResult,
    sample_validation_failed: ValidationResult,
) -> None:
    req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        validation_result=sample_validation_failed,
    )
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 2
    types = {r.recommendation_type for r in res.recommendations}
    assert RecommendationType.VERIFY_DATA in types
    assert RecommendationType.REVIEW_EVIDENCE in types


def test_32_collection_contains_identical_recommendations(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    assert res.recommendation_collection.recommendations == res.recommendations


def test_33_status_success_on_valid_generation(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    assert res.status == ResultStatus.SUCCESS
    assert len(res.issues) == 0


def test_34_status_success_when_no_rules_fire(
    sample_assessment_supported: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_supported)
    res = RecommendationGenerationEngine.generate(req)
    assert res.status == ResultStatus.SUCCESS
    assert len(res.recommendations) == 0


def test_35_rule_f_evaluated_and_deferred(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    assert RecommendationGenerationEngine.RULE_F_CHECK_CONFIGURATION_RECORD in res.rules_evaluated
    assert (
        RecommendationGenerationEngine.RULE_F_CHECK_CONFIGURATION_RECORD not in res.rules_triggered
    )


def test_36_constraints_and_prerequisites_populated(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    rec = res.recommendations[0]
    assert len(rec.constraints) > 0
    assert len(rec.prerequisites) > 0


def test_37_serialization_roundtrip_preserves_structure(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    data = res.model_dump()
    assert "recommendations" in data
    assert "recommendation_collection" in data
    assert "rules_evaluated" in data
    assert "rules_triggered" in data


def test_38_policy_metadata_safeguard() -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationPolicy(metadata={"failure_probability": "0.1"})


def test_39_policy_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationPolicy(unsupported_extra_field="test")  # type: ignore[call-arg]


def test_40_request_whitespace_investigation_id_rejected(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationRequest(
            hypothesis_assessment=sample_assessment_unresolved,
            investigation_id="   ",
        )


def test_41_rule_d_fires_on_weakly_supported_assessment(
    mock_prov: ProvenanceRecord,
) -> None:
    weak_asmt = HypothesisAssessmentResult(
        hypothesis_id="hyp-weak-01",
        assessment_state=HypothesisAssessmentState.WEAKLY_SUPPORTED,
        supporting_evidence_ids=("ev-01",),
        contradicting_evidence_ids=("ev-02",),
        configuration_used=HypothesisAssessmentConfiguration(),
        provenance=mock_prov,
        status=ResultStatus.SUCCESS,
    )
    req = RecommendationGenerationRequest(
        hypothesis_assessment=weak_asmt,
        policy=RecommendationGenerationPolicy(enable_evidence_review_rule=False),
    )
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.REQUEST_ENGINEERING_REVIEW


def test_42_parent_record_ids_in_result_provenance(
    sample_assessment_unresolved: HypothesisAssessmentResult,
    sample_validation_failed: ValidationResult,
) -> None:
    req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        validation_result=sample_validation_failed,
    )
    res = RecommendationGenerationEngine.generate(req)
    assert sample_assessment_unresolved.provenance.record_id in res.provenance.parent_record_ids
    assert sample_validation_failed.provenance.record_id in res.provenance.parent_record_ids


def test_43_collection_id_is_non_empty(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendation_collection.collection_id) > 0


def test_44_no_dataframes_in_generation_result(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    for k in res.model_dump():
        assert "dataframe" not in k
        assert "df" not in k


def test_45_empty_quality_summary_handled_cleanly(mock_prov: ProvenanceRecord) -> None:
    qr = QualityResult(
        quality_score=0.85,
        summary="",
        provenance=mock_prov,
        status=ResultStatus.WARNING,
    )
    req = RecommendationGenerationRequest(quality_result=qr)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 1
    assert res.recommendations[0].recommendation_type == RecommendationType.VERIFY_DATA


def test_46_quality_score_none_does_not_fire_rule_a(mock_prov: ProvenanceRecord) -> None:
    qr = QualityResult(
        quality_score=None,
        provenance=mock_prov,
        status=ResultStatus.SUCCESS,
    )
    req = RecommendationGenerationRequest(quality_result=qr)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 0


def test_47_clean_quality_score_does_not_fire_rule_a(mock_prov: ProvenanceRecord) -> None:
    qr = QualityResult(
        quality_score=0.99,
        provenance=mock_prov,
        status=ResultStatus.SUCCESS,
    )
    req = RecommendationGenerationRequest(quality_result=qr)
    res = RecommendationGenerationEngine.generate(req)
    assert len(res.recommendations) == 0


def test_48_public_api_exports_accessible() -> None:
    import aeris.core.recommendation as rec

    assert hasattr(rec, "RecommendationGenerationEngine")
    assert hasattr(rec, "RecommendationGenerationPolicy")
    assert hasattr(rec, "RecommendationGenerationRequest")
    assert hasattr(rec, "RecommendationGenerationResult")


def test_49_rule_c_evidence_ids_contain_supporting_only(
    sample_assessment_insufficient: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_insufficient)
    res = RecommendationGenerationEngine.generate(req)
    rec = res.recommendations[0]
    assert rec.evidence_ids == ()


def test_50_recommendation_epistemic_level_is_recommendation(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    for r in res.recommendations:
        assert r.epistemic_level == EpistemicLevel.RECOMMENDATION


def test_51_all_recommendation_titles_are_non_empty(
    sample_assessment_unresolved: HypothesisAssessmentResult,
    sample_validation_failed: ValidationResult,
) -> None:
    req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        validation_result=sample_validation_failed,
    )
    res = RecommendationGenerationEngine.generate(req)
    for r in res.recommendations:
        assert len(r.title.strip()) > 0
        assert len(r.description.strip()) > 0
        assert len(r.rationale.strip()) > 0


def test_52_all_recommendation_ids_are_unique(
    sample_assessment_unresolved: HypothesisAssessmentResult,
    sample_validation_failed: ValidationResult,
    sample_quality_degraded: QualityResult,
) -> None:
    req = RecommendationGenerationRequest(
        hypothesis_assessment=sample_assessment_unresolved,
        validation_result=sample_validation_failed,
        quality_result=sample_quality_degraded,
    )
    res = RecommendationGenerationEngine.generate(req)
    ids = [r.recommendation_id for r in res.recommendations]
    assert len(ids) == len(set(ids))


def test_53_rule_e_not_triggered_on_success_status(mock_prov: ProvenanceRecord) -> None:
    success_asmt = HypothesisAssessmentResult(
        hypothesis_id="hyp-success-01",
        assessment_state=HypothesisAssessmentState.SUPPORTED,
        supporting_evidence_ids=("ev-01",),
        missing_evidence_ids=(),
        configuration_used=HypothesisAssessmentConfiguration(),
        provenance=mock_prov,
        status=ResultStatus.SUCCESS,
    )
    policy = RecommendationGenerationPolicy(enable_repeat_analysis_rule=True)
    req = RecommendationGenerationRequest(hypothesis_assessment=success_asmt, policy=policy)
    res = RecommendationGenerationEngine.generate(req)
    assert RecommendationGenerationEngine.RULE_E_REPEAT_ANALYSIS not in res.rules_triggered


def test_54_recommendation_has_no_operational_commands(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    for r in res.recommendations:
        assert not hasattr(r, "execute_command")
        assert not hasattr(r, "authorize_flight")


def test_55_collection_provenance_links_recommendation_provenances(
    sample_assessment_unresolved: HypothesisAssessmentResult,
) -> None:
    req = RecommendationGenerationRequest(hypothesis_assessment=sample_assessment_unresolved)
    res = RecommendationGenerationEngine.generate(req)
    coll_prov = res.recommendation_collection.provenance
    for r in res.recommendations:
        assert r.provenance.record_id in coll_prov.parent_record_ids
