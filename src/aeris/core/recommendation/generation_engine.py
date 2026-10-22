"""
AERIS Controlled Engineering Recommendation Generation Engine.

Evaluates explicitly supplied evidence and assessment records against
deterministic, engineer-configured rules to produce bounded recommendations.
"""

from __future__ import annotations

from typing import Any

from aeris.core.evidence.models import EvidenceType
from aeris.core.hypothesis.models import HypothesisAssessmentState
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.recommendation.generation_models import (
    RecommendationGenerationPolicy,
    RecommendationGenerationRequest,
    RecommendationGenerationResult,
)
from aeris.core.recommendation.models import (
    Recommendation,
    RecommendationCollection,
    RecommendationStatus,
    RecommendationType,
)
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


def _extract_provenance_id(result: Any, default_prefix: str) -> str:
    """Safely extracts provenance record_id from result containers."""
    if hasattr(result, "provenance") and result.provenance is not None:
        return str(result.provenance.record_id)
    return f"urn:aeris:unknown:{default_prefix}"


class RecommendationGenerationEngine:
    """
    Deterministic engine synthesizing bounded engineering investigation recommendations.
    """

    RULE_A_VERIFY_DATA = "RULE_A_VERIFY_DATA"
    RULE_B_REVIEW_EVIDENCE = "RULE_B_REVIEW_EVIDENCE"
    RULE_C_COLLECT_ADDITIONAL_EVIDENCE = "RULE_C_COLLECT_ADDITIONAL_EVIDENCE"
    RULE_D_REQUEST_ENGINEERING_REVIEW = "RULE_D_REQUEST_ENGINEERING_REVIEW"
    RULE_E_REPEAT_ANALYSIS = "RULE_E_REPEAT_ANALYSIS"
    RULE_F_CHECK_CONFIGURATION_RECORD = "RULE_F_CHECK_CONFIGURATION_RECORD"

    @classmethod
    def generate(cls, request: RecommendationGenerationRequest) -> RecommendationGenerationResult:
        """
        Evaluate supplied inputs against active policy rules to produce recommendations.
        """
        policy: RecommendationGenerationPolicy = request.policy
        recs: list[Recommendation] = []
        rules_evaluated: list[str] = []
        rules_triggered: list[str] = []
        issues: list[Issue] = []
        status = ResultStatus.SUCCESS

        # -------------------------------------------------------------------
        # Rule A: Data Verification (VERIFY_DATA)
        # -------------------------------------------------------------------
        rules_evaluated.append(cls.RULE_A_VERIFY_DATA)
        if policy.enable_data_verification_rule:
            if request.validation_result is not None:
                vr = request.validation_result
                if not vr.is_valid or vr.violation_count > 0:
                    rules_triggered.append(cls.RULE_A_VERIFY_DATA)
                    vr_prov_id = _extract_provenance_id(vr, "validation")
                    rec_id = f"rec-verify-data-val-{vr_prov_id[:8]}"
                    prov = create_provenance(
                        activity_name="RecommendationGeneration",
                        source_uri="urn:aeris:engine:recommendation:rule_a",
                        epistemic_level=EpistemicLevel.RECOMMENDATION,
                        parent_record_ids=[vr_prov_id],
                        parameters={"rule_id": cls.RULE_A_VERIFY_DATA},
                    )
                    recs.append(
                        Recommendation(
                            recommendation_id=rec_id,
                            recommendation_type=RecommendationType.VERIFY_DATA,
                            title="Verify Data Integrity for Ingested Telemetry",
                            description=(
                                f"Audit data validation defects across {len(vr.validated_columns)} "
                                "columns to reconcile schema violations."
                            ),
                            rationale=(
                                f"Validation reported is_valid={vr.is_valid} with "
                                f"{vr.violation_count} violations."
                            ),
                            status=RecommendationStatus.PROPOSED,
                            other_assessment_ids=(vr_prov_id,),
                            constraints=("Do not modify raw telemetry archive records.",),
                            prerequisites=("Obtain raw acquisition ingestion logs.",),
                            provenance=prov,
                            investigation_id=request.investigation_id,
                        )
                    )

            if request.quality_result is not None:
                qr = request.quality_result
                if qr.quality_score is not None and qr.quality_score < 0.9:
                    rule_key = f"{cls.RULE_A_VERIFY_DATA}_QUALITY"
                    rules_triggered.append(rule_key)
                    qr_prov_id = _extract_provenance_id(qr, "quality")
                    rec_id = f"rec-verify-data-qual-{qr_prov_id[:8]}"
                    prov = create_provenance(
                        activity_name="RecommendationGeneration",
                        source_uri="urn:aeris:engine:recommendation:rule_a_quality",
                        epistemic_level=EpistemicLevel.RECOMMENDATION,
                        parent_record_ids=[qr_prov_id],
                        parameters={"rule_id": rule_key},
                    )
                    recs.append(
                        Recommendation(
                            recommendation_id=rec_id,
                            recommendation_type=RecommendationType.VERIFY_DATA,
                            title="Verify Telemetry Signal Quality and Completeness",
                            description=(
                                "Inspect telemetry channels flagging degraded quality metrics "
                                "or sample dropouts."
                            ),
                            rationale=(
                                f"Data quality assessment score is {qr.quality_score:.2f} "
                                f"(threshold: 0.90). {qr.summary}".strip()
                            ),
                            status=RecommendationStatus.PROPOSED,
                            other_assessment_ids=(qr_prov_id,),
                            constraints=("Preserve existing channel quality metadata.",),
                            prerequisites=("Extract channel dropout summary table.",),
                            provenance=prov,
                            investigation_id=request.investigation_id,
                        )
                    )

        # -------------------------------------------------------------------
        # Rule B: Evidence Review (REVIEW_EVIDENCE)
        # -------------------------------------------------------------------
        rules_evaluated.append(cls.RULE_B_REVIEW_EVIDENCE)
        if policy.enable_evidence_review_rule and request.hypothesis_assessment is not None:
            ha = request.hypothesis_assessment
            ha_prov_id = _extract_provenance_id(ha, "assessment")
            has_mixed_evidence = (
                len(ha.supporting_evidence_ids) > 0 and len(ha.contradicting_evidence_ids) > 0
            )
            if ha.assessment_state == HypothesisAssessmentState.UNRESOLVED or has_mixed_evidence:
                rules_triggered.append(cls.RULE_B_REVIEW_EVIDENCE)
                rec_id = f"rec-review-ev-{ha.hypothesis_id}"
                combined_ev = tuple(
                    sorted(set(ha.supporting_evidence_ids + ha.contradicting_evidence_ids))
                )
                prov = create_provenance(
                    activity_name="RecommendationGeneration",
                    source_uri="urn:aeris:engine:recommendation:rule_b",
                    epistemic_level=EpistemicLevel.RECOMMENDATION,
                    parent_record_ids=[ha_prov_id],
                    parameters={"rule_id": cls.RULE_B_REVIEW_EVIDENCE},
                )
                recs.append(
                    Recommendation(
                        recommendation_id=rec_id,
                        recommendation_type=RecommendationType.REVIEW_EVIDENCE,
                        title=f"Review Conflicting Evidence for Hypothesis '{ha.hypothesis_id}'",
                        description=(
                            "Conduct detailed engineering review to resolve conflicting "
                            "supporting and contradicting evidence records."
                        ),
                        rationale=(
                            f"Assessment state is {ha.assessment_state.value} with "
                            f"{len(ha.supporting_evidence_ids)} supporting and "
                            f"{len(ha.contradicting_evidence_ids)} contradicting records."
                        ),
                        status=RecommendationStatus.PROPOSED,
                        evidence_ids=combined_ev,
                        hypothesis_ids=(ha.hypothesis_id,),
                        hypothesis_assessment_ids=(ha_prov_id,),
                        constraints=(
                            "Do not assert causal attribution until conflicts are resolved.",
                        ),
                        prerequisites=(
                            "Retrieve calibrated sensor traces for referenced evidence.",
                        ),
                        provenance=prov,
                        investigation_id=request.investigation_id,
                    )
                )

        # -------------------------------------------------------------------
        # Rule C: Additional Evidence Collection (COLLECT_ADDITIONAL_EVIDENCE)
        # -------------------------------------------------------------------
        rules_evaluated.append(cls.RULE_C_COLLECT_ADDITIONAL_EVIDENCE)
        if policy.enable_additional_evidence_rule and request.hypothesis_assessment is not None:
            ha = request.hypothesis_assessment
            ha_prov_id = _extract_provenance_id(ha, "assessment")
            if ha.assessment_state == HypothesisAssessmentState.INSUFFICIENT_EVIDENCE:
                rules_triggered.append(cls.RULE_C_COLLECT_ADDITIONAL_EVIDENCE)
                rec_id = f"rec-collect-ev-{ha.hypothesis_id}"
                prov = create_provenance(
                    activity_name="RecommendationGeneration",
                    source_uri="urn:aeris:engine:recommendation:rule_c",
                    epistemic_level=EpistemicLevel.RECOMMENDATION,
                    parent_record_ids=[ha_prov_id],
                    parameters={"rule_id": cls.RULE_C_COLLECT_ADDITIONAL_EVIDENCE},
                )
                recs.append(
                    Recommendation(
                        recommendation_id=rec_id,
                        recommendation_type=RecommendationType.COLLECT_ADDITIONAL_EVIDENCE,
                        title=(f"Collect Additional Evidence for Hypothesis '{ha.hypothesis_id}'"),
                        description=(
                            "Identify supplementary flight test points or ground inspection "
                            "records to satisfy minimum evidence thresholds."
                        ),
                        rationale=(
                            f"Hypothesis '{ha.hypothesis_id}' evaluated to "
                            f"{ha.assessment_state.value} under configured policy "
                            f"(min total required: {ha.configuration_used.minimum_total_evidence})."
                        ),
                        status=RecommendationStatus.PROPOSED,
                        evidence_ids=tuple(sorted(set(ha.supporting_evidence_ids))),
                        hypothesis_ids=(ha.hypothesis_id,),
                        hypothesis_assessment_ids=(ha_prov_id,),
                        constraints=(
                            "Investigation proposal only; does not authorize "
                            "flight envelope expansion.",
                        ),
                        prerequisites=("Review flight test instrumentation manifest.",),
                        provenance=prov,
                        investigation_id=request.investigation_id,
                    )
                )

        # -------------------------------------------------------------------
        # Rule D: Engineering Review (REQUEST_ENGINEERING_REVIEW)
        # -------------------------------------------------------------------
        rules_evaluated.append(cls.RULE_D_REQUEST_ENGINEERING_REVIEW)
        if policy.enable_engineering_review_rule and request.hypothesis_assessment is not None:
            ha = request.hypothesis_assessment
            ha_prov_id = _extract_provenance_id(ha, "assessment")
            if ha.assessment_state in (
                HypothesisAssessmentState.INCONCLUSIVE,
                HypothesisAssessmentState.WEAKLY_SUPPORTED,
            ):
                rules_triggered.append(cls.RULE_D_REQUEST_ENGINEERING_REVIEW)
                rec_id = f"rec-eng-review-{ha.hypothesis_id}"
                prov = create_provenance(
                    activity_name="RecommendationGeneration",
                    source_uri="urn:aeris:engine:recommendation:rule_d",
                    epistemic_level=EpistemicLevel.RECOMMENDATION,
                    parent_record_ids=[ha_prov_id],
                    parameters={"rule_id": cls.RULE_D_REQUEST_ENGINEERING_REVIEW},
                )
                recs.append(
                    Recommendation(
                        recommendation_id=rec_id,
                        recommendation_type=RecommendationType.REQUEST_ENGINEERING_REVIEW,
                        title=(f"Request Disciplinary Engineering Review for '{ha.hypothesis_id}'"),
                        description=(
                            "Submit assessed findings to appropriate engineering discipline "
                            "for physical feasibility review."
                        ),
                        rationale=(
                            f"Hypothesis '{ha.hypothesis_id}' state is "
                            f"{ha.assessment_state.value}; requires domain specialist evaluation."
                        ),
                        status=RecommendationStatus.PROPOSED,
                        evidence_ids=tuple(sorted(set(ha.supporting_evidence_ids))),
                        hypothesis_ids=(ha.hypothesis_id,),
                        hypothesis_assessment_ids=(ha_prov_id,),
                        constraints=("Specialist review must precede operational disposition.",),
                        prerequisites=("Compile parameter correlation summary.",),
                        provenance=prov,
                        investigation_id=request.investigation_id,
                    )
                )

        # -------------------------------------------------------------------
        # Rule E: Repeat Analysis (REPEAT_ANALYSIS) [Gated / Deferred]
        # -------------------------------------------------------------------
        rules_evaluated.append(cls.RULE_E_REPEAT_ANALYSIS)
        if policy.enable_repeat_analysis_rule and request.hypothesis_assessment is not None:
            ha = request.hypothesis_assessment
            ha_prov_id = _extract_provenance_id(ha, "assessment")
            if ha.status == ResultStatus.PARTIAL and len(ha.missing_evidence_ids) > 0:
                rules_triggered.append(cls.RULE_E_REPEAT_ANALYSIS)
                rec_id = f"rec-repeat-analysis-{ha.hypothesis_id}"
                prov = create_provenance(
                    activity_name="RecommendationGeneration",
                    source_uri="urn:aeris:engine:recommendation:rule_e",
                    epistemic_level=EpistemicLevel.RECOMMENDATION,
                    parent_record_ids=[ha_prov_id],
                    parameters={"rule_id": cls.RULE_E_REPEAT_ANALYSIS},
                )
                recs.append(
                    Recommendation(
                        recommendation_id=rec_id,
                        recommendation_type=RecommendationType.REPEAT_ANALYSIS,
                        title=(f"Repeat Assessment Analysis for Hypothesis '{ha.hypothesis_id}'"),
                        description=(
                            "Re-execute hypothesis assessment once missing referenced "
                            f"evidence ({len(ha.missing_evidence_ids)} records) is supplied."
                        ),
                        rationale=(
                            f"Prior assessment was degraded (status: {ha.status.value}) due to "
                            f"missing evidence records: {ha.missing_evidence_ids}."
                        ),
                        status=RecommendationStatus.PROPOSED,
                        hypothesis_ids=(ha.hypothesis_id,),
                        hypothesis_assessment_ids=(ha_prov_id,),
                        constraints=(
                            "Ensure all missing evidence identifiers exist prior to re-run.",
                        ),
                        prerequisites=("Construct and register missing evidence records.",),
                        provenance=prov,
                        investigation_id=request.investigation_id,
                    )
                )

        # -------------------------------------------------------------------
        # Rule F: Configuration Record Review (CHECK_CONFIGURATION_RECORD)
        # -------------------------------------------------------------------
        rules_evaluated.append(cls.RULE_F_CHECK_CONFIGURATION_RECORD)
        if policy.enable_configuration_review_rule:
            discrepancy_evidence_ids: list[str] = []
            discrepancy_descriptions: list[str] = []
            target_vehicle_id: str | None = None
            target_config_id: str | None = None

            # 1. Inspect evidence_collection for configuration discrepancy records
            if request.evidence_collection is not None:
                for ev in request.evidence_collection.evidence:
                    if ev.evidence_type == EvidenceType.CONFIGURATION_DISCREPANCY:
                        discrepancy_evidence_ids.append(ev.evidence_id)
                        discrepancy_descriptions.append(ev.description)
                        if target_vehicle_id is None and ev.vehicle_id:
                            target_vehicle_id = ev.vehicle_id
                        if target_config_id is None and ev.configuration_id:
                            target_config_id = ev.configuration_id

            # 2. Inspect vehicle_evaluation_result for discrepancies if supplied
            veh_res = request.vehicle_evaluation_result
            if veh_res is not None:
                if target_vehicle_id is None:
                    target_vehicle_id = veh_res.vehicle_id
                if target_config_id is None:
                    target_config_id = veh_res.configuration_id

                for d in veh_res.configuration_discrepancies:
                    discrepancy_descriptions.append(d)

            # Rule F fires ONLY when configuration discrepancy evidence/records are present
            if discrepancy_descriptions or discrepancy_evidence_ids:
                rules_triggered.append(cls.RULE_F_CHECK_CONFIGURATION_RECORD)

                veh_str = target_vehicle_id or "Vehicle"
                cfg_str = target_config_id or "declared"
                rec_id = f"rec-check-config-{veh_str}-{cfg_str}".replace(" ", "_")

                parent_ids: list[str] = []
                if veh_res is not None:
                    parent_ids.append(_extract_provenance_id(veh_res, "vehicle_evaluation"))
                if request.evidence_collection is not None:
                    for ev_id in discrepancy_evidence_ids:
                        parent_ids.append(f"urn:aeris:evidence:{ev_id}")

                prov = create_provenance(
                    activity_name="RecommendationGeneration",
                    source_uri="urn:aeris:engine:recommendation:rule_f",
                    epistemic_level=EpistemicLevel.RECOMMENDATION,
                    parent_record_ids=list(sorted(set(parent_ids))),
                    parameters={"rule_id": cls.RULE_F_CHECK_CONFIGURATION_RECORD},
                )

                disc_count = max(len(discrepancy_evidence_ids), len(discrepancy_descriptions))
                recs.append(
                    Recommendation(
                        recommendation_id=rec_id,
                        recommendation_type=RecommendationType.CHECK_CONFIGURATION_RECORD,
                        title=f"Review Configuration Envelope Records for Vehicle '{veh_str}'",
                        description=(
                            "Conduct engineering review of vehicle configuration records, "
                            "envelope definitions, and test data to reconcile conflicting "
                            "configuration states."
                        ),
                        rationale=(
                            f"Identified {disc_count} configuration discrepancy indicator(s) "
                            f"between declared flight observation data and active configuration "
                            f"'{cfg_str}'. Verification is required to ensure applicable limits "
                            "match the true physical configuration."
                        ),
                        status=RecommendationStatus.PROPOSED,
                        evidence_ids=tuple(sorted(set(discrepancy_evidence_ids))),
                        other_assessment_ids=tuple(sorted(set(parent_ids))),
                        constraints=(
                            "Verification proposal only; do not assert root cause or alter "
                            "physical vehicle configuration without engineering disposition.",
                        ),
                        prerequisites=(
                            "Retrieve verified vehicle configuration logs and active envelope "
                            "definition.",
                        ),
                        provenance=prov,
                        investigation_id=request.investigation_id,
                    )
                )

        # Enforce maximum recommendations limit from policy
        if len(recs) > policy.max_recommendations:
            recs = recs[: policy.max_recommendations]
            issues.append(
                Issue(
                    issue_code="WARN_MAX_RECOMMENDATIONS_EXCEEDED",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Generated recommendations exceeded policy limit of "
                        f"{policy.max_recommendations}; output was truncated."
                    ),
                )
            )
            status = ResultStatus.WARNING

        # Build collection
        coll_id = f"rec-coll-{request.investigation_id or 'default'}-{len(recs):02d}"
        coll_parent_ids = [r.provenance.record_id for r in recs]
        coll_prov = create_provenance(
            activity_name="RecommendationCollectionAggregation",
            source_uri="urn:aeris:engine:recommendation:collection",
            epistemic_level=EpistemicLevel.RECOMMENDATION,
            parent_record_ids=coll_parent_ids,
        )
        collection = RecommendationCollection(
            collection_id=coll_id,
            recommendations=tuple(recs),
            investigation_id=request.investigation_id,
            provenance=coll_prov,
        )

        # Build overall result provenance
        result_parents = list(coll_parent_ids)
        if request.hypothesis_assessment is not None:
            result_parents.append(
                _extract_provenance_id(request.hypothesis_assessment, "assessment")
            )
        if request.validation_result is not None:
            result_parents.append(_extract_provenance_id(request.validation_result, "validation"))
        if request.quality_result is not None:
            result_parents.append(_extract_provenance_id(request.quality_result, "quality"))
        if request.vehicle_evaluation_result is not None:
            result_parents.append(
                _extract_provenance_id(request.vehicle_evaluation_result, "vehicle_evaluation")
            )

        result_prov = create_provenance(
            activity_name="RecommendationGenerationExecution",
            source_uri="urn:aeris:engine:recommendation:execution",
            epistemic_level=EpistemicLevel.RECOMMENDATION,
            parent_record_ids=list(sorted(set(result_parents))),
            parameters={
                "rules_evaluated_count": len(rules_evaluated),
                "rules_triggered_count": len(rules_triggered),
                "recommendation_count": len(recs),
            },
        )

        return RecommendationGenerationResult(
            recommendations=tuple(recs),
            recommendation_collection=collection,
            rules_evaluated=tuple(rules_evaluated),
            rules_triggered=tuple(rules_triggered),
            policy_used=policy,
            provenance=result_prov,
            status=status,
            issues=issues,
        )
