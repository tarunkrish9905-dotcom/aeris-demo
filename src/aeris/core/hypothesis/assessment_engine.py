from aeris.core.hypothesis.assessment_models import (
    HypothesisAssessmentRequest,
    HypothesisAssessmentResult,
)
from aeris.core.hypothesis.models import HypothesisAssessmentState, HypothesisEvidenceRelationship
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


class HypothesisAssessmentEngine:
    """Engine for assessing hypotheses against explicitly provided evidence."""

    @classmethod
    def assess(cls, request: HypothesisAssessmentRequest) -> HypothesisAssessmentResult:
        """
        Evaluate the hypothesis based on explicitly referenced evidence and policy.
        """
        issues: list[Issue] = []
        status = ResultStatus.SUCCESS

        hyp = request.hypothesis
        coll = request.evidence_collection
        config = request.configuration

        # O(1) evidence lookup
        evidence_map = {e.evidence_id: e for e in coll.evidence}

        supporting: list[str] = []
        contradicting: list[str] = []
        relevant_neutral: list[str] = []
        missing: list[str] = []

        # Classify linked evidence
        for link in hyp.evidence_links:
            e_id = link.evidence_id
            if e_id not in evidence_map:
                missing.append(e_id)
                issues.append(
                    Issue(
                        issue_code="ERR_MISSING_EVIDENCE",
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"Hypothesis references evidence '{e_id}' which was "
                            "not provided in the collection."
                        ),
                    )
                )
                status = ResultStatus.PARTIAL
                continue

            rel = link.relationship_type
            if rel == HypothesisEvidenceRelationship.SUPPORTING:
                supporting.append(e_id)
            elif rel == HypothesisEvidenceRelationship.CONTRADICTING:
                contradicting.append(e_id)
            elif rel in (
                HypothesisEvidenceRelationship.RELEVANT,
                HypothesisEvidenceRelationship.NEUTRAL,
            ):
                relevant_neutral.append(e_id)

        # Count actual available evidence
        total_valid = len(supporting) + len(contradicting) + len(relevant_neutral)
        num_support = len(supporting)
        num_contradict = len(contradicting)

        # Apply Policy
        assessment_state = HypothesisAssessmentState.UNASSESSED

        if total_valid == 0:
            assessment_state = HypothesisAssessmentState.INSUFFICIENT_EVIDENCE
            if missing:
                status = ResultStatus.ERROR
        elif total_valid < config.minimum_total_evidence:
            assessment_state = HypothesisAssessmentState.INSUFFICIENT_EVIDENCE
        elif num_contradict > 0:
            if num_support > 0:
                # Mixed evidence
                assessment_state = HypothesisAssessmentState.UNRESOLVED
            else:
                assessment_state = HypothesisAssessmentState.CONTRADICTED
        else:
            # No contradicting evidence
            if num_support >= config.minimum_supporting_evidence:
                assessment_state = HypothesisAssessmentState.SUPPORTED
            else:
                assessment_state = HypothesisAssessmentState.INCONCLUSIVE

        # Edge case: If config allows contradiction (strict=False) and we have support
        if (
            not config.strict_no_contradiction
            and num_contradict > 0
            and num_support >= config.minimum_supporting_evidence
        ):
            # If the user policy explicitly says they don't require strict no-contradiction
            # we can declare it supported (or weakly supported). But we default to WEAKLY_SUPPORTED.
            assessment_state = HypothesisAssessmentState.WEAKLY_SUPPORTED

        # Provenance
        parent_ids = [hyp.provenance.record_id, coll.collection_id]
        prov = create_provenance(
            activity_name="HypothesisAssessment",
            source_uri="urn:aeris:engine:hypothesis:assessment",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=parent_ids,
            parameters={
                "total_valid_evidence": total_valid,
                "supporting_count": num_support,
                "contradicting_count": num_contradict,
                "missing_count": len(missing),
            },
        )

        return HypothesisAssessmentResult(
            hypothesis_id=hyp.hypothesis_id,
            assessment_state=assessment_state,
            supporting_evidence_ids=tuple(supporting),
            contradicting_evidence_ids=tuple(contradicting),
            relevant_neutral_evidence_ids=tuple(relevant_neutral),
            missing_evidence_ids=tuple(missing),
            configuration_used=config,
            status=status,
            issues=issues,
            provenance=prov,
        )
