"""
AERIS Milestone 12 - Step 12j End-to-End Investigation Intelligence Architecture Audit.

Verifies end-to-end traceability across the complete investigation workflow:
Contributor analysis
  -> Contributor evaluation and ranking
  -> Explicit hypothesis construction
  -> Evidence linkage
  -> Hypothesis assessment
  -> Recommendation generation
  -> Final pipeline result
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from aeris.core.anomaly.models import AnomalySeverity
from aeris.core.anomaly.threshold import (
    ThresholdDetectionRequest,
    ThresholdOperator,
    ThresholdRule,
)
from aeris.core.contributors.engine import (
    ContributorAnalysisMethod,
    ContributorAnalysisRequest,
)
from aeris.core.contributors.evaluation_engine import ContributorEvaluationEngine
from aeris.core.contributors.evaluation_models import (
    AssociationStrength,
    ContributorEvaluationRequest,
    RankingPolicy,
)
from aeris.core.contributors.models import (
    Contributor,
    ContributorType,
)
from aeris.core.contributors.pipeline import PipelineContributorAnalysisRequest
from aeris.core.evidence.pipeline import (
    EvidenceSourceSelection,
    PipelineEvidenceRequest,
)
from aeris.core.hypothesis.assessment_engine import HypothesisAssessmentEngine
from aeris.core.hypothesis.assessment_models import (
    HypothesisAssessmentConfiguration,
    HypothesisAssessmentRequest,
)
from aeris.core.hypothesis.models import (
    Hypothesis,
    HypothesisAssessmentState,
    HypothesisContributorLink,
    HypothesisContributorRelationship,
    HypothesisEvidenceLink,
    HypothesisEvidenceRelationship,
    HypothesisLifecycleStatus,
)
from aeris.core.pipeline.orchestrator import (
    PipelineAnomalyDetectionRequest,
    run_csv_pipeline,
)
from aeris.core.provenance.tracker import (
    EpistemicLevel,
    create_provenance,
)
from aeris.core.recommendation.generation_models import (
    RecommendationGenerationPolicy,
)
from aeris.core.recommendation.models import (
    RecommendationStatus,
    RecommendationType,
)
from aeris.core.recommendation.pipeline import PipelineRecommendationRequest
from aeris.core.results.types import ResultStatus


@pytest.fixture
def flight_test_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "flight_telemetry_investigation.csv"
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-03-15T10:00:00Z",
                "2026-03-15T10:00:01Z",
                "2026-03-15T10:00:02Z",
                "2026-03-15T10:00:03Z",
                "2026-03-15T10:00:04Z",
                "2026-03-15T10:00:05Z",
            ],
            "altitude_ft": [1000.0, 1050.0, 1100.0, 1160.0, 1220.0, 1290.0],
            "airspeed_kts": [100.0, 105.0, 110.0, 118.0, 125.0, 130.0],
            "pitch_deg": [2.0, 3.0, 4.0, 5.5, 6.0, 7.0],
            "roll_deg": [0.0, 0.1, 0.0, -0.1, 0.0, 0.2],
            "engine_rpm": [2400.0, 2450.0, 2500.0, 2550.0, 2600.0, 2650.0],
        }
    )
    df.to_csv(p, index=False)
    return p


def test_end_to_end_investigation_intelligence_workflow(
    flight_test_csv: pathlib.Path,
) -> None:
    # =========================================================================
    # Step 1: Upstream Pipeline with Contributor Analysis & Anomaly Detection
    # =========================================================================
    contrib_req = PipelineContributorAnalysisRequest(
        requests=(
            ContributorAnalysisRequest(
                analysis_id="ca-alt-airspeed",
                target_column="altitude_ft",
                factor_column="airspeed_kts",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
            ContributorAnalysisRequest(
                analysis_id="ca-alt-pitch",
                target_column="altitude_ft",
                factor_column="pitch_deg",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
            ContributorAnalysisRequest(
                analysis_id="ca-alt-roll",
                target_column="altitude_ft",
                factor_column="roll_deg",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
        )
    )

    anom_req = PipelineAnomalyDetectionRequest(
        threshold_requests=(
            ThresholdDetectionRequest(
                analysis_id="th-airspeed-high",
                target_column="airspeed_kts",
                rule=ThresholdRule(
                    rule_id="rule-airspeed-115",
                    target_column="airspeed_kts",
                    operator=ThresholdOperator.GREATER_THAN,
                    threshold_value=115.0,
                    severity=AnomalySeverity.HIGH,
                ),
            ),
        )
    )

    # Initial pipeline run to detect anomaly
    res_phase1 = run_csv_pipeline(
        flight_test_csv,
        contributor_analysis_request=contrib_req,
        anomaly_request=anom_req,
    )
    assert res_phase1.status == ResultStatus.SUCCESS
    assert res_phase1.contributor_analysis is not None
    assert len(res_phase1.contributor_analysis.results) == 3
    assert res_phase1.anomaly is not None
    anom_id = res_phase1.anomaly.detector_results[0].anomalies[0].anomaly_id

    # Pipeline run with Evidence construction bound to the anomaly
    ev_req = PipelineEvidenceRequest(
        collection_id="ev-coll-climb-investigation",
        selections=EvidenceSourceSelection(anomaly_ids=[anom_id]),
    )
    res_phase2 = run_csv_pipeline(
        flight_test_csv,
        contributor_analysis_request=contrib_req,
        anomaly_request=anom_req,
        evidence_request=ev_req,
    )
    assert res_phase2.status == ResultStatus.SUCCESS
    assert res_phase2.evidence is not None
    assert len(res_phase2.evidence.collection.evidence) == 1
    evidence_item = res_phase2.evidence.collection.evidence[0]
    assert evidence_item.source_record_id == anom_id
    assert evidence_item.epistemic_level == EpistemicLevel.CALCULATED

    # =========================================================================
    # Step 2: Contributor Evaluation and Controlled Ranking
    # =========================================================================
    assert res_phase2.contributor_analysis is not None
    eval_req = ContributorEvaluationRequest(
        analysis_results=res_phase2.contributor_analysis.results,
        request_ranking=True,
        ranking_policy=RankingPolicy.MAGNITUDE,
    )
    eval_res = ContributorEvaluationEngine.evaluate(eval_req)
    assert eval_res.status == ResultStatus.SUCCESS
    assert eval_res.is_ranked is True
    assert len(eval_res.evaluations) == 3

    # Verify deterministic ranking
    ranks = [e.rank_position for e in eval_res.evaluations]
    assert ranks == [1, 2, 3]
    top_eval = eval_res.evaluations[0]
    assert top_eval.rank_position == 1
    assert top_eval.association_strength in (
        AssociationStrength.STRONG,
        AssociationStrength.VERY_STRONG,
    )
    assert top_eval.provenance is not None
    assert top_eval.provenance.epistemic_level == EpistemicLevel.CALCULATED

    # =========================================================================
    # Step 3: Explicit Hypothesis Formulation
    # =========================================================================
    top_contributor = Contributor(
        contributor_id=f"contrib-{top_eval.factor_column}",
        name=f"Factor {top_eval.factor_column}",
        description=f"Observed statistical correlation with {top_eval.target_column}",
        contributor_type=ContributorType.PARAMETER,
    )

    hyp_prov = create_provenance(
        activity_name="EngineeringHypothesisFormulation",
        source_uri=str(flight_test_csv),
        epistemic_level=EpistemicLevel.HYPOTHESIS,
        parent_record_ids=[top_eval.provenance.record_id],
    )

    evidence_link = HypothesisEvidenceLink(
        link_id="link-ev-01",
        hypothesis_id="hyp-climb-rate-correlation",
        evidence_id=evidence_item.evidence_id,
        relationship_type=HypothesisEvidenceRelationship.SUPPORTING,
        provenance=hyp_prov,
        notes="High airspeed exceeds threshold during climb profile",
    )

    contrib_link = HypothesisContributorLink(
        link_id="link-contrib-01",
        hypothesis_id="hyp-climb-rate-correlation",
        contributor_id=top_contributor.contributor_id,
        relationship_type=HypothesisContributorRelationship.ASSOCIATED_FACTOR,
        provenance=hyp_prov,
        notes="Statistical correlation identified during initial screening",
    )

    hypothesis = Hypothesis(
        hypothesis_id="hyp-climb-rate-correlation",
        title="Climb Airspeed Exceedance Association",
        statement=(
            "Elevated climb rate is associated with exceedance of the 115 kts airspeed threshold."
        ),
        status=HypothesisLifecycleStatus.OPEN,
        assessment_state=HypothesisAssessmentState.UNASSESSED,
        epistemic_level=EpistemicLevel.HYPOTHESIS,
        provenance=hyp_prov,
        evidence_links=(evidence_link,),
        contributor_links=(contrib_link,),
        investigation_id="inv-climb-profile-01",
    )
    assert hypothesis.epistemic_level == EpistemicLevel.HYPOTHESIS

    # =========================================================================
    # Step 4: Hypothesis Assessment and Evidence Evaluation
    # =========================================================================
    assess_req = HypothesisAssessmentRequest(
        hypothesis=hypothesis,
        evidence_collection=res_phase2.evidence.collection,
        configuration=HypothesisAssessmentConfiguration(
            minimum_total_evidence=1,
            minimum_supporting_evidence=1,
            strict_no_contradiction=True,
        ),
    )
    assess_res = HypothesisAssessmentEngine.assess(assess_req)
    assert assess_res.status == ResultStatus.SUCCESS
    assert assess_res.assessment_state == HypothesisAssessmentState.SUPPORTED
    assert assess_res.supporting_evidence_ids == (evidence_item.evidence_id,)
    assert len(assess_res.contradicting_evidence_ids) == 0
    assert assess_res.provenance is not None
    assert assess_res.provenance.epistemic_level == EpistemicLevel.CALCULATED

    # =========================================================================
    # Step 5: Recommendation Generation via Orchestrated Pipeline Stage
    # =========================================================================
    rec_policy = RecommendationGenerationPolicy(
        max_recommendations=5,
        enable_data_verification_rule=True,
        enable_evidence_review_rule=True,
        enable_additional_evidence_rule=True,
        enable_engineering_review_rule=True,
    )

    pipeline_rec_req = PipelineRecommendationRequest(
        hypothesis_assessment=assess_res,
        hypothesis=hypothesis,
        evidence_collection=res_phase2.evidence.collection,
        policy=rec_policy,
        investigation_id="inv-climb-profile-01",
        use_pipeline_validation=True,
        use_pipeline_quality=True,
    )

    final_result = run_csv_pipeline(
        flight_test_csv,
        contributor_analysis_request=contrib_req,
        anomaly_request=anom_req,
        evidence_request=ev_req,
        recommendation_request=pipeline_rec_req,
    )

    # =========================================================================
    # Step 6: End-to-End Invariants & Traceability Verification
    # =========================================================================
    # 1. Pipeline status
    assert final_result.status == ResultStatus.SUCCESS
    assert final_result.recommendation is not None
    assert final_result.recommendation.status == ResultStatus.SUCCESS

    # 2. Stage completeness and isolation
    assert final_result.ingestion is not None
    assert final_result.validation is not None
    assert final_result.quality is not None
    assert final_result.anomaly is not None
    assert final_result.evidence is not None
    assert final_result.contributor_analysis is not None

    # 3. Recommendations epistemic level
    rec_coll = final_result.recommendation.recommendation_collection
    assert rec_coll.investigation_id == "inv-climb-profile-01"
    for r in final_result.recommendation.recommendations:
        assert r.provenance.epistemic_level == EpistemicLevel.RECOMMENDATION
        assert r.status == RecommendationStatus.PROPOSED
        assert r.recommendation_type in (
            RecommendationType.VERIFY_DATA,
            RecommendationType.REVIEW_EVIDENCE,
            RecommendationType.COLLECT_ADDITIONAL_EVIDENCE,
            RecommendationType.REQUEST_ENGINEERING_REVIEW,
            RecommendationType.REPEAT_ANALYSIS,
        )

    # 4. Provenance continuity across the entire pipeline
    assert final_result.provenance is not None
    assert final_result.recommendation.provenance is not None
    assert final_result.contributor_analysis.provenance is not None
    assert final_result.evidence.provenance is not None
    pipeline_parents = final_result.provenance.parent_record_ids
    assert final_result.recommendation.provenance.record_id in pipeline_parents
    assert final_result.contributor_analysis.provenance.record_id in pipeline_parents
    assert final_result.evidence.provenance.record_id in pipeline_parents

    # 5. Immutability: Prior outputs not mutated
    assert res_phase2.contributor_analysis is not None
    assert len(res_phase2.contributor_analysis.results) == 3
    assert len(res_phase2.evidence.collection.evidence) == 1
