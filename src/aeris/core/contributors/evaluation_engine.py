from __future__ import annotations

from aeris.core.contributors.evaluation_models import (
    AssociationStrength,
    ContributorEvaluation,
    ContributorEvaluationRequest,
    ContributorEvaluationResult,
)
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


class ContributorEvaluationEngine:
    """Evaluates and optionally ranks contributor analysis results."""

    @classmethod
    def evaluate(cls, request: ContributorEvaluationRequest) -> ContributorEvaluationResult:
        config = request.configuration
        evaluations: list[ContributorEvaluation] = []
        overall_issues: list[Issue] = []

        # 1. Process each request independently
        for res in request.analysis_results:
            eval_issues: list[Issue] = []
            status = res.status
            strength = None
            coef = res.coefficient
            obs = res.usable_observation_count

            if (
                status in (ResultStatus.ERROR, ResultStatus.NOT_AVAILABLE)
                or coef is None
                or obs is None
            ):
                status = ResultStatus.NOT_AVAILABLE
                eval_issues.append(
                    Issue(
                        issue_code="ERR_RESULT_UNAVAILABLE",
                        severity=IssueSeverity.WARNING,
                        message=(
                            "Cannot evaluate a result that is missing coefficient or "
                            "observation count."
                        ),
                    )
                )
            else:
                if obs < config.minimum_usable_observations:
                    status = ResultStatus.NOT_AVAILABLE
                    eval_issues.append(
                        Issue(
                            issue_code="ERR_INSUFFICIENT_OBSERVATIONS",
                            severity=IssueSeverity.WARNING,
                            message=(
                                f"Usable observations ({obs}) below configured minimum "
                                f"({config.minimum_usable_observations})."
                            ),
                        )
                    )
                else:
                    abs_c = abs(coef)
                    if abs_c < config.very_weak_upper:
                        strength = AssociationStrength.VERY_WEAK
                    elif abs_c < config.weak_upper:
                        strength = AssociationStrength.WEAK
                    elif abs_c < config.moderate_upper:
                        strength = AssociationStrength.MODERATE
                    elif abs_c < config.strong_upper:
                        strength = AssociationStrength.STRONG
                    else:
                        strength = AssociationStrength.VERY_STRONG

            prov = create_provenance(
                source_uri=res.provenance.source_uri if res.provenance else "unknown",
                activity_name="contributor_evaluation",
                parent_record_ids=[res.provenance.record_id] if res.provenance else [],
                epistemic_level=EpistemicLevel.CALCULATED,
            )

            evaluation = ContributorEvaluation(
                analysis_id=res.analysis_id,
                target_column=res.target_column,
                factor_column=res.factor_column,
                coefficient=coef,
                association_strength=strength,
                usable_observation_count=obs,
                excluded_observation_count=res.excluded_observation_count,
                status=status,
                issues=eval_issues,
                provenance=prov,
                rank_position=None,
            )
            evaluations.append(evaluation)

        # 2. Ranking
        if request.request_ranking:
            valid_evals = [
                e for e in evaluations if e.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL)
            ]
            invalid_evals = [
                e
                for e in evaluations
                if e.status not in (ResultStatus.SUCCESS, ResultStatus.PARTIAL)
            ]

            # Ordinal ranking: Strict deterministic tie-breaking
            valid_evals.sort(
                key=lambda e: (
                    -abs(e.coefficient) if e.coefficient is not None else 0.0,
                    -(e.usable_observation_count or 0),
                    e.analysis_id,
                    e.target_column,
                    e.factor_column,
                )
            )

            ranked_valid = []
            for i, e in enumerate(valid_evals, start=1):
                ranked_valid.append(e.model_copy(update={"rank_position": i}))

            evaluations = ranked_valid + invalid_evals

        # 3. Overall Status
        has_success = any(e.status == ResultStatus.SUCCESS for e in evaluations)
        has_partial = any(e.status == ResultStatus.PARTIAL for e in evaluations)
        has_invalid = any(
            e.status in (ResultStatus.ERROR, ResultStatus.NOT_AVAILABLE) for e in evaluations
        )

        if not evaluations or (has_success and not has_invalid and not has_partial):
            overall_status = ResultStatus.SUCCESS
        elif (has_success or has_partial) and has_invalid:
            overall_status = ResultStatus.PARTIAL
        elif not has_success and not has_partial:
            overall_status = ResultStatus.NOT_AVAILABLE
        else:
            overall_status = ResultStatus.PARTIAL

        # Collect parent record ids from all evaluations for the result container
        parent_ids = [e.provenance.record_id for e in evaluations if e.provenance]
        result_prov = create_provenance(
            source_uri="unknown",
            activity_name="contributor_evaluation_batch",
            parent_record_ids=parent_ids,
            epistemic_level=EpistemicLevel.CALCULATED,
        )

        return ContributorEvaluationResult(
            status=overall_status,
            issues=overall_issues,
            provenance=result_prov,
            evaluations=tuple(evaluations),
            is_ranked=request.request_ranking,
            ranking_policy=request.ranking_policy if request.request_ranking else None,
        )
