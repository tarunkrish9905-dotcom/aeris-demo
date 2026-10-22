import datetime
import uuid

from aeris.core.expected_observed.models import (
    ComparisonMethod,
    Deviation,
    ExpectedObservedComparison,
    ExpectedValue,
    ObservedValue,
    TemporalExpectedObservedPoint,
    TemporalExpectedObservedRequest,
    TemporalExpectedObservedResult,
    Tolerance,
    ToleranceType,
    VectorComparisonRequest,
    VectorComparisonResult,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


class ExpectedObservedEngine:
    """Computational engine for Expected-vs-Observed calculations."""

    @classmethod
    def compare(
        cls,
        expected: ExpectedValue,
        observed: ObservedValue,
        method: ComparisonMethod,
        tolerance: Tolerance | None = None,
        parent_record_ids: list[str] | None = None,
    ) -> ExpectedObservedComparison:
        """
        Evaluate scalar expected-vs-observed values deterministically.
        No pandas, no AnomalyRecord logic.
        """
        issues: list[Issue] = []

        if expected.parameter_id != observed.parameter_id:
            issues.append(
                Issue(
                    issue_code="ERR_PARAMETER_MISMATCH",
                    severity=IssueSeverity.ERROR,
                    message=f"Parameter mismatch: {expected.parameter_id} vs {observed.parameter_id}",
                )
            )

        if expected.unit != observed.unit:
            if not ((expected.unit is None) and (observed.unit is None)):
                issues.append(
                    Issue(
                        issue_code="ERR_INCOMPATIBLE_UNITS",
                        severity=IssueSeverity.ERROR,
                        message="Units are incompatible.",
                    )
                )

        if issues:
            return cls._build_error_result(expected, observed, tolerance, issues, parent_record_ids)

        e_val = expected.value
        o_val = observed.value
        dev_value: float | None = None

        if method == ComparisonMethod.ABSOLUTE_DIFFERENCE:
            dev_value = o_val - e_val
        elif method == ComparisonMethod.RELATIVE_DIFFERENCE:
            if e_val == 0.0:
                issues.append(
                    Issue(
                        issue_code="ERR_ZERO_DENOMINATOR",
                        severity=IssueSeverity.ERROR,
                        message="Relative difference undefined for expected=0.",
                    )
                )
            else:
                dev_value = (o_val - e_val) / e_val
        elif method == ComparisonMethod.PERCENT_DIFFERENCE:
            if e_val == 0.0:
                issues.append(
                    Issue(
                        issue_code="ERR_ZERO_DENOMINATOR",
                        severity=IssueSeverity.ERROR,
                        message="Percentage difference undefined for expected=0.",
                    )
                )
            else:
                dev_value = ((o_val - e_val) / e_val) * 100.0
        elif method == ComparisonMethod.RATIO:
            if e_val == 0.0:
                issues.append(
                    Issue(
                        issue_code="ERR_ZERO_DENOMINATOR",
                        severity=IssueSeverity.ERROR,
                        message="Ratio undefined for expected=0.",
                    )
                )
            else:
                dev_value = o_val / e_val

        if issues:
            return cls._build_error_result(expected, observed, tolerance, issues, parent_record_ids)

        assert dev_value is not None

        dev_unit = None
        if method == ComparisonMethod.ABSOLUTE_DIFFERENCE:
            dev_unit = expected.unit
        elif method == ComparisonMethod.PERCENT_DIFFERENCE:
            dev_unit = "%"

        deviation = Deviation(
            expected_value=e_val,
            observed_value=o_val,
            deviation_value=dev_value,
            deviation_unit=dev_unit,
            comparison_method=method,
        )

        tolerance_evaluated = False
        within_tolerance: bool | None = None

        if tolerance is not None:
            if tolerance.tolerance_type == ToleranceType.ABSOLUTE:
                if tolerance.unit and tolerance.unit != expected.unit:
                    issues.append(
                        Issue(
                            issue_code="ERR_INCOMPATIBLE_TOLERANCE_UNIT",
                            severity=IssueSeverity.ERROR,
                            message="Tolerance unit incompatible with absolute value.",
                        )
                    )
                else:
                    diff = o_val - e_val
                    lower = (
                        tolerance.lower_bound
                        if tolerance.lower_bound is not None
                        else -float("inf")
                    )
                    upper = (
                        tolerance.upper_bound if tolerance.upper_bound is not None else float("inf")
                    )
                    within_tolerance = lower <= diff <= upper
                    tolerance_evaluated = True

            elif tolerance.tolerance_type == ToleranceType.RELATIVE:
                if e_val == 0.0:
                    issues.append(
                        Issue(
                            issue_code="ERR_ZERO_DENOMINATOR_TOLERANCE",
                            severity=IssueSeverity.ERROR,
                            message="Relative tolerance undefined for expected=0.",
                        )
                    )
                else:
                    rel_diff = (o_val - e_val) / e_val
                    lower = (
                        tolerance.lower_bound
                        if tolerance.lower_bound is not None
                        else -float("inf")
                    )
                    upper = (
                        tolerance.upper_bound if tolerance.upper_bound is not None else float("inf")
                    )
                    within_tolerance = lower <= rel_diff <= upper
                    tolerance_evaluated = True

            elif tolerance.tolerance_type == ToleranceType.PERCENTAGE:
                if e_val == 0.0:
                    issues.append(
                        Issue(
                            issue_code="ERR_ZERO_DENOMINATOR_TOLERANCE",
                            severity=IssueSeverity.ERROR,
                            message="Percentage tolerance undefined for expected=0.",
                        )
                    )
                else:
                    pct_diff = ((o_val - e_val) / e_val) * 100.0
                    lower = (
                        tolerance.lower_bound
                        if tolerance.lower_bound is not None
                        else -float("inf")
                    )
                    upper = (
                        tolerance.upper_bound if tolerance.upper_bound is not None else float("inf")
                    )
                    within_tolerance = lower <= pct_diff <= upper
                    tolerance_evaluated = True

        status = ResultStatus.PARTIAL if issues else ResultStatus.SUCCESS

        prov = cls._create_provenance(parent_record_ids or [])

        return ExpectedObservedComparison(
            comparison_id=str(uuid.uuid4()),
            expected=expected,
            observed=observed,
            deviation=deviation,
            tolerance=tolerance,
            tolerance_evaluated=tolerance_evaluated,
            within_tolerance=within_tolerance,
            status=status,
            issues=issues,
            provenance=prov,
        )

    @classmethod
    def _build_error_result(
        cls,
        expected: ExpectedValue,
        observed: ObservedValue,
        tolerance: Tolerance | None,
        issues: list[Issue],
        parent_record_ids: list[str] | None,
    ) -> ExpectedObservedComparison:
        return ExpectedObservedComparison(
            comparison_id=str(uuid.uuid4()),
            expected=expected,
            observed=observed,
            deviation=None,
            tolerance=tolerance,
            tolerance_evaluated=False,
            within_tolerance=None,
            status=ResultStatus.ERROR,
            issues=issues,
            provenance=cls._create_provenance(parent_record_ids or []),
        )

    @classmethod
    def _create_provenance(cls, parent_record_ids: list[str]) -> ProvenanceRecord:
        return ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            activity_name="EXPECTED_OBSERVED_COMPARISON",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=list(parent_record_ids),
            created_at=datetime.datetime.now(datetime.UTC),
            source_uri="aeris://core/expected_observed/engine",
        )


class VectorExpectedObservedEngine:
    """Computational engine for evaluating explicit sequences of expected vs observed values."""

    @classmethod
    def compare_vectors(
        cls,
        request: VectorComparisonRequest,
        parent_record_ids: list[str] | None = None,
    ) -> VectorComparisonResult:
        """
        Evaluates a vector request element-by-element using ExpectedObservedEngine.
        """
        if parent_record_ids is None:
            parent_record_ids = []

        overall_issues: list[Issue] = []
        comparisons = []

        # Length check (should be handled by Pydantic, but double check)
        if len(request.expected_values) != len(request.observed_values):
            prov = ProvenanceRecord(
                record_id=str(uuid.uuid4()),
                activity_name="VECTOR_EXPECTED_OBSERVED_EVALUATION",
                epistemic_level=EpistemicLevel.CALCULATED,
                parent_record_ids=parent_record_ids,
                created_at=datetime.datetime.now(datetime.UTC),
                source_uri="in-memory",
            )
            return VectorComparisonResult(
                vector_id=request.vector_id,
                parameter_id=request.parameter_id,
                method=request.method,
                status=ResultStatus.ERROR,
                issues=[
                    Issue(
                        issue_code="ERR_LENGTH_MISMATCH",
                        severity=IssueSeverity.ERROR,
                        message="Vector length mismatch.",
                    )
                ],
                provenance=prov,
                elements=(),
            )

        for _i, (exp, obs) in enumerate(zip(request.expected_values, request.observed_values, strict=False)):
            comp = ExpectedObservedEngine.compare(
                expected=exp,
                observed=obs,
                method=request.method,
                tolerance=request.tolerance,
                parent_record_ids=parent_record_ids,
            )
            comparisons.append(comp)

        if not comparisons:
            # Empty vectors
            prov = ProvenanceRecord(
                record_id=str(uuid.uuid4()),
                activity_name="VECTOR_EXPECTED_OBSERVED_EVALUATION",
                epistemic_level=EpistemicLevel.CALCULATED,
                parent_record_ids=parent_record_ids,
                created_at=datetime.datetime.now(datetime.UTC),
                source_uri="in-memory",
            )
            return VectorComparisonResult(
                vector_id=request.vector_id,
                parameter_id=request.parameter_id,
                method=request.method,
                status=ResultStatus.SUCCESS,
                issues=[],
                provenance=prov,
                elements=(),
            )

        statuses = [c.status for c in comparisons]

        if all(s == ResultStatus.NOT_AVAILABLE for s in statuses):
            status = ResultStatus.NOT_AVAILABLE
        elif ResultStatus.ERROR in statuses:
            if ResultStatus.SUCCESS in statuses or ResultStatus.WARNING in statuses:
                status = ResultStatus.PARTIAL
            else:
                status = ResultStatus.ERROR
        elif ResultStatus.PARTIAL in statuses:
            status = ResultStatus.PARTIAL
        elif ResultStatus.WARNING in statuses:
            status = ResultStatus.WARNING
        else:
            status = ResultStatus.SUCCESS

        prov = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            activity_name="VECTOR_EXPECTED_OBSERVED_EVALUATION",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=parent_record_ids,
            created_at=datetime.datetime.now(datetime.UTC),
            source_uri="in-memory",
        )

        return VectorComparisonResult(
            vector_id=request.vector_id,
            parameter_id=request.parameter_id,
            method=request.method,
            status=status,
            issues=overall_issues,
            provenance=prov,
            elements=tuple(comparisons),
        )


class TemporalExpectedObservedEngine:
    """Computational engine for evaluating time-aware Expected-vs-Observed values."""

    @classmethod
    def compare_temporal(
        cls,
        request: TemporalExpectedObservedRequest,
        parent_record_ids: list[str] | None = None,
    ) -> TemporalExpectedObservedResult:
        """
        Evaluates a temporal request matching by exact timestamp.
        """
        if parent_record_ids is None:
            parent_record_ids = []

        overall_issues: list[Issue] = []

        if not request.expected_values and not request.observed_values:
            prov = ProvenanceRecord(
                record_id=str(uuid.uuid4()),
                activity_name="TEMPORAL_EXPECTED_OBSERVED_EVALUATION",
                epistemic_level=EpistemicLevel.CALCULATED,
                parent_record_ids=parent_record_ids,
                created_at=datetime.datetime.now(datetime.UTC),
                source_uri="in-memory",
            )
            return TemporalExpectedObservedResult(
                comparison_id=request.comparison_id,
                parameter_id=request.parameter_id,
                method=request.method,
                matched_points=(),
                unmatched_expected=(),
                unmatched_observed=(),
                expected_count=0,
                observed_count=0,
                matched_count=0,
                expected_only_count=0,
                observed_only_count=0,
                status=ResultStatus.NOT_AVAILABLE,
                issues=[
                    Issue(
                        issue_code="ERR_EMPTY_TEMPORAL_INPUT",
                        severity=IssueSeverity.WARNING,
                        message="Both expected and observed vectors are empty.",
                    )
                ],
                provenance=prov,
            )

        e_times = [e.timestamp for e in request.expected_values]
        o_times = [o.timestamp for o in request.observed_values]

        if len(set(e_times)) != len(e_times):
            overall_issues.append(
                Issue(
                    issue_code="ERR_DUPLICATE_EXPECTED_TIMESTAMPS",
                    severity=IssueSeverity.ERROR,
                    message="Duplicate expected timestamps detected.",
                )
            )
        if len(set(o_times)) != len(o_times):
            overall_issues.append(
                Issue(
                    issue_code="ERR_DUPLICATE_OBSERVED_TIMESTAMPS",
                    severity=IssueSeverity.ERROR,
                    message="Duplicate observed timestamps detected.",
                )
            )

        if overall_issues:
            prov = ProvenanceRecord(
                record_id=str(uuid.uuid4()),
                activity_name="TEMPORAL_EXPECTED_OBSERVED_EVALUATION",
                epistemic_level=EpistemicLevel.CALCULATED,
                parent_record_ids=parent_record_ids,
                created_at=datetime.datetime.now(datetime.UTC),
                source_uri="in-memory",
            )
            return TemporalExpectedObservedResult(
                comparison_id=request.comparison_id,
                parameter_id=request.parameter_id,
                method=request.method,
                matched_points=(),
                unmatched_expected=request.expected_values,
                unmatched_observed=request.observed_values,
                expected_count=len(request.expected_values),
                observed_count=len(request.observed_values),
                matched_count=0,
                expected_only_count=len(request.expected_values),
                observed_only_count=len(request.observed_values),
                status=ResultStatus.ERROR,
                issues=overall_issues,
                provenance=prov,
            )

        exp_map = {e.timestamp: e for e in request.expected_values}
        obs_map = {o.timestamp: o for o in request.observed_values}

        all_timestamps = sorted(list(set(exp_map.keys()) | set(obs_map.keys())))

        matched_points = []
        unmatched_exp = []
        unmatched_obs = []

        for t in all_timestamps:
            if t in exp_map and t in obs_map:
                comp = ExpectedObservedEngine.compare(
                    expected=exp_map[t],
                    observed=obs_map[t],
                    method=request.method,
                    tolerance=request.tolerance,
                    parent_record_ids=parent_record_ids,
                )
                matched_points.append(TemporalExpectedObservedPoint(timestamp=t, comparison=comp))
            elif t in exp_map:
                unmatched_exp.append(exp_map[t])
            elif t in obs_map:
                unmatched_obs.append(obs_map[t])

        if not matched_points:
            status = ResultStatus.NOT_AVAILABLE
        else:
            c_statuses = [m.comparison.status for m in matched_points]
            if ResultStatus.ERROR in c_statuses:
                if ResultStatus.SUCCESS in c_statuses or ResultStatus.WARNING in c_statuses:
                    status = ResultStatus.PARTIAL
                else:
                    status = ResultStatus.ERROR
            elif ResultStatus.PARTIAL in c_statuses or unmatched_exp or unmatched_obs:
                status = ResultStatus.PARTIAL
            elif ResultStatus.WARNING in c_statuses:
                status = ResultStatus.WARNING
            else:
                status = ResultStatus.SUCCESS

        prov = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            activity_name="TEMPORAL_EXPECTED_OBSERVED_EVALUATION",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=parent_record_ids,
            created_at=datetime.datetime.now(datetime.UTC),
            source_uri="in-memory",
        )

        return TemporalExpectedObservedResult(
            comparison_id=request.comparison_id,
            parameter_id=request.parameter_id,
            method=request.method,
            matched_points=tuple(matched_points),
            unmatched_expected=tuple(unmatched_exp),
            unmatched_observed=tuple(unmatched_obs),
            expected_count=len(request.expected_values),
            observed_count=len(request.observed_values),
            matched_count=len(matched_points),
            expected_only_count=len(unmatched_exp),
            observed_only_count=len(unmatched_obs),
            status=status,
            issues=overall_issues,
            provenance=prov,
        )
