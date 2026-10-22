import uuid

import pandas as pd

from aeris.core.contributors.models import (
    ContributorAnalysisMethod,
    ContributorAnalysisRequest,
    ContributorAnalysisResult,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


class ContributorAnalysisEngine:
    """
    Controlled analytical engine for evaluating associations between
    explicitly supplied targets and factors.
    Does NOT infer causality, discover factors, or generate hypotheses.
    """

    @classmethod
    def analyze(
        cls, request: ContributorAnalysisRequest, df: pd.DataFrame
    ) -> ContributorAnalysisResult:
        issues: list[Issue] = []

        # 1. Validate requested columns exist
        missing_cols = []
        for col in [request.target_column, request.factor_column]:
            if col not in df.columns:
                missing_cols.append(col)

        if request.timestamp_column and request.timestamp_column not in df.columns:
            missing_cols.append(request.timestamp_column)

        if missing_cols:
            issues.append(
                Issue(
                    issue_code="ERR_MISSING_COLUMN",
                    severity=IssueSeverity.ERROR,
                    message=f"Missing columns required for analysis: {missing_cols}",
                )
            )
            return cls._build_error_result(request, issues)

        # 2. Timestamp validation (if provided)
        if request.timestamp_column:
            ts_series = df[request.timestamp_column]
            if not pd.api.types.is_datetime64_any_dtype(ts_series):
                issues.append(
                    Issue(
                        issue_code="ERR_TIMESTAMP_INVALID",
                        severity=IssueSeverity.ERROR,
                        message="Timestamp column is not datetime64 type.",
                    )
                )
                return cls._build_error_result(request, issues)
            # Check for tz-awareness and UTC
            if ts_series.dt.tz is None:
                issues.append(
                    Issue(
                        issue_code="ERR_TIMESTAMP_NAIVE",
                        severity=IssueSeverity.ERROR,
                        message="Timestamp column must be timezone-aware.",
                    )
                )
                return cls._build_error_result(request, issues)
            # Strict UTC check
            tz = ts_series.dt.tz
            if str(tz) != "UTC" and getattr(tz, "zone", "") != "UTC":
                issues.append(
                    Issue(
                        issue_code="ERR_TIMESTAMP_NON_UTC",
                        severity=IssueSeverity.ERROR,
                        message="Timestamp column must be UTC.",
                    )
                )
                return cls._build_error_result(request, issues)

        # 3. Numeric semantics check
        for col in [request.target_column, request.factor_column]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(
                    Issue(
                        issue_code="ERR_NON_NUMERIC_INPUT",
                        severity=IssueSeverity.ERROR,
                        message=f"Column '{col}' must be strictly numeric.",
                    )
                )
                return cls._build_error_result(request, issues)

            # Specifically reject boolean which is a subclass of numeric in pandas sometimes
            if pd.api.types.is_bool_dtype(df[col]):
                issues.append(
                    Issue(
                        issue_code="ERR_BOOLEAN_INPUT_REJECTED",
                        severity=IssueSeverity.ERROR,
                        message=(
                            f"Column '{col}' is boolean, which is unsupported for "
                            "strict numeric correlation."
                        ),
                    )
                )
                return cls._build_error_result(request, issues)

        # 4. Extract subset and drop NaNs (pairwise exclusion)
        subset = df[[request.target_column, request.factor_column]].dropna()

        total_rows = len(df)
        usable = len(subset)
        excluded = total_rows - usable

        # 5. Check minimum observation count
        if usable < request.minimum_observations:
            issues.append(
                Issue(
                    issue_code="ERR_INSUFFICIENT_OBSERVATIONS",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Usable pairs ({usable}) below minimum required "
                        f"({request.minimum_observations})."
                    ),
                )
            )
            return cls._build_not_available_result(request, issues, total_rows, usable, excluded)

        # 6. Check for zero variance
        if subset[request.target_column].nunique() <= 1:
            issues.append(
                Issue(
                    issue_code="ERR_ZERO_VARIANCE",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Target column '{request.target_column}' has zero "
                        "variance in usable pairs."
                    ),
                )
            )
            return cls._build_not_available_result(request, issues, total_rows, usable, excluded)

        if subset[request.factor_column].nunique() <= 1:
            issues.append(
                Issue(
                    issue_code="ERR_ZERO_VARIANCE",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Factor column '{request.factor_column}' has zero "
                        "variance in usable pairs."
                    ),
                )
            )
            return cls._build_not_available_result(request, issues, total_rows, usable, excluded)

        # 7. Perform calculation
        if request.analysis_method == ContributorAnalysisMethod.PEARSON:
            coef = subset[request.factor_column].corr(
                subset[request.target_column], method="pearson"
            )
        elif request.analysis_method == ContributorAnalysisMethod.SPEARMAN:
            issues.append(
                Issue(
                    issue_code="ERR_METHOD_UNAVAILABLE_SCIPY_MISSING",
                    severity=IssueSeverity.ERROR,
                    message=(
                        "Spearman correlation requires scipy, which is "
                        "intentionally excluded from the architecture."
                    ),
                )
            )
            return cls._build_error_result(request, issues)
        else:
            issues.append(
                Issue(
                    issue_code="ERR_UNSUPPORTED_METHOD",
                    severity=IssueSeverity.ERROR,
                    message=f"Analysis method {request.analysis_method.name} is not supported.",
                )
            )
            return cls._build_error_result(request, issues)

        status = ResultStatus.SUCCESS
        if excluded > 0:
            status = ResultStatus.PARTIAL

        prov = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            source_uri=request.source_reference or "unknown",
            activity_name=f"ContributorAnalysisEngine_{request.analysis_method.name}",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=list(request.parent_record_ids),
        )

        return ContributorAnalysisResult(
            status=status,
            issues=issues,
            provenance=prov,
            analysis_id=request.analysis_id,
            target_column=request.target_column,
            factor_column=request.factor_column,
            analysis_method=request.analysis_method,
            total_input_rows=total_rows,
            usable_observation_count=usable,
            excluded_observation_count=excluded,
            coefficient=float(coef),
            p_value=None,
            metadata={"disclaimer": "Result indicates statistical association, NOT causality."},
        )

    @classmethod
    def _build_error_result(
        cls, request: ContributorAnalysisRequest, issues: list[Issue]
    ) -> ContributorAnalysisResult:
        prov = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            source_uri=request.source_reference or "unknown",
            activity_name="ContributorAnalysisEngine_FAILED",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=list(request.parent_record_ids),
        )
        return ContributorAnalysisResult(
            status=ResultStatus.ERROR,
            issues=issues,
            provenance=prov,
            analysis_id=request.analysis_id,
            target_column=request.target_column,
            factor_column=request.factor_column,
            analysis_method=request.analysis_method,
            total_input_rows=None,
            usable_observation_count=None,
            excluded_observation_count=None,
            coefficient=None,
            p_value=None,
        )

    @classmethod
    def _build_not_available_result(
        cls,
        request: ContributorAnalysisRequest,
        issues: list[Issue],
        total: int,
        usable: int,
        excluded: int,
    ) -> ContributorAnalysisResult:
        prov = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            source_uri=request.source_reference or "unknown",
            activity_name="ContributorAnalysisEngine_NOT_AVAILABLE",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=list(request.parent_record_ids),
        )
        return ContributorAnalysisResult(
            status=ResultStatus.NOT_AVAILABLE,
            issues=issues,
            provenance=prov,
            analysis_id=request.analysis_id,
            target_column=request.target_column,
            factor_column=request.factor_column,
            analysis_method=request.analysis_method,
            total_input_rows=total,
            usable_observation_count=usable,
            excluded_observation_count=excluded,
            coefficient=None,
            p_value=None,
        )
