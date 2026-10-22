import numpy as np
import pandas as pd

from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus
from aeris.core.statistics.models import (
    MetricType,
    StatisticalAnalysisRequest,
    StatisticalAnalysisResult,
    StatisticalMetric,
)


class StatisticalAnalysisEngine:
    """
    Focused statistical execution engine for descriptive statistics.
    Calculates explicit metrics on explicit target columns without mutating input data.
    """

    def execute(
        self,
        request: StatisticalAnalysisRequest,
        data: pd.DataFrame,
        parent_record_ids: list[str] | None = None,
        source_uri: str | None = None,
    ) -> StatisticalAnalysisResult:

        issues = []
        metrics: list[StatisticalMetric] = []

        # Validate target column
        if request.target_column not in data.columns:
            issues.append(
                Issue(
                    issue_code="ERR_TARGET_COLUMN_MISSING",
                    severity=IssueSeverity.ERROR,
                    message=f"Target column '{request.target_column}' is missing from the dataset.",
                )
            )
            return self._build_result(
                request, ResultStatus.ERROR, issues, [], parent_record_ids, source_uri
            )

        series = data[request.target_column]

        # Numeric check if any numeric metric is requested
        numeric_metrics = {
            MetricType.MEAN,
            MetricType.MEDIAN,
            MetricType.MINIMUM,
            MetricType.MAXIMUM,
            MetricType.VARIANCE,
            MetricType.STANDARD_DEVIATION,
            MetricType.PERCENTILE,
        }
        requested_set = set(request.requested_metrics)
        needs_numeric = bool(requested_set & numeric_metrics)

        if needs_numeric and not pd.api.types.is_numeric_dtype(series):
            issues.append(
                Issue(
                    issue_code="ERR_TARGET_DATA_TYPE_INVALID",
                    severity=IssueSeverity.ERROR,
                    message=f"Target column '{request.target_column}' is not numeric.",
                )
            )
            return self._build_result(
                request, ResultStatus.ERROR, issues, [], parent_record_ids, source_uri
            )

        # Validate percentiles explicitly if PERCENTILE is requested
        if MetricType.PERCENTILE in requested_set:
            if not request.requested_percentiles:
                issues.append(
                    Issue(
                        issue_code="ERR_INVALID_PERCENTILE_CONFIGURATION",
                        severity=IssueSeverity.ERROR,
                        message="Percentile requested but no percentiles explicitly configured.",
                    )
                )
                return self._build_result(
                    request, ResultStatus.ERROR, issues, [], parent_record_ids, source_uri
                )
            for p in request.requested_percentiles:
                if not (0.0 <= p <= 100.0):
                    issues.append(
                        Issue(
                            issue_code="ERR_INVALID_PERCENTILE_CONFIGURATION",
                            severity=IssueSeverity.ERROR,
                            message=f"Invalid percentile {p}. Must be between 0 and 100.",
                        )
                    )
                    return self._build_result(
                        request, ResultStatus.ERROR, issues, [], parent_record_ids, source_uri
                    )

        # Missingness
        valid_series = series.dropna()
        valid_count = len(valid_series)
        missing_count = len(series) - valid_count

        unavailable_count = 0
        total_requested = 0

        # Calculate metrics explicitly
        for metric_type in request.requested_metrics:
            if metric_type == MetricType.COUNT:
                metrics.append(
                    StatisticalMetric(
                        metric_id=f"{request.analysis_id}_{metric_type.value}",
                        metric_type=metric_type,
                        value=float(valid_count),
                    )
                )
                total_requested += 1

            elif metric_type == MetricType.MISSING_COUNT:
                metrics.append(
                    StatisticalMetric(
                        metric_id=f"{request.analysis_id}_{metric_type.value}",
                        metric_type=metric_type,
                        value=float(missing_count),
                    )
                )
                total_requested += 1

            elif metric_type == MetricType.MEAN:
                total_requested += 1
                if valid_count > 0:
                    val = float(valid_series.mean())
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_MEAN",
                            metric_type=metric_type,
                            value=val,
                        )
                    )
                else:
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_MEAN",
                            metric_type=metric_type,
                            value=None,
                        )
                    )
                    unavailable_count += 1

            elif metric_type == MetricType.MEDIAN:
                total_requested += 1
                if valid_count > 0:
                    val = float(valid_series.median())
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_MEDIAN",
                            metric_type=metric_type,
                            value=val,
                        )
                    )
                else:
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_MEDIAN",
                            metric_type=metric_type,
                            value=None,
                        )
                    )
                    unavailable_count += 1

            elif metric_type == MetricType.MINIMUM:
                total_requested += 1
                if valid_count > 0:
                    val = float(valid_series.min())
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_MIN",
                            metric_type=metric_type,
                            value=val,
                        )
                    )
                else:
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_MIN",
                            metric_type=metric_type,
                            value=None,
                        )
                    )
                    unavailable_count += 1

            elif metric_type == MetricType.MAXIMUM:
                total_requested += 1
                if valid_count > 0:
                    val = float(valid_series.max())
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_MAX",
                            metric_type=metric_type,
                            value=val,
                        )
                    )
                else:
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_MAX",
                            metric_type=metric_type,
                            value=None,
                        )
                    )
                    unavailable_count += 1

            elif metric_type == MetricType.VARIANCE:
                total_requested += 1
                if valid_count > 1:
                    val = float(valid_series.var(ddof=1))
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_VAR",
                            metric_type=metric_type,
                            value=val,
                        )
                    )
                else:
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_VAR",
                            metric_type=metric_type,
                            value=None,
                        )
                    )
                    unavailable_count += 1

            elif metric_type == MetricType.STANDARD_DEVIATION:
                total_requested += 1
                if valid_count > 1:
                    val = float(valid_series.std(ddof=1))
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_STD",
                            metric_type=metric_type,
                            value=val,
                        )
                    )
                else:
                    metrics.append(
                        StatisticalMetric(
                            metric_id=f"{request.analysis_id}_STD",
                            metric_type=metric_type,
                            value=None,
                        )
                    )
                    unavailable_count += 1

            elif metric_type == MetricType.PERCENTILE:
                for p in request.requested_percentiles:
                    total_requested += 1
                    if valid_count > 0:
                        val = float(np.percentile(valid_series, p))
                        metrics.append(
                            StatisticalMetric(
                                metric_id=f"{request.analysis_id}_P_{p}",
                                metric_type=metric_type,
                                value=val,
                                parameters={"percentile": p},
                            )
                        )
                    else:
                        metrics.append(
                            StatisticalMetric(
                                metric_id=f"{request.analysis_id}_P_{p}",
                                metric_type=metric_type,
                                value=None,
                                parameters={"percentile": p},
                            )
                        )
                        unavailable_count += 1
            else:
                issues.append(
                    Issue(
                        issue_code="ERR_UNSUPPORTED_METRIC",
                        severity=IssueSeverity.WARNING,
                        message=f"Metric {metric_type} is unsupported.",
                    )
                )

        if unavailable_count > 0:
            issues.append(
                Issue(
                    issue_code="ERR_INSUFFICIENT_DATA",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Insufficient valid data to calculate "
                        f"{unavailable_count} requested metrics."
                    ),
                )
            )

        if total_requested > 0 and unavailable_count == total_requested:
            status = ResultStatus.NOT_AVAILABLE
        elif unavailable_count > 0 or issues:
            status = ResultStatus.WARNING
        else:
            status = ResultStatus.SUCCESS

        return self._build_result(request, status, issues, metrics, parent_record_ids, source_uri)

    def _build_result(
        self,
        request: StatisticalAnalysisRequest,
        status: ResultStatus,
        issues: list[Issue],
        metrics: list[StatisticalMetric],
        parent_record_ids: list[str] | None,
        source_uri: str | None,
    ) -> StatisticalAnalysisResult:

        prov = create_provenance(
            source_uri=source_uri or "in-memory",
            activity_name="statistical_analysis",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=parent_record_ids or [],
        )

        return StatisticalAnalysisResult(
            status=status,
            issues=issues,
            result_id=f"res_{request.analysis_id}_{prov.record_id[:8]}",
            analysis_id=request.analysis_id,
            target_column=request.target_column,
            metrics=metrics,
            provenance=prov,
        )
