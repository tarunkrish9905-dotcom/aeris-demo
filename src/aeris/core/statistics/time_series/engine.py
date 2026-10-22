import datetime
from typing import Any

import pandas as pd

from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus
from aeris.core.statistics.models import MetricType
from aeris.core.statistics.time_series.models import (
    SamplingIrregularity,
    SamplingMetadata,
    TemporalScopeType,
    TimeSeriesAnalysisRequest,
    TimeSeriesAnalysisResult,
    TimeSeriesMetric,
    WindowType,
)


class TimeSeriesStatisticalAnalysisEngine:
    """Explicit rolling/windowed statistical execution engine."""

    def execute(
        self,
        request: TimeSeriesAnalysisRequest,
        data: pd.DataFrame,
        parent_record_ids: list[str] | None = None,
        source_uri: str | None = None,
    ) -> TimeSeriesAnalysisResult:
        if parent_record_ids is None:
            parent_record_ids = []

        issues: list[Issue] = []
        status = ResultStatus.SUCCESS

        # 1. Validation
        if request.timestamp_column not in data.columns:
            issues.append(
                Issue(
                    issue_code="ERR_MISSING_TIMESTAMP_COLUMN",
                    severity=IssueSeverity.ERROR,
                    message=f"Timestamp column '{request.timestamp_column}' is missing.",
                )
            )
            return self._build_error_result(request, issues, parent_record_ids, source_uri)

        if request.target_column not in data.columns:
            issues.append(
                Issue(
                    issue_code="ERR_MISSING_TARGET_COLUMN",
                    severity=IssueSeverity.ERROR,
                    message=f"Target column '{request.target_column}' is missing.",
                )
            )
            return self._build_error_result(request, issues, parent_record_ids, source_uri)

        # 2. Extract Data
        # Keep index aligned, but only operate on the timestamp and target
        ts_series = data[request.timestamp_column]
        target_series = data[request.target_column]

        # 3. Check Timestamp Constraints
        if not pd.api.types.is_datetime64_any_dtype(ts_series):
            # Attempt to check if we can cleanly use them as datetimes
            # But the contract says it MUST contain timezone-aware UTC timestamps
            # So if it's string, we fail. The orchestrator / pipeline must have parsed it.
            issues.append(
                Issue(
                    issue_code="ERR_INVALID_TIMESTAMP_DTYPE",
                    severity=IssueSeverity.ERROR,
                    message="Timestamp column is not a valid datetime dtype.",
                )
            )
            return self._build_error_result(request, issues, parent_record_ids, source_uri)

        # Check UTC
        if ts_series.dt.tz is None or ts_series.dt.tz != datetime.UTC:
            issues.append(
                Issue(
                    issue_code="ERR_NON_UTC_TIMESTAMP",
                    severity=IssueSeverity.ERROR,
                    message="Timestamps must be explicitly UTC timezone-aware.",
                )
            )
            return self._build_error_result(request, issues, parent_record_ids, source_uri)

        # Check Monotonicity
        if not ts_series.is_monotonic_increasing:
            issues.append(
                Issue(
                    issue_code="ERR_UNORDERED_TIMESTAMPS",
                    severity=IssueSeverity.ERROR,
                    message="Timestamps are not monotonically increasing. Sorting is not permitted.",  # noqa: E501
                )
            )
            return self._build_error_result(request, issues, parent_record_ids, source_uri)

        # 4. Filter by Temporal Scope
        mask = pd.Series(True, index=data.index)
        if request.temporal_scope.scope_type == TemporalScopeType.EXPLICIT_INTERVAL:
            if request.temporal_scope.start_timestamp is not None:
                mask &= ts_series >= request.temporal_scope.start_timestamp
            if request.temporal_scope.end_timestamp is not None:
                mask &= ts_series <= request.temporal_scope.end_timestamp

        if not mask.any():
            issues.append(
                Issue(
                    issue_code="ERR_EMPTY_TEMPORAL_SCOPE",
                    severity=IssueSeverity.ERROR,
                    message="No eligible observations exist within the requested temporal scope.",
                )
            )
            res = self._build_error_result(request, issues, parent_record_ids, source_uri)
            return res.model_copy(update={"status": ResultStatus.NOT_AVAILABLE})

        # 5. Extract scoped subset (keep original index)
        scoped_ts = ts_series[mask]
        scoped_target = target_series[mask]

        if not pd.api.types.is_numeric_dtype(scoped_target):
            issues.append(
                Issue(
                    issue_code="ERR_NON_NUMERIC_TARGET",
                    severity=IssueSeverity.ERROR,
                    message="Target column is not numeric.",
                )
            )
            return self._build_error_result(request, issues, parent_record_ids, source_uri)

        # 6. Analyze Sampling Metadata
        irregularities = []
        dup_count = int(scoped_ts.duplicated().sum())
        if dup_count > 0:
            irregularities.append(SamplingIrregularity.DUPLICATE_TIMESTAMPS)

        diffs = scoped_ts.diff().dt.total_seconds()
        nominal_interval = None
        gap_count = 0
        if len(diffs) > 1:
            diff_counts = diffs.dropna().value_counts()
            if not diff_counts.empty:
                nominal_interval = float(diff_counts.idxmax())
                gaps = diffs[diffs > nominal_interval * 1.5]
                gap_count = len(gaps)
                if gap_count > 0:
                    irregularities.append(SamplingIrregularity.GAPS_PRESENT)

        is_regular = (
            dup_count == 0 and gap_count == 0 and len(diffs) > 1 and diffs.dropna().nunique() == 1
        )
        if not is_regular and len(diffs) > 1:
            irregularities.append(SamplingIrregularity.IRREGULAR)

        sampling_metadata = SamplingMetadata(
            is_regular=is_regular,
            irregularities=irregularities,
            duplicate_count=dup_count,
            gap_count=gap_count,
            nominal_interval_seconds=nominal_interval,
        )

        # 7. Check ambiguous duplicates for duration window
        if (
            request.window_configuration
            and request.window_configuration.window_type == WindowType.DURATION
            and dup_count > 0
        ):
            issues.append(
                Issue(
                    issue_code="ERR_DURATION_WINDOW_DUPLICATES",
                    severity=IssueSeverity.ERROR,
                    message="Duration windows cannot safely process duplicate timestamps.",
                )
            )
            return self._build_error_result(request, issues, parent_record_ids, source_uri)

        # 8. Compute Rolling Statistics
        df_scoped = pd.DataFrame(
            {"target": scoped_target, "ts": scoped_ts}, index=scoped_target.index
        )

        window_kwargs: dict[str, Any] = {}
        if request.window_configuration:
            if request.window_configuration.window_type == WindowType.OBSERVATION_COUNT:
                window_kwargs["window"] = request.window_configuration.window_size_observations
                window_kwargs["min_periods"] = request.window_configuration.min_observations
            elif request.window_configuration.window_type == WindowType.DURATION:
                # Use string offset for pandas rolling
                window_kwargs["window"] = f"{request.window_configuration.window_size_seconds}s"
                window_kwargs["on"] = "ts"
                window_kwargs["min_periods"] = request.window_configuration.min_observations
                window_kwargs["closed"] = "both"  # Match AERIS closed-interval semantics

        metrics = []
        if request.window_configuration:
            roller = (
                df_scoped[["target", "ts"]].rolling(**window_kwargs)
                if "on" in window_kwargs
                else df_scoped["target"].rolling(**window_kwargs)
            )

            for m_type in request.requested_metrics:
                if m_type == MetricType.COUNT:
                    res_series = (
                        roller["target"].count() if "on" in window_kwargs else roller.count()
                    )
                elif m_type == MetricType.MEAN:
                    res_series = roller["target"].mean() if "on" in window_kwargs else roller.mean()
                elif m_type == MetricType.MINIMUM:
                    res_series = roller["target"].min() if "on" in window_kwargs else roller.min()
                elif m_type == MetricType.MAXIMUM:
                    res_series = roller["target"].max() if "on" in window_kwargs else roller.max()
                elif m_type == MetricType.VARIANCE:
                    res_series = (
                        roller["target"].var(ddof=1)
                        if "on" in window_kwargs
                        else roller.var(ddof=1)
                    )
                elif m_type == MetricType.STANDARD_DEVIATION:
                    res_series = (
                        roller["target"].std(ddof=1)
                        if "on" in window_kwargs
                        else roller.std(ddof=1)
                    )
                else:
                    issues.append(
                        Issue(
                            issue_code="ERR_UNSUPPORTED_METRIC",
                            severity=IssueSeverity.ERROR,
                            message=f"Metric {m_type} is not supported for rolling analysis.",
                        )
                    )
                    return self._build_error_result(request, issues, parent_record_ids, source_uri)

                # Replace NaN with None for pure domain model
                values_tuple = tuple(None if pd.isna(x) else float(x) for x in res_series)
                if any(x is None for x in values_tuple):
                    status = ResultStatus.PARTIAL

                metrics.append(
                    TimeSeriesMetric(
                        metric_id=f"{request.analysis_id}_{m_type.name}",
                        metric_type=m_type,
                        values=values_tuple,
                    )
                )

        prov = create_provenance(
            source_uri=source_uri or "unknown",
            activity_name="time_series_rolling_analysis",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=parent_record_ids,
        )

        return TimeSeriesAnalysisResult(
            status=status,
            issues=issues,
            result_id=f"res_{request.analysis_id}",
            analysis_id=request.analysis_id,
            target_column=request.target_column,
            timestamp_column=request.timestamp_column,
            temporal_scope=request.temporal_scope,
            window_configuration=request.window_configuration,
            sampling_metadata=sampling_metadata,
            metrics=metrics,
            observation_count=len(scoped_target),
            provenance=prov,
        )

    def _build_error_result(
        self,
        request: TimeSeriesAnalysisRequest,
        issues: list[Issue],
        parent_record_ids: list[str],
        source_uri: str | None,
    ) -> TimeSeriesAnalysisResult:
        prov = create_provenance(
            source_uri=source_uri or "unknown",
            activity_name="time_series_rolling_analysis",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=parent_record_ids,
        )
        return TimeSeriesAnalysisResult(
            status=ResultStatus.ERROR,
            issues=issues,
            result_id=f"res_{request.analysis_id}_err",
            analysis_id=request.analysis_id,
            target_column=request.target_column,
            timestamp_column=request.timestamp_column,
            temporal_scope=request.temporal_scope,
            window_configuration=request.window_configuration,
            sampling_metadata=None,
            metrics=[],
            observation_count=0,
            provenance=prov,
        )
