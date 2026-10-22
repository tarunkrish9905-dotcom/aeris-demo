import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aeris.core.anomaly.models import (
    AnomalyClassification,
    AnomalyRecord,
    AnomalyResult,
    AnomalySeverity,
    DetectionMethod,
    EvidenceReference,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


class TimeSeriesAnomalyDetectionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    timestamp_column: str = Field(..., min_length=1)
    window_size: int = Field(..., gt=0)
    z_threshold: float

    minimum_observations: int = Field(default=2, ge=2)
    severity: AnomalySeverity = Field(default=AnomalySeverity.INFO)
    classification: AnomalyClassification = Field(default=AnomalyClassification.CONTEXTUAL_ANOMALY)
    ddof: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_z_threshold(self) -> "TimeSeriesAnomalyDetectionRequest":
        if not np.isfinite(self.z_threshold):
            raise ValueError("z_threshold must be finite.")
        if self.z_threshold <= 0:
            raise ValueError("z_threshold must be strictly greater than zero.")
        return self


class TimeSeriesAnomalyDetectionEngine:
    """Explicit time-series anomaly detection engine using trailing rolling Z-scores."""

    def execute(
        self,
        request: TimeSeriesAnomalyDetectionRequest,
        data: pd.DataFrame,
        parent_record_ids: list[str] | None = None,
        source_uri: str | None = None,
    ) -> AnomalyResult:

        issues: list[Issue] = []

        # 1. Validation
        if request.target_column not in data.columns:
            issues.append(
                Issue(
                    issue_code="ERR_MISSING_TARGET",
                    severity=IssueSeverity.ERROR,
                    message=f"Target column '{request.target_column}' is missing.",
                    field_name=request.target_column,
                )
            )
            return self._build_result(
                request,
                ResultStatus.ERROR,
                (),
                issues,
                parent_record_ids,
                source_uri,
                data.shape[0],
                0,
            )

        if request.timestamp_column not in data.columns:
            issues.append(
                Issue(
                    issue_code="ERR_MISSING_TIMESTAMP",
                    severity=IssueSeverity.ERROR,
                    message=f"Timestamp column '{request.timestamp_column}' is missing.",
                    field_name=request.timestamp_column,
                )
            )
            return self._build_result(
                request,
                ResultStatus.ERROR,
                (),
                issues,
                parent_record_ids,
                source_uri,
                data.shape[0],
                0,
            )

        target_series = data[request.target_column]
        if not pd.api.types.is_numeric_dtype(target_series):
            issues.append(
                Issue(
                    issue_code="ERR_NON_NUMERIC_TARGET",
                    severity=IssueSeverity.ERROR,
                    message="Target must be numeric.",
                    field_name=request.target_column,
                )
            )
            return self._build_result(
                request,
                ResultStatus.ERROR,
                (),
                issues,
                parent_record_ids,
                source_uri,
                data.shape[0],
                0,
            )

        if pd.api.types.is_bool_dtype(target_series):
            issues.append(
                Issue(
                    issue_code="ERR_BOOLEAN_TARGET",
                    severity=IssueSeverity.ERROR,
                    message="Target must be numeric, not boolean.",
                    field_name=request.target_column,
                )
            )
            return self._build_result(
                request,
                ResultStatus.ERROR,
                (),
                issues,
                parent_record_ids,
                source_uri,
                data.shape[0],
                0,
            )

        ts_series = data[request.timestamp_column]
        if not pd.api.types.is_datetime64_any_dtype(ts_series):
            issues.append(
                Issue(
                    issue_code="ERR_INVALID_TIMESTAMP_DTYPE",
                    severity=IssueSeverity.ERROR,
                    message="Timestamp column must be datetime.",
                    field_name=request.timestamp_column,
                )
            )
            return self._build_result(
                request,
                ResultStatus.ERROR,
                (),
                issues,
                parent_record_ids,
                source_uri,
                data.shape[0],
                0,
            )

        if ts_series.dt.tz is None:
            issues.append(
                Issue(
                    issue_code="ERR_NAIVE_TIMESTAMP",
                    severity=IssueSeverity.ERROR,
                    message="Timestamp must be timezone-aware.",
                    field_name=request.timestamp_column,
                )
            )
            return self._build_result(
                request,
                ResultStatus.ERROR,
                (),
                issues,
                parent_record_ids,
                source_uri,
                data.shape[0],
                0,
            )

        if str(ts_series.dt.tz) != "UTC" and str(ts_series.dt.tz) != "datetime.timezone.utc":
            issues.append(
                Issue(
                    issue_code="ERR_NON_UTC_TIMESTAMP",
                    severity=IssueSeverity.ERROR,
                    message="Timestamp must be UTC.",
                    field_name=request.timestamp_column,
                )
            )
            return self._build_result(
                request,
                ResultStatus.ERROR,
                (),
                issues,
                parent_record_ids,
                source_uri,
                data.shape[0],
                0,
            )

        # Ordering and Duplicates Check
        if not ts_series.is_monotonic_increasing:
            issues.append(
                Issue(
                    issue_code="ERR_TIMESTAMP_UNORDERED",
                    severity=IssueSeverity.ERROR,
                    message="Timestamps must be ordered chronologically.",
                    field_name=request.timestamp_column,
                )
            )
            return self._build_result(
                request,
                ResultStatus.ERROR,
                (),
                issues,
                parent_record_ids,
                source_uri,
                data.shape[0],
                0,
            )

        if ts_series.duplicated().any():
            issues.append(
                Issue(
                    issue_code="ERR_DUPLICATE_TIMESTAMPS",
                    severity=IssueSeverity.ERROR,
                    message="Duplicate timestamps are not allowed.",
                    field_name=request.timestamp_column,
                )
            )
            return self._build_result(
                request,
                ResultStatus.ERROR,
                (),
                issues,
                parent_record_ids,
                source_uri,
                data.shape[0],
                0,
            )

        total_count = data.shape[0]
        if total_count == 0:
            issues.append(
                Issue(
                    issue_code="ERR_NO_DATA",
                    severity=IssueSeverity.WARNING,
                    message="No observations available.",
                )
            )
            return self._build_result(
                request,
                ResultStatus.NOT_AVAILABLE,
                (),
                issues,
                parent_record_ids,
                source_uri,
                total_count,
                0,
            )

        # 2. Rolling Evaluation
        anomalies: list[AnomalyRecord] = []
        evaluable_count = 0
        unevaluable_count = 0

        prov = ProvenanceRecord(
            record_id=f"prov_{request.analysis_id}_{datetime.datetime.now(datetime.UTC).timestamp()}",
            activity_name="time_series_anomaly_detection",
            parent_record_ids=parent_record_ids or [],
            source_uri=source_uri or "unknown",
            epistemic_level=EpistemicLevel.CALCULATED,
        )

        anomaly_id_counter = 0

        # Create numpy arrays for fast evaluation
        target_values = target_series.to_numpy(dtype=float)
        timestamps = ts_series.dt.to_pydatetime()

        for i in range(total_count):
            obs = target_values[i]

            if np.isnan(obs):
                unevaluable_count += 1
                continue

            # Trailing window excluding current observation: [max(0, i - window_size), i - 1]
            start_idx = max(0, i - request.window_size)
            end_idx = i

            if start_idx == end_idx:
                unevaluable_count += 1
                continue

            window_data = target_values[start_idx:end_idx]
            valid_window_data = window_data[~np.isnan(window_data)]

            if len(valid_window_data) < request.minimum_observations:
                unevaluable_count += 1
                continue

            mean_val = float(np.mean(valid_window_data))
            std_val = float(np.std(valid_window_data, ddof=request.ddof))

            if std_val == 0.0:
                unevaluable_count += 1
                continue

            evaluable_count += 1
            z_score = (obs - mean_val) / std_val

            if abs(z_score) > request.z_threshold:
                anomaly_id_counter += 1
                original_index = data.index[i]
                ev_ref = EvidenceReference(
                    source_uri=source_uri,
                    parent_record_ids=parent_record_ids,
                    row_index=original_index,  # type: ignore
                )
                record = AnomalyRecord(
                    anomaly_id=f"{request.analysis_id}_{anomaly_id_counter}",
                    analysis_id=request.analysis_id,
                    target_column=request.target_column,
                    classification=request.classification,
                    severity=request.severity,
                    detection_method=DetectionMethod.TIME_SERIES,
                    timestamp=timestamps[i],
                    observed_value=float(obs),
                    deviation_value=float(z_score),
                    deviation_metric="rolling_z_score",
                    reference_mean=mean_val,
                    reference_standard_deviation=std_val,
                    evidence_reference=ev_ref,
                    provenance=prov,
                )
                anomalies.append(record)

        if evaluable_count == 0:
            status = ResultStatus.NOT_AVAILABLE
        elif unevaluable_count > 0:
            status = ResultStatus.PARTIAL
        else:
            status = ResultStatus.SUCCESS

        return self._build_result(
            request,
            status,
            tuple(anomalies),
            issues,
            parent_record_ids,
            source_uri,
            total_count,
            evaluable_count,
        )

    def _build_result(
        self,
        request: TimeSeriesAnomalyDetectionRequest,
        status: ResultStatus,
        anomalies: tuple[AnomalyRecord, ...],
        issues: list[Issue],
        parent_record_ids: list[str] | None,
        source_uri: str | None,
        total_count: int,
        evaluable_count: int,
    ) -> AnomalyResult:

        prov = ProvenanceRecord(
            record_id=f"prov_res_{request.analysis_id}_{datetime.datetime.now(datetime.UTC).timestamp()}",
            activity_name="time_series_anomaly_result",
            parent_record_ids=parent_record_ids or [],
            source_uri=source_uri or "unknown",
            epistemic_level=EpistemicLevel.CALCULATED,
        )

        metadata = {
            "total_observations": total_count,
            "evaluable_observations": evaluable_count,
            "anomalies_detected": len(anomalies),
            "window_size": request.window_size,
            "z_threshold": request.z_threshold,
            "minimum_observations": request.minimum_observations,
            "ddof": request.ddof,
        }

        return AnomalyResult(
            analysis_id=request.analysis_id,
            target_column=request.target_column,
            detection_method=DetectionMethod.TIME_SERIES,
            status=status,
            anomalies=anomalies,
            issues=issues,
            provenance=prov,
            metadata=metadata,
        )
