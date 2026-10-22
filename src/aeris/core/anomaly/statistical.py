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


class StatisticalAnomalyDetectionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    z_threshold: float

    timestamp_column: str | None = None
    severity: AnomalySeverity = Field(default=AnomalySeverity.INFO)
    classification: AnomalyClassification = Field(default=AnomalyClassification.POINT_ANOMALY)
    description: str | None = None
    minimum_observations: int = Field(default=2, ge=2)
    ddof: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_z_threshold(self) -> "StatisticalAnomalyDetectionRequest":
        if not np.isfinite(self.z_threshold):
            raise ValueError("z_threshold must be finite.")
        if self.z_threshold <= 0:
            raise ValueError("z_threshold must be strictly greater than zero.")
        return self


class StatisticalAnomalyDetectionEngine:
    """Explicit statistical anomaly detection engine using Z-scores."""

    def execute(
        self,
        request: StatisticalAnomalyDetectionRequest,
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
                None,
                None,
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
                None,
                None,
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
                None,
                None,
            )

        # 2. Timestamp Validation
        if request.timestamp_column is not None:
            if request.timestamp_column not in data.columns:
                issues.append(
                    Issue(
                        issue_code="ERR_MISSING_TIMESTAMP",
                        severity=IssueSeverity.ERROR,
                        message="Timestamp column missing.",
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
                    None,
                    None,
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
                    None,
                    None,
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
                    None,
                    None,
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
                    None,
                    None,
                )

        # 3. Reference Population
        valid_mask = target_series.notna()
        evaluable_count = int(valid_mask.sum())
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
                None,
                None,
            )

        if evaluable_count < request.minimum_observations:
            issues.append(
                Issue(
                    issue_code="ERR_INSUFFICIENT_OBSERVATIONS",
                    severity=IssueSeverity.ERROR,
                    message=f"Need {request.minimum_observations} observations, found {evaluable_count}.",
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
                evaluable_count,
                None,
                None,
            )

        valid_data = target_series[valid_mask]
        mean_val = float(valid_data.mean())
        std_val = float(valid_data.std(ddof=request.ddof))

        if std_val == 0.0:
            issues.append(
                Issue(
                    issue_code="ERR_ZERO_STANDARD_DEVIATION",
                    severity=IssueSeverity.ERROR,
                    message="Standard deviation is zero, cannot compute z-scores.",
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
                evaluable_count,
                mean_val,
                std_val,
            )

        # 4. Z-score Evaluation
        anomalies: list[AnomalyRecord] = []

        # Common provenance for all anomalies
        prov = ProvenanceRecord(
            record_id=f"prov_{request.analysis_id}_{datetime.datetime.now(datetime.UTC).timestamp()}",
            activity_name="statistical_anomaly_detection",
            parent_record_ids=parent_record_ids or [],
            source_uri=source_uri or "unknown",
            epistemic_level=EpistemicLevel.CALCULATED,
        )

        anomaly_id_counter = 0

        for idx, row in data[valid_mask].iterrows():
            obs = float(row[request.target_column])
            ts = row[request.timestamp_column] if request.timestamp_column else None
            if ts is not None:
                ts = ts.to_pydatetime()

            z_score = (obs - mean_val) / std_val

            if abs(z_score) > request.z_threshold:
                anomaly_id_counter += 1
                ev_ref = EvidenceReference(
                    source_uri=source_uri,
                    parent_record_ids=parent_record_ids,
                    row_index=idx,  # type: ignore
                )
                record = AnomalyRecord(
                    anomaly_id=f"{request.analysis_id}_{anomaly_id_counter}",
                    analysis_id=request.analysis_id,
                    target_column=request.target_column,
                    classification=request.classification,
                    severity=request.severity,
                    detection_method=DetectionMethod.STATISTICAL,
                    timestamp=ts,
                    observed_value=obs,
                    deviation_value=z_score,
                    deviation_metric="z_score",
                    reference_mean=mean_val,
                    reference_standard_deviation=std_val,
                    evidence_reference=ev_ref,
                    provenance=prov,
                )
                anomalies.append(record)

        status = ResultStatus.SUCCESS
        if evaluable_count < total_count:
            status = ResultStatus.PARTIAL
            issues.append(
                Issue(
                    issue_code="WARN_MISSING_TARGET_VALUES",
                    severity=IssueSeverity.WARNING,
                    message=f"{total_count - evaluable_count} missing values skipped.",
                )
            )

        return self._build_result(
            request,
            status,
            tuple(anomalies),
            issues,
            parent_record_ids,
            source_uri,
            total_count,
            evaluable_count,
            mean_val,
            std_val,
        )

    def _build_result(
        self,
        request: StatisticalAnomalyDetectionRequest,
        status: ResultStatus,
        anomalies: tuple[AnomalyRecord, ...],
        issues: list[Issue],
        parent_record_ids: list[str] | None,
        source_uri: str | None,
        total_count: int,
        evaluable_count: int,
        mean_val: float | None,
        std_val: float | None,
    ) -> AnomalyResult:

        prov = ProvenanceRecord(
            record_id=f"prov_res_{request.analysis_id}_{datetime.datetime.now(datetime.UTC).timestamp()}",
            activity_name="statistical_anomaly_result",
            parent_record_ids=parent_record_ids or [],
            source_uri=source_uri or "unknown",
            epistemic_level=EpistemicLevel.CALCULATED,
        )

        metadata = {
            "total_observations": total_count,
            "evaluable_observations": evaluable_count,
            "anomalies_detected": len(anomalies),
            "z_threshold": request.z_threshold,
            "ddof": request.ddof,
        }
        if mean_val is not None:
            metadata["reference_mean"] = mean_val
        if std_val is not None:
            metadata["reference_standard_deviation"] = std_val

        return AnomalyResult(
            analysis_id=request.analysis_id,
            target_column=request.target_column,
            detection_method=DetectionMethod.STATISTICAL,
            status=status,
            anomalies=anomalies,
            issues=issues,
            provenance=prov,
            metadata=metadata,
        )
