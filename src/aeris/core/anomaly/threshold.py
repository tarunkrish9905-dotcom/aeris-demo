import datetime
from enum import StrEnum

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


class ThresholdOperator(StrEnum):
    GREATER_THAN = "GREATER_THAN"
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL"
    LESS_THAN = "LESS_THAN"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"
    BETWEEN = "BETWEEN"


class ThresholdRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    operator: ThresholdOperator

    threshold_value: float | None = None
    threshold_lower: float | None = None
    threshold_upper: float | None = None

    unit: str | None = None
    severity: AnomalySeverity = Field(default=AnomalySeverity.INFO)
    description: str | None = None
    analysis_id: str | None = None
    classification: AnomalyClassification = Field(default=AnomalyClassification.POINT_ANOMALY)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ThresholdRule":
        if self.operator == ThresholdOperator.BETWEEN:
            if self.threshold_lower is None or self.threshold_upper is None:
                raise ValueError("BETWEEN operator requires threshold_lower and threshold_upper.")
            if self.threshold_lower >= self.threshold_upper:
                raise ValueError("BETWEEN operator requires lower < upper.")
            if self.threshold_value is not None:
                raise ValueError("BETWEEN operator should not use threshold_value.")
        else:
            if self.threshold_value is None:
                raise ValueError(f"{self.operator} operator requires threshold_value.")
            if self.threshold_lower is not None or self.threshold_upper is not None:
                raise ValueError(f"{self.operator} operator does not use bounds.")
        return self


class ThresholdDetectionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    rule: ThresholdRule
    timestamp_column: str | None = None

    classification: AnomalyClassification = Field(default=AnomalyClassification.POINT_ANOMALY)
    severity: AnomalySeverity = Field(default=AnomalySeverity.INFO)
    description: str | None = None

    @model_validator(mode="after")
    def match_targets(self) -> "ThresholdDetectionRequest":
        if self.target_column != self.rule.target_column:
            raise ValueError("Request target_column must match rule target_column.")
        return self


class ThresholdAnomalyDetectionEngine:
    """Explicit threshold-based anomaly detection engine."""

    def execute(
        self,
        request: ThresholdDetectionRequest,
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

        # 3. Execution
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
            )

        if evaluable_count == 0:
            issues.append(
                Issue(
                    issue_code="ERR_NO_EVALUABLE_DATA",
                    severity=IssueSeverity.WARNING,
                    message="All target values are missing.",
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

        # Evaluate rules
        anomalies: list[AnomalyRecord] = []
        op = request.rule.operator
        val = request.rule.threshold_value
        lower = request.rule.threshold_lower
        upper = request.rule.threshold_upper

        # Common provenance for all anomalies
        prov = ProvenanceRecord(
            record_id=f"prov_{request.analysis_id}_{datetime.datetime.now(datetime.UTC).timestamp()}",
            activity_name="threshold_anomaly_detection",
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

            is_anomaly = False
            dev = None

            if op == ThresholdOperator.GREATER_THAN:
                if obs > val:  # type: ignore
                    is_anomaly = True
                    dev = obs - val  # type: ignore
            elif op == ThresholdOperator.GREATER_THAN_OR_EQUAL:
                if obs >= val:  # type: ignore
                    is_anomaly = True
                    dev = obs - val  # type: ignore
            elif op == ThresholdOperator.LESS_THAN:
                if obs < val:  # type: ignore
                    is_anomaly = True
                    dev = val - obs  # type: ignore
            elif op == ThresholdOperator.LESS_THAN_OR_EQUAL:
                if obs <= val:  # type: ignore
                    is_anomaly = True
                    dev = val - obs  # type: ignore
            elif op == ThresholdOperator.EQUAL:
                if obs == val:  # type: ignore
                    is_anomaly = True
                    dev = 0.0
            elif op == ThresholdOperator.NOT_EQUAL:
                if obs != val:  # type: ignore
                    is_anomaly = True
                    dev = abs(obs - val)  # type: ignore
            elif op == ThresholdOperator.BETWEEN:
                if not (lower <= obs <= upper):  # type: ignore
                    is_anomaly = True
                    # distance to nearest bound
                    if obs < lower:  # type: ignore
                        dev = lower - obs  # type: ignore
                    else:
                        dev = obs - upper  # type: ignore

            if is_anomaly:
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
                    severity=request.rule.severity,
                    detection_method=DetectionMethod.RULE_BASED,
                    timestamp=ts,
                    observed_value=obs,
                    deviation_value=dev,
                    deviation_metric="absolute",
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
        )

    def _build_result(
        self,
        request: ThresholdDetectionRequest,
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
            activity_name="threshold_anomaly_result",
            parent_record_ids=parent_record_ids or [],
            source_uri=source_uri or "unknown",
            epistemic_level=EpistemicLevel.CALCULATED,
        )

        return AnomalyResult(
            analysis_id=request.analysis_id,
            target_column=request.target_column,
            detection_method=DetectionMethod.RULE_BASED,
            status=status,
            anomalies=anomalies,
            issues=issues,
            provenance=prov,
            metadata={
                "total_observations": total_count,
                "evaluable_observations": evaluable_count,
                "anomalies_detected": len(anomalies),
            },
        )
