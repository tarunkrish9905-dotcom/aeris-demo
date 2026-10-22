import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import BasePipelineResult, Issue, ResultStatus


class AnomalyClassification(StrEnum):
    """Classification of the anomaly type."""

    POINT_ANOMALY = "POINT_ANOMALY"
    CONTEXTUAL_ANOMALY = "CONTEXTUAL_ANOMALY"
    COLLECTIVE_ANOMALY = "COLLECTIVE_ANOMALY"


class AnomalySeverity(StrEnum):
    """Analytical significance of the anomaly. Does NOT imply safety risk or failure."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DetectionMethod(StrEnum):
    """The methodology used to detect the anomaly."""

    RULE_BASED = "RULE_BASED"
    STATISTICAL = "STATISTICAL"
    TIME_SERIES = "TIME_SERIES"
    MODEL_BASED = "MODEL_BASED"
    OTHER = "OTHER"


class EvidenceReference(BaseModel):
    """Points to the source or evidence used to identify the anomaly."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    source_uri: str | None = Field(default=None, description="URI of the source data.")
    parent_record_ids: list[str] | None = Field(
        default=None, description="Lineage pointing to upstream execution records."
    )
    row_index: int | None = Field(default=None, description="Specific row index if applicable.")


class AnomalyRecord(BaseModel):
    """
    Represents a single detected anomaly or deviation.
    Does NOT represent a fault, root cause, hypothesis, or recommendation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    anomaly_id: str = Field(..., min_length=1)
    analysis_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)

    classification: AnomalyClassification
    severity: AnomalySeverity
    detection_method: DetectionMethod

    # Temporal location (can be point or interval)
    timestamp: datetime.datetime | None = None
    start_timestamp: datetime.datetime | None = None
    end_timestamp: datetime.datetime | None = None

    # Values
    observed_value: float | str | None = None
    expected_value: float | str | None = None
    deviation_value: float | None = None
    deviation_metric: str | None = None
    score: float | None = Field(
        default=None, description="Analytical score produced by the detection method."
    )

    # Statistical Context
    reference_mean: float | None = None
    reference_standard_deviation: float | None = None

    explanation: str | None = None
    evidence_reference: EvidenceReference | None = None
    source_reference: str | None = None

    provenance: ProvenanceRecord
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.CALCULATED,
        description="Must be CALCULATED. Anomalies are analytical outputs, not inferences.",
    )

    @field_validator("epistemic_level")
    @classmethod
    def validate_epistemic_level(cls, v: EpistemicLevel) -> EpistemicLevel:
        if v not in (EpistemicLevel.CALCULATED, EpistemicLevel.OBSERVED):
            raise ValueError(
                "Anomaly records must be CALCULATED (or OBSERVED if raw), "
                "not INFERRED, HYPOTHESIS, or RECOMMENDATION."
            )
        return v

    @field_validator("timestamp", "start_timestamp", "end_timestamp")
    @classmethod
    def validate_utc_timestamps(cls, v: datetime.datetime | None) -> datetime.datetime | None:
        if v is not None and (v.tzinfo is None or v.tzinfo.utcoffset(v) is None):
            raise ValueError("Timestamps must be timezone-aware UTC.")
        return v


class AnomalyResult(BasePipelineResult):
    """
    Result model containing a collection of detected anomalies.
    Zero anomalies with ResultStatus.SUCCESS is perfectly valid.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    detection_method: DetectionMethod
    anomalies: tuple[AnomalyRecord, ...] = Field(default_factory=tuple)

    # Allow overriding properties from BasePipelineResult without breaking immutability checks
    status: ResultStatus
    issues: list[Issue] = Field(default_factory=list)
    provenance: ProvenanceRecord | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineAnomalyDetectionResult(BasePipelineResult):
    """Aggregate result containing specific detector outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector_results: tuple[AnomalyResult, ...] = Field(default_factory=tuple)

    # Base properties
    status: ResultStatus
    issues: list[Issue] = Field(default_factory=list)
    provenance: ProvenanceRecord | None = None
