"""
AERIS Core Result and Outcome Domain Models.

Defines standardized containers for pipeline stage outcomes (IngestionResult,
ValidationResult, QualityResult), status flags, issue reporting, and links
to epistemic provenance.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from aeris.core.anomaly.models import PipelineAnomalyDetectionResult
    from aeris.core.contributors.pipeline import PipelineContributorAnalysisResult
    from aeris.core.expected_observed.models import PipelineExpectedObservedResult
    from aeris.core.recommendation.generation_models import RecommendationGenerationResult
    from aeris.core.vehicle_evaluation.models import VehicleEvaluationResult


from aeris.core.provenance.tracker import (
    EpistemicIntegrityError,
    EpistemicLevel,
    ProvenanceRecord,
)

# ---------------------------------------------------------------------------
# Status and Severity Enumerations
# ---------------------------------------------------------------------------


class ResultStatus(StrEnum):
    """Execution status for pipeline stages and operations."""

    SUCCESS = "SUCCESS"
    """Operation completed cleanly with full validity."""

    WARNING = "WARNING"
    """Operation completed but non-critical advisories or anomalies were flagged."""

    ERROR = "ERROR"
    """Operation failed or encountered unrecoverable data defects."""

    NOT_AVAILABLE = "NOT_AVAILABLE"
    """Result could not be calculated due to missing prerequisite inputs or scope limits."""

    PARTIAL = "PARTIAL"
    """Operation degraded gracefully, delivering partial results from available data."""


class IssueSeverity(StrEnum):
    """Severity classification for issues identified during processing."""

    INFO = "INFO"
    """Informational note or operational context observation."""

    WARNING = "WARNING"
    """Degraded data quality, minor gap, or non-fatal anomaly."""

    ERROR = "ERROR"
    """Defect preventing normal analysis or violating schema/consistency rules."""

    CRITICAL = "CRITICAL"
    """Fatal defect rendering the data unparseable or completely invalid."""


# ---------------------------------------------------------------------------
# Issue Model
# ---------------------------------------------------------------------------


class Issue(BaseModel):
    """Structured representation of a data defect, anomaly, or observation.

    Used across ingestion, validation, and quality assessment to communicate
    issues with location and traceability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue_code: str = Field(
        description="Stable machine-readable identifier (e.g., 'ERR_TIMESTAMP_INVALID').",
        min_length=1,
    )
    severity: IssueSeverity = Field(
        description="Severity classification of the issue.",
    )
    message: str = Field(
        description="Human-readable explanation of the issue.",
        min_length=1,
    )
    field_name: str | None = Field(
        default=None,
        description="Name of the affected telemetry column or parameter, if applicable.",
    )
    location: str | int | None = Field(
        default=None,
        description="Row index, timestamp, or byte offset where the issue occurred.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional technical context or parameter values.",
    )


# ---------------------------------------------------------------------------
# Base and Specialized Result Containers
# ---------------------------------------------------------------------------


class BasePipelineResult(BaseModel):
    """Foundational result container for all AERIS core pipeline operations.

    Encapsulates outcome status, detected issues, execution metadata, and
    epistemic provenance.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    status: ResultStatus = Field(
        description="Overall execution status of the pipeline operation.",
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="List of issues, warnings, or notes identified during processing.",
    )
    provenance: ProvenanceRecord | None = Field(
        default=None,
        description="Lineage and epistemic metadata tracking how this result was produced.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary operational metadata or execution timing.",
    )

    @property
    def is_successful(self) -> bool:
        """Return True if the status is SUCCESS and no ERROR/CRITICAL issues exist."""
        return self.status == ResultStatus.SUCCESS and not self.has_errors

    @property
    def has_errors(self) -> bool:
        """Return True if status is ERROR or any issue has ERROR/CRITICAL severity."""
        if self.status == ResultStatus.ERROR:
            return True
        return any(i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL) for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Return True if status is WARNING or any issue has WARNING severity."""
        if self.status == ResultStatus.WARNING:
            return True
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    @property
    def error_count(self) -> int:
        """Return count of issues with ERROR or CRITICAL severity."""
        return sum(
            1 for i in self.issues if i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        """Return count of issues with WARNING severity."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)


class IngestionResult(BasePipelineResult):
    """Result container for data ingestion operations (CSV, telemetry files)."""

    data: Any = Field(
        default=None,
        description="Ingested data structure (e.g. pandas DataFrame or raw records).",
    )
    row_count: int = Field(
        default=0,
        ge=0,
        description="Total number of data rows ingested.",
    )
    column_names: list[str] = Field(
        default_factory=list,
        description="Detected column or parameter names.",
    )
    source_path: str | None = Field(
        default=None,
        description="File path or URI of the ingested source.",
    )


class ValidationResult(BasePipelineResult):
    """Result container for schema and data validation operations."""

    is_valid: bool = Field(
        default=False,
        description="Boolean summary indicating whether the dataset passed validation.",
    )
    validated_row_count: int = Field(
        default=0,
        ge=0,
        description="Number of rows evaluated during validation.",
    )
    violation_count: int = Field(
        default=0,
        ge=0,
        description="Total number of validation rule violations identified.",
    )
    validated_columns: list[str] = Field(
        default_factory=list,
        description="Columns evaluated against schema requirements.",
    )


class ParameterValidationResult(BasePipelineResult):
    """
    Structured outcome of the parameter semantic validation stage.

    Evaluates whether raw column data semantically matches the definitions
    in the ParameterRegistry, without applying corrections.
    """

    validated_parameters: list[str] = Field(
        default_factory=list,
        description="List of requested identifiers that were successfully validated.",
    )
    failed_parameters: list[str] = Field(
        default_factory=list,
        description="List of requested identifiers that failed semantic validation.",
    )


class QualityResult(BasePipelineResult):
    """Result container for data quality assessment operations."""

    quality_score: float | None = Field(
        default=None,
        description="Aggregate quality score between 0.0 (unusable) and 1.0 (perfect).",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Specific quality metrics (e.g., completeness, duplicate_rate).",
    )
    summary: str = Field(
        default="",
        description="High-level engineering summary of the quality assessment.",
    )

    @field_validator("quality_score")
    @classmethod
    def validate_quality_score(cls, v: float | None) -> float | None:
        """Ensure quality_score is bounded within [0.0, 1.0] when present."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"quality_score must be between 0.0 and 1.0, got {v}")
        return v


class TransformationType(StrEnum):
    """Types of explicit data normalizations."""

    UNIT_CONVERSION = "UNIT_CONVERSION"
    COLUMN_RENAME = "COLUMN_RENAME"
    TYPE_CONVERSION = "TYPE_CONVERSION"
    TIMESTAMP_STANDARDIZATION = "TIMESTAMP_STANDARDIZATION"


class TransformationRecord(BaseModel):
    """Explicit representation of a normalization transformation applied to data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    transformation_id: str = Field(description="Unique identifier for the applied transformation.")
    transformation_type: TransformationType = Field(
        description="The category of the transformation."
    )
    source_column: str = Field(description="The name of the column before transformation.")
    destination_column: str = Field(description="The name of the column after transformation.")
    source_unit: str | None = Field(default=None, description="The original unit, if applicable.")
    destination_unit: str | None = Field(
        default=None, description="The target unit, if applicable."
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Metadata or parameters used in the transformation."
    )
    is_lossless: bool | None = Field(
        default=None, description="Whether the transformation was fully lossless."
    )


class NormalizationResult(BasePipelineResult):
    """Result container for data normalization operations."""

    data: Any = Field(
        default=None,
        description="Normalized data structure (e.g. pandas DataFrame).",
    )
    transformations: list[TransformationRecord] = Field(
        default_factory=list,
        description="Explicit records of all applied transformations.",
    )
    input_row_count: int = Field(default=0, ge=0)
    output_row_count: int = Field(default=0, ge=0)
    input_columns: list[str] = Field(default_factory=list)
    output_columns: list[str] = Field(default_factory=list)


class DataPipelineResult(BasePipelineResult):
    """Result container for end-to-end data processing pipelines."""

    ingestion: IngestionResult = Field(
        description="Outcome of the data ingestion stage.",
    )
    validation: ValidationResult | None = Field(
        default=None,
        description="Outcome of the schema and data validation stage, if executed.",
    )
    quality: QualityResult | None = Field(
        default=None,
        description="Outcome of the data quality assessment stage, if executed.",
    )
    normalization: NormalizationResult | None = Field(
        default=None,
        description="Outcome of the normalization stage, if executed.",
    )
    parameter_validation: ParameterValidationResult | None = Field(
        default=None,
        description="Outcome of the parameter semantic validation stage, if executed.",
    )
    association: Any = Field(
        default=None,
        description="Outcome of the Test Point Association stage, if executed.",
    )
    engineering: Any = Field(
        default=None,
        description="Outcome of the Engineering Calculation stage, if executed.",
    )
    statistics: Any = Field(
        default=None,
        description="Outcome of the Statistical Analysis stage, if executed.",
    )
    time_series: Any = Field(
        default=None,
        description="Outcome of the Time-Series Statistical Analysis stage, if executed.",
    )
    expected_observed: PipelineExpectedObservedResult | None = Field(
        default=None,
        description="Outcome of the Expected-vs-Observed stage, if executed.",
    )
    anomaly: PipelineAnomalyDetectionResult | None = Field(
        default=None,
        description="Outcome of the Anomaly Detection stage, if executed.",
    )
    evidence: Any = Field(
        default=None,
        description="Outcome of the explicit Evidence Construction stage, if executed.",
    )
    contributor_analysis: PipelineContributorAnalysisResult | None = Field(
        default=None,
        description="Outcome of the explicit Contributor Analysis stage, if executed.",
    )
    recommendation: RecommendationGenerationResult | None = Field(
        default=None,
        description="Outcome of the explicit Recommendation Generation stage, if executed.",
    )
    vehicle_evaluation: VehicleEvaluationResult | None = Field(
        default=None,
        description="Outcome of the explicit Vehicle Engineering Evaluation stage, if executed.",
    )

    @property
    def recommendation_result(self) -> RecommendationGenerationResult | None:
        """Alias for recommendation result container."""
        return self.recommendation

    @property
    def vehicle_evaluation_result(self) -> VehicleEvaluationResult | None:
        """Alias for vehicle evaluation result container."""
        return self.vehicle_evaluation


# Re-export epistemic types for convenient single-source access
__all__ = [
    "BasePipelineResult",
    "DataPipelineResult",
    "EpistemicIntegrityError",
    "EpistemicLevel",
    "IngestionResult",
    "Issue",
    "IssueSeverity",
    "NormalizationResult",
    "ParameterValidationResult",
    "ProvenanceRecord",
    "QualityResult",
    "ResultStatus",
    "TransformationRecord",
    "TransformationType",
    "ValidationResult",
]
