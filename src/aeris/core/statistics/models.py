from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import BasePipelineResult


class MetricType(StrEnum):
    """Explicitly supported statistical metric types."""

    COUNT = "COUNT"
    MISSING_COUNT = "MISSING_COUNT"
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    MINIMUM = "MINIMUM"
    MAXIMUM = "MAXIMUM"
    VARIANCE = "VARIANCE"
    STANDARD_DEVIATION = "STANDARD_DEVIATION"
    PERCENTILE = "PERCENTILE"


class StatisticalMetric(BaseModel):
    """Explicit representation of a calculated statistical metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_id: str = Field(..., min_length=1, description="Unique identifier for the metric.")
    metric_type: MetricType = Field(..., description="The type of the metric.")
    value: float | None = Field(default=None, description="The calculated numerical value, if any.")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Metadata or parameters (e.g., percentile=95)."
    )


class StatisticalAnalysisRequest(BaseModel):
    """
    Explicit request to perform statistical analysis.
    Does not automatically execute or infer targets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str = Field(
        ..., min_length=1, description="Unique identifier for this specific analysis request."
    )
    target_column: str = Field(
        ..., min_length=1, description="Explicit column or parameter to analyze."
    )
    source_identifier: str | None = Field(
        default=None, description="Optional identifier of the source data."
    )
    scope_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional scoping boundaries or grouping."
    )
    requested_metrics: list[MetricType] = Field(
        default_factory=list, description="Specific metrics explicitly requested for calculation."
    )
    requested_percentiles: list[float] = Field(
        default_factory=list,
        description="Explicit percentile values if PERCENTILE metric is requested.",
    )


class StatisticalAnalysisResult(BasePipelineResult):
    """
    Generic result container for statistical analysis.
    Inherits status, issues, and provenance tracking.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    result_id: str = Field(..., min_length=1)
    analysis_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    metrics: list[StatisticalMetric] = Field(default_factory=list)
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.CALCULATED,
        description="Statistical results are strictly calculated.",
    )

    @field_validator("epistemic_level")
    @classmethod
    def strict_epistemic_level(cls, v: EpistemicLevel) -> EpistemicLevel:
        """Ensure statistical results are always calculations, never observed facts."""
        if v.name in ("OBSERVED", "HISTORICAL", "INFERRED", "HYPOTHESIS", "RECOMMENDATION"):
            raise ValueError(f"Statistical results must not produce {v.name} epistemic level.")
        return v
