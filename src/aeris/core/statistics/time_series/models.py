import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import BasePipelineResult
from aeris.core.statistics.models import MetricType


class TimeSeriesMetric(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    metric_id: str = Field(..., min_length=1)
    metric_type: MetricType = Field(...)
    values: tuple[float | None, ...] = Field(default_factory=tuple)
    parameters: dict[str, Any] = Field(default_factory=dict)


class TemporalScopeType(StrEnum):
    ENTIRE_DATASET = "ENTIRE_DATASET"
    EXPLICIT_INTERVAL = "EXPLICIT_INTERVAL"


class TemporalScope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scope_type: TemporalScopeType = Field(description="The type of temporal boundary.")
    start_timestamp: datetime.datetime | None = Field(
        default=None, description="Start timestamp of the interval (inclusive)."
    )
    end_timestamp: datetime.datetime | None = Field(
        default=None, description="End timestamp of the interval (inclusive)."
    )

    @field_validator("start_timestamp", "end_timestamp")
    @classmethod
    def validate_utc_timestamp(cls, v: datetime.datetime | None) -> datetime.datetime | None:
        if v is not None:
            if v.tzinfo is None:
                raise ValueError("Timestamps must be timezone-aware.")
            if v.tzinfo != datetime.UTC:
                raise ValueError(f"Timestamps must be strictly UTC, got {v.tzinfo}.")
        return v

    @model_validator(mode="after")
    def validate_interval_semantics(self) -> "TemporalScope":
        if self.scope_type == TemporalScopeType.EXPLICIT_INTERVAL:
            if self.start_timestamp is None or self.end_timestamp is None:
                raise ValueError("Explicit intervals require both start and end timestamps.")
            if self.start_timestamp > self.end_timestamp:
                raise ValueError("Interval start timestamp cannot be strictly after end timestamp.")
        return self


class WindowType(StrEnum):
    DURATION = "DURATION"
    OBSERVATION_COUNT = "OBSERVATION_COUNT"


class WindowConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    window_type: WindowType = Field(description="The type of window sizing.")
    window_size_seconds: float | None = Field(
        default=None, description="Size of the window in seconds (for DURATION)."
    )
    window_size_observations: int | None = Field(
        default=None, description="Size of the window in row count (for OBSERVATION_COUNT)."
    )
    min_observations: int = Field(
        default=1,
        description="Minimum number of valid observations required within the window.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "WindowConfiguration":
        if self.window_type == WindowType.DURATION and (
            self.window_size_seconds is None or self.window_size_seconds <= 0
        ):
            raise ValueError("Duration windows require a positive window_size_seconds.")
        if self.window_type == WindowType.OBSERVATION_COUNT and (
            self.window_size_observations is None or self.window_size_observations <= 0
        ):
            raise ValueError(
                "Observation count windows require a positive window_size_observations."
            )
        return self


class SamplingIrregularity(StrEnum):
    NOMINAL = "NOMINAL"
    IRREGULAR = "IRREGULAR"
    DUPLICATE_TIMESTAMPS = "DUPLICATE_TIMESTAMPS"
    GAPS_PRESENT = "GAPS_PRESENT"
    INSUFFICIENT_TIMESTAMPS = "INSUFFICIENT_TIMESTAMPS"


class SamplingMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    is_regular: bool = Field(
        description="True if the sampling follows a nominal, regular interval."
    )
    irregularities: list[SamplingIrregularity] = Field(
        default_factory=list,
        description="Explicit list of identified temporal data characteristics.",
    )
    duplicate_count: int = Field(
        default=0, ge=0, description="Number of duplicate timestamps identified."
    )
    gap_count: int = Field(
        default=0,
        ge=0,
        description="Number of identified temporal gaps exceeding nominal bounds.",
    )
    nominal_interval_seconds: float | None = Field(
        default=None,
        description="The expected or detected nominal sampling interval in seconds.",
    )


class TimeSeriesAnalysisRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    timestamp_column: str = Field(..., min_length=1)

    temporal_scope: TemporalScope = Field(
        default_factory=lambda: TemporalScope(scope_type=TemporalScopeType.ENTIRE_DATASET)
    )
    window_configuration: WindowConfiguration | None = Field(
        default=None, description="Optional window configuration for rolling analysis."
    )
    requested_metrics: list[MetricType] = Field(
        default_factory=list,
        description="Specific metrics explicitly requested for calculation.",
    )
    requested_percentiles: list[float] = Field(
        default_factory=list,
        description="Explicit percentile values if PERCENTILE metric is requested.",
    )


class TimeSeriesAnalysisResult(BasePipelineResult):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    result_id: str = Field(..., min_length=1)
    analysis_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    timestamp_column: str = Field(..., min_length=1)

    temporal_scope: TemporalScope = Field(description="The actual scope applied during analysis.")
    window_configuration: WindowConfiguration | None = Field(default=None)
    sampling_metadata: SamplingMetadata | None = Field(
        default=None, description="Descriptive metadata about the temporal sampling observed."
    )

    metrics: list[TimeSeriesMetric] = Field(default_factory=list)
    data: Any = Field(default=None, description="Optional dataframe holding aligned results.")
    observation_count: int = Field(default=0, ge=0, description="Number of observations analyzed.")

    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.CALCULATED,
        description="Time-series statistical results are strictly calculated.",
    )

    @field_validator("epistemic_level")
    @classmethod
    def strict_epistemic_level(cls, v: EpistemicLevel) -> EpistemicLevel:
        if v.name in ("OBSERVED", "HISTORICAL", "INFERRED", "HYPOTHESIS", "RECOMMENDATION"):
            raise ValueError(f"Time-series results must not produce {v.name} epistemic level.")
        return v
