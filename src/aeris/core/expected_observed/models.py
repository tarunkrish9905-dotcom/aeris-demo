from __future__ import annotations

import datetime
import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import BasePipelineResult, Issue, ResultStatus


class ExpectationSource(StrEnum):
    """Controlled representation for the origin of an expectation."""

    TEST_PROCEDURE = "TEST_PROCEDURE"
    TEST_CARD = "TEST_CARD"
    ENGINEERING_MODEL = "ENGINEERING_MODEL"
    VEHICLE_BASELINE = "VEHICLE_BASELINE"
    HISTORICAL_BASELINE = "HISTORICAL_BASELINE"
    EXPLICIT_INPUT = "EXPLICIT_INPUT"


class ComparisonMethod(StrEnum):
    """Controlled representation for comparison semantics."""

    ABSOLUTE_DIFFERENCE = "ABSOLUTE_DIFFERENCE"
    RELATIVE_DIFFERENCE = "RELATIVE_DIFFERENCE"
    PERCENT_DIFFERENCE = "PERCENT_DIFFERENCE"
    RATIO = "RATIO"


class ToleranceType(StrEnum):
    """Tolerance representation."""

    ABSOLUTE = "ABSOLUTE"
    RELATIVE = "RELATIVE"
    PERCENTAGE = "PERCENTAGE"


class Tolerance(BaseModel):
    """Immutable explicit tolerance representation.
    Tolerance is separated from anomaly severity and deviation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tolerance_type: ToleranceType
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: str | None = None

    @field_validator("lower_bound", "upper_bound")
    @classmethod
    def validate_finite(cls, v: float | None) -> float | None:
        """Reject NaN or infinity for tolerances."""
        if v is not None and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Tolerance bounds must be finite numbers.")
        return v


class ExpectedValue(BaseModel):
    """Explicit expectation of a value, preserving source and epistemic status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_id: str = Field(min_length=1)
    parameter_id: str = Field(min_length=1)
    value: float
    unit: str | None = None
    source: ExpectationSource
    source_reference: str = Field(min_length=1)
    epistemic_level: EpistemicLevel
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        """Reject NaN or infinity."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Expected value must be a finite number.")
        return v


class ObservedValue(BaseModel):
    """Explicit observed value, preserving source reference and epistemic status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observed_id: str = Field(min_length=1)
    parameter_id: str = Field(min_length=1)
    value: float
    unit: str | None = None
    source_reference: str = Field(min_length=1)
    epistemic_level: EpistemicLevel = Field(default=EpistemicLevel.OBSERVED)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        """Reject NaN or infinity."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Observed value must be a finite number.")
        return v


class Deviation(BaseModel):
    """Mathematical deviation representation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_value: float
    observed_value: float
    deviation_value: float
    deviation_unit: str | None = None
    comparison_method: ComparisonMethod

    @field_validator("expected_value", "observed_value", "deviation_value")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        """Reject NaN or infinity."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Deviation values must be finite numbers.")
        return v


class ExpectedObservedComparison(BaseModel):
    """The overarching domain model for comparing expected vs observed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_id: str = Field(min_length=1)
    expected: ExpectedValue
    observed: ObservedValue
    deviation: Deviation | None = None
    tolerance: Tolerance | None = None
    tolerance_evaluated: bool = False
    within_tolerance: bool | None = None
    status: ResultStatus
    issues: list[Issue] = Field(default_factory=list)
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def validate_parameter_consistency(self) -> ExpectedObservedComparison:
        """Ensure expected and observed values are structurally consistent unless returning an ERROR."""
        if (
            self.status != ResultStatus.ERROR
            and self.expected.parameter_id != self.observed.parameter_id
        ):
            raise ValueError("Expected and observed parameter_ids must match.")
        return self


class ScalarComparisonRequest(BaseModel):
    """Explicit request for a scalar Expected-vs-Observed comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    comparison_id: str = Field(min_length=1)
    expected: ExpectedValue
    observed_parameter_id: str = Field(min_length=1)
    observed_row_index: int
    observed_unit: str | None = None
    method: ComparisonMethod
    tolerance: Tolerance | None = None


class PipelineExpectedObservedRequest(BaseModel):
    """Request containing multiple scalar comparisons."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    scalar_requests: tuple[ScalarComparisonRequest, ...] = Field(default_factory=tuple)


class PipelineExpectedObservedResult(BasePipelineResult):
    """Result container for Expected-vs-Observed pipeline execution."""

    comparisons: tuple[ExpectedObservedComparison, ...] = Field(default_factory=tuple)


class VectorComparisonRequest(BaseModel):
    """Explicit request for a vector Expected-vs-Observed comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    vector_id: str = Field(min_length=1)
    parameter_id: str = Field(min_length=1)
    expected_values: tuple[ExpectedValue, ...]
    observed_values: tuple[ObservedValue, ...]
    method: ComparisonMethod
    tolerance: Tolerance | None = None

    @model_validator(mode="after")
    def validate_vectors(self) -> VectorComparisonRequest:
        if len(self.expected_values) != len(self.observed_values):
            raise ValueError("Expected and observed vectors must have the same length.")
        for e in self.expected_values:
            if e.parameter_id != self.parameter_id:
                raise ValueError("All expected values must match the request parameter_id.")
        for o in self.observed_values:
            if o.parameter_id != self.parameter_id:
                raise ValueError("All observed values must match the request parameter_id.")
        return self


class VectorComparisonResult(BaseModel):
    """Result container for a Vector Expected-vs-Observed evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    vector_id: str
    parameter_id: str
    method: ComparisonMethod
    elements: tuple[ExpectedObservedComparison, ...] = Field(default_factory=tuple)
    status: ResultStatus
    issues: list[Issue] = Field(default_factory=list)
    provenance: ProvenanceRecord


class TemporalExpectedValue(ExpectedValue):
    """Explicit expectation of a value at an exact timestamp."""

    timestamp: datetime.datetime

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime.datetime) -> datetime.datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        if v.utcoffset() != datetime.timedelta(0):
            raise ValueError("Timestamp must be UTC.")
        return v


class TemporalObservedValue(ObservedValue):
    """Explicit observed value at an exact timestamp."""

    timestamp: datetime.datetime

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime.datetime) -> datetime.datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        if v.utcoffset() != datetime.timedelta(0):
            raise ValueError("Timestamp must be UTC.")
        return v


class TemporalExpectedObservedPoint(BaseModel):
    """Explicitly ties an exact timestamp to a scalar comparison result."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    timestamp: datetime.datetime
    comparison: ExpectedObservedComparison


class TemporalExpectedObservedRequest(BaseModel):
    """Explicit request for a time-aware Expected-vs-Observed comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    comparison_id: str = Field(min_length=1)
    parameter_id: str = Field(min_length=1)
    expected_values: tuple[TemporalExpectedValue, ...]
    observed_values: tuple[TemporalObservedValue, ...]
    method: ComparisonMethod
    tolerance: Tolerance | None = None

    @model_validator(mode="after")
    def validate_parameter_identity(self) -> TemporalExpectedObservedRequest:
        for e in self.expected_values:
            if e.parameter_id != self.parameter_id:
                raise ValueError("All expected values must match the request parameter_id.")
        for o in self.observed_values:
            if o.parameter_id != self.parameter_id:
                raise ValueError("All observed values must match the request parameter_id.")
        return self


class TemporalExpectedObservedResult(BaseModel):
    """Result container for a time-aware Expected-vs-Observed evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    comparison_id: str
    parameter_id: str
    method: ComparisonMethod

    matched_points: tuple[TemporalExpectedObservedPoint, ...] = Field(default_factory=tuple)
    unmatched_expected: tuple[TemporalExpectedValue, ...] = Field(default_factory=tuple)
    unmatched_observed: tuple[TemporalObservedValue, ...] = Field(default_factory=tuple)

    expected_count: int
    observed_count: int
    matched_count: int
    expected_only_count: int
    observed_only_count: int

    status: ResultStatus
    issues: list[Issue] = Field(default_factory=list)
    provenance: ProvenanceRecord
