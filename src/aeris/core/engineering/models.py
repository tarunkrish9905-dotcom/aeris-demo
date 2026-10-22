from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import BasePipelineResult


class EngineeringCalculationDefinition(BaseModel):
    """
    Explicit, deterministic definition of an engineering calculation intended to be performed.
    Does not execute calculations or contain empirical data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    calculation_id: str = Field(
        ...,
        min_length=1,
        description="Unique deterministic identifier for this specific calculation.",
    )
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    output_parameter_id: str = Field(..., min_length=1)
    output_unit: str = Field(..., min_length=1)
    input_parameter_ids: list[str] = Field(
        ..., description="Explicit identifiers for required input parameters."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input_parameter_ids")
    @classmethod
    def validate_inputs(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("input_parameter_ids cannot be empty")
        if len(set(v)) != len(v):
            raise ValueError("input_parameter_ids cannot contain duplicates")
        return v


class EngineeringInputReference(BaseModel):
    """
    Explicit reference to an input used in an engineering calculation for provenance traceability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_id: str = Field(..., min_length=1)
    source_identifier: str = Field(
        ..., min_length=1, description="Identifier of the source dataset or result."
    )
    column_name: str | None = Field(default=None)
    unit: str | None = Field(default=None)
    epistemic_level: EpistemicLevel = Field(...)


class EngineeringResult(BasePipelineResult):
    """
    Base generic result of an engineering calculation or analysis.
    Inherits status, issues, and provenance from BasePipelineResult.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    result_id: str = Field(..., min_length=1)
    calculation_id: str = Field(..., min_length=1)
    payload: Any = Field(default=None)
    unit: str | None = Field(default=None)
    input_references: list[EngineeringInputReference] = Field(default_factory=list)


class EngineeringCalculationResult(EngineeringResult):
    """
    Execution metadata and result of a specific deterministic engineering calculation.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    output_value: Any = Field(default=None, description="The strictly calculated output value.")
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.CALCULATED,
        description="The epistemic level of this calculation, strictly constrained.",
    )

    @field_validator("epistemic_level")
    @classmethod
    def strict_epistemic_level(cls, v: EpistemicLevel) -> EpistemicLevel:
        """Ensure calculations are never silently passed off as observed or historical facts."""
        # We allow PROCEDURAL if it exists in the codebase, though CALCULATED is the primary.
        # But we explicitly reject OBSERVED, HISTORICAL, INFERRED, HYPOTHESIS, RECOMMENDATION.
        if v.name in ("OBSERVED", "HISTORICAL", "INFERRED", "HYPOTHESIS", "RECOMMENDATION"):
            raise ValueError(f"Engineering calculations must not produce {v.name} epistemic level.")

        return v
