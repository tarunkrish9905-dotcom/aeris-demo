"""
Domain models for the Controlled Vehicle Engineering Evaluation Engine.

Provides immutable request, observation, and result models for evaluating
validated flight observations against configured vehicle engineering limits
and SOP procedure requirements. Maintains strict separation between raw comparison
outcomes, severity classifications, and downstream engineering judgments.
"""

from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.registry import ParameterRegistry
from aeris.core.results.types import BasePipelineResult
from aeris.core.sop.models import (
    ProcedureRequirementProfile,
    RequirementCategory,
    RequirementVerificationStatus,
)
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.identifiers import validate_domain_identifier
from aeris.core.vehicle.profile_models import VehicleProfile


class EvaluationOutcome(StrEnum):
    """
    Controlled classification of an evaluation outcome.

    Kept strictly distinct from operational severity, risk, or regulatory compliance.
    """

    WITHIN_BOUNDS = "WITHIN_BOUNDS"
    """Measured or calculated value is inside the allowable engineering limit boundary."""

    OUTSIDE_BOUNDS = "OUTSIDE_BOUNDS"
    """Measured or calculated value breaches the configured engineering limit boundary."""

    SATISFIED = "SATISFIED"
    """Procedural criteria or target tolerance band was evaluated and satisfied."""

    VIOLATED = "VIOLATED"
    """Procedural criteria or target tolerance band was evaluated and breached/deviated."""

    INCONCLUSIVE = "INCONCLUSIVE"
    """Evaluation could not determine outcome due to missing telemetry or unit mismatch."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    """Limit or procedure requirement was not applicable to the flight configuration/scope."""

    NOT_EVALUATED = "NOT_EVALUATED"
    """No corresponding limit exists, or qualitative requirement lacks confirmation input."""


class EvaluationObservation(BaseModel):
    """
    Immutable representation of an observation or calculated engineering value
    supplied for vehicle engineering evaluation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    __test__ = False

    observation_id: str = Field(
        ...,
        description="Deterministic identifier for this observation.",
    )
    parameter_id: str = Field(
        ...,
        description="Telemetric or engineering parameter identifier.",
    )
    value: float | None = Field(
        default=None,
        description="Observed numeric value (None for qualitative observations).",
    )
    unit: str | None = Field(
        default=None,
        description="Engineering unit for numeric values (e.g. 'kts', 'ft').",
    )
    timestamp: datetime.datetime | None = Field(
        default=None,
        description="UTC timestamp of the observation.",
    )
    row_index: int | None = Field(
        default=None,
        description="Row index in the originating telemetry dataset.",
    )
    source_reference: str = Field(
        ...,
        min_length=1,
        description="Traceable source identifier (e.g. 'flight_data.csv', 'calc_q').",
    )
    source_record_id: str | None = Field(
        default=None,
        description="Optional upstream record identifier.",
    )
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.OBSERVED,
        description="Epistemic standing of the input value (OBSERVED or CALCULATED).",
    )
    is_qualitative: bool = Field(
        default=False,
        description="True if this is a qualitative/checklist state observation.",
    )
    qualitative_state: str | None = Field(
        default=None,
        description="Descriptive state for qualitative items (e.g. 'CONFIRMED', 'DOWN').",
    )
    configuration_id: str | None = Field(
        default=None,
        description="Optional configuration state associated with this observation.",
    )
    test_card_id: str | None = Field(
        default=None,
        description="Optional test card ID associated with this observation.",
    )
    test_point_id: str | None = Field(
        default=None,
        description="Optional test point ID associated with this observation.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Descriptive metadata for this observation.",
    )

    @field_validator(
        "observation_id",
        "parameter_id",
        "configuration_id",
        "test_card_id",
        "test_point_id",
    )
    @classmethod
    def _validate_ids(cls, v: str | None, info: ValidationInfo) -> str | None:
        if v is not None:
            field_name = info.field_name if info.field_name else "identifier"
            return validate_domain_identifier(v, field_name)
        return None

    @field_validator("source_reference")
    @classmethod
    def _validate_source_ref(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("source_reference cannot be empty or solely whitespace.")
        return v_stripped

    @field_validator("unit", "qualitative_state")
    @classmethod
    def _validate_opt_text(cls, v: str | None) -> str | None:
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                raise ValueError("Text fields cannot be empty or solely whitespace.")
            return v_stripped
        return None


class LimitEvaluationRecord(BaseModel):
    """
    Immutable evaluation record of an observation against an engineering limit.

    Preserves mathematical margin, configured limit severity, and evaluation outcome.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    __test__ = False

    evaluation_id: str = Field(
        ...,
        description="Deterministic identifier for this limit evaluation record.",
    )
    limit_id: str = Field(
        ...,
        description="Identifier of the evaluated ParameterLimitDefinition.",
    )
    parameter_id: str = Field(
        ...,
        description="Evaluated parameter identifier.",
    )
    limit_type: LimitType = Field(
        ...,
        description="Boundary comparison type (UPPER_BOUND, LOWER_BOUND, BAND).",
    )
    severity: LimitSeverity = Field(
        ...,
        description="Configured limit severity (ADVISORY, CAUTION, WARNING, EXCEEDANCE).",
    )
    outcome: EvaluationOutcome = Field(
        ...,
        description="Evaluation outcome (WITHIN_BOUNDS, OUTSIDE_BOUNDS, INCONCLUSIVE, etc.).",
    )
    reason: str = Field(
        ...,
        description="Deterministic explanatory reason for the evaluation outcome.",
    )

    # Observed value details
    observed_value: float | None = Field(
        default=None,
        description="Observed numeric value evaluated.",
    )
    observed_unit: str | None = Field(
        default=None,
        description="Engineering unit of the observed value.",
    )

    # Boundary details
    limit_value: float | None = Field(
        default=None,
        description="Configured nominal limit value (if single-sided).",
    )
    lower_bound: float | None = Field(
        default=None,
        description="Lower limit boundary value.",
    )
    upper_bound: float | None = Field(
        default=None,
        description="Upper limit boundary value.",
    )
    limit_unit: str = Field(
        ...,
        description="Configured unit of the engineering limit.",
    )

    # Comparison results
    is_exceeded: bool = Field(
        ...,
        description="True if outcome is OUTSIDE_BOUNDS; False otherwise.",
    )
    margin: float | None = Field(
        default=None,
        description="Signed mathematical margin (+ = within, 0 = on bound, - = exceeded).",
    )

    # Lineage and context
    vehicle_id: str = Field(
        ...,
        description="Vehicle identifier to which the evaluated limit applies.",
    )
    profile_id: str = Field(
        ...,
        description="Vehicle profile identifier.",
    )
    profile_version: str = Field(
        ...,
        description="Vehicle profile version.",
    )
    configuration_id: str | None = Field(
        default=None,
        description="Configuration ID if limit is configuration-scoped.",
    )
    source_reference: str = Field(
        ...,
        description="Authoritative source reference for the limit.",
    )
    observation_id: str | None = Field(
        default=None,
        description="Identifier of the observation that was evaluated.",
    )
    row_index: int | None = Field(
        default=None,
        description="Row index of the evaluated observation.",
    )
    timestamp: datetime.datetime | None = Field(
        default=None,
        description="Timestamp of the evaluated observation.",
    )
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.LIMIT,
        description="AERIS epistemic level: LIMIT.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation metadata.",
    )


class ProcedureEvaluationRecord(BaseModel):
    """
    Immutable evaluation record of an observation against an SOP requirement.

    Preserves verification status, category, target criteria, and qualitative state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    __test__ = False

    evaluation_id: str = Field(
        ...,
        description="Deterministic identifier for this procedure evaluation record.",
    )
    requirement_id: str = Field(
        ...,
        description="Identifier of the evaluated ProcedureRequirement.",
    )
    document_id: str = Field(
        ...,
        description="Authoritative procedure document identifier.",
    )
    document_version: str = Field(
        ...,
        description="Revision/version of the procedure document.",
    )
    category: RequirementCategory = Field(
        ...,
        description="Procedural requirement category.",
    )
    verification_status: RequirementVerificationStatus = Field(
        ...,
        description="Verification outcome status (SATISFIED, VIOLATED, UNVERIFIED, etc.).",
    )
    outcome: EvaluationOutcome = Field(
        ...,
        description="Evaluation outcome (SATISFIED, VIOLATED, INCONCLUSIVE, NOT_EVALUATED).",
    )
    reason: str = Field(
        ...,
        description="Deterministic explanatory reason for the verification outcome.",
    )

    # Quantitative details
    observed_value: float | None = Field(
        default=None,
        description="Observed numeric value evaluated.",
    )
    observed_unit: str | None = Field(
        default=None,
        description="Engineering unit of the observed value.",
    )
    target_value: float | None = Field(
        default=None,
        description="Nominal target value specified in requirement.",
    )
    tolerance_lower: float | None = Field(
        default=None,
        description="Lower tolerance delta.",
    )
    tolerance_upper: float | None = Field(
        default=None,
        description="Upper tolerance delta.",
    )
    effective_lower_bound: float | None = Field(
        default=None,
        description="Effective lower bound (min_value or target - tolerance).",
    )
    effective_upper_bound: float | None = Field(
        default=None,
        description="Effective upper bound (max_value or target + tolerance).",
    )
    requirement_unit: str | None = Field(
        default=None,
        description="Engineering unit specified by the procedure requirement.",
    )
    margin: float | None = Field(
        default=None,
        description="Signed mathematical margin to procedure boundary.",
    )
    is_satisfied: bool | None = Field(
        default=None,
        description="True if SATISFIED, False if VIOLATED, None if unverified/inconclusive.",
    )

    # Qualitative details
    is_qualitative: bool = Field(
        default=False,
        description="True if requirement is a qualitative checklist item.",
    )
    qualitative_state: str | None = Field(
        default=None,
        description="Observed qualitative state if supplied.",
    )

    # Context & Lineage
    vehicle_id: str | None = Field(
        default=None,
        description="Vehicle identifier if requirement is vehicle-scoped.",
    )
    profile_version: str | None = Field(
        default=None,
        description="Vehicle profile version.",
    )
    configuration_id: str | None = Field(
        default=None,
        description="Configuration identifier if requirement is configuration-scoped.",
    )
    test_card_id: str | None = Field(
        default=None,
        description="Test card ID if requirement is card-scoped.",
    )
    test_point_id: str | None = Field(
        default=None,
        description="Test point ID if requirement is point-scoped.",
    )
    parameter_id: str | None = Field(
        default=None,
        description="Evaluated parameter ID.",
    )
    source_reference: str = Field(
        ...,
        description="Authoritative source reference in procedure document.",
    )
    observation_id: str | None = Field(
        default=None,
        description="Identifier of the evaluated observation.",
    )
    row_index: int | None = Field(
        default=None,
        description="Row index of the evaluated observation.",
    )
    timestamp: datetime.datetime | None = Field(
        default=None,
        description="Timestamp of the evaluated observation.",
    )
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.PROCEDURAL,
        description="AERIS epistemic level: PROCEDURAL.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation metadata.",
    )


class VehicleEvaluationRequest(BaseModel):
    """
    Explicit request specification for vehicle engineering evaluation.

    Requires explicit vehicle profile and version bindings; does not silently guess context.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")
    __test__ = False

    request_id: str = Field(
        ...,
        description="Deterministic identifier for this evaluation request.",
    )
    vehicle_profile: VehicleProfile = Field(
        ...,
        description="Authoritative VehicleProfile specifying vehicle identity and version.",
    )
    configuration_envelope: ConfigurationEnvelope | None = Field(
        default=None,
        description="Optional ConfigurationEnvelope defining active vehicle configuration.",
    )
    limit_profile: EngineeringLimitProfile | None = Field(
        default=None,
        description="Optional EngineeringLimitProfile defining applicable parameter limits.",
    )
    procedure_profile: ProcedureRequirementProfile | None = Field(
        default=None,
        description="Optional ProcedureRequirementProfile defining SOP requirements.",
    )
    observations: tuple[EvaluationObservation, ...] = Field(
        default_factory=tuple,
        description="Ordered tuple of explicit observations to evaluate.",
    )
    test_card_id: str | None = Field(
        default=None,
        description="Optional test card context filtering applicability.",
    )
    test_point_id: str | None = Field(
        default=None,
        description="Optional test point context filtering applicability.",
    )
    context_id: str | None = Field(
        default=None,
        description="Optional operational test context ID.",
    )
    parameter_registry: ParameterRegistry | None = Field(
        default=None,
        description="Optional ParameterRegistry for canonical unit and parameter cross-checks.",
    )
    evaluate_limits: bool = Field(
        default=True,
        description="Whether to perform engineering limit evaluations.",
    )
    evaluate_procedures: bool = Field(
        default=True,
        description="Whether to perform procedure requirement evaluations.",
    )
    strict_units: bool = Field(
        default=True,
        description="Whether to reject unit mismatches as INCONCLUSIVE (True) vs error.",
    )
    provenance: ProvenanceRecord | None = Field(
        default=None,
        description="Optional upstream provenance record.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Request metadata.",
    )

    @field_validator("request_id")
    @classmethod
    def _validate_request_id(cls, v: str) -> str:
        return validate_domain_identifier(v, "request_id")


class VehicleEvaluationResult(BasePipelineResult):
    """
    Authoritative, immutable result of a vehicle engineering evaluation.

    Encapsulates all evaluated limits, procedure requirements, configuration
    discrepancies, and overall execution status.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")
    __test__ = False

    evaluation_id: str = Field(
        ...,
        description="Deterministic identifier for this vehicle evaluation execution.",
    )
    vehicle_id: str = Field(
        ...,
        description="Evaluated vehicle identifier.",
    )
    profile_id: str = Field(
        ...,
        description="Evaluated vehicle profile identifier.",
    )
    profile_version: str = Field(
        ...,
        description="Evaluated vehicle profile version.",
    )
    configuration_id: str | None = Field(
        default=None,
        description="Active configuration ID if configuration was supplied.",
    )

    limit_evaluations: tuple[LimitEvaluationRecord, ...] = Field(
        default_factory=tuple,
        description="All individual engineering limit evaluations performed.",
    )
    procedure_evaluations: tuple[ProcedureEvaluationRecord, ...] = Field(
        default_factory=tuple,
        description="All individual procedure requirement evaluations performed.",
    )
    configuration_discrepancies: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Any configuration discrepancies detected between data and envelope.",
    )
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="High-level evaluation metrics and counts.",
    )

    @property
    def total_evaluations(self) -> int:
        """Total number of limit and procedure evaluations performed."""
        return len(self.limit_evaluations) + len(self.procedure_evaluations)

    @property
    def limit_exceedance_count(self) -> int:
        """Number of engineering limit evaluations resulting in an exceedance."""
        return sum(1 for rec in self.limit_evaluations if rec.is_exceeded)

    @property
    def procedure_violation_count(self) -> int:
        """Number of procedure requirement evaluations resulting in a violation."""
        return sum(1 for rec in self.procedure_evaluations if rec.is_satisfied is False)

    @property
    def inconclusive_count(self) -> int:
        """Number of evaluations resulting in INCONCLUSIVE status."""
        lim_inc = sum(
            1 for rec in self.limit_evaluations if rec.outcome == EvaluationOutcome.INCONCLUSIVE
        )
        proc_inc = sum(
            1 for rec in self.procedure_evaluations if rec.outcome == EvaluationOutcome.INCONCLUSIVE
        )
        return lim_inc + proc_inc

    @property
    def has_exceedances(self) -> bool:
        """Return True if any engineering limits were exceeded."""
        return self.limit_exceedance_count > 0

    @property
    def has_violations(self) -> bool:
        """Return True if any procedure requirements were violated."""
        return self.procedure_violation_count > 0
