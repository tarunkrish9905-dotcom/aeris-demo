"""
SOP and Test Procedure Requirement domain models.

Provides immutable, version-aware domain models for explicitly supplied SOP and
test-procedure requirements. Preserves strict epistemic and safety boundaries:
requirements represent authoritative procedural reference criteria, not real-time
cockpit alerts, regulatory airworthiness certifications, or proof that requirements
were satisfied. Verification state is strictly segregated from requirement definitions.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.registry import ParameterRegistry
from aeris.core.sop.exceptions import (
    ConflictingRequirementError,
    DuplicateRequirementError,
    IncompatibleProcedureUnitError,
    InvalidProcedureRequirementError,
    ProcedureProfileMismatchError,
    UnknownParameterProcedureError,
    UnknownTestOperationReferenceError,
)
from aeris.core.test_operations.test_card import TestCard
from aeris.core.test_operations.test_point import TestPoint
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.identifiers import validate_domain_identifier
from aeris.core.vehicle.profile_models import VehicleProfile, validate_profile_version

_DOC_VERSION_REGEX = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_document_version(v: str, field_name: str = "document_version") -> str:
    """Enforce non-empty, well-formed document revision or version strings."""
    v_stripped = v.strip()
    if not v_stripped:
        raise ValueError(f"{field_name} cannot be empty or solely whitespace.")
    if not _DOC_VERSION_REGEX.match(v_stripped):
        raise ValueError(
            f"{field_name} must contain only alphanumeric characters, "
            "dots, dashes, and underscores."
        )
    return v_stripped


class RequirementCategory(StrEnum):
    """
    Explicit enumeration of supported procedural requirement categories.

    Distinguishes operational test phases and procedural criteria types.
    """

    ENTRY_CONDITION = "ENTRY_CONDITION"
    """Prerequisites and initial flight conditions required prior to initiating a test point."""

    EXECUTION = "EXECUTION"
    """Procedural actions and pilot/vehicle techniques required during maneuver execution."""

    TOLERANCE_BAND = "TOLERANCE_BAND"
    """Allowable parameter variation bands (e.g. altitude or airspeed tolerances)."""

    ABORT_CRITERIA = "ABORT_CRITERIA"
    """Explicit procedural conditions requiring immediate maneuver termination or test abort."""

    EXIT_CONDITION = "EXIT_CONDITION"
    """Completion and recovery criteria required at the conclusion of a test point."""

    OPERATIONAL_CHECK = "OPERATIONAL_CHECK"
    """Non-numeric procedural checklist items or discrete system configuration checks."""


class RequirementVerificationStatus(StrEnum):
    """
    Evaluation status of a procedural requirement.

    Strictly decoupled from the requirement definition itself: a requirement
    definition is reference metadata, whereas this status represents an evaluation result.
    """

    UNVERIFIED = "UNVERIFIED"
    """Requirement is defined but has not yet undergone post-flight telemetry/data evaluation."""

    SATISFIED = "SATISFIED"
    """Evaluated against flight evidence and determined to be satisfied within tolerance."""

    VIOLATED = "VIOLATED"
    """Evaluated against flight evidence and determined to have deviated or breached tolerance."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    """The requirement was not applicable to the flown flight condition or configuration."""

    INCONCLUSIVE = "INCONCLUSIVE"
    """Insufficient, missing, or corrupt telemetry prevented a definitive determination."""


class ProcedureRequirement(BaseModel):
    """
    Immutable domain model defining an explicit SOP or test procedure requirement.

    This model serves purely as authoritative procedural reference data.
    It does NOT store mutable verification status or observed telemetry.
    Supports both numeric tolerance criteria and qualitative checklist items.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    __test__ = False

    requirement_id: str = Field(
        ...,
        description="Deterministic, stable identifier for this procedural requirement.",
    )
    document_id: str = Field(
        ...,
        description="Authoritative document or procedure identifier (e.g. 'SOP-FT-001').",
    )
    document_version: str = Field(
        ...,
        description="Document version or revision (e.g. '1.0.0', 'Rev-B').",
    )
    source_reference: str = Field(
        ...,
        description="Exact source location (e.g. 'Section 3.2.1, Para 4', 'Table 4-2').",
    )
    description: str = Field(
        ...,
        description="Requirement statement faithfully supplied by the authoritative source.",
    )
    category: RequirementCategory = Field(
        ...,
        description="Explicit classification of the procedural requirement.",
    )

    # Applicability metadata
    vehicle_id: str | None = Field(
        default=None,
        description="Optional explicit vehicle ID to which this requirement applies.",
    )
    profile_id: str | None = Field(
        default=None,
        description="Optional vehicle profile ID to which this requirement applies.",
    )
    profile_version: str | None = Field(
        default=None,
        description="Optional vehicle profile version to which this requirement applies.",
    )
    configuration_id: str | None = Field(
        default=None,
        description="Optional configuration envelope ID to which this requirement applies.",
    )
    test_card_id: str | None = Field(
        default=None,
        description="Optional test card ID to which this requirement applies.",
    )
    test_point_id: str | None = Field(
        default=None,
        description="Optional test point ID to which this requirement applies.",
    )
    parameter_id: str | None = Field(
        default=None,
        description="Optional telemetry/parameter ID evaluated by this requirement.",
    )

    # Numeric target, tolerance, or band parameters (None for qualitative requirements)
    target_value: float | None = Field(
        default=None,
        description="Nominal target numeric value if specified by procedure.",
    )
    tolerance: float | None = Field(
        default=None,
        description="Symmetric allowable numeric tolerance (+/-) around target_value.",
    )
    tolerance_lower: float | None = Field(
        default=None,
        description="Asymmetric lower allowable delta (effective lower = target - tol_lower).",
    )
    tolerance_upper: float | None = Field(
        default=None,
        description="Asymmetric upper allowable delta (effective upper = target + tol_upper).",
    )
    min_value: float | None = Field(
        default=None,
        description="Explicit lower numeric bound.",
    )
    max_value: float | None = Field(
        default=None,
        description="Explicit upper numeric bound.",
    )
    unit: str | None = Field(
        default=None,
        description="Mandatory engineering unit for numeric requirements (e.g. 'kts', 'ft').",
    )

    # Domain provenance metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary descriptive metadata. Must not contain observed telemetry.",
    )

    @field_validator(
        "requirement_id",
        "document_id",
        "vehicle_id",
        "profile_id",
        "configuration_id",
        "test_card_id",
        "test_point_id",
        "parameter_id",
    )
    @classmethod
    def _validate_identifiers(cls, v: str | None, info: ValidationInfo) -> str | None:
        if v is not None:
            field_name = info.field_name if info.field_name else "identifier"
            return validate_domain_identifier(v, field_name)
        return None

    @field_validator("document_version")
    @classmethod
    def _validate_doc_version(cls, v: str) -> str:
        return validate_document_version(v, "document_version")

    @field_validator("profile_version")
    @classmethod
    def _validate_profile_version_field(cls, v: str | None) -> str | None:
        if v is not None:
            return validate_profile_version(v, "profile_version")
        return None

    @field_validator("source_reference", "description")
    @classmethod
    def _validate_non_empty_strings(cls, v: str, info: ValidationInfo) -> str:
        v_stripped = v.strip()
        field_name = info.field_name if info.field_name else "field"
        if not v_stripped:
            raise ValueError(f"{field_name} cannot be empty or solely whitespace.")
        return v_stripped

    @field_validator("unit")
    @classmethod
    def _validate_unit(cls, v: str | None) -> str | None:
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                raise ValueError("unit cannot be empty or solely whitespace.")
            return v_stripped
        return None

    @field_validator("metadata")
    @classmethod
    def _validate_metadata_safety(cls, v: dict[str, Any]) -> dict[str, Any]:
        forbidden_keys = {
            "telemetry",
            "observed_data",
            "time_series",
            "samples",
            "flight_data",
            "observed_value",
        }
        for k in v:
            if k.lower() in forbidden_keys:
                raise ValueError(
                    f"metadata contains forbidden operational key '{k}'. "
                    "ProcedureRequirement must not store observed flight telemetry."
                )
        return v

    @property
    def is_numeric(self) -> bool:
        """Return True if this requirement specifies numeric target/bounds or parameter."""
        return (
            self.parameter_id is not None
            or self.target_value is not None
            or self.tolerance is not None
            or self.tolerance_lower is not None
            or self.tolerance_upper is not None
            or self.min_value is not None
            or self.max_value is not None
        )

    @model_validator(mode="after")
    def _validate_numeric_and_boundaries(self) -> ProcedureRequirement:
        # 1. Validate numeric finiteness
        numeric_fields = (
            ("target_value", self.target_value),
            ("tolerance", self.tolerance),
            ("tolerance_lower", self.tolerance_lower),
            ("tolerance_upper", self.tolerance_upper),
            ("min_value", self.min_value),
            ("max_value", self.max_value),
        )
        for name, val in numeric_fields:
            if val is not None and (math.isnan(val) or math.isinf(val)):
                raise InvalidProcedureRequirementError(
                    f"{name} must be a finite number, not NaN or Infinity."
                )

        # 2. Enforce unit rules
        if self.is_numeric:
            if self.unit is None or not self.unit.strip():
                raise InvalidProcedureRequirementError(
                    f"Numeric requirement '{self.requirement_id}' "
                    "requires an explicit non-empty unit."
                )
        elif self.unit is not None:
            raise InvalidProcedureRequirementError(
                f"Non-numeric procedural requirement '{self.requirement_id}' "
                "must not specify a unit."
            )

        # 3. Validate tolerances
        for t_name, t_val in (
            ("tolerance", self.tolerance),
            ("tolerance_lower", self.tolerance_lower),
            ("tolerance_upper", self.tolerance_upper),
        ):
            if t_val is not None:
                if t_val < 0.0:
                    raise InvalidProcedureRequirementError(
                        f"{t_name} must be non-negative, got {t_val}."
                    )
                if self.target_value is None:
                    raise InvalidProcedureRequirementError(
                        f"{t_name} cannot be specified without target_value."
                    )

        # 4. Check tolerance consistency
        if self.tolerance is not None and (
            (self.tolerance_lower is not None and self.tolerance_lower != self.tolerance)
            or (self.tolerance_upper is not None and self.tolerance_upper != self.tolerance)
        ):
            raise ConflictingRequirementError(
                f"Requirement '{self.requirement_id}' specifies conflicting symmetric tolerance "
                f"({self.tolerance}) and asymmetric tolerances "
                f"(lower={self.tolerance_lower}, upper={self.tolerance_upper})."
            )

        # 5. Validate min_value and max_value bounds
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ConflictingRequirementError(
                f"Requirement '{self.requirement_id}' min_value ({self.min_value}) "
                f"cannot exceed max_value ({self.max_value})."
            )

        # 6. Validate target_value with respect to min/max
        if self.target_value is not None:
            if self.min_value is not None and self.target_value < self.min_value:
                raise ConflictingRequirementError(
                    f"Requirement '{self.requirement_id}' target_value ({self.target_value}) "
                    f"is less than min_value ({self.min_value})."
                )
            if self.max_value is not None and self.target_value > self.max_value:
                raise ConflictingRequirementError(
                    f"Requirement '{self.requirement_id}' target_value ({self.target_value}) "
                    f"exceeds max_value ({self.max_value})."
                )

        return self

    @property
    def effective_lower_bound(self) -> float | None:
        """Effective lower bound calculated from min_value or target - tolerance."""
        if self.min_value is not None:
            return self.min_value
        if self.target_value is not None:
            tol_low = self.tolerance_lower if self.tolerance_lower is not None else self.tolerance
            if tol_low is not None:
                return self.target_value - tol_low
        return None

    @property
    def effective_upper_bound(self) -> float | None:
        """Effective upper bound calculated from max_value or target + tolerance."""
        if self.max_value is not None:
            return self.max_value
        if self.target_value is not None:
            tol_up = self.tolerance_upper if self.tolerance_upper is not None else self.tolerance
            if tol_up is not None:
                return self.target_value + tol_up
        return None

    def is_satisfied_by_value(self, value: float) -> bool:
        """
        Check if an observed numeric value satisfies this requirement.

        Raises ValueError if called on a non-numeric requirement or non-finite value.
        """
        if not self.is_numeric:
            raise ValueError(
                f"Requirement '{self.requirement_id}' is non-numeric; "
                "cannot check numeric satisfaction."
            )
        if math.isnan(value) or math.isinf(value):
            raise ValueError("Cannot evaluate satisfaction for non-finite values (NaN or Inf).")

        lower = self.effective_lower_bound
        upper = self.effective_upper_bound

        if lower is not None and upper is not None:
            return lower <= value <= upper
        elif lower is not None:
            return value >= lower
        elif upper is not None:
            return value <= upper
        elif self.target_value is not None:
            return value == self.target_value
        return True

    def calculate_margin(self, value: float) -> float | None:
        """
        Calculate the margin of an observed numeric value to the nearest requirement boundary.

        Positive margin indicates value is inside the allowable band.
        Negative margin indicates a violation or exceedance magnitude.
        Returns None for non-numeric requirements or exact targets without tolerances.
        """
        if not self.is_numeric:
            return None
        if math.isnan(value) or math.isinf(value):
            raise ValueError("Cannot calculate margin for non-finite values (NaN or Inf).")

        lower = self.effective_lower_bound
        upper = self.effective_upper_bound

        if lower is not None and upper is not None:
            return min(value - lower, upper - value)
        elif lower is not None:
            return value - lower
        elif upper is not None:
            return upper - value
        elif self.target_value is not None:
            return -abs(value - self.target_value)
        return None


class ProcedureRequirementVerification(BaseModel):
    """
    Immutable model representing the evaluation/verification state of a procedure requirement.

    Strictly segregated from the reference ProcedureRequirement definition.
    A ProcedureRequirement definition represents what was required by the SOP;
    this model represents the empirical result of evaluating test data against that requirement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    __test__ = False

    verification_id: str = Field(
        ...,
        description="Deterministic, unique identifier for this verification assessment record.",
    )
    requirement_id: str = Field(
        ...,
        description="Explicit reference to the verified ProcedureRequirement ID.",
    )
    document_id: str = Field(
        ...,
        description="Authoritative document ID referenced by the requirement.",
    )
    status: RequirementVerificationStatus = Field(
        default=RequirementVerificationStatus.UNVERIFIED,
        description="Evaluation outcome status of the procedural requirement.",
    )
    observed_value: float | None = Field(
        default=None,
        description="Observed numeric value from telemetry during the evaluation window.",
    )
    observed_unit: str | None = Field(
        default=None,
        description="Unit of the observed telemetry value.",
    )
    is_satisfied: bool | None = Field(
        default=None,
        description="True if satisfied, False if violated, None if unverified/not applicable.",
    )
    discrepancy_message: str | None = Field(
        default=None,
        description="Descriptive explanation if requirement was violated or inconclusive.",
    )
    evaluated_at: str | None = Field(
        default=None,
        description="Timestamp or execution context when verification was performed.",
    )
    evidence_id: str | None = Field(
        default=None,
        description="Optional reference to an associated EvidenceRecord.",
    )
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.PROCEDURAL,
        description="AERIS epistemic standing: strictly PROCEDURAL.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary evaluation metadata.",
    )

    @field_validator("verification_id", "requirement_id", "document_id")
    @classmethod
    def _validate_ids(cls, v: str, info: ValidationInfo) -> str:
        field_name = info.field_name if info.field_name else "identifier"
        return validate_domain_identifier(v, field_name)

    @model_validator(mode="after")
    def _validate_status_coherence(self) -> ProcedureRequirementVerification:
        if self.status == RequirementVerificationStatus.SATISFIED and self.is_satisfied is False:
            raise ValueError("is_satisfied cannot be False when status is SATISFIED.")
        elif self.status == RequirementVerificationStatus.VIOLATED and self.is_satisfied is True:
            raise ValueError("is_satisfied cannot be True when status is VIOLATED.")
        return self


class ProcedureRequirementProfile(BaseModel):
    """
    Immutable domain model representing a validated collection of procedure requirements.

    Preserves document identity, revision, and vehicle/test-operation applicability.
    Enforces deterministic deduplication, conflict prevention, and structured lookups.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    __test__ = False

    profile_id: str = Field(
        ...,
        description="Deterministic, stable identifier for this requirement profile.",
    )
    document_id: str = Field(
        ...,
        description="Authoritative document or procedure identifier (e.g. 'SOP-FT-001').",
    )
    document_version: str = Field(
        ...,
        description="Document version or revision (e.g. '1.0.0', 'Rev-B').",
    )
    name: str = Field(
        default="Procedure Requirement Profile",
        description="Human-readable requirement profile name.",
    )
    description: str | None = Field(
        default=None,
        description="General description of the procedure requirement collection scope.",
    )
    source_reference: str | None = Field(
        default=None,
        description="Master source document or archive reference.",
    )

    # Optional profile-level scoping
    vehicle_id: str | None = Field(
        default=None,
        description="Optional vehicle ID scoping all requirements in this collection.",
    )
    profile_version: str | None = Field(
        default=None,
        description="Optional vehicle profile version scoping this collection.",
    )
    configuration_id: str | None = Field(
        default=None,
        description="Optional vehicle configuration ID scoping this collection.",
    )

    requirements: tuple[ProcedureRequirement, ...] = Field(
        default_factory=tuple,
        description="Ordered tuple of validated ProcedureRequirement objects.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary domain metadata. Must not contain observed telemetry.",
    )

    @field_validator("profile_id", "document_id", "vehicle_id", "configuration_id")
    @classmethod
    def _validate_profile_ids(cls, v: str | None, info: ValidationInfo) -> str | None:
        if v is not None:
            field_name = info.field_name if info.field_name else "identifier"
            return validate_domain_identifier(v, field_name)
        return None

    @field_validator("document_version")
    @classmethod
    def _validate_profile_doc_version(cls, v: str) -> str:
        return validate_document_version(v, "document_version")

    @field_validator("profile_version")
    @classmethod
    def _validate_profile_version_opt(cls, v: str | None) -> str | None:
        if v is not None:
            return validate_profile_version(v, "profile_version")
        return None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("name cannot be empty or solely whitespace.")
        return v_stripped

    @field_validator("source_reference", "description")
    @classmethod
    def _validate_opt_text(cls, v: str | None) -> str | None:
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                raise ValueError("Text fields cannot be empty or solely whitespace.")
            return v_stripped
        return None

    @model_validator(mode="after")
    def _validate_requirements_collection(self) -> ProcedureRequirementProfile:
        seen_req_ids: set[str] = set()
        numeric_scopes: set[tuple[RequirementCategory, str | None, str | None, str, str | None]] = (
            set()
        )
        qualitative_scopes: set[
            tuple[RequirementCategory, str | None, str | None, str, str | None]
        ] = set()

        for req in self.requirements:
            # 1. Uniqueness of requirement_id
            if req.requirement_id in seen_req_ids:
                raise DuplicateRequirementError(
                    f"Duplicate requirement_id found in profile: '{req.requirement_id}'."
                )
            seen_req_ids.add(req.requirement_id)

            # 2. Strict document alignment
            if req.document_id != self.document_id:
                raise ProcedureProfileMismatchError(
                    f"Requirement '{req.requirement_id}' document_id '{req.document_id}' "
                    f"does not match profile document_id '{self.document_id}'."
                )
            if req.document_version != self.document_version:
                raise ProcedureProfileMismatchError(
                    f"Requirement '{req.requirement_id}' document_version '{req.document_version}' "
                    f"does not match profile document_version '{self.document_version}'."
                )

            # 3. Vehicle / profile scoping alignment
            if (
                self.vehicle_id is not None
                and req.vehicle_id is not None
                and req.vehicle_id != self.vehicle_id
            ):
                raise ProcedureProfileMismatchError(
                    f"Requirement '{req.requirement_id}' vehicle_id '{req.vehicle_id}' "
                    f"does not match profile vehicle_id '{self.vehicle_id}'."
                )
            if (
                self.profile_version is not None
                and req.profile_version is not None
                and req.profile_version != self.profile_version
            ):
                raise ProcedureProfileMismatchError(
                    f"Requirement '{req.requirement_id}' profile_version '{req.profile_version}' "
                    f"does not match profile profile_version '{self.profile_version}'."
                )
            if (
                self.configuration_id is not None
                and req.configuration_id is not None
                and req.configuration_id != self.configuration_id
            ):
                raise ProcedureProfileMismatchError(
                    f"Requirement '{req.requirement_id}' configuration_id '{req.configuration_id}' "
                    f"does not match profile configuration_id '{self.configuration_id}'."
                )

            # 4. Scope conflict prevention
            if req.parameter_id is not None:
                num_scope = (
                    req.category,
                    req.test_card_id,
                    req.test_point_id,
                    req.parameter_id,
                    req.configuration_id,
                )
                if num_scope in numeric_scopes:
                    raise ConflictingRequirementError(
                        f"Conflicting or duplicate numeric requirement found for parameter "
                        f"'{req.parameter_id}' in category '{req.category}' under scope "
                        f"(card={req.test_card_id}, point={req.test_point_id}, "
                        f"config={req.configuration_id})."
                    )
                numeric_scopes.add(num_scope)
            else:
                qual_scope = (
                    req.category,
                    req.test_card_id,
                    req.test_point_id,
                    req.description.lower().strip(),
                    req.configuration_id,
                )
                if qual_scope in qualitative_scopes:
                    raise ConflictingRequirementError(
                        f"Duplicate non-numeric checklist item found for category '{req.category}' "
                        f"under scope (card={req.test_card_id}, point={req.test_point_id}, "
                        f"config={req.configuration_id}): '{req.description}'."
                    )
                qualitative_scopes.add(qual_scope)

        return self

    def validate_vehicle_profile_compatibility(self, profile: VehicleProfile) -> None:
        """
        Validate this requirement profile against an authoritative VehicleProfile.

        Checks:
        - Vehicle identity and version alignment.
        - Supported parameter registration in the vehicle profile.
        """
        if self.vehicle_id is not None and self.vehicle_id != profile.vehicle_id:
            raise ProcedureProfileMismatchError(
                f"Profile vehicle_id '{self.vehicle_id}' does not match "
                f"expected VehicleProfile vehicle_id '{profile.vehicle_id}'."
            )
        if self.profile_version is not None and self.profile_version != profile.profile_version:
            raise ProcedureProfileMismatchError(
                f"Profile profile_version '{self.profile_version}' does not match "
                f"expected VehicleProfile profile_version '{profile.profile_version}'."
            )

        for req in self.requirements:
            if req.vehicle_id is not None and req.vehicle_id != profile.vehicle_id:
                raise ProcedureProfileMismatchError(
                    f"Requirement '{req.requirement_id}' vehicle_id '{req.vehicle_id}' "
                    f"does not match expected VehicleProfile vehicle_id '{profile.vehicle_id}'."
                )
            if (
                profile.supported_parameter_ids
                and req.parameter_id is not None
                and req.parameter_id not in profile.supported_parameter_ids
            ):
                raise UnknownParameterProcedureError(
                    f"Requirement '{req.requirement_id}' parameter '{req.parameter_id}' "
                    f"is not supported by vehicle profile '{profile.profile_id}'."
                )

    def validate_configuration_compatibility(self, envelope: ConfigurationEnvelope) -> None:
        """
        Validate this requirement profile against an authoritative ConfigurationEnvelope.

        Checks vehicle identity and configuration-level scoping.
        """
        if self.vehicle_id is not None and self.vehicle_id != envelope.vehicle_id:
            raise ProcedureProfileMismatchError(
                f"Profile vehicle_id '{self.vehicle_id}' does not match "
                f"ConfigurationEnvelope vehicle_id '{envelope.vehicle_id}'."
            )
        if self.configuration_id is not None and self.configuration_id != envelope.configuration_id:
            raise ProcedureProfileMismatchError(
                f"Profile configuration_id '{self.configuration_id}' does not match "
                f"ConfigurationEnvelope configuration_id '{envelope.configuration_id}'."
            )

        for req in self.requirements:
            if (
                req.configuration_id is not None
                and req.configuration_id != envelope.configuration_id
            ):
                raise ProcedureProfileMismatchError(
                    f"Requirement '{req.requirement_id}' configuration_id '{req.configuration_id}' "
                    f"conflicts with ConfigurationEnvelope '{envelope.configuration_id}'."
                )

    def validate_test_operations_compatibility(
        self,
        test_cards: Sequence[TestCard] | None = None,
        test_points: Sequence[TestPoint] | None = None,
    ) -> None:
        """
        Validate requirement references against explicit flight-test operations entities.

        Checks:
        - Referenced test_card_id values exist in the supplied test_cards.
        - Referenced test_point_id values exist in the supplied test_points.
        - When both card and point are referenced, verifies the point is listed in the card.
        """
        card_map: dict[str, TestCard] = {}
        if test_cards is not None:
            card_map = {tc.test_card_id: tc for tc in test_cards}
            for req in self.requirements:
                if req.test_card_id is not None and req.test_card_id not in card_map:
                    raise UnknownTestOperationReferenceError(
                        f"Requirement '{req.requirement_id}' references unknown "
                        f"test_card_id '{req.test_card_id}'."
                    )

        tp_map: dict[str, TestPoint] = {}
        if test_points is not None:
            tp_map = {tp.test_point_id: tp for tp in test_points}
            for req in self.requirements:
                if req.test_point_id is not None and req.test_point_id not in tp_map:
                    raise UnknownTestOperationReferenceError(
                        f"Requirement '{req.requirement_id}' references unknown "
                        f"test_point_id '{req.test_point_id}'."
                    )

        if test_cards is not None and test_points is not None:
            for req in self.requirements:
                if req.test_card_id is not None and req.test_point_id is not None:
                    card = card_map.get(req.test_card_id)
                    if card is not None and req.test_point_id not in card.test_point_ids:
                        raise ConflictingRequirementError(
                            f"Requirement '{req.requirement_id}' references test_point_id "
                            f"'{req.test_point_id}', which is not listed in test_card "
                            f"'{req.test_card_id}'."
                        )

    def validate_registry_compatibility(self, registry: ParameterRegistry) -> None:
        """
        Validate all referenced parameters and units against a ParameterRegistry.

        Checks:
        - Referenced parameter IDs are registered.
        - Declared requirement units match the parameter's canonical_unit.
        """
        for req in self.requirements:
            if req.parameter_id is not None:
                if not registry.contains(req.parameter_id):
                    raise UnknownParameterProcedureError(
                        f"Requirement '{req.requirement_id}' references unknown "
                        f"parameter '{req.parameter_id}'."
                    )
                if req.unit is not None:
                    param_def = registry.get(req.parameter_id)
                    if (
                        param_def.canonical_unit is not None
                        and req.unit != param_def.canonical_unit
                    ):
                        raise IncompatibleProcedureUnitError(
                            f"Requirement '{req.requirement_id}' specifies unit '{req.unit}' "
                            f"for parameter '{req.parameter_id}', which conflicts with registry "
                            f"canonical unit '{param_def.canonical_unit}'."
                        )

    def get_requirement_by_id(self, requirement_id: str) -> ProcedureRequirement | None:
        """Retrieve a specific requirement by its requirement_id, or None if not found."""
        for req in self.requirements:
            if req.requirement_id == requirement_id:
                return req
        return None

    def get_requirements_for_test_point(
        self,
        test_point_id: str,
        configuration_id: str | None = None,
        include_all_configurations: bool = False,
    ) -> tuple[ProcedureRequirement, ...]:
        """
        Retrieve all requirements explicitly applicable to a given test_point_id.

        If configuration_id is provided, returns requirements matching configuration_id
        as well as baseline requirements where configuration_id is None.
        If configuration_id is None, uses profile.configuration_id if present;
        otherwise returns all requirements for this test point.
        """
        effective_config = (
            configuration_id if configuration_id is not None else self.configuration_id
        )
        results: list[ProcedureRequirement] = []
        for req in self.requirements:
            if req.test_point_id == test_point_id and (
                include_all_configurations
                or effective_config is None
                or req.configuration_id is None
                or req.configuration_id == effective_config
            ):
                results.append(req)
        return tuple(results)

    def get_requirements_for_test_card(
        self,
        test_card_id: str,
        configuration_id: str | None = None,
        include_all_configurations: bool = False,
    ) -> tuple[ProcedureRequirement, ...]:
        """Retrieve all requirements explicitly applicable to a given test_card_id."""
        effective_config = (
            configuration_id if configuration_id is not None else self.configuration_id
        )
        results: list[ProcedureRequirement] = []
        for req in self.requirements:
            if req.test_card_id == test_card_id and (
                include_all_configurations
                or effective_config is None
                or req.configuration_id is None
                or req.configuration_id == effective_config
            ):
                results.append(req)
        return tuple(results)

    def get_requirements_for_parameter(
        self,
        parameter_id: str,
        configuration_id: str | None = None,
        include_all_configurations: bool = False,
    ) -> tuple[ProcedureRequirement, ...]:
        """Retrieve all requirements applicable to a given parameter_id."""
        effective_config = (
            configuration_id if configuration_id is not None else self.configuration_id
        )
        results: list[ProcedureRequirement] = []
        for req in self.requirements:
            if req.parameter_id == parameter_id and (
                include_all_configurations
                or effective_config is None
                or req.configuration_id is None
                or req.configuration_id == effective_config
            ):
                results.append(req)
        return tuple(results)

    def get_requirements(
        self,
        *,
        category: RequirementCategory | None = None,
        test_card_id: str | None = None,
        test_point_id: str | None = None,
        parameter_id: str | None = None,
        configuration_id: str | None = None,
    ) -> tuple[ProcedureRequirement, ...]:
        """Deterministic filtered query across multiple criteria."""
        results: list[ProcedureRequirement] = []
        for req in self.requirements:
            if category is not None and req.category != category:
                continue
            if test_card_id is not None and req.test_card_id != test_card_id:
                continue
            if test_point_id is not None and req.test_point_id != test_point_id:
                continue
            if parameter_id is not None and req.parameter_id != parameter_id:
                continue
            if configuration_id is not None and req.configuration_id != configuration_id:
                continue
            results.append(req)
        return tuple(results)


# Architectural alias for flexibility and alignment with planning documents
ProcedureRequirementSet = ProcedureRequirementProfile
