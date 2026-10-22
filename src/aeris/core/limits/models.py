"""
Engineering Limits domain models.

Provides immutable, versioned domain models for parameter-specific engineering limits
and limit profile collections. Preserves epistemic boundaries: configured limits represent
explicit engineering references, not real-time cockpit alarms, root causes, or proofs
of airworthiness.
"""

from __future__ import annotations

import math
from collections import defaultdict
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from aeris.core.limits.exceptions import (
    ConflictingLimitError,
    DuplicateLimitError,
    IncompatibleLimitUnitError,
    LimitProfileMismatchError,
    UnknownParameterLimitError,
)
from aeris.core.registry import ParameterRegistry
from aeris.core.vehicle.identifiers import validate_domain_identifier
from aeris.core.vehicle.profile_models import VehicleProfile, validate_profile_version


class LimitType(StrEnum):
    """
    Explicit enumeration of supported boundary comparison semantics.

    Distinguishes the boundary geometric type (UPPER_BOUND, LOWER_BOUND, BAND)
    from operational severity or significance classification.
    """

    UPPER_BOUND = "UPPER_BOUND"
    LOWER_BOUND = "LOWER_BOUND"
    BAND = "BAND"


class LimitSeverity(StrEnum):
    """
    Analytical classification of limit significance.

    Does NOT imply regulatory compliance, certification, or flight clearance.
    """

    ADVISORY = "ADVISORY"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    EXCEEDANCE = "EXCEEDANCE"


class ParameterLimitDefinition(BaseModel):
    """
    Immutable domain model defining an explicit engineering limit for a parameter.

    Binds a parameter to a numeric boundary, unit, applicable vehicle profile, and version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    limit_id: str = Field(
        ...,
        description="Deterministic, stable identifier for this specific limit.",
    )
    parameter_id: str = Field(
        ...,
        description="Reference to the target parameter ID.",
    )
    limit_type: LimitType = Field(
        ...,
        description="Boundary comparison semantics (UPPER_BOUND, LOWER_BOUND, BAND).",
    )
    severity: LimitSeverity = Field(
        default=LimitSeverity.EXCEEDANCE,
        description="Classification of limit significance.",
    )
    unit: str = Field(
        ...,
        description="Explicit unit of measurement. Required for all limits.",
    )
    vehicle_id: str = Field(
        ...,
        description="Reference to the target vehicle ID.",
    )
    profile_id: str = Field(
        ...,
        description="Reference to the parent vehicle profile ID.",
    )
    profile_version: str = Field(
        ...,
        description="Explicit vehicle profile version.",
    )
    configuration_id: str | None = Field(
        default=None,
        description="Optional configuration applicability. If None, applies globally.",
    )
    lower_bound: float | None = Field(
        default=None,
        description="Lower numeric boundary for LOWER_BOUND and BAND limits.",
    )
    upper_bound: float | None = Field(
        default=None,
        description="Upper numeric boundary for UPPER_BOUND and BAND limits.",
    )
    limit_value: float | None = Field(
        default=None,
        description="Scalar boundary value for single-sided limits.",
    )
    description: str | None = Field(
        default=None,
        description="Clear engineering description of the limit's intended interpretation.",
    )
    source_reference: str = Field(
        ...,
        description="Document, manual, or engineering source reference.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary domain metadata. Must not contain observed telemetry.",
    )

    @field_validator("limit_id", "parameter_id", "vehicle_id", "profile_id")
    @classmethod
    def _validate_ids(cls, v: str, info: ValidationInfo) -> str:
        field_name = info.field_name if info.field_name else "identifier"
        return validate_domain_identifier(v, field_name)

    @field_validator("configuration_id")
    @classmethod
    def _validate_config_id(cls, v: str | None) -> str | None:
        if v is not None:
            return validate_domain_identifier(v, "configuration_id")
        return None

    @field_validator("profile_version")
    @classmethod
    def _validate_profile_version(cls, v: str) -> str:
        return validate_profile_version(v, "profile_version")

    @field_validator("unit", "source_reference")
    @classmethod
    def _validate_required_strings(cls, v: str, info: ValidationInfo) -> str:
        field_name = info.field_name if info.field_name else "field"
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError(f"{field_name} cannot be empty or solely whitespace.")
        return v_stripped

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: str | None) -> str | None:
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                raise ValueError("description cannot be empty or solely whitespace.")
            return v_stripped
        return None

    @model_validator(mode="before")
    @classmethod
    def _map_scalar_limit_value(cls, data: Any) -> Any:
        if isinstance(data, dict):
            limit_val = data.get("limit_value")
            l_type = data.get("limit_type")
            if limit_val is not None:
                if (
                    l_type in (LimitType.UPPER_BOUND, "UPPER_BOUND")
                    and data.get("upper_bound") is None
                ):
                    data["upper_bound"] = limit_val
                elif (
                    l_type in (LimitType.LOWER_BOUND, "LOWER_BOUND")
                    and data.get("lower_bound") is None
                ):
                    data["lower_bound"] = limit_val
        return data

    @model_validator(mode="after")
    def _validate_boundaries(self) -> ParameterLimitDefinition:
        # Validate finite floats
        for b_name, b_val in (
            ("lower_bound", self.lower_bound),
            ("upper_bound", self.upper_bound),
            ("limit_value", self.limit_value),
        ):
            if b_val is not None and (math.isnan(b_val) or math.isinf(b_val)):
                raise ValueError(f"{b_name} must be a finite number, not NaN or Infinity.")

        if self.limit_type == LimitType.UPPER_BOUND:
            if self.upper_bound is None:
                raise ValueError("UPPER_BOUND limit requires upper_bound to be specified.")
            if self.lower_bound is not None:
                raise ValueError("UPPER_BOUND limit must not specify lower_bound.")
            if self.limit_value is not None and self.limit_value != self.upper_bound:
                raise ValueError("limit_value must match upper_bound for UPPER_BOUND limit.")

        elif self.limit_type == LimitType.LOWER_BOUND:
            if self.lower_bound is None:
                raise ValueError("LOWER_BOUND limit requires lower_bound to be specified.")
            if self.upper_bound is not None:
                raise ValueError("LOWER_BOUND limit must not specify upper_bound.")
            if self.limit_value is not None and self.limit_value != self.lower_bound:
                raise ValueError("limit_value must match lower_bound for LOWER_BOUND limit.")

        elif self.limit_type == LimitType.BAND:
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("BAND limit requires both lower_bound and upper_bound.")
            if self.limit_value is not None:
                raise ValueError("BAND limit must not specify scalar limit_value.")
            if self.lower_bound >= self.upper_bound:
                raise ValueError(
                    f"BAND limit lower_bound ({self.lower_bound}) must be strictly less "
                    f"than upper_bound ({self.upper_bound})."
                )

        return self

    def is_exceeded(self, value: float) -> bool:
        """
        Check if an observed numeric value exceeds this limit definition.

        Returns True if the value violates the limit boundary:
        - UPPER_BOUND: value > upper_bound
        - LOWER_BOUND: value < lower_bound
        - BAND: value < lower_bound or value > upper_bound
        """
        if math.isnan(value) or math.isinf(value):
            raise ValueError("Cannot evaluate exceedance for non-finite values (NaN or Inf).")

        if self.limit_type == LimitType.UPPER_BOUND:
            assert self.upper_bound is not None
            return value > self.upper_bound
        elif self.limit_type == LimitType.LOWER_BOUND:
            assert self.lower_bound is not None
            return value < self.lower_bound
        elif self.limit_type == LimitType.BAND:
            assert self.lower_bound is not None and self.upper_bound is not None
            return value < self.lower_bound or value > self.upper_bound
        return False

    def calculate_margin(self, value: float) -> float:
        """
        Calculate the margin between the observed value and the limit boundary.

        Positive margin indicates value is inside the allowable limit.
        Negative margin indicates an exceedance by that magnitude.
        """
        if math.isnan(value) or math.isinf(value):
            raise ValueError("Cannot calculate margin for non-finite values (NaN or Inf).")

        if self.limit_type == LimitType.UPPER_BOUND:
            assert self.upper_bound is not None
            return self.upper_bound - value
        elif self.limit_type == LimitType.LOWER_BOUND:
            assert self.lower_bound is not None
            return value - self.lower_bound
        elif self.limit_type == LimitType.BAND:
            assert self.lower_bound is not None and self.upper_bound is not None
            return min(value - self.lower_bound, self.upper_bound - value)
        raise ValueError(f"Unsupported limit_type '{self.limit_type}'")


class EngineeringLimitProfile(BaseModel):
    """
    Immutable domain model representing a collection of engineering limits for a vehicle profile.

    Enforces deterministic deduplication, scope validation, and monotonicity checks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    limit_profile_id: str = Field(
        ...,
        description="Deterministic, stable identifier for this limit profile.",
    )
    vehicle_id: str = Field(
        ...,
        description="Reference to the stable vehicle identity.",
    )
    profile_id: str = Field(
        ...,
        description="Reference to the parent vehicle profile ID.",
    )
    profile_version: str = Field(
        ...,
        description="Explicit vehicle profile version.",
    )
    name: str = Field(
        default="Engineering Limit Profile",
        description="Human-readable limit profile name.",
    )
    description: str | None = Field(
        default=None,
        description="General description of the limit profile.",
    )
    source_reference: str | None = Field(
        default=None,
        description="Source or revision tracker reference.",
    )
    limits: tuple[ParameterLimitDefinition, ...] = Field(
        default_factory=tuple,
        description="Ordered tuple of parameter limit definitions.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary domain metadata. Must not contain observed telemetry.",
    )

    @field_validator("limit_profile_id", "vehicle_id", "profile_id")
    @classmethod
    def _validate_ids(cls, v: str, info: ValidationInfo) -> str:
        field_name = info.field_name if info.field_name else "identifier"
        return validate_domain_identifier(v, field_name)

    @field_validator("profile_version")
    @classmethod
    def _validate_profile_version(cls, v: str) -> str:
        return validate_profile_version(v, "profile_version")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("name cannot be empty or solely whitespace.")
        return v_stripped

    @field_validator("source_reference")
    @classmethod
    def _validate_source_reference(cls, v: str | None) -> str | None:
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                raise ValueError("source_reference cannot be empty or solely whitespace.")
            return v_stripped
        return None

    @model_validator(mode="after")
    def _validate_limits_collection(self) -> EngineeringLimitProfile:
        seen_ids: set[str] = set()
        seen_scopes: set[tuple[str, str | None, LimitType, LimitSeverity]] = set()
        grouped: dict[tuple[str, str | None], list[ParameterLimitDefinition]] = defaultdict(list)

        for limit in self.limits:
            # 1. Uniqueness of limit_id
            if limit.limit_id in seen_ids:
                raise DuplicateLimitError(
                    f"Duplicate limit_id found in profile: '{limit.limit_id}'."
                )
            seen_ids.add(limit.limit_id)

            # 2. Strict alignment with profile vehicle_id, profile_id, profile_version
            if limit.vehicle_id != self.vehicle_id:
                raise LimitProfileMismatchError(
                    f"Limit '{limit.limit_id}' vehicle_id '{limit.vehicle_id}' "
                    f"does not match profile vehicle_id '{self.vehicle_id}'."
                )
            if limit.profile_id != self.profile_id:
                raise LimitProfileMismatchError(
                    f"Limit '{limit.limit_id}' profile_id '{limit.profile_id}' "
                    f"does not match profile profile_id '{self.profile_id}'."
                )
            if limit.profile_version != self.profile_version:
                raise LimitProfileMismatchError(
                    f"Limit '{limit.limit_id}' profile_version '{limit.profile_version}' "
                    f"does not match profile profile_version '{self.profile_version}'."
                )

            # 3. Duplicate applicability scope check
            scope_key = (
                limit.parameter_id,
                limit.configuration_id,
                limit.limit_type,
                limit.severity,
            )
            if scope_key in seen_scopes:
                raise ConflictingLimitError(
                    f"Duplicate limit definition for parameter '{limit.parameter_id}', "
                    f"configuration '{limit.configuration_id}', limit_type '{limit.limit_type}', "
                    f"severity '{limit.severity}'."
                )
            seen_scopes.add(scope_key)

            grouped[(limit.parameter_id, limit.configuration_id)].append(limit)

        # 4. Check semantic conflicts within the same parameter and configuration scope
        for (param, config), group in grouped.items():
            # Disallow multiple BAND limits in same scope
            bands = [lim for lim in group if lim.limit_type == LimitType.BAND]
            if len(bands) > 1:
                raise ConflictingLimitError(
                    f"Multiple overlapping BAND limits defined for parameter '{param}' "
                    f"and configuration '{config}'."
                )

            # Check UPPER_BOUND monotonicity (Caution upper bound <= Warning upper bound)
            cautions_upper = [
                lim
                for lim in group
                if lim.limit_type == LimitType.UPPER_BOUND and lim.severity == LimitSeverity.CAUTION
            ]
            warnings_upper = [
                lim
                for lim in group
                if lim.limit_type == LimitType.UPPER_BOUND and lim.severity == LimitSeverity.WARNING
            ]
            if cautions_upper and warnings_upper:
                c_val = cautions_upper[0].upper_bound
                w_val = warnings_upper[0].upper_bound
                assert c_val is not None and w_val is not None
                if c_val > w_val:
                    raise ConflictingLimitError(
                        f"Contradictory upper bounds for parameter '{param}': "
                        f"CAUTION upper bound ({c_val}) cannot exceed "
                        f"WARNING upper bound ({w_val})."
                    )

            # Check LOWER_BOUND monotonicity (Caution lower bound >= Warning lower bound)
            cautions_lower = [
                lim
                for lim in group
                if lim.limit_type == LimitType.LOWER_BOUND and lim.severity == LimitSeverity.CAUTION
            ]
            warnings_lower = [
                lim
                for lim in group
                if lim.limit_type == LimitType.LOWER_BOUND and lim.severity == LimitSeverity.WARNING
            ]
            if cautions_lower and warnings_lower:
                c_val = cautions_lower[0].lower_bound
                w_val = warnings_lower[0].lower_bound
                assert c_val is not None and w_val is not None
                if c_val < w_val:
                    raise ConflictingLimitError(
                        f"Contradictory lower bounds for parameter '{param}': "
                        f"CAUTION lower bound ({c_val}) cannot be below "
                        f"WARNING lower bound ({w_val})."
                    )

        return self

    def validate_vehicle_profile_compatibility(self, profile: VehicleProfile) -> None:
        """
        Validate that this limit profile is strictly compatible with the provided VehicleProfile.

        Raises
        ------
        LimitProfileMismatchError
            If vehicle_id, profile_id, or profile_version do not exactly match.
        """
        if self.vehicle_id != profile.vehicle_id:
            raise LimitProfileMismatchError(
                f"Limit profile vehicle_id '{self.vehicle_id}' does not match "
                f"vehicle profile vehicle_id '{profile.vehicle_id}'."
            )
        if self.profile_id != profile.profile_id:
            raise LimitProfileMismatchError(
                f"Limit profile profile_id '{self.profile_id}' does not match "
                f"vehicle profile profile_id '{profile.profile_id}'."
            )
        if self.profile_version != profile.profile_version:
            raise LimitProfileMismatchError(
                f"Limit profile profile_version '{self.profile_version}' does not match "
                f"vehicle profile profile_version '{profile.profile_version}'."
            )

    def validate_registry_compatibility(self, registry: ParameterRegistry) -> None:
        """
        Validate that all parameter references and units match the provided ParameterRegistry.

        Raises
        ------
        UnknownParameterLimitError
            If any limit references a parameter_id not registered in the registry.
        IncompatibleLimitUnitError
            If any limit specifies a unit incompatible with the registry's canonical_unit.
        """
        for limit in self.limits:
            if not registry.contains(limit.parameter_id):
                raise UnknownParameterLimitError(
                    f"Limit '{limit.limit_id}' references unknown parameter '{limit.parameter_id}' "
                    f"not found in ParameterRegistry."
                )

            param_def = registry.get(limit.parameter_id)
            if param_def.canonical_unit is not None and limit.unit != param_def.canonical_unit:
                raise IncompatibleLimitUnitError(
                    f"Limit '{limit.limit_id}' declares unit '{limit.unit}', but parameter "
                    f"'{limit.parameter_id}' in registry has canonical_unit "
                    f"'{param_def.canonical_unit}'."
                )

    def get_limits_for_parameter(
        self,
        parameter_id: str,
        configuration_id: str | None = None,
        include_all_configurations: bool = False,
    ) -> tuple[ParameterLimitDefinition, ...]:
        """
        Retrieve all applicable limits for a given parameter and optional configuration ID.

        If include_all_configurations is True, returns all limits for the parameter regardless
        of configuration. Otherwise, returns limits matching the specified configuration_id
        plus baseline limits where configuration_id is None.
        """
        if include_all_configurations:
            return tuple(lim for lim in self.limits if lim.parameter_id == parameter_id)

        matching = [
            lim
            for lim in self.limits
            if lim.parameter_id == parameter_id
            and (lim.configuration_id == configuration_id or lim.configuration_id is None)
        ]
        return tuple(matching)
