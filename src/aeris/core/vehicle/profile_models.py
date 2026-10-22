"""
Vehicle Profile domain models.

Provides immutable, versioned domain models defining a vehicle profile.
A VehicleProfile captures the explicit identity, profile version, and supported
parameter references for an aircraft or test vehicle.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from aeris.core.vehicle.identifiers import validate_domain_identifier

# Pattern requiring semantic-style or dotted versioning: e.g. "1.0.0", "v1.2", "2026.1", "1.0"
_VERSION_REGEX = re.compile(r"^v?[0-9]+(\.[0-9A-Za-z_-]+)+$")


def validate_profile_version(v: str, field_name: str = "profile_version") -> str:
    """Enforce non-empty, formatted profile/configuration version strings."""
    v_stripped = v.strip()
    if not v_stripped:
        raise ValueError(f"{field_name} cannot be empty or solely whitespace.")
    if not _VERSION_REGEX.match(v_stripped):
        raise ValueError(
            f"{field_name} must follow dotted version formatting (e.g., '1.0.0', 'v1.0')."
        )
    return v_stripped


class VehicleProfile(BaseModel):
    """
    Immutable domain model representing a versioned vehicle profile.

    Represents explicit configuration and metadata for an aircraft or test vehicle.
    Does NOT contain observed flight telemetry, physical calculations, or operational limits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(
        ...,
        description="Deterministic, stable identifier for this vehicle profile.",
    )
    vehicle_id: str = Field(
        ...,
        description="Reference to the stable vehicle identity.",
    )
    profile_version: str = Field(
        ...,
        description="Explicit semantic/revision version of the profile (e.g. '1.0.0').",
    )
    name: str = Field(
        ...,
        description="Human-readable profile name.",
    )
    description: str | None = Field(
        default=None,
        description="Detailed description of the vehicle profile scope and purpose.",
    )
    source_reference: str | None = Field(
        default=None,
        description="Opaque document reference, engineering change order, or revision tracker.",
    )
    supported_parameter_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Ordered tuple of explicit parameter IDs supported on this vehicle.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary domain metadata. Must not contain observed telemetry.",
    )

    @field_validator("profile_id", "vehicle_id")
    @classmethod
    def _validate_ids(cls, v: str, info: ValidationInfo) -> str:
        field_name = info.field_name if info.field_name else "identifier"
        return validate_domain_identifier(v, field_name)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("name cannot be empty or solely whitespace.")
        return v_stripped

    @field_validator("profile_version")
    @classmethod
    def _validate_version(cls, v: str) -> str:
        return validate_profile_version(v, "profile_version")

    @field_validator("source_reference")
    @classmethod
    def _validate_source_reference(cls, v: str | None) -> str | None:
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                raise ValueError("source_reference cannot be empty or solely whitespace.")
            return v_stripped
        return None

    @field_validator("supported_parameter_ids")
    @classmethod
    def _validate_supported_parameters(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        validated: list[str] = []
        for param_id in v:
            clean_id = validate_domain_identifier(param_id, "supported_parameter_id")
            if clean_id in seen:
                raise ValueError(
                    f"Duplicate parameter_id found in supported_parameter_ids: '{clean_id}'."
                )
            seen.add(clean_id)
            validated.append(clean_id)
        return tuple(validated)
