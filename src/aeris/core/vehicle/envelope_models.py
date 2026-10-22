"""
Configuration Envelope domain models.

Defines the ConfigurationEnvelope model, which links an explicit configuration
to a parent VehicleProfile and profile version, validating compatibility and
preserving configuration distinctness without assuming equivalence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from aeris.core.vehicle.exceptions import ProfileMismatchError
from aeris.core.vehicle.identifiers import validate_domain_identifier
from aeris.core.vehicle.profile_models import VehicleProfile, validate_profile_version


class ConfigurationEnvelope(BaseModel):
    """
    Immutable domain model representing a validated configuration envelope.

    Binds a specific configuration state to an authoritative vehicle profile
    and version, preventing silent cross-configuration equivalence or profile mismatches.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    configuration_id: str = Field(
        ...,
        description="Deterministic, stable identifier for this specific configuration state.",
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
        description="Explicit profile version required or targeted by this configuration.",
    )
    configuration_version: str | None = Field(
        default=None,
        description="Optional revision/version of this specific configuration envelope.",
    )
    name: str | None = Field(
        default=None,
        description="Human-readable configuration name.",
    )
    description: str | None = Field(
        default=None,
        description="General description of the configuration envelope.",
    )
    source_reference: str | None = Field(
        default=None,
        description="Document or engineering source reference.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary domain metadata. Must not contain observed telemetry.",
    )

    @field_validator("configuration_id", "vehicle_id", "profile_id")
    @classmethod
    def _validate_ids(cls, v: str, info: ValidationInfo) -> str:
        field_name = info.field_name if info.field_name else "identifier"
        return validate_domain_identifier(v, field_name)

    @field_validator("profile_version")
    @classmethod
    def _validate_profile_ver(cls, v: str) -> str:
        return validate_profile_version(v, "profile_version")

    @field_validator("configuration_version")
    @classmethod
    def _validate_config_ver(cls, v: str | None) -> str | None:
        if v is not None:
            return validate_profile_version(v, "configuration_version")
        return None

    @field_validator("name", "source_reference")
    @classmethod
    def _validate_non_empty_strings(cls, v: str | None, info: ValidationInfo) -> str | None:
        field_name = info.field_name if info.field_name else "field"
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                raise ValueError(f"{field_name} cannot be empty or solely whitespace.")
            return v_stripped
        return None

    def validate_profile_compatibility(self, profile: VehicleProfile) -> None:
        """
        Validate that this configuration envelope is strictly compatible with
        the provided VehicleProfile.

        Raises
        ------
        ProfileMismatchError
            If vehicle_id, profile_id, or profile_version do not exactly match.
        """
        if self.vehicle_id != profile.vehicle_id:
            raise ProfileMismatchError(
                f"Configuration envelope vehicle_id '{self.vehicle_id}' does not match "
                f"profile vehicle_id '{profile.vehicle_id}'."
            )
        if self.profile_id != profile.profile_id:
            raise ProfileMismatchError(
                f"Configuration envelope profile_id '{self.profile_id}' does not match "
                f"profile profile_id '{profile.profile_id}'."
            )
        if self.profile_version != profile.profile_version:
            raise ProfileMismatchError(
                f"Configuration envelope profile_version '{self.profile_version}' does not match "
                f"profile profile_version '{profile.profile_version}'."
            )
