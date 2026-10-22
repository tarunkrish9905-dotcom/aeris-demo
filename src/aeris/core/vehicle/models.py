"""
Vehicle identity and configuration models.

These models represent the aircraft/vehicle context. They are explicitly
supplied configuration and are strictly segregated from observed flight
data (telemetry) and physical engineering computations.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Vehicle(BaseModel):
    """
    Foundational domain model representing a vehicle/aircraft's identity
    and configuration context.

    This model is immutable after creation and prohibits the inclusion of
    unregistered/extra fields (e.g., speculative performance limits or
    telemetry metrics) to maintain strict domain separation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vehicle_id: str = Field(
        ...,
        description="Deterministic, stable identifier for the vehicle.",
    )
    name: str | None = Field(
        default=None,
        description="Common name or designation.",
    )
    manufacturer: str | None = Field(
        default=None,
        description="Manufacturer name.",
    )
    model: str | None = Field(
        default=None,
        description="Model designation.",
    )
    variant: str | None = Field(
        default=None,
        description="Specific variant designation.",
    )
    configuration_identifier: str | None = Field(
        default=None,
        description="Specific configuration state or string.",
    )
    description: str | None = Field(
        default=None,
        description="General description of the vehicle.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary domain metadata. Must not contain observed telemetry.",
    )

    @field_validator("vehicle_id")
    @classmethod
    def _validate_vehicle_id(cls, v: str) -> str:
        """Enforce strict formatting for stable vehicle identifiers."""
        from aeris.core.vehicle.identifiers import validate_domain_identifier

        return validate_domain_identifier(v, "vehicle_id")
