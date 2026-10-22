"""
Vehicle Configuration domain models.

These models explicitly define the configuration states of a vehicle
during a test or flight. They are completely separated from telemetry,
engineering calculation logic, and automatic test point inference.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VehicleConfiguration(BaseModel):
    """
    Foundational domain model representing a specific, explicit configuration
    state of a vehicle.

    This model is frozen (immutable) after creation. It forbids extra fields
    such as performance limits (e.g. max_speed_kts, stall_speed) to enforce
    strict separation between configuration metadata and future engineering/
    performance domains.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    configuration_id: str = Field(
        ...,
        description="Deterministic, stable identifier for this specific configuration.",
    )
    vehicle_id: str | None = Field(
        default=None,
        description="Optional explicit reference to the parent vehicle's stable identifier.",
    )
    name: str | None = Field(
        default=None,
        description="Common name or human-readable designation for the configuration.",
    )
    description: str | None = Field(
        default=None,
        description="General description of the configuration state.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary configuration metadata. Must not contain observed telemetry.",
    )

    @field_validator("configuration_id")
    @classmethod
    def _validate_configuration_id(cls, v: str) -> str:
        """Enforce strict formatting for stable configuration identifiers."""
        from aeris.core.vehicle.identifiers import validate_domain_identifier

        return validate_domain_identifier(v, "configuration_id")
