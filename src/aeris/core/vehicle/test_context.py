"""
Explicit Context domain models.

These models explicitly define the context in which a test operation
is conducted (Vehicle + Configuration context). They strictly
represent metadata and do not include telemetry or performance calculations.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from aeris.core.vehicle.identifiers import validate_domain_identifier


class TestContext(BaseModel):
    """
    Foundational domain model representing the explicit context
    (vehicle and configuration) of a flight or test operation.

    This model is frozen (immutable) after creation and explicitly forbids
    unregistered extra fields (such as speed limits, altitude limits,
    telemetry, etc.) to maintain pure separation of concerns.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    __test__ = False

    context_id: str = Field(
        ...,
        description="Deterministic, stable identifier for the test context.",
    )
    vehicle_id: str = Field(
        ...,
        description="Reference to the stable vehicle identity.",
    )
    configuration_id: str = Field(
        ...,
        description="Reference to the explicit vehicle configuration state.",
    )
    name: str | None = Field(
        default=None,
        description="Common name or human-readable designation for the context.",
    )
    description: str | None = Field(
        default=None,
        description="General description of the test context.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary context metadata. Must not contain observed telemetry.",
    )

    @field_validator("context_id", "vehicle_id", "configuration_id")
    @classmethod
    def _validate_identifiers(cls, v: str, info: ValidationInfo) -> str:
        """Enforce strict formatting for stable identifiers."""
        field_name = info.field_name if info.field_name else "identifier"
        return validate_domain_identifier(v, field_name)
