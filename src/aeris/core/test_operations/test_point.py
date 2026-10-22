"""
Test Point domain model.

Represents an explicitly defined point or event in a flight-test activity.
This is purely a definition/context object. It strictly excludes measured
telemetry, engineering calculations, analysis results, and performance limits.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from aeris.core.vehicle.identifiers import validate_domain_identifier


class TestPoint(BaseModel):
    """
    Immutable domain model defining an explicit flight-test point.

    This model serves as the definitional metadata for a specific maneuver
    or test condition. It mathematically rejects operational telemetry,
    engineering boundaries, and observed results to enforce domain separation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    __test__ = False

    test_point_id: str = Field(
        ...,
        description="Deterministic, stable identifier for the test point.",
    )
    context_id: str = Field(
        ...,
        description="Explicit reference to the test context (vehicle + configuration).",
    )
    name: str = Field(
        ...,
        description="Name or short designation of the test point.",
    )
    description: str | None = Field(
        default=None,
        description="General description of the test point event.",
    )
    point_type: str | None = Field(
        default=None,
        description="Descriptive classification metadata for this test point.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary descriptive metadata. Must not contain observed telemetry.",
    )

    @field_validator("test_point_id", "context_id")
    @classmethod
    def _validate_identifiers(cls, v: str, info: ValidationInfo) -> str:
        """Enforce strict formatting for stable identifiers."""
        field_name = info.field_name if info.field_name else "identifier"
        return validate_domain_identifier(v, field_name)
