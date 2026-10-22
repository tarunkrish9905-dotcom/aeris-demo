"""
Test Card domain model.

Represents an explicitly defined flight-test activity/test card.
This is an operational definition object. It mathematically forbids telemetry,
test results, physical engineering calculations, and timestamp associations
to preserve absolute separation of concerns.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from aeris.core.vehicle.identifiers import validate_domain_identifier


class TestCard(BaseModel):
    """
    Immutable domain model defining an explicit flight-test card.

    This model serves as the definitional metadata for a planned test event,
    organizing a set of explicit Test Points. It rejects operational telemetry,
    engineering boundaries, and observed results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    __test__ = False

    test_card_id: str = Field(
        ...,
        description="Deterministic, stable identifier for the test card.",
    )
    context_id: str = Field(
        ...,
        description="Explicit reference to the test context (vehicle + configuration).",
    )
    name: str = Field(
        ...,
        description="Name or short designation of the test card.",
    )
    description: str | None = Field(
        default=None,
        description="General description of the test card.",
    )
    objective: str | None = Field(
        default=None,
        description="Test objective statement.",
    )
    test_point_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Ordered collection of explicitly referenced Test Point IDs.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary descriptive metadata. Must not contain observed telemetry.",
    )

    @field_validator("test_card_id", "context_id")
    @classmethod
    def _validate_identifiers(cls, v: str, info: ValidationInfo) -> str:
        """Enforce strict formatting for stable identifiers."""
        field_name = info.field_name if info.field_name else "identifier"
        return validate_domain_identifier(v, field_name)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("name cannot be empty or solely whitespace.")
        return v_stripped

    @field_validator("test_point_ids")
    @classmethod
    def _validate_test_point_ids(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        validated_ids: list[str] = []
        for tp_id in v:
            clean_id = validate_domain_identifier(tp_id, "test_point_id")
            if clean_id in seen:
                raise ValueError(f"Duplicate test_point_id found: {clean_id}")
            seen.add(clean_id)
            validated_ids.append(clean_id)
        return tuple(validated_ids)
