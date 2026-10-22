"""
Test Point Observation domain model.

Represents an explicit temporal association mapping a TestPoint definition
to an observed temporal region (provenance/temporal bounds).
This model does not contain or slice telemetry data, but instead serves
as the relational bridge pointing to a time window within a data source.
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from aeris.core.vehicle.identifiers import validate_domain_identifier


class TestPointObservation(BaseModel):
    """
    Immutable domain model defining an explicit temporal observation of a Test Point.

    Defines a closed interval `[start_timestamp, end_timestamp]` explicitly mapping
    a defined Test Point to an occurrence in an external data source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    __test__ = False

    observation_id: str = Field(
        ...,
        description="Deterministic, stable identifier for the observation record.",
    )
    test_point_id: str = Field(
        ...,
        description="Reference to the explicit Test Point definition.",
    )
    source_reference: str = Field(
        ...,
        description="Opaque source reference (e.g., filename, run id).",
    )
    start_timestamp: datetime.datetime = Field(
        ...,
        description="Timezone-aware UTC start boundary of the observed interval (inclusive).",
    )
    end_timestamp: datetime.datetime = Field(
        ...,
        description="Timezone-aware UTC end boundary of the observed interval (inclusive).",
    )

    @field_validator("observation_id", "test_point_id")
    @classmethod
    def _validate_identifiers(cls, v: str, info: ValidationInfo) -> str:
        """Enforce strict formatting for stable identifiers."""
        field_name = info.field_name if info.field_name else "identifier"
        return validate_domain_identifier(v, field_name)

    @field_validator("source_reference")
    @classmethod
    def _validate_source_reference(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("source_reference cannot be empty or solely whitespace.")
        return v_stripped

    @field_validator("start_timestamp", "end_timestamp")
    @classmethod
    def _validate_timezone(cls, v: datetime.datetime, info: ValidationInfo) -> datetime.datetime:
        """
        Require strict UTC timezone-awareness.
        Refuses naive datetimes or implicitly converted non-UTC offsets.
        """
        field_name = info.field_name if info.field_name else "timestamp"
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError(f"{field_name} must be timezone-aware.")

        # Must explicitly be UTC (offset == 0)
        offset = v.tzinfo.utcoffset(v)
        if offset is None or offset.total_seconds() != 0:
            raise ValueError(f"{field_name} must explicitly use UTC timezone (offset 0).")
        return v

    @model_validator(mode="after")
    def _validate_temporal_order(self) -> TestPointObservation:
        """Enforce that the interval is structurally logical (start <= end)."""
        if self.start_timestamp > self.end_timestamp:
            raise ValueError("start_timestamp must be less than or equal to end_timestamp.")
        return self
