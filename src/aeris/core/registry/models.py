"""
AERIS Parameter Registry Models.

Defines the immutable semantic definitions for telemetry and flight-test parameters.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParameterDataType(StrEnum):
    """Controlled representation of expected parameter data types."""

    NUMERIC = "NUMERIC"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    STRING = "STRING"
    TIMESTAMP = "TIMESTAMP"


class ParameterDefinition(BaseModel):
    """Immutable semantic definition of a flight-test or telemetry parameter.

    This definition establishes identity, meaning, and expected types, but does
    NOT normalize data automatically or prescribe aircraft-specific limits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_id: str = Field(
        description="Stable deterministic identifier for the parameter (e.g., 'airspeed').",
        min_length=1,
    )
    display_name: str = Field(
        description="Human-readable name for UI and reports.",
        min_length=1,
    )
    description: str = Field(
        description="Detailed semantic definition of the parameter.",
    )
    physical_quantity: str = Field(
        description="The category of physical measurement (e.g., 'velocity', 'pressure').",
    )
    data_type: ParameterDataType = Field(
        description="The expected foundational data type for this parameter.",
    )
    canonical_unit: str | None = Field(
        default=None,
        description="The standard unit of measurement, provided purely as metadata.",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names or source string identifiers (e.g., legacy column names).",
    )
    source_type: str | None = Field(
        default=None,
        description="Category of the source (e.g., 'sensor', 'calculated', 'bus').",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional auxiliary dictionary for tags, domains, or arbitrary configuration.",
    )
