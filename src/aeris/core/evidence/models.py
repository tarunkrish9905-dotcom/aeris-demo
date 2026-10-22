import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord


class EvidenceType(StrEnum):
    """Categories describing what kind of source the evidence represents."""

    OBSERVATION = "OBSERVATION"
    CALCULATION = "CALCULATION"
    STATISTICAL_RESULT = "STATISTICAL_RESULT"
    ANOMALY = "ANOMALY"
    EXPECTED_OBSERVED_COMPARISON = "EXPECTED_OBSERVED_COMPARISON"
    PROCEDURAL_REFERENCE = "PROCEDURAL_REFERENCE"
    LIMIT_REFERENCE = "LIMIT_REFERENCE"
    DATA_QUALITY = "DATA_QUALITY"
    OTHER_EXPLICIT_SOURCE = "OTHER_EXPLICIT_SOURCE"
    CONFIGURATION_DISCREPANCY = "CONFIGURATION_DISCREPANCY"


class EvidenceRecord(BaseModel):
    """A structured representation of an existing piece of traceable information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(min_length=1)
    evidence_type: EvidenceType
    description: str = Field(min_length=1)

    # Traceability fields
    source_reference: str = Field(min_length=1)
    source_record_id: str | None = None
    source_parameter_id: str | None = None
    source_row_index: int | None = None
    timestamp: datetime.datetime | None = None
    unit: str | None = None

    # Core domain metadata
    epistemic_level: EpistemicLevel
    provenance: ProvenanceRecord

    # Vehicle and evaluation lineage fields
    evaluation_id: str | None = None
    vehicle_id: str | None = None
    profile_id: str | None = None
    profile_version: str | None = None
    configuration_id: str | None = None
    limit_profile_id: str | None = None
    limit_id: str | None = None
    document_id: str | None = None
    document_version: str | None = None
    requirement_id: str | None = None
    evaluation_outcome: str | None = None
    observed_value: float | None = None
    limit_value: float | None = None
    margin: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime.datetime | None) -> datetime.datetime | None:
        if v is not None:
            if v.tzinfo is None:
                raise ValueError("Timestamp must be timezone-aware.")
            if v.utcoffset() != datetime.timedelta(0):
                raise ValueError("Timestamp must be UTC.")
        return v


class EvidenceCollection(BaseModel):
    """An immutable collection of distinct evidence records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: str = Field(min_length=1)
    evidence: tuple[EvidenceRecord, ...]

    @model_validator(mode="after")
    def validate_unique_evidence(self) -> "EvidenceCollection":
        evidence_ids = [e.evidence_id for e in self.evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("ERR_DUPLICATE_EVIDENCE_ID")
        return self
