"""
AERIS Provenance Tracking and Epistemic Level Definitions.

Records data lineage, processing activities, source files, and epistemic
standing for all pipeline outputs.

DISCLAIMER:
This implementation provides execution-level traceability (source URIs, file
checksums, timestamps, parameters, and activity metadata). It does NOT claim
or provide regulatory or cryptographic proof of custody.
"""

from __future__ import annotations

import hashlib
import pathlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Epistemic Level Definitions & Enforcement
# ---------------------------------------------------------------------------


class EpistemicIntegrityError(ValueError):
    """Raised when an interpretive or hypothetical result is improperly treated as fact."""


class EpistemicLevel(StrEnum):
    """AERIS evidence hierarchy defining the epistemic standing of all platform data.

    Maintains the strict AERIS distinction:
        RAW DATA -> VALIDATED DATA -> CALCULATED RESULT -> OBSERVATION
        -> INFERENCE -> HYPOTHESIS -> RECOMMENDATION

    An inference or hypothesis must never be presented as an established fact.
    """

    OBSERVED = "OBSERVED"
    """Direct empirical evidence captured by sensors or data acquisition systems."""

    CALCULATED = "CALCULATED"
    """Derived mathematically from validated observations via deterministic algorithms."""

    PROCEDURAL = "PROCEDURAL"
    """Prescriptive requirements from SOPs, flight test cards, or standard procedures."""

    LIMIT = "LIMIT"
    """Established operational, structural, aerodynamic, or engine limitation boundaries."""

    HISTORICAL = "HISTORICAL"
    """Prior validated flight test baselines, fleet data, or reference benchmarks."""

    INFERRED = "INFERRED"
    """Algorithmic or statistical deduction of behavior/anomalies (not directly observed)."""

    HYPOTHESIS = "HYPOTHESIS"
    """Candidate explanation or potential root cause requiring supporting evidence."""

    RECOMMENDATION = "RECOMMENDATION"
    """Engineering decision-support guidance synthesized from evidence and investigation."""

    def is_empirical(self) -> bool:
        """Return True if this level represents direct or mathematically calculated measurements."""
        return self in (EpistemicLevel.OBSERVED, EpistemicLevel.CALCULATED)

    def is_factual(self) -> bool:
        """Return True if this level represents established ground-truth or historical fact."""
        return self in (
            EpistemicLevel.OBSERVED,
            EpistemicLevel.CALCULATED,
            EpistemicLevel.HISTORICAL,
        )

    def is_interpretive(self) -> bool:
        """Return True if this level represents deductions, hypotheses, or recommendations."""
        return self in (
            EpistemicLevel.INFERRED,
            EpistemicLevel.HYPOTHESIS,
            EpistemicLevel.RECOMMENDATION,
        )

    def is_prescriptive(self) -> bool:
        """Return True if this level represents procedures, limits, or envelope requirements."""
        return self in (EpistemicLevel.PROCEDURAL, EpistemicLevel.LIMIT)

    def assert_factual(self, context: str = "") -> None:
        """Raise EpistemicIntegrityError if this level is interpretive rather than factual."""
        if self.is_interpretive():
            msg = (
                f"Epistemic violation: cannot treat interpretive level '{self.value}' "
                f"as an established fact"
            )
            if context:
                msg += f" in context: {context}"
            raise EpistemicIntegrityError(msg)


# ---------------------------------------------------------------------------
# Provenance Record Model & Utilities
# ---------------------------------------------------------------------------


class ProvenanceRecord(BaseModel):
    """Structured record of lineage, execution context, and epistemic standing.

    Captures where a result came from, what operation produced it, the parameters
    used, and upstream parent lineage links.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this provenance record (UUIDv4).",
    )
    source_uri: str = Field(
        description="Source file path, dataset URI, or upstream data origin.",
    )
    activity_name: str = Field(
        description="Name of the process, validator, or engine that produced the result.",
    )
    epistemic_level: EpistemicLevel = Field(
        description="Epistemic categorization of the result.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of result generation.",
    )
    source_checksum: str | None = Field(
        default=None,
        description="SHA-256 hex digest of the source file if applicable.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters or runtime arguments used during generation.",
    )
    parent_record_ids: list[str] = Field(
        default_factory=list,
        description="List of upstream provenance record IDs for traceability.",
    )


def compute_file_checksum(file_path: pathlib.Path) -> str:
    """Compute the SHA-256 hexadecimal digest of a file.

    Raises FileNotFoundError if file_path does not exist.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot compute checksum; file not found: {file_path}")

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_provenance(
    source_uri: str,
    activity_name: str,
    epistemic_level: EpistemicLevel,
    *,
    source_checksum: str | None = None,
    parameters: dict[str, Any] | None = None,
    parent_record_ids: list[str] | None = None,
) -> ProvenanceRecord:
    """Factory helper to construct a ProvenanceRecord with validated defaults."""
    return ProvenanceRecord(
        source_uri=source_uri,
        activity_name=activity_name,
        epistemic_level=epistemic_level,
        source_checksum=source_checksum,
        parameters=parameters or {},
        parent_record_ids=parent_record_ids or [],
    )
