from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord


class HypothesisLifecycleStatus(StrEnum):
    """The lifecycle state of a hypothesis in an investigation."""

    OPEN = "OPEN"
    UNDER_INVESTIGATION = "UNDER_INVESTIGATION"
    CLOSED = "CLOSED"


class HypothesisAssessmentState(StrEnum):
    """The analytical assessment of the hypothesis based on evidence."""

    UNASSESSED = "UNASSESSED"
    SUPPORTED = "SUPPORTED"
    WEAKLY_SUPPORTED = "WEAKLY_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"
    CONTRADICTED = "CONTRADICTED"
    REJECTED = "REJECTED"


class HypothesisEvidenceRelationship(StrEnum):
    """How an explicitly referenced evidence item relates to a hypothesis."""

    SUPPORTING = "SUPPORTING"
    CONTRADICTING = "CONTRADICTING"
    RELEVANT = "RELEVANT"
    NEUTRAL = "NEUTRAL"


class HypothesisContributorRelationship(StrEnum):
    """How a contributor (factor) relates to an explicitly stated hypothesis."""

    RELEVANT_FACTOR = "RELEVANT_FACTOR"
    ASSOCIATED_FACTOR = "ASSOCIATED_FACTOR"
    INVESTIGATION_FACTOR = "INVESTIGATION_FACTOR"
    CANDIDATE_FACTOR = "CANDIDATE_FACTOR"


class HypothesisEvidenceLink(BaseModel):
    """A stable, immutable reference between a Hypothesis and an EvidenceRecord."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    link_id: str = Field(..., min_length=1)
    hypothesis_id: str = Field(..., min_length=1)
    evidence_id: str = Field(..., min_length=1)
    relationship_type: HypothesisEvidenceRelationship
    provenance: ProvenanceRecord
    notes: str = ""


class HypothesisContributorLink(BaseModel):
    """A stable, immutable reference between a Hypothesis and a Contributor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    link_id: str = Field(..., min_length=1)
    hypothesis_id: str = Field(..., min_length=1)
    contributor_id: str = Field(..., min_length=1)
    relationship_type: HypothesisContributorRelationship
    provenance: ProvenanceRecord
    notes: str = ""


def _validate_metadata_safeguards(v: dict[str, Any] | Mapping[str, Any]) -> Mapping[str, Any]:
    """Ensures metadata bypasses do not inject unauthorized epistemic or causal statements."""
    if not isinstance(v, (dict, MappingProxyType)):
        v = dict(v) if v else {}

    blocked_keywords = {
        "cause",
        "causal",
        "root_cause",
        "responsible",
        "recommendation",
        "corrective",
        "probability",
        "confidence",
        "likelihood",
        "diagnosis",
        "certainty",
    }

    for key in v:
        k_lower = key.lower()
        if any(bad in k_lower for bad in blocked_keywords):
            raise ValueError(
                "ERR_METADATA_SAFEGUARD: Hypothesis metadata must not contain causal, "
                f"recommendation, or confidence claims. Invalid key found: '{key}'"
            )
    return MappingProxyType(dict(v))


class Hypothesis(BaseModel):
    """
    An explicit, structured engineering proposition.

    A hypothesis is an explicit statement to be investigated against evidence.
    It is NOT an established fact, a causal conclusion, or a recommendation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    statement: str = Field(..., min_length=1)
    description: str = ""
    status: HypothesisLifecycleStatus = HypothesisLifecycleStatus.OPEN
    assessment_state: HypothesisAssessmentState = HypothesisAssessmentState.UNASSESSED
    epistemic_level: EpistemicLevel = EpistemicLevel.HYPOTHESIS
    provenance: ProvenanceRecord
    evidence_links: tuple[HypothesisEvidenceLink, ...] = Field(default_factory=tuple)
    contributor_links: tuple[HypothesisContributorLink, ...] = Field(default_factory=tuple)
    investigation_id: str | None = Field(default=None, min_length=1)
    version: str | None = None
    supersedes_hypothesis_id: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("epistemic_level")
    @classmethod
    def validate_epistemic_level(cls, v: EpistemicLevel) -> EpistemicLevel:
        if v != EpistemicLevel.HYPOTHESIS:
            raise ValueError(
                "ERR_EPISTEMIC_MISMATCH: A Hypothesis must have EpistemicLevel.HYPOTHESIS. "
                f"Got: {v.name}"
            )
        return v

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "ERR_INVALID_STATEMENT: Hypothesis statement cannot be blank or whitespace-only."
            )
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        return _validate_metadata_safeguards(v)


class HypothesisCollection(BaseModel):
    """An immutable collection of hypotheses, preventing duplicates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypotheses: tuple[Hypothesis, ...]
    investigation_id: str | None = None
    provenance: ProvenanceRecord
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_duplicates(self) -> HypothesisCollection:
        seen = set()
        for h in self.hypotheses:
            if h.hypothesis_id in seen:
                raise ValueError(
                    "ERR_DUPLICATE_HYPOTHESIS: Duplicate hypothesis_id "
                    f"'{h.hypothesis_id}' found in collection."
                )
            seen.add(h.hypothesis_id)
        return self

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        return _validate_metadata_safeguards(v)
