"""
AERIS Controlled Engineering Recommendation Domain Foundation.

Defines the immutable domain models representing bounded engineering
investigation recommendations resulting from post-flight evidence,
contributor analysis, and hypothesis assessment.

A recommendation is an explicit, auditable proposal for an investigation
or verification activity (e.g., verify data, review SOP, repeat calculation).
It does NOT assert causation, determine root cause, authorize aircraft flight,
issue maintenance releases, or execute operational commands.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord


class RecommendationType(StrEnum):
    """Controlled categories describing investigation or verification activities."""

    VERIFY_DATA = "VERIFY_DATA"
    REVIEW_EVIDENCE = "REVIEW_EVIDENCE"
    REPEAT_ANALYSIS = "REPEAT_ANALYSIS"
    CHECK_CONFIGURATION_RECORD = "CHECK_CONFIGURATION_RECORD"
    REVIEW_TEST_PROCEDURE = "REVIEW_TEST_PROCEDURE"
    REQUEST_ENGINEERING_REVIEW = "REQUEST_ENGINEERING_REVIEW"
    COLLECT_ADDITIONAL_EVIDENCE = "COLLECT_ADDITIONAL_EVIDENCE"
    OTHER_INVESTIGATION = "OTHER_INVESTIGATION"


class RecommendationStatus(StrEnum):
    """Controlled lifecycle status of an engineering investigation recommendation."""

    PROPOSED = "PROPOSED"
    UNDER_REVIEW = "UNDER_REVIEW"
    ACCEPTED_FOR_INVESTIGATION = "ACCEPTED_FOR_INVESTIGATION"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"
    SUPERSEDED = "SUPERSEDED"


def _validate_metadata_safeguards(v: dict[str, Any] | Mapping[str, Any]) -> Mapping[str, Any]:
    """Ensures metadata bypasses do not inject unauthorized epistemic or causal statements."""
    if not isinstance(v, (dict, MappingProxyType)):
        v = dict(v) if v else {}

    blocked_keywords = {
        "cause",
        "causal",
        "root_cause",
        "responsible",
        "probability",
        "confidence",
        "likelihood",
        "certainty",
        "confirmed_root_cause",
        "causal_probability",
        "most_likely_cause",
        "cause_confidence",
        "failure_probability",
        "airworthiness",
        "authorization",
        "flight_clearance",
        "dispatch",
        "control_command",
    }

    for key in v:
        k_lower = key.lower()
        if any(bad in k_lower for bad in blocked_keywords):
            raise ValueError(
                "ERR_METADATA_SAFEGUARD: Recommendation metadata must not contain causal, "
                "confidence, root-cause, or operational authorization claims. "
                f"Invalid key found: '{key}'"
            )
    return MappingProxyType(dict(v))


def _validate_string_tuple(field_name: str, v: Any) -> tuple[str, ...]:
    """Ensures reference sequences are immutable tuples of unique, non-empty strings."""
    if v is None:
        return ()
    items = tuple(v) if isinstance(v, (list, tuple, set)) else (v,)
    seen = set()
    deduped = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"ERR_INVALID_REFERENCE: {field_name} must contain "
                "non-empty, non-whitespace strings."
            )
        if item in seen:
            raise ValueError(
                f"ERR_DUPLICATE_REFERENCE: {field_name} cannot contain "
                f"duplicate reference: '{item}'"
            )
        seen.add(item)
        deduped.append(item)
    return tuple(deduped)


class Recommendation(BaseModel):
    """
    An immutable, bounded engineering recommendation for post-flight investigation.

    A recommendation represents a structured proposal for verification or further
    analysis. It must be explicitly traceable to upstream evidence, contributor
    factors, and hypothesis assessments.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    recommendation_id: str = Field(..., min_length=1)
    recommendation_type: RecommendationType
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    status: RecommendationStatus = RecommendationStatus.PROPOSED
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    hypothesis_ids: tuple[str, ...] = Field(default_factory=tuple)
    hypothesis_assessment_ids: tuple[str, ...] = Field(default_factory=tuple)
    other_assessment_ids: tuple[str, ...] = Field(default_factory=tuple)
    constraints: tuple[str, ...] = Field(default_factory=tuple)
    prerequisites: tuple[str, ...] = Field(default_factory=tuple)
    epistemic_level: EpistemicLevel = EpistemicLevel.RECOMMENDATION
    provenance: ProvenanceRecord
    investigation_id: str | None = Field(default=None, min_length=1)
    supersedes_recommendation_id: str | None = Field(default=None, min_length=1)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("recommendation_id", "title", "description", "rationale")
    @classmethod
    def validate_non_whitespace_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("String field cannot be empty or whitespace-only.")
        return v

    @field_validator("investigation_id", "supersedes_recommendation_id")
    @classmethod
    def validate_optional_non_whitespace(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("Optional identifier cannot be empty or whitespace-only.")
        return v

    @field_validator("epistemic_level")
    @classmethod
    def validate_epistemic_level(cls, v: EpistemicLevel) -> EpistemicLevel:
        if v != EpistemicLevel.RECOMMENDATION:
            raise ValueError(
                "ERR_EPISTEMIC_MISMATCH: A Recommendation must have EpistemicLevel.RECOMMENDATION. "
                f"Got: {v.name}"
            )
        return v

    @field_validator(
        "evidence_ids",
        "hypothesis_ids",
        "hypothesis_assessment_ids",
        "other_assessment_ids",
        "constraints",
        "prerequisites",
        mode="before",
    )
    @classmethod
    def validate_references(cls, v: Any, info: Any) -> tuple[str, ...]:
        field_name = info.field_name if hasattr(info, "field_name") else "references"
        return _validate_string_tuple(field_name, v)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        return _validate_metadata_safeguards(v)


class RecommendationCollection(BaseModel):
    """An immutable collection of recommendations, preventing duplicate identifiers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: str = Field(..., min_length=1)
    recommendations: tuple[Recommendation, ...] = Field(default_factory=tuple)
    investigation_id: str | None = Field(default=None, min_length=1)
    provenance: ProvenanceRecord
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Collection identifier cannot be empty or whitespace-only.")
        return v

    @field_validator("investigation_id")
    @classmethod
    def validate_optional_investigation_id(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("Investigation identifier cannot be empty or whitespace-only.")
        return v

    @model_validator(mode="after")
    def validate_duplicates(self) -> RecommendationCollection:
        seen = set()
        for r in self.recommendations:
            if r.recommendation_id in seen:
                raise ValueError(
                    "ERR_DUPLICATE_RECOMMENDATION: Duplicate recommendation_id "
                    f"'{r.recommendation_id}' found in collection."
                )
            seen.add(r.recommendation_id)
        return self

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        return _validate_metadata_safeguards(v)
