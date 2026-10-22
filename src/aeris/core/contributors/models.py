from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import BasePipelineResult


class ContributorType(StrEnum):
    """Controlled vocabulary for categories of potential contributing factors."""

    VEHICLE = "VEHICLE"
    CONFIGURATION = "CONFIGURATION"
    ENVIRONMENT = "ENVIRONMENT"
    OPERATING_CONDITION = "OPERATING_CONDITION"
    TEST_PROCEDURE = "TEST_PROCEDURE"
    TEST_EXECUTION = "TEST_EXECUTION"
    HUMAN_FACTOR = "HUMAN_FACTOR"
    SENSOR_OR_INSTRUMENTATION = "SENSOR_OR_INSTRUMENTATION"
    DATA_QUALITY = "DATA_QUALITY"
    PARAMETER = "PARAMETER"
    SYSTEM_STATE = "SYSTEM_STATE"
    HISTORICAL_PATTERN = "HISTORICAL_PATTERN"
    EXTERNAL_CONTEXT = "EXTERNAL_CONTEXT"
    OTHER = "OTHER"


class Contributor(BaseModel):
    """
    Immutable domain representation of a potential contributing factor.

    This model identifies a candidate factor. It DOES NOT assert causality.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contributor_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    contributor_type: ContributorType
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metadata_semantics(self) -> "Contributor":
        """Ensure metadata is not used as a dumping ground for causal conclusions."""
        forbidden_keys = {
            "root_cause",
            "probable_cause",
            "causal_confidence",
            "diagnosis",
            "failure_mode",
            "responsibility",
            "corrective_action",
            "recommendation",
            "conclusion",
            "cause_probability",
            "confidence",
        }
        for k in self.metadata:
            if k in forbidden_keys:
                raise ValueError(
                    f"ERR_CAUSAL_METADATA_FORBIDDEN: Metadata key '{k}' implies "
                    "causality/conclusion, which is not permitted in Contributor representation."
                )
        return self


class RelationshipType(StrEnum):
    """Controlled classification of the non-causal relationship between a factor and evidence."""

    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    TEMPORALLY_ALIGNED_WITH = "TEMPORALLY_ALIGNED_WITH"
    CONTEXTUALLY_RELEVANT = "CONTEXTUALLY_RELEVANT"
    SUPPORTS_CONSIDERATION = "SUPPORTS_CONSIDERATION"
    CONTRADICTS_CONSIDERATION = "CONTRADICTS_CONSIDERATION"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class ContributorAssessmentState(StrEnum):
    """
    Controlled assessment state describing where a factor stands in an investigation.
    These DO NOT represent causal conclusions.
    """

    IDENTIFIED = "IDENTIFIED"
    UNDER_REVIEW = "UNDER_REVIEW"
    SUPPORTED = "SUPPORTED"
    WEAKLY_SUPPORTED = "WEAKLY_SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class RelationshipStrength(StrEnum):
    """Analytical relationship strength, NOT causal probability."""

    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class ContributorEvidenceLink(BaseModel):
    """
    Immutable representation of a non-causal relationship between a factor and specific evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    link_id: str = Field(..., min_length=1)
    contributor_id: str = Field(..., min_length=1)
    evidence_reference: str = Field(..., min_length=1)
    relationship_type: RelationshipType
    description: str = Field(..., min_length=1)
    strength: RelationshipStrength | None = None
    assessment_state: ContributorAssessmentState = Field(
        default=ContributorAssessmentState.IDENTIFIED
    )

    epistemic_level: EpistemicLevel
    provenance: ProvenanceRecord
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_epistemic_integrity(self) -> "ContributorEvidenceLink":
        """
        Ensure the relationship does not silently elevate the epistemic level.
        Also prevents metadata abuse for causality.
        """
        # A relationship itself is an inference or hypothesis.
        if self.epistemic_level in (EpistemicLevel.OBSERVED, EpistemicLevel.CALCULATED):
            raise ValueError(
                f"ERR_INVALID_EPISTEMIC_LEVEL: Contributor link cannot have epistemic "
                f"level '{self.epistemic_level.name}'. Relationships are inherently analytical "
                "inferences, hypotheses, or contextual."
            )

        if self.epistemic_level != self.provenance.epistemic_level:
            raise ValueError(
                "ERR_EPISTEMIC_MISMATCH: Link's epistemic_level must match its "
                "provenance.epistemic_level exactly."
            )

        forbidden_keys = {
            "root_cause",
            "probable_cause",
            "causal_confidence",
            "diagnosis",
            "failure_mode",
            "responsibility",
            "corrective_action",
            "recommendation",
            "conclusion",
            "cause_probability",
            "confidence",
        }
        for k in self.metadata:
            if k in forbidden_keys:
                raise ValueError(
                    f"ERR_CAUSAL_METADATA_FORBIDDEN: Metadata key '{k}' implies "
                    "causality/conclusion."
                )

        return self


class ContributorInvestigation(BaseModel):
    """
    An immutable collection of contributors and their associated evidence links.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    investigation_id: str = Field(..., min_length=1)
    contributors: tuple[Contributor, ...] = Field(default_factory=tuple)
    links: tuple[ContributorEvidenceLink, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_integrity(self) -> "ContributorInvestigation":
        # Ensure collection orders are deterministic and no duplicate IDs
        c_ids = [c.contributor_id for c in self.contributors]
        if len(set(c_ids)) != len(c_ids):
            raise ValueError("ERR_DUPLICATE_CONTRIBUTOR_ID")

        link_ids = [lk.link_id for lk in self.links]
        if len(set(link_ids)) != len(link_ids):
            raise ValueError("ERR_DUPLICATE_LINK_ID")

        # Ensure all links reference known contributors in this collection
        known_contributors = set(c_ids)
        for link in self.links:
            if link.contributor_id not in known_contributors:
                raise ValueError(
                    f"ERR_UNKNOWN_CONTRIBUTOR: Link '{link.link_id}' references unknown "
                    f"contributor '{link.contributor_id}'."
                )

        return self


class ContributorAnalysisMethod(StrEnum):
    """Supported analytical methods for evaluating contributor association."""

    PEARSON = "PEARSON"
    SPEARMAN = "SPEARMAN"


class ContributorAnalysisRequest(BaseModel):
    """Explicit request to analyze a specific target/factor combination."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    factor_column: str = Field(..., min_length=1)
    analysis_method: ContributorAnalysisMethod
    minimum_observations: int = Field(
        default=3, ge=3, description="Minimum pairs required for statistical calculation."
    )
    timestamp_column: str | None = None
    source_reference: str | None = None
    parent_record_ids: tuple[str, ...] = Field(default_factory=tuple)


class ContributorAnalysisResult(BasePipelineResult):
    """Analytical result of a contributor evaluation. Does NOT imply causality."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    analysis_id: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    factor_column: str = Field(..., min_length=1)
    analysis_method: ContributorAnalysisMethod
    total_input_rows: int | None = None
    usable_observation_count: int | None = None
    excluded_observation_count: int | None = None
    coefficient: float | None = None
    p_value: float | None = None

    @field_validator("provenance")
    @classmethod
    def validate_provenance_epistemic_level(
        cls, v: ProvenanceRecord | None
    ) -> ProvenanceRecord | None:
        if v is not None and v.epistemic_level != EpistemicLevel.CALCULATED:
            raise ValueError(
                f"ERR_EPISTEMIC_MISMATCH: Analysis result provenance must have epistemic "
                f"level CALCULATED, got {v.epistemic_level.name}. Correlation/association "
                "is a calculation, not an observation or recommendation."
            )
        return v
