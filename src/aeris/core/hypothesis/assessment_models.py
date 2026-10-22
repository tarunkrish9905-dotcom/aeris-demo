from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aeris.core.evidence.models import EvidenceCollection
from aeris.core.hypothesis.models import Hypothesis, HypothesisAssessmentState
from aeris.core.results.types import BasePipelineResult


class HypothesisAssessmentConfiguration(BaseModel):
    """Explicit, immutable policy for evaluating a hypothesis against evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_total_evidence: int = Field(default=1, ge=0)
    minimum_supporting_evidence: int = Field(default=1, ge=0)
    strict_no_contradiction: bool = Field(default=True)
    treat_relevant_as_neutral: bool = Field(default=True)


class HypothesisAssessmentRequest(BaseModel):
    """Request to assess a hypothesis against a collection of evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis: Hypothesis
    evidence_collection: EvidenceCollection
    configuration: HypothesisAssessmentConfiguration = Field(
        default_factory=HypothesisAssessmentConfiguration
    )


class HypothesisAssessmentResult(BasePipelineResult):
    """The structured result of assessing a hypothesis against evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    hypothesis_id: str
    assessment_state: HypothesisAssessmentState
    supporting_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    contradicting_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    relevant_neutral_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    configuration_used: HypothesisAssessmentConfiguration
