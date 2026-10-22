"""
AERIS Recommendation Generation Domain Models.

Defines the configuration policy, generation request, and generation result
models for the deterministic, policy-controlled recommendation engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aeris.core.evidence.models import EvidenceCollection
from aeris.core.hypothesis.assessment_models import HypothesisAssessmentResult
from aeris.core.hypothesis.models import Hypothesis
from aeris.core.provenance.tracker import ProvenanceRecord
from aeris.core.recommendation.models import (
    Recommendation,
    RecommendationCollection,
    _validate_metadata_safeguards,
)
from aeris.core.results.types import BasePipelineResult, QualityResult, ValidationResult
from aeris.core.vehicle_evaluation.models import VehicleEvaluationResult


class RecommendationGenerationPolicy(BaseModel):
    """
    Immutable configuration governing which deterministic recommendation rules are enabled.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enable_data_verification_rule: bool = True
    enable_evidence_review_rule: bool = True
    enable_additional_evidence_rule: bool = True
    enable_engineering_review_rule: bool = True
    enable_repeat_analysis_rule: bool = False
    enable_configuration_review_rule: bool = False
    max_recommendations: int = Field(default=20, ge=1)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        return _validate_metadata_safeguards(v)


class RecommendationGenerationRequest(BaseModel):
    """
    Explicit, immutable contract supplying verified assessment and evidence inputs
    for recommendation generation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    hypothesis_assessment: HypothesisAssessmentResult | None = None
    hypothesis: Hypothesis | None = None
    evidence_collection: EvidenceCollection | None = None
    validation_result: ValidationResult | None = None
    quality_result: QualityResult | None = None
    vehicle_evaluation_result: VehicleEvaluationResult | None = None
    policy: RecommendationGenerationPolicy = Field(default_factory=RecommendationGenerationPolicy)
    investigation_id: str | None = Field(default=None, min_length=1)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("investigation_id")
    @classmethod
    def validate_optional_investigation_id(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("Investigation identifier cannot be empty or whitespace-only.")
        return v

    @model_validator(mode="after")
    def validate_at_least_one_input(self) -> RecommendationGenerationRequest:
        if not any(
            [
                self.hypothesis_assessment,
                self.hypothesis,
                self.evidence_collection,
                self.validation_result,
                self.quality_result,
                self.vehicle_evaluation_result,
            ]
        ):
            raise ValueError(
                "ERR_EMPTY_REQUEST: At least one analytical or assessment input must be provided."
            )
        return self

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        return _validate_metadata_safeguards(v)


class RecommendationGenerationResult(BasePipelineResult):
    """
    Structured, immutable result emitted by the RecommendationGenerationEngine.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    recommendations: tuple[Recommendation, ...] = Field(default_factory=tuple)
    recommendation_collection: RecommendationCollection
    rules_evaluated: tuple[str, ...] = Field(default_factory=tuple)
    rules_triggered: tuple[str, ...] = Field(default_factory=tuple)
    policy_used: RecommendationGenerationPolicy
    provenance: ProvenanceRecord
