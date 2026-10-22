"""
AERIS Recommendation Pipeline Integration Models.

Defines the explicit pipeline request models for integrating
the recommendation generation stage into DataPipelineOrchestrator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aeris.core.evidence.models import EvidenceCollection
from aeris.core.hypothesis.assessment_models import HypothesisAssessmentResult
from aeris.core.hypothesis.models import Hypothesis
from aeris.core.recommendation.generation_models import (
    RecommendationGenerationPolicy,
    RecommendationGenerationResult,
)
from aeris.core.recommendation.models import _validate_metadata_safeguards
from aeris.core.results.types import QualityResult, ValidationResult
from aeris.core.vehicle_evaluation.models import VehicleEvaluationResult


class PipelineRecommendationRequest(BaseModel):
    """
    Explicit pipeline request contract for controlled recommendation generation.

    Supports either directly supplied inputs or binding to upstream pipeline
    stage outcomes (validation, quality, evidence, vehicle evaluation).
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
    use_pipeline_validation: bool = False
    use_pipeline_quality: bool = False
    use_pipeline_evidence: bool = False
    use_pipeline_vehicle_evaluation: bool = False

    @field_validator("investigation_id")
    @classmethod
    def validate_optional_investigation_id(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("Investigation identifier cannot be empty or whitespace-only.")
        return v

    @model_validator(mode="after")
    def validate_inputs_present_or_bound(self) -> PipelineRecommendationRequest:
        has_direct_inputs = any(
            [
                self.hypothesis_assessment,
                self.hypothesis,
                self.evidence_collection,
                self.validation_result,
                self.quality_result,
                self.vehicle_evaluation_result,
            ]
        )
        has_pipeline_bindings = any(
            [
                self.use_pipeline_validation,
                self.use_pipeline_quality,
                self.use_pipeline_evidence,
                self.use_pipeline_vehicle_evaluation,
            ]
        )
        if not (has_direct_inputs or has_pipeline_bindings):
            raise ValueError(
                "ERR_EMPTY_REQUEST: At least one analytical or assessment input, "
                "or pipeline input binding flag, must be specified."
            )
        return self

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        return _validate_metadata_safeguards(v)


__all__ = [
    "PipelineRecommendationRequest",
    "RecommendationGenerationResult",
]
