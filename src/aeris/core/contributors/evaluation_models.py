from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aeris.core.contributors.models import ContributorAnalysisResult
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import BasePipelineResult


class AssociationStrength(StrEnum):
    """Categorical classification of statistical association strength."""

    VERY_WEAK = "VERY_WEAK"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


class RankingPolicy(StrEnum):
    """Supported deterministic ranking policies."""

    MAGNITUDE = "MAGNITUDE"


class ContributorEvaluationConfiguration(BaseModel):
    """Explicit deterministic thresholds for evaluating contributor association strength."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    very_weak_upper: float = Field(default=0.20, ge=0.0, le=1.0)
    weak_upper: float = Field(default=0.40, ge=0.0, le=1.0)
    moderate_upper: float = Field(default=0.60, ge=0.0, le=1.0)
    strong_upper: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_usable_observations: int = Field(default=3, ge=3)

    @model_validator(mode="after")
    def validate_thresholds(self) -> ContributorEvaluationConfiguration:
        """Ensure threshold boundaries are strictly increasing and valid."""
        if (
            math.isnan(self.very_weak_upper)
            or math.isnan(self.weak_upper)
            or math.isnan(self.moderate_upper)
            or math.isnan(self.strong_upper)
        ):
            raise ValueError("ERR_INVALID_CONFIG: Thresholds cannot be NaN.")

        if not (
            0.0
            <= self.very_weak_upper
            < self.weak_upper
            < self.moderate_upper
            < self.strong_upper
            <= 1.0
        ):
            raise ValueError(
                "ERR_INVALID_CONFIG: Thresholds must be strictly increasing between 0.0 and 1.0."
            )

        return self


class ContributorEvaluationRequest(BaseModel):
    """Explicit request to evaluate and optionally rank a set of analysis results."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    analysis_results: tuple[ContributorAnalysisResult, ...]
    configuration: ContributorEvaluationConfiguration = Field(
        default_factory=ContributorEvaluationConfiguration
    )
    request_ranking: bool = False
    ranking_policy: RankingPolicy = RankingPolicy.MAGNITUDE


class ContributorEvaluation(BasePipelineResult):
    """Evaluation outcome for a single contributor analysis result."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    analysis_id: str
    target_column: str
    factor_column: str
    coefficient: float | None = None
    association_strength: AssociationStrength | None = None
    rank_position: int | None = None
    usable_observation_count: int | None = None
    excluded_observation_count: int | None = None

    @field_validator("provenance")
    @classmethod
    def validate_provenance_epistemic_level(
        cls, v: ProvenanceRecord | None
    ) -> ProvenanceRecord | None:
        if v is not None and v.epistemic_level != EpistemicLevel.CALCULATED:
            raise ValueError(
                f"ERR_EPISTEMIC_MISMATCH: Evaluation result provenance must have epistemic "
                f"level CALCULATED, got {v.epistemic_level.name}. Evaluation is a calculation "
                "over calculations, NOT a causal inference."
            )
        return v


class ContributorEvaluationResult(BasePipelineResult):
    """Container for the full evaluation output, including ranking if requested."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    evaluations: tuple[ContributorEvaluation, ...] = Field(default_factory=tuple)
    is_ranked: bool = False
    ranking_policy: RankingPolicy | None = None
