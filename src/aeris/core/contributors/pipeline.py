from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aeris.core.contributors.models import ContributorAnalysisRequest, ContributorAnalysisResult
from aeris.core.results.types import BasePipelineResult


class PipelineContributorAnalysisRequest(BaseModel):
    """Explicit pipeline request containing multiple contributor analysis tasks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: tuple[ContributorAnalysisRequest, ...] = Field(default_factory=tuple)


class PipelineContributorAnalysisResult(BasePipelineResult):
    """Container for all contributor analysis results in the pipeline stage."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    results: tuple[ContributorAnalysisResult, ...] = Field(default_factory=tuple)
