from pydantic import BaseModel, ConfigDict, Field

from aeris.core.evidence.models import EvidenceCollection
from aeris.core.provenance.tracker import ProvenanceRecord
from aeris.core.results.types import BasePipelineResult, Issue, ResultStatus


class StatisticalMetricSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    analysis_id: str
    metric_id: str


class EvidenceSourceSelection(BaseModel):
    """Explicit selection of a source for evidence construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    anomaly_ids: list[str] = Field(default_factory=list)
    engineering_result_ids: list[str] = Field(default_factory=list)
    statistical_metrics: list[StatisticalMetricSelection] = Field(default_factory=list)
    expected_observed_comparison_ids: list[str] = Field(default_factory=list)
    include_vehicle_evaluation: bool = Field(default=False)
    limit_evaluation_ids: list[str] = Field(default_factory=list)
    procedure_evaluation_ids: list[str] = Field(default_factory=list)
    include_configuration_discrepancies: bool = Field(default=False)


class PipelineEvidenceRequest(BaseModel):
    """Explicit request to construct evidence during the pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: str = Field(..., min_length=1)
    selections: EvidenceSourceSelection


class PipelineEvidenceResult(BasePipelineResult):
    """Result of the evidence construction stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection: EvidenceCollection | None = None
    status: ResultStatus
    issues: list[Issue] = Field(default_factory=list)
    provenance: ProvenanceRecord | None = None
