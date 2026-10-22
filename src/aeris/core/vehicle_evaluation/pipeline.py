"""
AERIS Vehicle Evaluation Pipeline Integration Models.

Defines the explicit pipeline request and configuration models for integrating
the Vehicle Engineering Evaluation stage into the DataPipelineOrchestrator.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aeris.core.limits.models import EngineeringLimitProfile
from aeris.core.registry import ParameterRegistry
from aeris.core.sop.models import ProcedureRequirementProfile
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.profile_models import VehicleProfile
from aeris.core.vehicle_evaluation.models import EvaluationObservation


class PipelineVehicleEvaluationRequest(BaseModel):
    """
    Explicit pipeline request contract for Vehicle Engineering Evaluation.

    Allows callers to supply authoritative vehicle profiles, configuration envelopes,
    engineering limit profiles, procedure profiles, and optional explicit observations
    for evaluation within the data pipeline orchestrator.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")
    __test__ = False

    vehicle_profile: VehicleProfile | None = Field(
        default=None,
        description="Authoritative VehicleProfile specifying vehicle identity and version.",
    )
    configuration_envelope: ConfigurationEnvelope | None = Field(
        default=None,
        description="Optional ConfigurationEnvelope defining active vehicle configuration.",
    )
    limit_profile: EngineeringLimitProfile | None = Field(
        default=None,
        description="Optional EngineeringLimitProfile defining applicable parameter limits.",
    )
    procedure_profile: ProcedureRequirementProfile | None = Field(
        default=None,
        description="Optional ProcedureRequirementProfile defining SOP requirements.",
    )
    test_card_id: str | None = Field(
        default=None,
        description="Optional test card context filtering applicability.",
    )
    test_point_id: str | None = Field(
        default=None,
        description="Optional test point context filtering applicability.",
    )
    context_id: str | None = Field(
        default=None,
        description="Optional operational test context ID.",
    )
    parameter_registry: ParameterRegistry | None = Field(
        default=None,
        description="Optional ParameterRegistry for canonical unit and parameter cross-checks.",
    )
    evaluate_limits: bool = Field(
        default=True,
        description="Whether to perform engineering limit evaluations.",
    )
    evaluate_procedures: bool = Field(
        default=True,
        description="Whether to perform procedure requirement evaluations.",
    )
    strict_units: bool = Field(
        default=True,
        description="Whether to reject unit mismatches as INCONCLUSIVE (True) vs error.",
    )
    explicit_observations: tuple[EvaluationObservation, ...] = Field(
        default_factory=tuple,
        description="Ordered tuple of explicit observations to evaluate.",
    )
    auto_extract_observations: bool = Field(
        default=True,
        description=(
            "Whether to automatically extract observations from the active pipeline dataset."
        ),
    )
    max_extracted_observations: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of observations to automatically extract from the dataset.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Request metadata.",
    )


__all__ = [
    "PipelineVehicleEvaluationRequest",
]
