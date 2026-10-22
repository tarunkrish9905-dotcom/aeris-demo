"""
Vehicle Engineering Evaluation domain package.

Provides deterministic evaluation of flight observations and calculated results
against authoritative vehicle engineering limits and SOP procedure requirements.
"""

from __future__ import annotations

from aeris.core.vehicle_evaluation.engine import (
    VehicleEngineeringEvaluationEngine,
)
from aeris.core.vehicle_evaluation.exceptions import (
    AmbiguousApplicabilityError,
    EvaluationConfigurationError,
    EvaluationDomainError,
    IncompatibleEvaluationUnitError,
    UnknownEvaluationParameterError,
)
from aeris.core.vehicle_evaluation.models import (
    EvaluationObservation,
    EvaluationOutcome,
    LimitEvaluationRecord,
    ProcedureEvaluationRecord,
    VehicleEvaluationRequest,
    VehicleEvaluationResult,
)
from aeris.core.vehicle_evaluation.pipeline import (
    PipelineVehicleEvaluationRequest,
)

__all__ = [
    "AmbiguousApplicabilityError",
    "EvaluationConfigurationError",
    "EvaluationDomainError",
    "EvaluationObservation",
    "EvaluationOutcome",
    "IncompatibleEvaluationUnitError",
    "LimitEvaluationRecord",
    "PipelineVehicleEvaluationRequest",
    "ProcedureEvaluationRecord",
    "UnknownEvaluationParameterError",
    "VehicleEngineeringEvaluationEngine",
    "VehicleEvaluationRequest",
    "VehicleEvaluationResult",
]
