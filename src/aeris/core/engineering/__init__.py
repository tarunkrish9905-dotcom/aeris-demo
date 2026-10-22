"""AERIS Core Engineering domain."""

from aeris.core.engineering.calculations import register_default_calculations
from aeris.core.engineering.executor import EngineeringExecutor
from aeris.core.engineering.models import (
    EngineeringCalculationDefinition,
    EngineeringCalculationResult,
    EngineeringInputReference,
    EngineeringResult,
)
from aeris.core.engineering.registry import get_calculation_registry

register_default_calculations()

__all__ = [
    "EngineeringCalculationDefinition",
    "EngineeringCalculationResult",
    "EngineeringExecutor",
    "EngineeringInputReference",
    "EngineeringResult",
    "get_calculation_registry",
]
