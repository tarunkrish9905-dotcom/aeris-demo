from typing import Protocol

import pandas as pd

from aeris.core.engineering.models import (
    EngineeringCalculationDefinition,
    EngineeringCalculationResult,
)


class CalculationImplementation(Protocol):
    def __call__(
        self,
        definition: EngineeringCalculationDefinition,
        data: pd.DataFrame,
        parent_record_ids: list[str] | None = None,
        source_uri: str | None = None,
    ) -> EngineeringCalculationResult: ...


class CalculationRegistry:
    def __init__(self) -> None:
        self._implementations: dict[str, CalculationImplementation] = {}

    def register(self, calculation_id: str, implementation: CalculationImplementation) -> None:
        self._implementations[calculation_id] = implementation

    def get(self, calculation_id: str) -> CalculationImplementation | None:
        return self._implementations.get(calculation_id)


_GLOBAL_REGISTRY = CalculationRegistry()


def get_calculation_registry() -> CalculationRegistry:
    return _GLOBAL_REGISTRY
